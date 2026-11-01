"""OpenAI 호환 채팅 API(`/v1/chat/completions`)로 번역하는 엔진 (로컬 Qwen용).

llama.cpp 서버(`llama-server`)처럼 OpenAI 호환 API를 제공하는 로컬 추론 서버에 HTTP로 요청한다.
모델별 실행 방식은 서버가 맡고, 이 어댑터는 프롬프트 구성, 출력 검증, 오류 처리만 맡는다.

- 여러 원문은 한 요청에 묶어 보내고, JSON 스키마로 "원문 수와 같은 개수의 문자열 배열"을 강제한다.
  서버가 스키마를 지키지 않을 수 있으므로 받은 결과도 다시 검증한다.
- 표준 라이브러리(urllib)만 쓴다. 요청은 중간에 멈출 수 없으므로 취소는 요청 전후에 확인한다.
- 로그에는 원문·번역문을 남기지 않는다 (개수, 시간, 토큰 수만).
"""

import json
import logging
import re
import socket
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from translator.core.cancellation import CancellationToken
from translator.core.errors import TranslationError
from translator.core.interfaces import Translator
from translator.core.models import TranslationRequest, TranslationResult
from translator.tuning import (
    LLM_BATCH_MAX_LINES,
    LLM_HEALTH_POLL_SECONDS,
    LLM_HEALTH_TIMEOUT_SECONDS,
    LLM_MAX_OUTPUT_TOKENS_PER_LINE,
    LLM_MAX_RETRIES,
    LLM_OUTPUT_LENGTH_RATIO,
    LLM_OUTPUT_MIN_CHARS,
    LLM_REQUEST_TIMEOUT_SECONDS,
    LLM_STARTUP_TIMEOUT_SECONDS,
    LLM_TEMPERATURE,
)

logger = logging.getLogger(__name__)

# 사용자용 오류 메시지 (번역 대상 텍스트를 담지 않는다)
SERVER_UNAVAILABLE_MESSAGE = "번역 서버에 연결할 수 없습니다. 번역 모델 설정과 로그 파일을 확인해 주세요."
SERVER_NOT_READY_MESSAGE = "번역 모델이 준비되지 않았습니다 (불러오는 데 시간이 너무 오래 걸림). 로그 파일을 확인해 주세요."
TIMEOUT_MESSAGE = "번역 시간이 너무 오래 걸려 중단했습니다. 잠시 뒤 다시 시도해 주세요."
INVALID_OUTPUT_MESSAGE = "번역 모델이 올바른 형식으로 답하지 않았습니다. 다시 시도해 주세요."
SERVER_ERROR_MESSAGE = "번역 서버가 오류를 돌려주었습니다. 로그 파일을 확인해 주세요."

# 언어 코드 → 프롬프트에 쓸 언어 이름
_LANGUAGE_NAMES = {
    "ko": "Korean",
    "en": "English",
    "ja": "Japanese",
    "zh": "Chinese",
}
# 원문 언어를 모를 때(auto) 프롬프트에 쓸 표현
_AUTO_SOURCE = "the source language"
# 한 줄 출력 외에 JSON 감싸기 등에 쓰는 여유 토큰 수 (JSON 형식이 정한 고정 비용)
_JSON_OVERHEAD_TOKENS = 32
# 문자 종류 판별 (번역 언어 검증용)
_HANGUL = re.compile(r"[가-힣ᄀ-ᇿ㄰-㆏]")
_LETTER = re.compile(r"[^\W\d_]")
# 한국어 번역에 원문에 없던 이 문자가 섞이면 다른 언어로 답한 것으로 본다 (키릴 문자, 한자, 가나).
_FOREIGN_RANGES = {
    "cyrillic": "Ѐ-ӿ",
    "han": "㐀-䶿一-鿿",
    "kana": "぀-ヿ",
}
_FOREIGN_SCRIPTS = {name: re.compile(f"[{chars}]") for name, chars in _FOREIGN_RANGES.items()}

# 말투·강도·UI·고유명사 처리를 보여 주는 예시 (직접 작성, 평가 문장과 겹치지 않음).
# 영어 → 한국어일 때만 넣는다. 모델이 욕설을 순화하거나 존댓말·반말을 섞는 문제를 줄이기 위함.
_EN_KO_EXAMPLES = (
    (
        ["Hold the line, you useless bastards!", "My lord, the scouts have returned.",
         "Inventory full", "Ask Marcus about the Crimson Pact."],
        ["버텨, 이 쓸모없는 놈들아!", "영주님, 정찰병들이 돌아왔습니다.",
         "인벤토리 가득 참", "마커스한테 진홍 협약에 대해 물어봐."],
    ),
)


def build_system_prompt(source_language: str, target_language: str) -> str:
    """번역 지시문을 만든다.

    매개변수:
        source_language: 원문 언어 코드 (`auto`면 언어를 지정하지 않는다).
        target_language: 번역 언어 코드.
    반환값:
        시스템 프롬프트 문자열.
    """
    source = _LANGUAGE_NAMES.get(source_language, _AUTO_SOURCE)
    target = _LANGUAGE_NAMES.get(target_language, target_language)
    return (
        f"You translate on-screen text from games and videos ({source}) into natural {target}.\n"
        "Rules:\n"
        "- Translate every item independently and keep the order. Return exactly one translation per item.\n"
        "- Keep the speaker's tone, politeness level, emotion and intensity. Do not soften, censor or "
        "tone down insults, slang or profanity.\n"
        f"- Keep names and proper nouns consistent; transliterate them into {target} unless they are "
        "commonly left as is.\n"
        "- Short UI labels and menu items stay short.\n"
        "- Output only the translations. No explanations, notes or romanization.\n"
        "- Context lines, if given, are earlier dialogue for reference only. Do not translate them."
    )


def build_user_prompt(texts: Sequence[str], context: Sequence[str]) -> str:
    """번역할 항목(번호 붙인 JSON)과 참고 문맥을 담은 사용자 메시지를 만든다.

    매개변수:
        texts: 번역할 원문들.
        context: 참고용 이전 대사 (번역하지 않음). 비어 있으면 넣지 않는다.
    반환값:
        사용자 메시지 문자열.
    """
    payload: dict[str, Any] = {"items": list(texts)}
    if context:
        payload = {"context": list(context), **payload}
    return (
        f"Translate the {len(texts)} item(s) below. "
        'Answer as JSON: {"translations": [...]} with the same number of strings.\n'
        + json.dumps(payload, ensure_ascii=False)
    )


def build_messages(
    texts: Sequence[str], source_language: str, target_language: str, context: Sequence[str]
) -> list[dict[str, str]]:
    """채팅 메시지 목록을 만든다: 지시문, (영어 → 한국어면) 예시 대화, 번역할 항목.

    매개변수:
        texts: 번역할 원문들.
        source_language: 원문 언어 코드.
        target_language: 번역 언어 코드.
        context: 참고용 이전 대사.
    반환값:
        OpenAI 호환 `messages` 목록.
    """
    messages = [{"role": "system", "content": build_system_prompt(source_language, target_language)}]
    if source_language in ("en", "auto") and target_language == "ko":
        for sources, targets in _EN_KO_EXAMPLES:
            messages.append({"role": "user", "content": build_user_prompt(sources, ())})
            messages.append(
                {"role": "assistant", "content": json.dumps({"translations": targets}, ensure_ascii=False)}
            )
    messages.append({"role": "user", "content": build_user_prompt(texts, context)})
    return messages


def max_translation_chars(texts: Sequence[str]) -> int:
    """번역문 하나에 허용할 최대 글자 수 (가장 긴 원문 기준).

    모델이 한 문자열을 끝내지 못하고 계속 생성하는 경우(실측: 30초 넘게 생성한 뒤 서버 오류)를
    생성 단계에서 빨리 끊기 위함이다.
    """
    longest = max((len(text) for text in texts), default=0)
    return max(LLM_OUTPUT_MIN_CHARS, round(longest * LLM_OUTPUT_LENGTH_RATIO))


def translation_pattern(texts: Sequence[str], target_language: str) -> str:
    """번역문 하나가 따라야 할 JSON 스키마 문자열 패턴 (서버의 제약 디코딩용).

    - 길이를 `max_translation_chars`로 제한한다.
    - 한국어 번역에서 원문에 없던 키릴 문자·한자·가나를 막는다 (Qwen3.5가 한자를 섞는 문제).
      원문에 그 문자가 있으면(고유명사 등) 막지 않는다.

    매개변수:
        texts: 원문들.
        target_language: 번역 언어 코드.
    반환값:
        정규식 패턴.
    """
    limit = max_translation_chars(texts)
    ranges: list[str] = []
    if target_language == "ko":
        joined = "".join(texts)
        ranges = [
            _FOREIGN_RANGES[name] for name, pattern in _FOREIGN_SCRIPTS.items() if not pattern.search(joined)
        ]
    # 막을 문자가 없으면 아무 문자나 허용한다.
    allowed = f"[^{''.join(ranges)}]" if ranges else "."
    return f"^{allowed}{{0,{limit}}}$"


def response_schema(count: int, pattern: str | None = None) -> dict[str, Any]:
    """정확히 `count`개의 문자열을 요구하는 JSON 스키마 (서버의 제약 디코딩용).

    매개변수:
        count: 번역문 개수.
        pattern: 각 번역문이 따라야 할 정규식 (없으면 None).
    """
    item: dict[str, Any] = {"type": "string"}
    if pattern is not None:
        item["pattern"] = pattern
    return {
        "type": "object",
        "properties": {
            "translations": {
                "type": "array",
                "items": item,
                "minItems": count,
                "maxItems": count,
            }
        },
        "required": ["translations"],
    }


def is_already_in_language(text: str, target_language: str) -> bool:
    """원문이 이미 번역 언어로 쓰였는지 (지금은 한국어만: 글자의 절반 이상이 한글).

    바탕 화면의 한국어 UI처럼 번역할 필요가 없는 텍스트를 모델에 보내지 않기 위함이다.
    """
    if target_language != "ko":
        return False
    letters = _LETTER.findall(text)
    return bool(letters) and sum(1 for letter in letters if _HANGUL.match(letter)) * 2 >= len(letters)


def script_problem(source: str, translation: str, target_language: str) -> str | None:
    """번역문이 번역 언어로 쓰였는지 확인한다 (지금은 한국어만 검사).

    매개변수:
        source: 원문.
        translation: 번역문.
        target_language: 번역 언어 코드.
    반환값:
        문제가 있으면 그 이유, 없으면 None.
    """
    if target_language != "ko":
        return None
    for name, pattern in _FOREIGN_SCRIPTS.items():
        if pattern.search(translation) and not pattern.search(source):
            return f"원문에 없던 문자({name})가 섞였습니다"
    # 단어 하나(이름·약어)는 원문 그대로 둘 수 있다. 문장을 그대로 돌려주면 번역하지 않은 것으로 본다.
    kept_as_is = translation == source and len(source.split()) == 1
    if _LETTER.search(source) and not _HANGUL.search(translation) and not kept_as_is:
        return "한글이 없습니다"
    return None


def parse_translations(content: str, count: int) -> list[str]:
    """모델 답변에서 번역문 목록을 꺼내 검증한다.

    매개변수:
        content: 모델이 돌려준 메시지 본문 (JSON 문자열이어야 한다).
        count: 기대하는 번역문 개수.
    반환값:
        앞뒤 공백을 지운 번역문 목록.
    예외:
        ValueError: JSON이 아니거나, 개수가 다르거나, 문자열이 아니거나, 빈 번역이 있을 때.
    """
    data = json.loads(content)
    if not isinstance(data, dict) or not isinstance(data.get("translations"), list):
        raise ValueError("translations 목록이 없습니다.")
    items = data["translations"]
    if len(items) != count:
        raise ValueError(f"번역 개수가 다릅니다: {len(items)} (기대 {count})")
    if not all(isinstance(item, str) for item in items):
        raise ValueError("문자열이 아닌 번역이 있습니다.")
    translations = [item.strip() for item in items]
    if any(not item for item in translations):
        raise ValueError("빈 번역이 있습니다.")
    return translations


@dataclass(frozen=True)
class ChatServerSettings:
    """번역 서버 연결 설정 (숫자 조정값은 tuning.py)."""

    # 서버 주소 (예: "http://127.0.0.1:8080"). `/v1/chat/completions`를 붙여 요청한다.
    base_url: str
    # 요청에 넣을 모델 이름 (llama.cpp 서버는 불러온 모델 하나만 쓰므로 표시용)
    model: str = "qwen"
    # 결과에 기록할 엔진 이름
    engine_name: str = "qwen"
    # Qwen3.5 같은 하이브리드 모델의 생각(thinking) 모드를 끈다 (지연 감소). 지원하지 않는 모델은 무시한다.
    disable_thinking: bool = True
    # 요청의 원문 언어가 "auto"일 때 대신 쓸 언어 (사용자 결정: 영어 → 한국어)
    default_source_language: str = "en"


# HTTP 요청 함수 형태: (URL, 본문 바이트 또는 None, 시간 제한 초) → (상태 코드, 응답 바이트)
HttpCall = Callable[[str, bytes | None, float], tuple[int, bytes]]


def urllib_call(url: str, body: bytes | None, timeout: float) -> tuple[int, bytes]:
    """urllib로 HTTP 요청을 보낸다. 본문이 있으면 JSON POST, 없으면 GET.

    매개변수:
        url: 요청 주소.
        body: JSON 본문 바이트 (None이면 GET).
        timeout: 시간 제한 (초).
    반환값:
        (HTTP 상태 코드, 응답 본문). 4xx·5xx도 예외 없이 상태 코드로 돌려준다.
    예외:
        urllib.error.URLError, TimeoutError, OSError: 연결 실패나 시간 초과.
    """
    request = urllib.request.Request(
        url, data=body, method="POST" if body is not None else "GET",
        headers={"Content-Type": "application/json"} if body is not None else {},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 (설정한 서버 주소만 사용)
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


class ChatCompletionTranslator(Translator):
    """OpenAI 호환 채팅 API로 번역한다. 여러 워커 스레드에서 호출해도 안전하다 (상태는 준비 여부뿐)."""

    def __init__(
        self,
        settings: ChatServerSettings,
        http_call: HttpCall = urllib_call,
        is_server_alive: Callable[[], bool] | None = None,
        batch_max_lines: int = LLM_BATCH_MAX_LINES,
    ) -> None:
        """서버 설정, HTTP 요청 함수(테스트에서 바꾼다), 서버 프로세스 생존 확인 함수를 받는다.

        매개변수:
            settings: 서버 주소와 모델 정보.
            http_call: HTTP 요청 함수.
            is_server_alive: 앱이 띄운 서버 프로세스가 살아 있는지 확인하는 함수. 서버가 죽었으면
                준비를 기다리지 않고 바로 오류를 낸다. 외부 서버면 None.
            batch_max_lines: 한 요청에 묶을 최대 원문 수.
        """
        # 서버 연결 설정
        self._settings = settings
        # HTTP 요청 함수
        self._http_call = http_call
        # 앱이 띄운 서버 프로세스의 생존 확인 함수 (외부 서버면 None)
        self._is_server_alive = is_server_alive
        # 한 요청에 묶을 최대 원문 수
        self._batch_max_lines = max(1, batch_max_lines)
        # 서버가 요청을 받을 준비가 되었는지 (연결 실패가 나면 다시 확인한다)
        self._ready = False

    @property
    def settings(self) -> ChatServerSettings:
        """서버 연결 설정."""
        return self._settings

    def translate(
        self, request: TranslationRequest, cancellation: CancellationToken
    ) -> TranslationResult:
        """원문 하나를 번역한다 (`translate_batch`에 하나만 넘긴다)."""
        return self.translate_batch([request], cancellation)[0]

    def translate_batch(
        self, requests: Sequence[TranslationRequest], cancellation: CancellationToken
    ) -> list[TranslationResult]:
        """여러 원문을 묶음 단위로 번역해 같은 순서로 돌려준다.

        언어와 문맥이 같은 연속 요청을 최대 `batch_max_lines`개씩 묶어 한 번에 보낸다.

        매개변수:
            requests: 번역 요청들.
            cancellation: 취소 토큰. 묶음마다 요청 전후에 확인한다.
        반환값:
            요청과 같은 순서의 번역 결과.
        예외:
            OperationCancelledError: 취소되었을 때.
            TranslationError: 서버 연결 실패, 시간 초과, 잘못된 출력이 다시 시도해도 계속될 때.
        """
        # 원문 순서대로 채울 번역문 (이미 번역 언어인 원문은 그대로 둔다)
        texts: list[str | None] = [
            request.source_text if is_already_in_language(request.source_text, request.target_language) else None
            for request in requests
        ]
        pending = [index for index, text in enumerate(texts) if text is None]
        if len(pending) < len(requests):
            logger.debug("이미 번역 언어인 원문 %d개는 번역하지 않습니다.", len(requests) - len(pending))
        for group in self._groups([(index, requests[index]) for index in pending]):
            cancellation.raise_if_cancelled()
            first = group[0][1]
            # 원문 언어를 모르면(auto) 설정한 기본 원문 언어로 본다.
            source_language = (
                self._settings.default_source_language if first.source_language == "auto" else first.source_language
            )
            translations = self._translate_group(
                [request.source_text for _, request in group],
                source_language,
                first.target_language,
                first.context,
                cancellation,
            )
            for (index, _), text in zip(group, translations):
                texts[index] = text
        return [
            TranslationResult(request.source_text, text, self._settings.engine_name)
            for request, text in zip(requests, texts)
        ]

    def _groups(
        self, requests: Sequence[tuple[int, TranslationRequest]]
    ) -> list[list[tuple[int, TranslationRequest]]]:
        """언어·문맥이 같은 연속 요청을 최대 크기 묶음으로 나눈다 (원래 순서 번호를 함께 둔다)."""
        groups: list[list[tuple[int, TranslationRequest]]] = []
        for item in requests:
            request = item[1]
            key = (request.source_language, request.target_language, request.context)
            last = groups[-1] if groups else None
            if last is not None and len(last) < self._batch_max_lines:
                head = last[0][1]
                if (head.source_language, head.target_language, head.context) == key:
                    last.append(item)
                    continue
            groups.append([item])
        return groups

    def _translate_group(
        self,
        texts: list[str],
        source_language: str,
        target_language: str,
        context: tuple[str, ...],
        cancellation: CancellationToken,
    ) -> list[str]:
        """한 묶음을 번역한다. 번역 언어가 맞지 않는 줄은 그 줄만 한 번 더 번역해 본다.

        그래도 맞지 않는 줄은 원문을 그대로 쓴다 (파일 경로·코드처럼 번역할 수 없는 텍스트 때문에
        화면 전체 번역이 실패하지 않게 하기 위함). 그런 줄의 수만 로그에 남긴다.
        """
        translations = self._request_translations(texts, source_language, target_language, context, cancellation)
        bad = [
            index for index, (source, text) in enumerate(zip(texts, translations))
            if script_problem(source, text, target_language) is not None
        ]
        if not bad:
            return translations
        logger.warning("번역 언어가 맞지 않는 %d/%d줄을 따로 다시 번역합니다.", len(bad), len(texts))
        retried = self._request_translations(
            [texts[index] for index in bad], source_language, target_language, context, cancellation
        )
        kept = 0
        for index, text in zip(bad, retried):
            if script_problem(texts[index], text, target_language) is None:
                translations[index] = text
            else:
                translations[index] = texts[index]
                kept += 1
        if kept:
            logger.warning("번역 언어가 계속 맞지 않는 %d줄은 원문을 그대로 표시합니다.", kept)
        return translations

    def _request_translations(
        self,
        texts: list[str],
        source_language: str,
        target_language: str,
        context: tuple[str, ...],
        cancellation: CancellationToken,
    ) -> list[str]:
        """번역 요청을 보내 형식이 올바른 번역문 목록을 받는다. 형식 오류는 정해진 횟수만큼만 다시 시도한다."""
        self._wait_until_ready(cancellation)
        body = json.dumps(self._request_body(texts, source_language, target_language, context)).encode()
        attempts = LLM_MAX_RETRIES + 1
        for attempt in range(1, attempts + 1):
            cancellation.raise_if_cancelled()
            # 요청 시작 시각 (지연 기록용)
            started = time.perf_counter()
            status, payload = self._post(body)
            elapsed_ms = (time.perf_counter() - started) * 1000
            # 요청이 끝나는 동안 취소되었으면 결과를 버린다.
            cancellation.raise_if_cancelled()
            if status != 200:
                logger.error(
                    "번역 서버 응답 %d (%d/%d회, %.0fms): %s",
                    status, attempt, attempts, elapsed_ms, _error_summary(payload),
                )
                if attempt == attempts:
                    raise TranslationError(SERVER_ERROR_MESSAGE)
                continue
            try:
                content, usage = _message_content(payload)
                translations = parse_translations(content, len(texts))
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                logger.warning(
                    "번역 출력이 올바르지 않습니다 (%d/%d회, 종료 사유 %s): %s",
                    attempt, attempts, _finish_reason(payload), exc,
                )
                if attempt == attempts:
                    raise TranslationError(INVALID_OUTPUT_MESSAGE) from exc
                continue
            logger.debug(
                "번역 %d줄, %.0fms, 토큰 입력 %s / 출력 %s",
                len(texts), elapsed_ms, usage.get("prompt_tokens"), usage.get("completion_tokens"),
            )
            return translations
        raise AssertionError("도달할 수 없는 코드")  # pragma: no cover

    def _request_body(
        self, texts: list[str], source_language: str, target_language: str, context: tuple[str, ...]
    ) -> dict[str, Any]:
        """채팅 API 요청 본문을 만든다."""
        body: dict[str, Any] = {
            "model": self._settings.model,
            "messages": build_messages(texts, source_language, target_language, context),
            "temperature": LLM_TEMPERATURE,
            "max_tokens": max_output_tokens(len(texts)),
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "translations",
                    "schema": response_schema(len(texts), translation_pattern(texts, target_language)),
                },
            },
            "stream": False,
        }
        if self._settings.disable_thinking:
            # llama.cpp 서버(--jinja)가 채팅 템플릿에 넘기는 값. 하이브리드 모델(Qwen3.5)의 생각 단계를 끈다.
            body["chat_template_kwargs"] = {"enable_thinking": False}
        return body

    def _post(self, body: bytes) -> tuple[int, bytes]:
        """채팅 API에 요청한다. 연결 실패·시간 초과는 `TranslationError`로 바꾼다."""
        url = self._settings.base_url.rstrip("/") + "/v1/chat/completions"
        try:
            return self._http_call(url, body, LLM_REQUEST_TIMEOUT_SECONDS)
        except (TimeoutError, socket.timeout) as exc:
            logger.error("번역 요청 시간 초과 (%ss)", LLM_REQUEST_TIMEOUT_SECONDS)
            raise TranslationError(TIMEOUT_MESSAGE) from exc
        except (urllib.error.URLError, OSError) as exc:
            # 서버가 꺼졌을 수 있으므로 다음 요청에서 준비 여부를 다시 확인한다.
            self._ready = False
            logger.error("번역 서버 연결 실패: %s", type(exc).__name__)
            raise TranslationError(SERVER_UNAVAILABLE_MESSAGE) from exc

    def _wait_until_ready(self, cancellation: CancellationToken) -> None:
        """서버가 요청을 받을 수 있을 때까지 기다린다 (모델을 불러오는 중이면 기다림).

        예외:
            OperationCancelledError: 기다리는 동안 취소되었을 때.
            TranslationError: 서버 프로세스가 죽었거나 시간 안에 준비되지 않았을 때.
        """
        if self._ready:
            return
        url = self._settings.base_url.rstrip("/") + "/health"
        # 준비를 기다릴 마감 시각 (단조 시계)
        deadline = time.monotonic() + LLM_STARTUP_TIMEOUT_SECONDS
        while True:
            cancellation.raise_if_cancelled()
            if self._is_server_alive is not None and not self._is_server_alive():
                logger.error("번역 서버 프로세스가 실행 중이 아닙니다.")
                raise TranslationError(SERVER_UNAVAILABLE_MESSAGE)
            try:
                status, _ = self._http_call(url, None, LLM_HEALTH_TIMEOUT_SECONDS)
            except (urllib.error.URLError, OSError, TimeoutError):
                status = 0  # 아직 서버가 포트를 열지 않았다.
            if status == 200:
                self._ready = True
                return
            if time.monotonic() >= deadline:
                logger.error(
                    "번역 서버가 %ss 안에 준비되지 않았습니다 (마지막 상태 %d).",
                    LLM_STARTUP_TIMEOUT_SECONDS, status,
                )
                raise TranslationError(SERVER_NOT_READY_MESSAGE)
            cancellation.wait(LLM_HEALTH_POLL_SECONDS)


def _error_summary(payload: bytes) -> str:
    """서버 오류 응답의 종류와 짧은 설명 (진단용). 서버 오류 설명에는 요청 내용이 담기지 않는다."""
    try:
        error = json.loads(payload).get("error", {})
        return f"{error.get('type')}: {str(error.get('message'))[:120]}"
    except (ValueError, AttributeError):
        return "(형식을 알 수 없는 오류 응답)"


def _finish_reason(payload: bytes) -> str:
    """응답의 생성 종료 사유 (예: stop, length). 알 수 없으면 "?" (진단용, 내용은 담지 않음)."""
    try:
        return str(json.loads(payload)["choices"][0].get("finish_reason"))
    except (ValueError, KeyError, IndexError, TypeError, AttributeError):
        return "?"


def _message_content(payload: bytes) -> tuple[str, dict[str, Any]]:
    """채팅 API 응답에서 메시지 본문과 토큰 사용량을 꺼낸다.

    예외:
        ValueError, KeyError, IndexError, TypeError: 응답 형식이 예상과 다를 때.
    """
    data = json.loads(payload)
    content = data["choices"][0]["message"]["content"]
    if not isinstance(content, str):
        raise TypeError("메시지 본문이 문자열이 아닙니다.")
    usage = data.get("usage") or {}
    return content, usage if isinstance(usage, dict) else {}


def max_output_tokens(count: int) -> int:
    """`count`줄 묶음에 허용할 최대 출력 토큰 수 (줄마다 상한 + JSON 감싸기 여유)."""
    return count * LLM_MAX_OUTPUT_TOKENS_PER_LINE + _JSON_OVERHEAD_TOKENS
