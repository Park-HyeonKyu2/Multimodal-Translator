"""OpenAI 호환 채팅 API 번역 엔진 테스트: 가짜 HTTP로 요청 구성, 출력 검증, 오류·재시도·서버 대기를 확인한다.

실제 서버나 모델 없이 실행한다. 실제 Qwen 확인은 `test_qwen_real.py`.
"""

import json
import logging
import re
import threading
import urllib.error

import pytest

from translator.adapters.translation import chat_completion
from translator.adapters.translation.chat_completion import (
    INVALID_OUTPUT_MESSAGE,
    SERVER_ERROR_MESSAGE,
    SERVER_NOT_READY_MESSAGE,
    SERVER_UNAVAILABLE_MESSAGE,
    TIMEOUT_MESSAGE,
    ChatCompletionTranslator,
    ChatServerSettings,
    max_translation_chars,
    parse_translations,
    script_problem,
    translation_pattern,
)
from translator.application.pipeline import TranslationPipeline
from translator.core.cancellation import CancellationToken
from translator.core.errors import OperationCancelledError, TranslationError
from translator.core.models import TranslationRequest

# 로그에 남으면 안 되는 예시 원문
SECRET = "Secret line for the log test"
BASE = "http://127.0.0.1:9999"


def reply(translations: list[str] | str, finish: str = "stop") -> tuple[int, bytes]:
    """채팅 API 성공 응답을 만든다. 문자열을 주면 메시지 본문 그대로 쓴다."""
    content = translations if isinstance(translations, str) else json.dumps({"translations": translations})
    body = {
        "choices": [{"message": {"content": content}, "finish_reason": finish}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    return 200, json.dumps(body).encode()


class FakeServer:
    """가짜 HTTP 함수: /health는 `health` 순서대로, 채팅 요청은 `replies` 순서대로 답하고 요청을 기록한다."""

    def __init__(self, replies: list, health: list | None = None) -> None:
        """돌려줄 채팅 응답(또는 던질 예외)과 /health 응답 목록을 받는다."""
        # 채팅 요청에 차례로 돌려줄 (상태, 본문) 또는 예외
        self.replies = list(replies)
        # /health 요청에 차례로 돌려줄 상태 코드 또는 예외 (마지막 값을 계속 쓴다)
        self.health = list(health or [200])
        # 받은 채팅 요청 본문 (파싱한 JSON)
        self.requests: list[dict] = []
        # /health 요청 수
        self.health_calls = 0

    def __call__(self, url: str, body: bytes | None, timeout: float) -> tuple[int, bytes]:
        """요청을 기록하고 정해 둔 응답을 돌려준다."""
        if url.endswith("/health"):
            self.health_calls += 1
            value = self.health.pop(0) if len(self.health) > 1 else self.health[0]
            if isinstance(value, BaseException):
                raise value
            return value, b"{}"
        assert url == BASE + "/v1/chat/completions"
        self.requests.append(json.loads(body))
        value = self.replies.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value


def make(server: FakeServer, **kwargs) -> ChatCompletionTranslator:
    """가짜 서버를 쓰는 번역 엔진을 만든다."""
    return ChatCompletionTranslator(ChatServerSettings(BASE, engine_name="qwen-test"), http_call=server, **kwargs)


def sent_items(body: dict) -> list[str]:
    """요청 본문의 마지막 사용자 메시지에 담긴 번역 항목 (첫 줄은 지시문, 둘째 줄은 JSON)."""
    return json.loads(body["messages"][-1]["content"].split("\n", 1)[1])["items"]


def requests_of(*texts: str, source: str = "auto") -> list[TranslationRequest]:
    """원문들로 번역 요청 목록을 만든다."""
    return [TranslationRequest(text, source_language=source) for text in texts]


def test_batch_is_sent_in_one_request_with_constrained_schema() -> None:
    """여러 원문을 한 요청으로 보내고, 개수·문자를 제한한 스키마와 생각 끄기를 함께 보낸다."""
    server = FakeServer([reply(["비켜!", "저장"])])

    results = make(server).translate_batch(requests_of("Move!", "Save"), CancellationToken())

    assert [(r.source_text, r.translated_text, r.engine) for r in results] == [
        ("Move!", "비켜!", "qwen-test"),
        ("Save", "저장", "qwen-test"),
    ]
    [body] = server.requests
    schema = body["response_format"]["json_schema"]["schema"]["properties"]["translations"]
    assert schema["minItems"] == schema["maxItems"] == 2
    assert schema["items"]["pattern"] == translation_pattern(["Move!", "Save"], "ko")
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["messages"][0]["role"] == "system"
    # 원문 언어를 모르면 기본 원문 언어(영어)로 보고, 영어 → 한국어 예시를 넣는다.
    assert "(English)" in body["messages"][0]["content"]
    assert [m["role"] for m in body["messages"][1:]] == ["user", "assistant", "user"]
    assert json.loads(body["messages"][-1]["content"].split("\n", 1)[1]) == {"items": ["Move!", "Save"]}


def test_large_batches_are_split() -> None:
    """한 요청의 최대 원문 수를 넘으면 나눠 보낸다 (순서는 유지)."""
    server = FakeServer([reply(["가", "나"]), reply(["다"])])

    results = make(server, batch_max_lines=2).translate_batch(requests_of("a", "b", "c"), CancellationToken())

    assert [r.translated_text for r in results] == ["가", "나", "다"]
    assert [len(json.loads(b["messages"][-1]["content"].split("\n", 1)[1])["items"]) for b in server.requests] == [2, 1]


def test_context_is_sent_but_not_translated() -> None:
    """문맥(이전 대사)은 참고용으로 함께 보내고 번역 결과에는 포함하지 않는다 (Context Memory 대비)."""
    server = FakeServer([reply(["좋아."])])
    request = TranslationRequest("Fine.", source_language="en", context=("Are you ready?",))

    [result] = make(server).translate_batch([request], CancellationToken())

    payload = json.loads(server.requests[0]["messages"][-1]["content"].split("\n", 1)[1])
    assert payload == {"context": ["Are you ready?"], "items": ["Fine."]}
    assert result.translated_text == "좋아."


@pytest.mark.parametrize(
    "bad",
    [
        reply("not json"),
        reply(["하나만"]),  # 개수가 다름
        reply(["", "둘"]),  # 빈 번역
        (200, b'{"choices": []}'),  # 응답 형식 오류
    ],
)
def test_invalid_output_is_retried_once_then_reported(bad) -> None:
    """출력 형식이 잘못되면 한 번만 다시 시도하고, 그래도 잘못되면 형식 오류를 알린다 (무한 재시도 없음)."""
    server = FakeServer([bad, bad])

    with pytest.raises(TranslationError, match=re.escape(INVALID_OUTPUT_MESSAGE)):
        make(server).translate_batch(requests_of("Get out", "Save"), CancellationToken())

    assert len(server.requests) == 2


def test_wrong_language_line_is_retried_alone() -> None:
    """번역 언어가 맞지 않는 줄(원문에 없던 한자)은 그 줄만 따로 다시 번역해 바꾼다."""
    server = FakeServer([reply(["闭嘴!", "저장"]), reply(["닥쳐!"])])

    results = make(server).translate_batch(requests_of("Shut up!", "Save"), CancellationToken())

    assert [r.translated_text for r in results] == ["닥쳐!", "저장"]
    retry_items = sent_items(server.requests[1])
    assert retry_items == ["Shut up!"]


def test_untranslatable_line_keeps_source_without_failing_batch(caplog: pytest.LogCaptureFixture) -> None:
    """다시 번역해도 언어가 맞지 않는 줄(예: 파일 경로)은 원문을 그대로 쓰고, 묶음의 다른 번역은 살린다."""
    path = "C:/Games/My Save Folder"
    server = FakeServer([reply([path, "저장"]), reply([path])])

    results = make(server).translate_batch(requests_of(path, "Save"), CancellationToken())

    assert [r.translated_text for r in results] == [path, "저장"]
    assert "원문을 그대로 표시합니다" in caplog.text


def test_text_already_in_korean_is_not_sent() -> None:
    """이미 한국어인 원문(바탕 화면의 한국어 UI 등)은 모델에 보내지 않고 그대로 쓴다."""
    server = FakeServer([reply(["저장"])])

    results = make(server).translate_batch(requests_of("설정 열기", "Save", "OK 확인"), CancellationToken())

    assert [r.translated_text for r in results] == ["설정 열기", "저장", "OK 확인"]
    sent = sent_items(server.requests[0])
    assert sent == ["Save"]


def test_only_korean_texts_need_no_server() -> None:
    """모두 한국어면 서버에 요청하지 않는다 (서버 준비도 확인하지 않음)."""
    server = FakeServer([], health=[503])

    results = make(server).translate_batch(requests_of("확인", "취소"), CancellationToken())

    assert [r.translated_text for r in results] == ["확인", "취소"]
    assert server.health_calls == 0


def test_retry_can_succeed() -> None:
    """첫 출력이 잘못되어도 다시 시도해 올바르면 그 결과를 쓴다."""
    server = FakeServer([reply(["하나만"]), reply(["가", "나"])])

    results = make(server).translate_batch(requests_of("a", "b"), CancellationToken())

    assert [r.translated_text for r in results] == ["가", "나"]


def test_server_error_status_is_retried_then_reported(caplog: pytest.LogCaptureFixture) -> None:
    """서버가 오류 상태를 돌려주면 한 번 다시 시도하고, 계속되면 서버 오류를 알린다. 오류 종류는 로그에 남긴다."""
    error = (500, json.dumps({"error": {"type": "server_error", "message": "format mismatch"}}).encode())
    server = FakeServer([error, error])

    with pytest.raises(TranslationError, match=re.escape(SERVER_ERROR_MESSAGE)):
        make(server).translate_batch(requests_of("a"), CancellationToken())

    assert "format mismatch" in caplog.text


def test_connection_failure_resets_readiness() -> None:
    """연결이 끊기면 서버 연결 오류를 알리고, 다음 요청에서 서버 준비 여부를 다시 확인한다."""
    server = FakeServer([urllib.error.URLError("refused"), reply(["가"])])
    translator = make(server)

    with pytest.raises(TranslationError, match=re.escape(SERVER_UNAVAILABLE_MESSAGE)):
        translator.translate_batch(requests_of("a"), CancellationToken())
    translator.translate_batch(requests_of("a"), CancellationToken())

    assert server.health_calls == 2


def test_timeout_is_reported() -> None:
    """요청 시간 초과는 시간 초과 오류로 알린다."""
    server = FakeServer([TimeoutError()])

    with pytest.raises(TranslationError, match=re.escape(TIMEOUT_MESSAGE)):
        make(server).translate_batch(requests_of("a"), CancellationToken())


def test_waits_for_server_to_become_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    """서버가 모델을 불러오는 동안(연결 거부, 503)은 기다리다가 준비되면 번역한다. 준비 확인은 한 번만 한다."""
    monkeypatch.setattr(chat_completion, "LLM_HEALTH_POLL_SECONDS", 0.001)
    server = FakeServer(
        [reply(["가"]), reply(["나"])],
        health=[urllib.error.URLError("refused"), 503, 200],
    )
    translator = make(server)

    translator.translate_batch(requests_of("a"), CancellationToken())
    translator.translate_batch(requests_of("b"), CancellationToken())

    assert server.health_calls == 3


def test_not_ready_in_time_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """정한 시간 안에 준비되지 않으면 준비 실패를 알린다 (무한 대기 없음)."""
    monkeypatch.setattr(chat_completion, "LLM_HEALTH_POLL_SECONDS", 0.001)
    monkeypatch.setattr(chat_completion, "LLM_STARTUP_TIMEOUT_SECONDS", 0.05)
    server = FakeServer([], health=[503])

    with pytest.raises(TranslationError, match=re.escape(SERVER_NOT_READY_MESSAGE)):
        make(server).translate_batch(requests_of("a"), CancellationToken())


def test_dead_server_process_fails_fast() -> None:
    """앱이 띄운 서버 프로세스가 죽었으면 기다리지 않고 바로 연결 오류를 알린다."""
    server = FakeServer([], health=[503])

    with pytest.raises(TranslationError, match=re.escape(SERVER_UNAVAILABLE_MESSAGE)):
        make(server, is_server_alive=lambda: False).translate_batch(requests_of("a"), CancellationToken())

    assert server.health_calls == 0


def test_waiting_for_server_can_be_cancelled(monkeypatch: pytest.MonkeyPatch) -> None:
    """서버를 기다리는 동안 취소되면 바로 빠져나온다."""
    monkeypatch.setattr(chat_completion, "LLM_HEALTH_POLL_SECONDS", 0.01)
    server = FakeServer([], health=[503])
    token = CancellationToken()
    outcome: list = []

    def work() -> None:
        """다른 스레드에서 번역을 기다린다."""
        try:
            make(server).translate_batch(requests_of("a"), token)
        except BaseException as exc:  # noqa: BLE001 (테스트에서 예외를 그대로 확인한다)
            outcome.append(exc)

    thread = threading.Thread(target=work)
    thread.start()
    token.cancel()
    thread.join(5)

    assert isinstance(outcome[0], OperationCancelledError)


def test_cancelled_request_is_not_sent() -> None:
    """이미 취소된 요청은 서버에 보내지 않는다."""
    server = FakeServer([reply(["가"])])
    token = CancellationToken()
    token.cancel()

    with pytest.raises(OperationCancelledError):
        make(server).translate_batch(requests_of("a"), token)

    assert server.requests == []


def test_texts_are_not_logged(caplog: pytest.LogCaptureFixture) -> None:
    """원문과 번역문은 로그에 남기지 않는다 (성공과 형식 오류 모두)."""
    server = FakeServer([reply(["번역문 비밀"]), reply(["번역문 비밀"]), reply(["번역문 비밀"]), reply(["하나"])])
    translator = make(server)

    with caplog.at_level(logging.DEBUG, logger="translator"):
        translator.translate_batch(requests_of(SECRET), CancellationToken())
        with pytest.raises(TranslationError):
            translator.translate_batch(requests_of(SECRET, "two"), CancellationToken())

    assert SECRET not in caplog.text and "번역문 비밀" not in caplog.text


def test_works_with_pipeline_translation_stage() -> None:
    """파이프라인의 번역 단계에 그대로 끼워 쓸 수 있고, 같은 원문은 캐시를 쓴다."""
    server = FakeServer([reply(["가", "나"])])
    pipeline = TranslationPipeline(capturer=None, recognizer=object(), translator=make(server))

    first = pipeline.translate_texts(["a", "b"], CancellationToken())
    second = pipeline.translate_texts(["a"], CancellationToken())

    assert first == {"a": ("가", "qwen-test"), "b": ("나", "qwen-test")}
    assert second == {"a": ("가", "qwen-test")}
    assert len(server.requests) == 1


def test_parse_translations_strips_and_validates() -> None:
    """번역문 앞뒤 공백을 지우고, 개수가 맞는 문자열 목록만 받아들인다."""
    assert parse_translations('{"translations": [" 가 ", "나"]}', 2) == ["가", "나"]
    with pytest.raises(ValueError):
        parse_translations('{"translations": ["가", 3]}', 2)
    with pytest.raises(ValueError):
        parse_translations('["가"]', 1)


@pytest.mark.parametrize(
    ("source", "translation", "expected"),
    [
        ("Save", "저장", None),
        ("250", "250", None),  # 글자가 없는 원문
        ("Elena", "Elena", None),  # 원문 그대로 둔 고유명사
        ("Shut up!", "闭嘴!", "han"),
        ("Hello", "Привет", "cyrillic"),
        ("Go", "Go now", "한글"),
        ("Visit 東京", "東京에 가 봐", None),  # 원문에 있던 한자는 허용
    ],
)
def test_script_problem(source: str, translation: str, expected: str | None) -> None:
    """한국어 번역에 한글이 없거나 원문에 없던 문자가 섞이면 문제로 본다."""
    problem = script_problem(source, translation, "ko")
    assert (problem is None) if expected is None else (expected in problem)


def test_translation_pattern_limits_length_and_scripts() -> None:
    """번역문 패턴은 길이를 가장 긴 원문 기준으로 제한하고, 원문에 없던 문자를 막는다."""
    limit = max_translation_chars(["short", "x" * 100])
    pattern = translation_pattern(["short", "x" * 100], "ko")

    assert limit == 200
    assert pattern.endswith(f"{{0,{limit}}}$")
    assert re.fullmatch(pattern, "닥치고 싸워!")
    assert not re.fullmatch(pattern, "闭嘴")
    assert not re.fullmatch(pattern, "가" * (limit + 1))
    # 한국어가 아니면 문자를 막지 않는다.
    assert translation_pattern(["a"], "en").startswith("^.")
