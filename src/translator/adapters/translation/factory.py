"""번역 엔진 선택: 설정(환경 변수)에 따라 Qwen(llama.cpp 서버) 또는 Mock 번역을 만든다.

- `TRANSLATOR_TRANSLATION_ENGINE`: `auto`(기본), `qwen`, `mock`
  - `auto`: Qwen을 쓸 수 있으면(아래 서버 주소가 있거나, llama.cpp와 모델 파일이 있음) Qwen, 아니면 경고 후 Mock
  - `qwen`: 항상 Qwen. 쓸 수 없으면 번역할 때마다 설정 안내 오류를 보여 준다.
  - `mock`: Mock 번역 (원문에 표식만 붙임)
- `TRANSLATOR_LLM_URL`: 이미 떠 있는 OpenAI 호환 서버 주소 (예: http://127.0.0.1:8080). 있으면 서버를 띄우지 않는다.
  이 컴퓨터 밖의 주소는 원문이 외부로 전송되므로 `TRANSLATOR_LLM_ALLOW_REMOTE=1`일 때만 쓴다.
- `TRANSLATOR_LLAMA_SERVER`: llama-server 실행 파일 경로 (기본: 기본 폴더의 가장 최근 llama.cpp 빌드)
- `TRANSLATOR_LLM_MODEL`: GGUF 모델 파일 경로 (기본: 기본 폴더의 `DEFAULT_MODEL_FILE`)

기본 폴더는 `%LOCALAPPDATA%\\MultimodalTranslator\\llm`이다 (`llama.cpp\\<빌드>\\llama-server.exe`, `models\\*.gguf`).
이 모듈은 서버를 시작하지 않는다. 시작과 종료는 진입점이 `TranslationSelection.server`로 한다.
"""

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from urllib.parse import urlparse

from translator.adapters.translation.chat_completion import ChatCompletionTranslator, ChatServerSettings
from translator.adapters.translation.llama_server import LlamaServerProcess
from translator.adapters.translation.mock import MockTranslator
from translator.core.cancellation import CancellationToken
from translator.core.errors import TranslationError
from translator.core.interfaces import Translator
from translator.core.models import TranslationRequest, TranslationResult

logger = logging.getLogger(__name__)

# 설정 환경 변수 이름
ENGINE_ENV = "TRANSLATOR_TRANSLATION_ENGINE"
LLM_URL_ENV = "TRANSLATOR_LLM_URL"
LLM_ALLOW_REMOTE_ENV = "TRANSLATOR_LLM_ALLOW_REMOTE"
LLAMA_SERVER_ENV = "TRANSLATOR_LLAMA_SERVER"
LLM_MODEL_ENV = "TRANSLATOR_LLM_MODEL"
# 기본 번역 모델 파일. 4단계 비교(docs/phase2-qwen-translation.md)에서 오역이 가장 적었다.
DEFAULT_MODEL_FILE = "Qwen3.5-9B-Q4_K_M.gguf"
# llama-server 실행 파일 이름
SERVER_EXECUTABLE = "llama-server.exe"
# 이 컴퓨터 안의 주소로 보는 호스트 이름
_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
# Qwen 설정 안내 (번역 대상 텍스트를 담지 않는다)
QWEN_NOT_CONFIGURED_MESSAGE = (
    "Qwen 번역 모델을 찾지 못했습니다. README의 'Qwen 번역 설정'을 참고해 llama.cpp와 모델 파일을 준비해 주세요."
)


class TranslationEngineKind(Enum):
    """선택할 수 있는 번역 엔진 종류."""

    AUTO = "auto"
    QWEN = "qwen"
    MOCK = "mock"


@dataclass(frozen=True)
class TranslationSelection:
    """고른 번역 엔진과, 앱이 띄우고 꺼야 하는 서버 프로세스 (없으면 None)."""

    # 고른 번역 엔진
    translator: Translator
    # 앱이 관리할 llama.cpp 서버 (외부 서버를 쓰거나 Mock이면 None)
    server: LlamaServerProcess | None = None


class UnavailableTranslator(Translator):
    """번역할 때마다 정해진 오류를 던지는 엔진 (설정이 없음을 상태 창에 알리는 용도)."""

    def __init__(self, message: str) -> None:
        """사용자에게 보여 줄 오류 메시지를 받는다."""
        # 번역 요청마다 던질 오류 메시지
        self._message = message

    def translate(
        self, request: TranslationRequest, cancellation: CancellationToken
    ) -> TranslationResult:
        """항상 `TranslationError`를 던진다 (취소되었으면 취소 예외)."""
        cancellation.raise_if_cancelled()
        raise TranslationError(self._message)


def default_llm_dir(environ: Mapping[str, str]) -> Path:
    """llama.cpp와 모델 파일의 기본 폴더 (%LOCALAPPDATA%\\MultimodalTranslator\\llm)."""
    base = environ.get("LOCALAPPDATA")
    return (Path(base) if base else Path.home()) / "MultimodalTranslator" / "llm"


def find_server_executable(llm_dir: Path) -> Path | None:
    """기본 폴더에서 가장 최근에 바뀐 llama.cpp 빌드의 llama-server 실행 파일을 찾는다."""
    candidates = sorted(
        (llm_dir / "llama.cpp").glob(f"*/{SERVER_EXECUTABLE}"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def is_local_url(url: str) -> bool:
    """주소가 이 컴퓨터 안(localhost)을 가리키는지 여부."""
    return (urlparse(url).hostname or "") in _LOCAL_HOSTS


def parse_engine_kind(value: str | None) -> TranslationEngineKind:
    """환경 변수 값을 엔진 종류로 바꾼다. 비었거나 모르는 값이면 `AUTO` (경고 로그)."""
    if value is None or not value.strip():
        return TranslationEngineKind.AUTO
    try:
        return TranslationEngineKind(value.strip().lower())
    except ValueError:
        logger.warning("%s 값 '%s'을(를) 알 수 없어 auto로 처리합니다 (가능한 값: auto, qwen, mock).", ENGINE_ENV, value)
        return TranslationEngineKind.AUTO


def select_translator(environ: Mapping[str, str] | None = None) -> TranslationSelection:
    """설정에 맞는 번역 엔진을 고른다. 서버를 시작하지 않으므로 UI 스레드에서 호출해도 된다.

    매개변수:
        environ: 설정을 읽을 환경 변수 (None이면 `os.environ`).
    반환값:
        고른 번역 엔진과, 앱이 관리할 서버 프로세스.
    """
    # 설정을 읽을 환경 변수 모음
    environ = os.environ if environ is None else environ
    kind = parse_engine_kind(environ.get(ENGINE_ENV))
    if kind is TranslationEngineKind.MOCK:
        logger.info("번역 엔진: Mock (%s=mock)", ENGINE_ENV)
        return TranslationSelection(MockTranslator())

    selection = _qwen_selection(environ)
    if selection is not None:
        return selection
    if kind is TranslationEngineKind.AUTO:
        logger.warning("Qwen 번역 설정을 찾지 못해 Mock 번역을 사용합니다 (README의 'Qwen 번역 설정' 참고).")
        return TranslationSelection(MockTranslator())
    logger.error("%s=qwen이지만 Qwen 번역 설정을 찾지 못했습니다.", ENGINE_ENV)
    return TranslationSelection(UnavailableTranslator(QWEN_NOT_CONFIGURED_MESSAGE))


def _qwen_selection(environ: Mapping[str, str]) -> TranslationSelection | None:
    """Qwen을 쓸 수 있으면 그 엔진(과 서버)을, 아니면 None을 돌려준다."""
    url = environ.get(LLM_URL_ENV, "").strip()
    if url:
        if not is_local_url(url) and environ.get(LLM_ALLOW_REMOTE_ENV) != "1":
            # 화면에서 읽은 원문이 외부로 전송되므로 명시적으로 허용했을 때만 쓴다.
            logger.error(
                "%s가 이 컴퓨터 밖을 가리켜 쓰지 않습니다 (원문이 외부로 전송됨). 허용하려면 %s=1",
                LLM_URL_ENV, LLM_ALLOW_REMOTE_ENV,
            )
            return None
        logger.info("번역 엔진: Qwen (외부 서버, %s)", "로컬" if is_local_url(url) else "원격 허용됨")
        return TranslationSelection(ChatCompletionTranslator(ChatServerSettings(url)))

    llm_dir = default_llm_dir(environ)
    executable = Path(environ[LLAMA_SERVER_ENV]) if environ.get(LLAMA_SERVER_ENV) else find_server_executable(llm_dir)
    model = Path(environ[LLM_MODEL_ENV]) if environ.get(LLM_MODEL_ENV) else llm_dir / "models" / DEFAULT_MODEL_FILE
    if executable is None or not executable.is_file() or not model.is_file():
        logger.info(
            "Qwen 실행 파일 또는 모델이 없습니다 (서버 %s, 모델 %s).",
            "있음" if executable is not None and executable.is_file() else "없음",
            "있음" if model.is_file() else "없음",
        )
        return None
    server = LlamaServerProcess(executable, model)
    translator = ChatCompletionTranslator(
        ChatServerSettings(server.base_url, engine_name=server.model_name),
        is_server_alive=server.is_running,
    )
    logger.info("번역 엔진: Qwen (모델 %s, 앱이 llama.cpp 서버를 띄움)", model.name)
    return TranslationSelection(translator, server)
