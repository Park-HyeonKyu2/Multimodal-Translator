"""번역 엔진 선택 테스트: 환경 변수, 기본 폴더의 llama.cpp·모델 파일, 외부 주소 제한 (서버는 띄우지 않음)."""

import re
from pathlib import Path

import pytest

from translator.adapters.translation.chat_completion import ChatCompletionTranslator
from translator.adapters.translation.factory import (
    DEFAULT_MODEL_FILE,
    ENGINE_ENV,
    LLAMA_SERVER_ENV,
    LLM_ALLOW_REMOTE_ENV,
    LLM_MODEL_ENV,
    LLM_URL_ENV,
    QWEN_NOT_CONFIGURED_MESSAGE,
    UnavailableTranslator,
    find_server_executable,
    is_local_url,
    select_translator,
)
from translator.adapters.translation.mock import MockTranslator
from translator.core.cancellation import CancellationToken
from translator.core.errors import TranslationError
from translator.core.models import TranslationRequest


def llm_folder(tmp_path: Path, with_server: bool = True, with_model: bool = True) -> dict[str, str]:
    """임시 LOCALAPPDATA 아래에 기본 폴더 구조를 만들고 환경 변수를 돌려준다."""
    llm = tmp_path / "MultimodalTranslator" / "llm"
    if with_server:
        (llm / "llama.cpp" / "b1").mkdir(parents=True)
        (llm / "llama.cpp" / "b1" / "llama-server.exe").write_bytes(b"")
    if with_model:
        (llm / "models").mkdir(parents=True)
        (llm / "models" / DEFAULT_MODEL_FILE).write_bytes(b"")
    return {"LOCALAPPDATA": str(tmp_path)}


def test_mock_when_requested(tmp_path: Path) -> None:
    """mock을 고르면 Qwen 설정이 있어도 Mock 번역을 쓴다 (기존 동작 유지)."""
    selection = select_translator({**llm_folder(tmp_path), ENGINE_ENV: "mock"})

    assert isinstance(selection.translator, MockTranslator)
    assert selection.server is None


def test_default_folder_gives_qwen_with_managed_server(tmp_path: Path) -> None:
    """기본 폴더에 llama.cpp와 기본 모델이 있으면 Qwen을 쓰고, 앱이 띄울 서버를 함께 돌려준다 (아직 시작하지 않음)."""
    selection = select_translator(llm_folder(tmp_path))

    assert isinstance(selection.translator, ChatCompletionTranslator)
    assert selection.server is not None and not selection.server.is_running()
    assert selection.translator.settings.base_url == selection.server.base_url
    assert selection.translator.settings.engine_name == Path(DEFAULT_MODEL_FILE).stem


def test_auto_falls_back_to_mock_without_files(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """기본(auto)은 llama.cpp나 모델이 없으면 경고를 남기고 Mock 번역을 쓴다."""
    for env in (llm_folder(tmp_path / "a", with_model=False), llm_folder(tmp_path / "b", with_server=False)):
        selection = select_translator(env)
        assert isinstance(selection.translator, MockTranslator)
    assert "Mock 번역을 사용합니다" in caplog.text


def test_explicit_qwen_without_files_reports_on_use(tmp_path: Path) -> None:
    """qwen을 직접 골랐는데 설정이 없으면 번역할 때마다 설정 안내 오류를 보여 준다."""
    selection = select_translator({**llm_folder(tmp_path, with_model=False), ENGINE_ENV: "qwen"})

    assert isinstance(selection.translator, UnavailableTranslator)
    with pytest.raises(TranslationError, match=re.escape(QWEN_NOT_CONFIGURED_MESSAGE)):
        selection.translator.translate(TranslationRequest("a"), CancellationToken())


def test_explicit_paths_override_default_folder(tmp_path: Path) -> None:
    """실행 파일과 모델 경로를 직접 정하면 그 파일을 쓴다."""
    executable = tmp_path / "custom-server.exe"
    model = tmp_path / "other-model.gguf"
    executable.write_bytes(b"")
    model.write_bytes(b"")

    selection = select_translator(
        {"LOCALAPPDATA": str(tmp_path / "empty"), LLAMA_SERVER_ENV: str(executable), LLM_MODEL_ENV: str(model)}
    )

    assert selection.server is not None
    assert selection.translator.settings.engine_name == "other-model"


def test_local_url_uses_external_server_without_launching(tmp_path: Path) -> None:
    """이미 떠 있는 로컬 서버 주소를 주면 그 서버를 쓰고 서버를 띄우지 않는다."""
    selection = select_translator({**llm_folder(tmp_path), LLM_URL_ENV: "http://localhost:8080"})

    assert isinstance(selection.translator, ChatCompletionTranslator)
    assert selection.translator.settings.base_url == "http://localhost:8080"
    assert selection.server is None


def test_remote_url_needs_explicit_permission(tmp_path: Path) -> None:
    """외부 주소는 원문이 전송되므로 허용 설정이 있을 때만 쓴다 (없으면 설정 없음으로 처리)."""
    env = {**llm_folder(tmp_path, with_model=False), LLM_URL_ENV: "https://api.example.com"}

    refused = select_translator(env)
    allowed = select_translator({**env, LLM_ALLOW_REMOTE_ENV: "1"})

    assert isinstance(refused.translator, MockTranslator)
    assert isinstance(allowed.translator, ChatCompletionTranslator)


@pytest.mark.parametrize(
    ("url", "local"),
    [("http://127.0.0.1:1", True), ("http://localhost", True), ("http://[::1]:8", True),
     ("http://192.168.0.2", False), ("https://api.example.com", False)],
)
def test_is_local_url(url: str, local: bool) -> None:
    """이 컴퓨터 안의 주소만 로컬로 본다."""
    assert is_local_url(url) is local


def test_newest_server_build_is_found(tmp_path: Path) -> None:
    """llama.cpp 빌드가 여럿이면 가장 최근에 바뀐 빌드의 실행 파일을 고른다."""
    import os
    import time

    llm = tmp_path / "llm"
    for index, build in enumerate(("b1", "b2")):
        folder = llm / "llama.cpp" / build
        folder.mkdir(parents=True)
        path = folder / "llama-server.exe"
        path.write_bytes(b"")
        stamp = time.time() - 100 + index * 50
        os.utime(path, (stamp, stamp))

    assert find_server_executable(llm).parent.name == "b2"
    assert find_server_executable(tmp_path / "none") is None
