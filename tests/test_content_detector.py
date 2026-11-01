"""콘텐츠 식별 테스트: 모델 검증, 실행 파일 → 콘텐츠 변환, 식별기 동작, Win32 호출 확인."""

import os
import sys

import pytest

from translator.adapters.content.mock import ScriptedContentDetector
from translator.adapters.content.windows import (
    CONTENT_ID_PREFIX,
    ForegroundProcess,
    WindowsContentDetector,
    identity_from_executable_path,
    query_process_image_path,
    read_foreground_process,
)
from translator.core.content import ContentIdentity, IdentificationMethod

# 테스트에서 우리 앱으로 취급할 가짜 프로세스 ID
OWN_PROCESS_ID = 100
# 테스트에서 다른 프로그램으로 취급할 가짜 프로세스 ID
OTHER_PROCESS_ID = 200


def make_identity(content_id: str = "app:game.exe") -> ContentIdentity:
    """테스트용 콘텐츠 정보를 만든다."""
    return ContentIdentity(content_id, "game", IdentificationMethod.PROCESS_NAME)


@pytest.mark.parametrize("content_id, display_name", [("", "game"), ("  ", "game"), ("app:a", ""), ("app:a", " ")])
def test_identity_rejects_empty_values(content_id: str, display_name: str) -> None:
    """빈 ID나 빈 이름은 식별되지 않은 콘텐츠가 섞이는 원인이 되므로 만들 수 없다."""
    with pytest.raises(ValueError):
        ContentIdentity(content_id, display_name, IdentificationMethod.MANUAL)


def test_executable_path_becomes_lowercase_process_content_id() -> None:
    """실행 파일 이름이 소문자 콘텐츠 ID가 되고, 표시 이름은 확장자 없는 원래 이름이다."""
    identity = identity_from_executable_path(r"C:\Games\Elden Ring\Game.EXE")

    assert identity == ContentIdentity(
        f"{CONTENT_ID_PREFIX}game.exe", "Game", IdentificationMethod.PROCESS_NAME
    )


def test_same_executable_name_in_different_case_is_same_content() -> None:
    """대소문자만 다른 경로는 같은 콘텐츠로 본다 (Windows 파일 이름은 대소문자를 구분하지 않는다)."""
    first = identity_from_executable_path(r"C:\Apps\Player.exe")
    second = identity_from_executable_path(r"D:\other\PLAYER.EXE")

    assert first is not None and second is not None
    assert first.content_id == second.content_id


@pytest.mark.parametrize(
    "path", ["", "C:\\", r"C:\Windows\System32\ApplicationFrameHost.exe", r"c:\x\APPLICATIONFRAMEHOST.EXE"]
)
def test_unidentifiable_executables_return_none(path: str) -> None:
    """파일 이름이 없거나 실제 앱을 알 수 없는 프로세스(UWP 틀)는 식별하지 않는다."""
    assert identity_from_executable_path(path) is None


def test_detector_identifies_foreground_process() -> None:
    """다른 프로그램이 활성 창이면 그 실행 파일로 식별한다."""
    detector = WindowsContentDetector(
        OWN_PROCESS_ID, lambda: ForegroundProcess(OTHER_PROCESS_ID, r"C:\Games\game.exe")
    )

    assert detector.detect() == make_identity()


def test_detector_ignores_own_windows() -> None:
    """우리 앱 자신의 창(오버레이 등)이 활성이면 식별하지 않는다 (콘텐츠가 우리 앱으로 바뀌면 안 된다)."""
    detector = WindowsContentDetector(
        OWN_PROCESS_ID, lambda: ForegroundProcess(OWN_PROCESS_ID, r"C:\Python\python.exe")
    )

    assert detector.detect() is None


def test_detector_returns_none_without_foreground_process() -> None:
    """활성 창이 없거나 권한이 없어 프로세스를 읽지 못하면 None이다."""
    detector = WindowsContentDetector(OWN_PROCESS_ID, lambda: None)

    assert detector.detect() is None


def test_detector_logs_and_returns_none_when_reading_fails(caplog: pytest.LogCaptureFixture) -> None:
    """Windows API 오류는 예외로 번지지 않고 경고 로그를 남긴 뒤 None이다 (감시가 멈추면 안 된다)."""

    def failing_reader() -> ForegroundProcess | None:
        """항상 OSError를 던지는 가짜 읽기 함수."""
        raise OSError("api failure")

    detector = WindowsContentDetector(OWN_PROCESS_ID, failing_reader)

    with caplog.at_level("WARNING"):
        assert detector.detect() is None

    assert "활성 창의 프로세스를 읽지 못했습니다" in caplog.text


def test_detector_defaults_to_current_process_as_own() -> None:
    """우리 앱 프로세스 ID를 주지 않으면 현재 프로세스를 우리 앱으로 본다."""
    detector = WindowsContentDetector(
        read_foreground=lambda: ForegroundProcess(os.getpid(), r"C:\Python\python.exe")
    )

    assert detector.detect() is None


def test_scripted_detector_replays_results_then_repeats_last() -> None:
    """Mock 식별기는 정해진 결과(None 포함)를 순서대로 돌려주고, 끝나면 마지막 결과를 반복한다."""
    first, second = make_identity("app:a.exe"), make_identity("app:b.exe")
    detector = ScriptedContentDetector([first, None, second])

    assert [detector.detect() for _ in range(5)] == [first, None, second, second, second]


def test_scripted_detector_requires_at_least_one_result() -> None:
    """결과가 없는 Mock 식별기는 만들 수 없다."""
    with pytest.raises(ValueError):
        ScriptedContentDetector([])


@pytest.mark.skipif(sys.platform != "win32", reason="Windows API 호출 확인")
def test_query_process_image_path_reads_current_python_executable() -> None:
    """실제 Windows API로 현재 프로세스의 실행 파일 경로를 읽는다 (파이썬 실행 파일)."""
    path = query_process_image_path(os.getpid())

    assert path is not None
    assert os.path.basename(path).lower().startswith("python")


@pytest.mark.skipif(sys.platform != "win32", reason="Windows API 호출 확인")
def test_query_process_image_path_returns_none_for_missing_process() -> None:
    """없는 프로세스 ID는 예외 없이 None이다."""
    assert query_process_image_path(0xFFFFFFF0) is None


@pytest.mark.skipif(sys.platform != "win32", reason="Windows API 호출 확인")
def test_read_foreground_process_returns_valid_value_or_none() -> None:
    """실제 활성 창을 읽는다. 값이 있으면 프로세스 ID와 경로가 채워져 있다 (활성 창은 환경마다 다르다)."""
    process = read_foreground_process()

    if process is not None:
        assert process.process_id > 0
        assert process.executable_path
