"""시스템·보조 창 제외 테스트: 시작 메뉴, 셸, 게임 오버레이가 활성 창이어도 콘텐츠로 보지 않는다 (회귀)."""

import pytest

from translator.adapters.content.windows import (
    ForegroundProcess,
    WindowsContentDetector,
    identity_from_executable_path,
)

# 테스트에서 우리 앱으로 취급할 가짜 프로세스 ID
OWN_PROCESS_ID = 100


@pytest.mark.parametrize(
    "executable_path",
    [
        r"C:\Windows\explorer.exe",
        r"C:\Windows\SystemApps\Microsoft.Windows.StartMenuExperienceHost_cw5n1h2txyewy\StartMenuExperienceHost.exe",
        r"C:\Windows\SystemApps\MicrosoftWindows.Client.CBS_cw5n1h2txyewy\SearchHost.exe",
        r"C:\Windows\SystemApps\ShellExperienceHost_cw5n1h2txyewy\ShellExperienceHost.exe",
        r"C:\Windows\SystemApps\MicrosoftWindows.Client.CBS_cw5n1h2txyewy\TextInputHost.exe",
        r"C:\Windows\SystemApps\Microsoft.LockApp_cw5n1h2txyewy\LockApp.exe",
        r"C:\Program Files\WindowsApps\Microsoft.XboxGamingOverlay\GameBar.exe",
        r"C:\Program Files (x86)\Steam\gameoverlayui.exe",
        r"C:\WINDOWS\EXPLORER.EXE",
    ],
)
def test_system_and_overlay_processes_are_not_content(executable_path: str) -> None:
    """셸·시스템 팝업·게임 오버레이 프로세스는 (대소문자와 폴더에 상관없이) 콘텐츠가 아니다."""
    assert identity_from_executable_path(executable_path) is None


def test_detector_ignores_start_menu_even_when_it_stays_in_front() -> None:
    """시작 메뉴가 활성 창이면 식별하지 않아, 감시기가 현재 콘텐츠를 유지한다 (세션이 바뀌지 않는다)."""
    detector = WindowsContentDetector(
        OWN_PROCESS_ID,
        lambda: ForegroundProcess(200, r"C:\Windows\SystemApps\X\StartMenuExperienceHost.exe"),
    )

    assert detector.detect() is None


def test_ordinary_programs_are_still_identified() -> None:
    """제외 목록에 없는 프로그램(게임, 브라우저)은 그대로 식별한다."""
    identity = identity_from_executable_path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")

    assert identity is not None and identity.content_id == "app:chrome.exe"
