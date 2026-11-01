"""Windows 활성 창의 프로세스로 콘텐츠를 식별한다.

활성(foreground) 창 → 그 창을 가진 프로세스 → 실행 파일 이름 순서로 확인하고, 실행 파일 이름으로
`content_id`를 만든다. 창 제목은 읽지 않는다. 브라우저 탭 제목 같은 열람 기록 성격의 값을 다루지
않기 위해서다 (브라우저는 프로그램 하나가 한 콘텐츠다. 탭·영상 구분은 별도 검토 대상).

한계:
- 서로 다른 폴더의 같은 이름 실행 파일은 같은 콘텐츠로 본다.
- UWP 앱은 `ApplicationFrameHost.exe` 창으로 감싸여 있어 실제 앱을 알 수 없으므로 식별하지 않는다.
- 셸(`explorer.exe`: 작업 표시줄·바탕 화면·Alt-Tab), 시작 메뉴·검색 같은 셸 팝업, 입력기, 게임 오버레이가
  활성 창일 때는 식별하지 않는다 (`_IGNORED_PROCESS_NAMES`). 그래서 파일 탐색기 창은 콘텐츠가 될 수 없다.
- 관리자 권한으로 실행 중인 프로세스는 실행 파일 경로를 읽지 못할 수 있다 (식별 실패로 처리).
"""

import ctypes
import functools
import logging
import os
from collections.abc import Callable
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import PureWindowsPath

from translator.core.content import ContentIdentity, IdentificationMethod
from translator.core.interfaces import ContentDetector

logger = logging.getLogger(__name__)

# 프로세스 이름·경로를 읽는 데 필요한 최소 권한 (Windows API 정의값)
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
# 실행 파일 전체 경로를 받을 버퍼 크기 (글자 수). 긴 경로를 허용하는 Windows 최대치
_IMAGE_PATH_BUFFER_CHARS = 32768
# 콘텐츠 ID가 프로세스 기반임을 나타내는 접두사
CONTENT_ID_PREFIX = "app:"
# 콘텐츠로 보지 않는 프로세스 이름 (소문자). 활성 창이 이 프로세스여도 콘텐츠를 바꾸지 않는다.
# 디바운스(약 1초)는 순간 전환만 막으므로, 사용자가 1초 넘게 열어 두는 시스템·보조 창이 콘텐츠를
# 바꾸어 번역이 취소되고 박스가 교체되는 것을 막는 목록이다.
# 이 이름들은 Windows와 각 프로그램의 알려진 프로세스 이름이며 이 PC에서 모두 확인한 것은 아니다.
# 실제 게임에서 콘텐츠가 엉뚱하게 바뀌면 여기에 추가한다.
_IGNORED_PROCESS_NAMES = frozenset({
    # UWP 앱의 틀 창: 실제 앱을 알 수 없다
    "applicationframehost.exe",
    # 작업 표시줄, 바탕 화면, Alt-Tab 화면을 그리는 Windows 셸
    "explorer.exe",
    # 시작 메뉴, 검색, 알림·빠른 설정 같은 셸 팝업
    "startmenuexperiencehost.exe", "searchhost.exe", "searchapp.exe", "shellexperiencehost.exe",
    # 입력기(이모지, 한글 입력 후보 창), 잠금 화면
    "textinputhost.exe", "lockapp.exe",
    # 게임 위에 뜨는 오버레이: Xbox Game Bar, Steam
    "gamebar.exe", "gamebarftserver.exe", "gameoverlayui.exe",
})


@dataclass(frozen=True)
class ForegroundProcess:
    """활성 창을 소유한 프로세스."""

    # 프로세스 ID
    process_id: int
    # 실행 파일 전체 경로
    executable_path: str


@functools.cache
def _load_win32() -> tuple[ctypes.WinDLL, ctypes.WinDLL]:
    """user32와 kernel32를 불러와 함수 원형(인자·반환 형식)을 지정한다.

    64비트 Windows에서 창·프로세스 핸들은 64비트 값이므로, 원형을 지정하지 않으면 값이 잘린다.

    반환값:
        (user32, kernel32).
    예외:
        OSError: Windows가 아니라서 라이브러리를 불러올 수 없을 때.
    """
    # 활성 창 조회 함수가 들어 있는 라이브러리
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    # 프로세스 조회 함수가 들어 있는 라이브러리
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    user32.GetForegroundWindow.argtypes = []
    user32.GetForegroundWindow.restype = wintypes.HWND
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user32.GetWindowThreadProcessId.restype = wintypes.DWORD
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD),
    ]
    kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    return user32, kernel32


def query_process_image_path(process_id: int) -> str | None:
    """프로세스의 실행 파일 전체 경로를 읽는다.

    매개변수:
        process_id: 프로세스 ID.
    반환값:
        실행 파일 경로. 프로세스가 없거나 권한이 없으면 None.
    예외:
        OSError: Windows가 아니라서 API를 부를 수 없을 때.
    """
    _, kernel32 = _load_win32()
    # 프로세스 핸들 (열지 못하면 0 또는 None)
    handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, process_id)
    if not handle:
        return None
    try:
        # 경로를 받을 버퍼와, 버퍼 크기·실제 경로 길이를 주고받는 값
        buffer = ctypes.create_unicode_buffer(_IMAGE_PATH_BUFFER_CHARS)
        length = wintypes.DWORD(_IMAGE_PATH_BUFFER_CHARS)
        if not kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(length)):
            return None
        return buffer.value
    finally:
        kernel32.CloseHandle(handle)


def read_foreground_process() -> ForegroundProcess | None:
    """활성 창을 소유한 프로세스를 읽는다.

    반환값:
        프로세스 정보. 활성 창이 없거나(잠금 화면 등) 프로세스 경로를 읽지 못하면 None.
    예외:
        OSError: Windows가 아니라서 API를 부를 수 없을 때.
    """
    user32, _ = _load_win32()
    # 지금 입력을 받는 창
    window = user32.GetForegroundWindow()
    if not window:
        return None
    # 창을 소유한 프로세스 ID를 받을 값
    process_id = wintypes.DWORD(0)
    user32.GetWindowThreadProcessId(window, ctypes.byref(process_id))
    if process_id.value == 0:
        return None
    # 프로세스의 실행 파일 경로
    executable_path = query_process_image_path(process_id.value)
    if not executable_path:
        return None
    return ForegroundProcess(process_id.value, executable_path)


def identity_from_executable_path(executable_path: str) -> ContentIdentity | None:
    """실행 파일 경로에서 콘텐츠를 만든다.

    매개변수:
        executable_path: 실행 파일 전체 경로.
    반환값:
        콘텐츠 정보. 파일 이름이 비었거나 실제 앱을 알 수 없는 프로세스면 None.
    """
    # 경로에서 뽑은 실행 파일 이름 (확장자 포함)과 확장자 없는 이름
    path = PureWindowsPath(executable_path)
    file_name = path.name
    if not file_name or file_name.lower() in _IGNORED_PROCESS_NAMES:
        return None
    return ContentIdentity(
        content_id=f"{CONTENT_ID_PREFIX}{file_name.lower()}",
        display_name=path.stem or file_name,
        method=IdentificationMethod.PROCESS_NAME,
    )


class WindowsContentDetector(ContentDetector):
    """활성 창의 프로세스로 콘텐츠를 식별한다.

    우리 앱 자신의 창(오버레이, 말풍선 등)이 활성 창일 때는 식별하지 않는다. 그렇지 않으면 사용자가
    오버레이 버튼을 누르는 순간 콘텐츠가 우리 앱으로 바뀐다.
    """

    def __init__(
        self,
        own_process_id: int | None = None,
        read_foreground: Callable[[], ForegroundProcess | None] = read_foreground_process,
    ) -> None:
        """우리 앱의 프로세스 ID와 활성 프로세스를 읽는 함수(테스트에서는 가짜)를 받는다.

        매개변수:
            own_process_id: 식별에서 제외할 프로세스 ID. None이면 현재 프로세스.
            read_foreground: 활성 창의 프로세스를 읽는 함수.
        """
        # 식별에서 제외할 우리 앱의 프로세스 ID
        self._own_process_id = os.getpid() if own_process_id is None else own_process_id
        # 활성 창의 프로세스를 읽는 함수
        self._read_foreground = read_foreground

    def detect(self) -> ContentIdentity | None:
        """활성 창의 프로세스로 콘텐츠를 식별한다. 식별하지 못하면 None."""
        try:
            process = self._read_foreground()
        except OSError:
            logger.warning("활성 창의 프로세스를 읽지 못했습니다.", exc_info=True)
            return None
        if process is None:
            logger.debug("활성 창이 없거나 프로세스 정보를 읽지 못했습니다 (권한 부족 등).")
            return None
        if process.process_id == self._own_process_id:
            return None
        identity = identity_from_executable_path(process.executable_path)
        if identity is None:
            logger.debug("실행 파일로 콘텐츠를 식별할 수 없습니다 (프로세스 ID %d).", process.process_id)
        return identity
