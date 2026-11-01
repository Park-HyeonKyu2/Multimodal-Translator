"""오버레이 창 공용 Windows 연동 도우미."""

import ctypes
import logging
import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QWidget

logger = logging.getLogger(__name__)

# Windows 10 2004 이상에서 지원. 창이 화면 캡처에 찍히지 않게 한다.
_WDA_EXCLUDEFROMCAPTURE = 0x00000011


def exclude_from_screen_capture(widget: QWidget) -> bool:
    """오버레이/결과창이 번역용 캡처 이미지에 포함되지 않도록 한다.

    실패해도 동작에는 지장이 없으므로 경고만 남긴다. 네이티브 창이 다시 만들어지면
    설정이 사라지므로 showEvent마다 호출한다.
    """
    if sys.platform != "win32" or QGuiApplication.platformName() != "windows":
        return False
    hwnd = int(widget.winId())
    if ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, _WDA_EXCLUDEFROMCAPTURE):
        return True
    logger.warning(
        "창을 캡처 제외로 설정하지 못했습니다 (Windows 오류 코드 %d). "
        "오버레이가 캡처 이미지에 포함될 수 있습니다.",
        ctypes.GetLastError(),
    )
    return False


def set_click_through(widget: QWidget, enabled: bool) -> None:
    """마우스 입력이 창을 통과해 아래 프로그램(게임 등)으로 전달되게 한다.

    `QWidget.setWindowFlag`는 보이는 창을 숨기고 다시 만들기 때문에 쓰지 않는다.
    위젯 플래그와 네이티브 창 플래그를 함께 바꿔 창을 유지한 채 즉시 적용한다.
    """
    flag = Qt.WindowType.WindowTransparentForInput
    flags = widget.windowFlags()
    flags = flags | flag if enabled else flags & ~flag
    widget.overrideWindowFlags(flags)
    handle = widget.windowHandle()
    if handle is not None:
        handle.setFlags(flags)
    # 통과를 끈 뒤에는 다시 그려야 반투명 창이 클릭을 받는다.
    widget.update()
