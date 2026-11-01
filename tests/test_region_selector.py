"""영역 선택 UI와 DPI 좌표 변환 테스트."""

import pytest
from PySide6.QtCore import QPoint, QRect, Qt

from translator.core.models import ScreenRegion
from translator.ui.region_selector import RegionSelector
from translator.ui.screen_geometry import (
    global_logical_rect_to_physical,
    logical_rect_to_physical,
)


@pytest.mark.parametrize(
    "local_rect, screen_geometry, dpr, expected",
    [
        # 100% 배율 주 모니터
        (QRect(10, 20, 100, 50), QRect(0, 0, 1920, 1080), 1.0, ScreenRegion(10, 20, 100, 50)),
        # 150% 배율: 원점은 물리 좌표 그대로, 내부 오프셋과 크기만 확대
        (QRect(10, 20, 100, 50), QRect(1920, 0, 1707, 960), 1.5, ScreenRegion(1935, 30, 150, 75)),
        # 주 모니터 왼쪽에 있는 모니터 (음수 좌표)
        (QRect(0, 0, 40, 40), QRect(-2560, -200, 1280, 720), 2.0, ScreenRegion(-2560, -200, 80, 80)),
        # 125% 배율에서 반올림 시 가장자리 기준으로 크기를 계산
        (QRect(1, 1, 3, 3), QRect(0, 0, 100, 100), 1.25, ScreenRegion(1, 1, 4, 4)),
    ],
)
def test_logical_rect_to_physical(local_rect, screen_geometry, dpr, expected) -> None:
    """모니터 원점과 배율로 논리 좌표를 물리 좌표로 바꾼다."""
    assert logical_rect_to_physical(local_rect, screen_geometry, dpr) == expected


def drag(qtbot, window, start: QPoint, end: QPoint) -> None:
    """선택 창 위에서 마우스를 누르고, 옮기고, 놓는다."""
    qtbot.mousePress(window, Qt.MouseButton.LeftButton, pos=start)
    qtbot.mouseMove(window, end)
    qtbot.mouseRelease(window, Qt.MouseButton.LeftButton, pos=end)


def test_drag_selection_emits_global_logical_rect_and_closes(qtbot) -> None:
    """드래그 결과는 가상 데스크톱 논리 좌표로 전달되고 선택 창은 닫힌다."""
    selector = RegionSelector()
    selector.start()
    window = selector.windows[0]
    origin = window.screen().geometry().topLeft()

    with qtbot.waitSignal(selector.region_selected, timeout=1000) as blocker:
        # 오른쪽 아래에서 왼쪽 위로 드래그해도 정규화되어야 한다.
        drag(qtbot, window, QPoint(110, 80), QPoint(10, 20))

    assert blocker.args == [QRect(10, 20, 100, 60).translated(origin.x(), origin.y())]
    assert not selector.is_active


def test_global_rect_uses_screen_under_rect_center(qapp) -> None:
    """실제(테스트용) 모니터 위 사각형은 그 모니터 배율로 변환된다."""
    screen = qapp.primaryScreen()
    geometry = screen.geometry()
    rect = QRect(geometry.x() + 10, geometry.y() + 20, 100, 50)

    expected = logical_rect_to_physical(
        QRect(10, 20, 100, 50), geometry, screen.devicePixelRatio()
    )
    assert global_logical_rect_to_physical(rect) == expected


def test_global_rect_with_empty_size_returns_none(qapp) -> None:
    """크기가 0인 사각형은 변환하지 않는다."""
    assert global_logical_rect_to_physical(QRect(0, 0, 0, 10)) is None


def test_tiny_selection_is_ignored(qtbot) -> None:
    """너무 작은 드래그는 선택으로 보지 않는다."""
    selector = RegionSelector()
    selector.start()
    received: list[object] = []
    selector.region_selected.connect(received.append)

    drag(qtbot, selector.windows[0], QPoint(10, 10), QPoint(12, 12))

    assert received == []
    assert selector.is_active
    selector.cancel()


def test_escape_cancels_selection(qtbot) -> None:
    """Esc를 누르면 선택이 취소된다."""
    selector = RegionSelector()
    selector.start()

    with qtbot.waitSignal(selector.cancelled, timeout=1000):
        qtbot.keyClick(selector.windows[0], Qt.Key.Key_Escape)
    assert not selector.is_active


def test_start_twice_does_not_open_duplicate_windows(qtbot) -> None:
    """선택 중에 다시 시작해도 선택 창이 늘어나지 않는다."""
    selector = RegionSelector()
    selector.start()
    count = len(selector.windows)
    selector.start()

    assert len(selector.windows) == count
    selector.cancel()
