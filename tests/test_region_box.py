"""바운딩 박스 테스트."""

from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtGui import QColor

from translator.core.models import TranslationDirection
from translator.tuning import REGION_BOX_MARGIN
from translator.ui.region_box import RegionBox, color_for_index
from translator.ui.screen_geometry import global_logical_rect_to_physical


def make_box(qtbot, rect: QRect = QRect(100, 100, 200, 80)) -> RegionBox:
    """지정한 내부 영역으로 박스를 만들어 표시한다."""
    box = RegionBox("box-1", QColor(255, 140, 0))
    qtbot.addWidget(box)
    box.set_capture_rect(rect)
    box.show()
    return box


def test_capture_rect_excludes_border_margin(qtbot) -> None:
    """캡처 영역은 테두리 여백을 뺀 내부이고, 물리 좌표로 변환된다."""
    rect = QRect(100, 100, 200, 80)
    box = make_box(qtbot, rect)

    m = REGION_BOX_MARGIN
    assert box.geometry() == rect.adjusted(-m, -m, m, m)
    assert box.capture_rect() == rect
    assert box.capture_region() == global_logical_rect_to_physical(rect)


def test_capture_region_is_available_while_locked(qtbot) -> None:
    """잠금 모드(흐린 점선, 클릭 통과)에서도 자동 감지를 위해 캡처 영역을 계산한다."""
    rect = QRect(100, 100, 200, 80)
    box = make_box(qtbot, rect)

    box.set_interactive(False)

    assert box.capture_region() == global_logical_rect_to_physical(rect)


def test_label_shows_id_direction_and_auto(qtbot) -> None:
    """편집 모드 이름표에 박스 ID, 출력 방향, 자동 감지 여부가 보인다."""
    box = make_box(qtbot)
    assert box.label_text() == "box-1 · ↓ 아래"

    box.set_direction(TranslationDirection.LEFT)
    box.set_auto_enabled(True)

    assert box.label_text() == "box-1 · ← 왼쪽 · 자동"
    assert box.accent == QColor(255, 140, 0)


def test_context_menu_actions_emit_requests(qtbot) -> None:
    """우클릭 메뉴로 출력 방향, 자동 감지, 지금 번역, 삭제를 요청한다."""
    box = make_box(qtbot)
    menu = box.build_context_menu()
    direction_menu = menu.actions()[0].menu()
    actions = {action.text(): action for action in menu.actions()}
    direction_actions = {action.text(): action for action in direction_menu.actions()}

    assert direction_actions["↓ 아래"].isChecked()
    with qtbot.waitSignal(box.direction_selected, timeout=1000) as blocker:
        direction_actions["↑ 위"].trigger()
    assert blocker.args == [TranslationDirection.TOP]
    with qtbot.waitSignal(box.auto_toggled, timeout=1000) as blocker:
        actions["자동 감지"].trigger()
    assert blocker.args == [True]
    with qtbot.waitSignal(box.translate_requested, timeout=1000):
        actions["지금 번역"].trigger()
    with qtbot.waitSignal(box.remove_requested, timeout=1000):
        actions["영역 삭제"].trigger()


def test_colors_cycle_per_box() -> None:
    """박스마다 다른 색을 차례로 쓴다."""
    assert color_for_index(0) != color_for_index(1)
    assert color_for_index(0) == color_for_index(6)


def test_moving_box_reports_geometry_change(qtbot) -> None:
    """박스를 옮기거나 크기를 바꾸면 변경 시그널을 보낸다."""
    box = make_box(qtbot)

    with qtbot.waitSignal(box.geometry_changed, timeout=1000):
        box.move(box.x() + 30, box.y())
    with qtbot.waitSignal(box.geometry_changed, timeout=1000):
        box.resize(box.width() + 40, box.height())


def test_edges_at_detects_resize_handles(qtbot) -> None:
    """가장자리와 모서리는 크기 조절, 안쪽은 이동으로 판단한다."""
    box = make_box(qtbot)
    w, h = box.width(), box.height()

    assert box.edges_at(QPoint(1, 1)) == Qt.Edge.LeftEdge | Qt.Edge.TopEdge
    assert box.edges_at(QPoint(w - 1, h - 1)) == Qt.Edge.RightEdge | Qt.Edge.BottomEdge
    assert box.edges_at(QPoint(w - 1, h // 2)) == Qt.Edge.RightEdge
    assert not box.edges_at(QPoint(w // 2, h // 2))  # 내부는 이동


def test_locking_makes_box_click_through(qtbot) -> None:
    """잠금 모드에서 박스는 흐리게 보이지만 클릭을 통과시킨다."""
    box = make_box(qtbot)

    box.set_interactive(False)
    assert box.windowHandle().flags() & Qt.WindowType.WindowTransparentForInput
    assert box.remove_button.isHidden()
    assert box.isVisible()

    box.set_interactive(True)
    assert not box.windowHandle().flags() & Qt.WindowType.WindowTransparentForInput
    assert box.remove_button.isVisible()


def test_remove_button_requests_removal(qtbot) -> None:
    """삭제 버튼은 박스 삭제를 요청한다."""
    box = make_box(qtbot)

    with qtbot.waitSignal(box.remove_requested, timeout=1000):
        qtbot.mouseClick(box.remove_button, Qt.MouseButton.LeftButton)
