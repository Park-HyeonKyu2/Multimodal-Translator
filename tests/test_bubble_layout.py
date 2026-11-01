"""번역 말풍선 배치 계산 테스트: 상하좌우, 화면 경계 보정, 너비 제한."""

import pytest
from PySide6.QtCore import QRect, QSize

from translator.core.models import TranslationDirection
from translator.ui.bubble_layout import opposite, place_bubble, preferred_width

TOP = TranslationDirection.TOP
BOTTOM = TranslationDirection.BOTTOM
LEFT = TranslationDirection.LEFT
RIGHT = TranslationDirection.RIGHT

AREA = QRect(0, 0, 1920, 1040)  # 작업 영역 (작업표시줄 제외)
TEXT = QRect(800, 500, 200, 30)  # 화면 가운데 원문
SIZE = QSize(240, 50)
GAP = 6


@pytest.mark.parametrize(
    "direction, expected",
    [
        (BOTTOM, QRect(780, 536, 240, 50)),  # 원문 아래 + 간격, 가로 가운데
        (TOP, QRect(780, 444, 240, 50)),  # 원문 위 - 간격 - 높이
        (RIGHT, QRect(1006, 490, 240, 50)),  # 원문 오른쪽 + 간격, 세로 가운데
        (LEFT, QRect(554, 490, 240, 50)),  # 원문 왼쪽 - 간격 - 너비
    ],
)
def test_bubble_is_placed_on_requested_side(direction, expected) -> None:
    """지정한 방향에 원문과 간격을 두고 놓인다."""
    rect, used = place_bubble(TEXT, SIZE, direction, AREA, GAP)

    assert rect == expected
    assert used is direction


@pytest.mark.parametrize(
    "direction, text, expected_direction",
    [
        (BOTTOM, QRect(800, 1000, 200, 30), TOP),  # 화면 아래 끝 → 위로
        (TOP, QRect(800, 5, 200, 30), BOTTOM),  # 화면 위 끝 → 아래로
        (RIGHT, QRect(1800, 500, 100, 30), LEFT),  # 화면 오른쪽 끝 → 왼쪽으로
        (LEFT, QRect(10, 500, 100, 30), RIGHT),  # 화면 왼쪽 끝 → 오른쪽으로
    ],
)
def test_bubble_flips_to_opposite_side_when_no_room(direction, text, expected_direction) -> None:
    """지정 방향에 자리가 없으면 반대 방향으로 옮기고, 결과는 화면 안에 있다."""
    rect, used = place_bubble(text, SIZE, direction, AREA, GAP)

    assert used is expected_direction
    assert AREA.contains(rect)


def test_cross_axis_is_clamped_inside_screen() -> None:
    """원문이 화면 왼쪽 끝에 있어도 아래 말풍선이 화면 밖으로 나가지 않는다."""
    rect, used = place_bubble(QRect(0, 500, 40, 30), SIZE, BOTTOM, AREA, GAP)

    assert used is BOTTOM
    assert rect.left() == 0


def test_bubble_larger_than_screen_is_clamped_to_top_left() -> None:
    """어디에도 들어가지 않을 만큼 크면 지정 방향 위치를 화면 안으로 당긴다."""
    rect, used = place_bubble(TEXT, QSize(2500, 2000), BOTTOM, AREA, GAP)

    assert used is BOTTOM
    assert (rect.left(), rect.top()) == (0, 0)


def test_secondary_monitor_area_with_offset() -> None:
    """작업 영역이 원점이 아닌 보조 모니터(음수 좌표 포함)에서도 경계를 지킨다."""
    area = QRect(-1280, 100, 1280, 720)
    text = QRect(-1270, 780, 100, 30)  # 보조 모니터 왼쪽 아래 구석

    rect, used = place_bubble(text, SIZE, BOTTOM, area, GAP)

    assert used is TOP
    assert area.contains(rect)


def test_preferred_width_follows_text_for_vertical_placement() -> None:
    """위/아래 배치는 원문 너비를 따르되 최소/최대 너비로 제한한다."""
    assert preferred_width(QRect(0, 0, 300, 20), BOTTOM, AREA, GAP, 80, 480) == 300
    assert preferred_width(QRect(0, 0, 20, 20), BOTTOM, AREA, GAP, 80, 480) == 80
    assert preferred_width(QRect(0, 0, 1500, 20), TOP, AREA, GAP, 80, 480) == 480


def test_preferred_width_uses_side_space_for_horizontal_placement() -> None:
    """왼쪽/오른쪽 배치는 옆 공간을 쓰되 최대 너비와 화면 너비를 넘지 않는다."""
    near_right_edge = QRect(1700, 0, 100, 20)  # 오른쪽 공간 114, 왼쪽 공간 1694
    assert preferred_width(near_right_edge, RIGHT, AREA, GAP, 80, 480) == 480
    narrow_area = QRect(0, 0, 300, 400)
    assert preferred_width(QRect(100, 0, 100, 20), LEFT, narrow_area, GAP, 80, 480) == 94


def test_opposite() -> None:
    """반대 방향."""
    assert {opposite(d) for d in (TOP, BOTTOM, LEFT, RIGHT)} == {TOP, BOTTOM, LEFT, RIGHT}
    assert opposite(TOP) is BOTTOM and opposite(LEFT) is RIGHT
