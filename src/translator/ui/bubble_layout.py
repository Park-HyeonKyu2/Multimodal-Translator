"""번역 말풍선의 크기와 위치를 정하는 순수 계산 (Qt 논리 좌표).

말풍선은 원문 텍스트 사각형을 기준으로 지정한 방향(상하좌우)에 간격을 두고 놓는다.
그 방향에 자리가 없으면 반대 방향 → 아래 → 위 → 오른쪽 → 왼쪽 순으로 시도하고,
어디에도 다 들어가지 않으면 지정한 방향 위치를 모니터 작업 영역 안으로 당긴다.
창을 만들지 않고 QRect/QSize만 쓰므로 단독으로 테스트할 수 있다.
"""

from PySide6.QtCore import QRect, QSize

from translator.core.models import TranslationDirection

_OPPOSITE = {
    TranslationDirection.TOP: TranslationDirection.BOTTOM,
    TranslationDirection.BOTTOM: TranslationDirection.TOP,
    TranslationDirection.LEFT: TranslationDirection.RIGHT,
    TranslationDirection.RIGHT: TranslationDirection.LEFT,
}
# 지정 방향과 반대 방향 다음에 시도할 순서
_FALLBACK_ORDER = (
    TranslationDirection.BOTTOM,
    TranslationDirection.TOP,
    TranslationDirection.RIGHT,
    TranslationDirection.LEFT,
)


def opposite(direction: TranslationDirection) -> TranslationDirection:
    """반대 방향."""
    return _OPPOSITE[direction]


def preferred_width(
    text_rect: QRect,
    direction: TranslationDirection,
    area: QRect,
    gap: int,
    min_width: int,
    max_width: int,
) -> int:
    """말풍선 너비를 정한다.

    위/아래 배치는 원문 너비를 따르고, 왼쪽/오른쪽 배치는 옆에 남은 공간(더 넓은 쪽)을
    쓴다. 결과는 [min_width, max_width] 범위이며 작업 영역 너비를 넘지 않는다.
    """
    if direction in (TranslationDirection.TOP, TranslationDirection.BOTTOM):
        width = text_rect.width()
    else:
        space_left = text_rect.left() - gap - area.left()
        space_right = area.left() + area.width() - (text_rect.left() + text_rect.width()) - gap
        width = max(space_left, space_right)
    width = max(min_width, min(max_width, width))
    return max(1, min(width, area.width()))


def place_bubble(
    text_rect: QRect,
    size: QSize,
    direction: TranslationDirection,
    area: QRect,
    gap: int,
) -> tuple[QRect, TranslationDirection]:
    """말풍선 사각형과 실제로 쓴 방향을 반환한다."""
    candidates = [direction, opposite(direction)]
    candidates += [d for d in _FALLBACK_ORDER if d not in candidates]
    for candidate in candidates:
        rect = _rect_for(text_rect, size, candidate, area, gap)
        if _inside(rect, area):
            return rect, candidate
    # 어느 방향에도 다 들어가지 않으면 지정 방향 위치를 화면 안으로 당긴다.
    return _clamp(_rect_for(text_rect, size, direction, area, gap), area), direction


def _rect_for(
    text_rect: QRect, size: QSize, direction: TranslationDirection, area: QRect, gap: int
) -> QRect:
    """한 방향에 놓았을 때의 사각형. 놓는 방향과 수직인 축은 원문 가운데에 맞추고 화면 안으로 당긴다."""
    width, height = size.width(), size.height()
    text_left, text_top = text_rect.left(), text_rect.top()
    text_right = text_left + text_rect.width()  # 원문 밖 첫 픽셀
    text_bottom = text_top + text_rect.height()
    center_x = text_left + text_rect.width() // 2
    center_y = text_top + text_rect.height() // 2

    if direction is TranslationDirection.BOTTOM:
        x, y = center_x - width // 2, text_bottom + gap
    elif direction is TranslationDirection.TOP:
        x, y = center_x - width // 2, text_top - gap - height
    elif direction is TranslationDirection.RIGHT:
        x, y = text_right + gap, center_y - height // 2
    else:
        x, y = text_left - gap - width, center_y - height // 2

    # 놓는 방향과 수직인 축은 화면 안으로 당겨도 원문을 가리지 않는다.
    if direction in (TranslationDirection.TOP, TranslationDirection.BOTTOM):
        x = _clamp_axis(x, width, area.left(), area.width())
    else:
        y = _clamp_axis(y, height, area.top(), area.height())
    return QRect(x, y, width, height)


def _clamp_axis(position: int, length: int, area_start: int, area_length: int) -> int:
    """한 축에서 [area_start, area_start + area_length) 안에 들어가도록 시작 위치를 당긴다."""
    return max(area_start, min(position, area_start + area_length - length))


def _inside(rect: QRect, area: QRect) -> bool:
    """사각형이 작업 영역 안에 완전히 들어가는지 여부."""
    return (
        rect.left() >= area.left()
        and rect.top() >= area.top()
        and rect.left() + rect.width() <= area.left() + area.width()
        and rect.top() + rect.height() <= area.top() + area.height()
    )


def _clamp(rect: QRect, area: QRect) -> QRect:
    """사각형을 작업 영역 안으로 당긴다 (크기가 더 크면 왼쪽 위에 맞춘다)."""
    x = _clamp_axis(rect.left(), rect.width(), area.left(), area.width())
    y = _clamp_axis(rect.top(), rect.height(), area.top(), area.height())
    return QRect(x, y, rect.width(), rect.height())
