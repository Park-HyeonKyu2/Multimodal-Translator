"""원문 텍스트 옆에 번역문을 보여 주는 반투명 말풍선 창.

말풍선은 표시 전용이다. 항상 클릭을 아래 프로그램으로 통과시키고, 포커스를 빼앗지
않으며, 화면 캡처에서 제외된다(번역문이 다시 OCR되지 않도록).
"""

from PySide6.QtCore import QRect, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPaintEvent
from PySide6.QtWidgets import QWidget

from translator.core.models import TranslationDirection
from translator.tuning import (
    BUBBLE_FONT_SIZE_PX,
    BUBBLE_GAP,
    BUBBLE_MAX_HEIGHT_RATIO,
    BUBBLE_MAX_WIDTH,
    BUBBLE_MIN_WIDTH,
    BUBBLE_PADDING,
    RESULT_WINDOW_DEFAULT_OPACITY,
)
from translator.ui.bubble_layout import place_bubble, preferred_width
from translator.ui.window_utils import exclude_from_screen_capture

# 박스 색 띠의 두께 (논리 픽셀). 어느 박스의 결과인지 색으로 연결해 보여 준다.
_ACCENT_WIDTH = 3
_ELLIPSIS = "…"


class TranslationBubble(QWidget):
    """번역문 하나를 표시하는 말풍선."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """클릭 통과·포커스 없음·항상 위 속성을 가진 투명 창을 만든다."""
        super().__init__(parent)
        self.setWindowTitle("번역")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowTransparentForInput
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        font = QFont(self.font())
        font.setPixelSize(BUBBLE_FONT_SIZE_PX)
        self.setFont(font)
        # 화면에 그릴 (필요하면 말줄임표로 자른) 번역문
        self._display_text = ""
        # 박스와 연결해 보여 주는 색
        self._accent = QColor(0, 170, 255)
        # 배경 불투명도 (0~1). 글자는 항상 선명하게 그린다.
        self._opacity = RESULT_WINDOW_DEFAULT_OPACITY
        # 실제로 놓인 방향 (화면 경계 때문에 지정 방향과 다를 수 있음)
        self._placed_direction: TranslationDirection | None = None

    @property
    def display_text(self) -> str:
        """현재 그리는 번역문 (잘렸으면 말줄임표 포함)."""
        return self._display_text

    @property
    def placed_direction(self) -> TranslationDirection | None:
        """화면 경계 보정 후 실제로 놓인 방향."""
        return self._placed_direction

    def set_background_opacity(self, opacity: float) -> None:
        """배경 불투명도를 바꾼다."""
        self._opacity = opacity
        self.update()

    def present(
        self,
        text: str,
        text_rect: QRect,
        area: QRect,
        direction: TranslationDirection,
        accent: QColor,
    ) -> None:
        """원문 사각형(논리 좌표) 옆에 번역문을 배치하고 보여 준다.

        `area`는 원문이 있는 모니터의 작업 영역이다. 번역문의 줄바꿈 문자는 그대로 줄을
        바꾼다. 번역문이 길면 말풍선 너비에서 줄을 바꾸고, 그래도 최대 높이를 넘으면
        말줄임표로 자른다.
        """
        self._accent = QColor(accent)
        metrics = QFontMetrics(self.font())
        # 바깥 너비 → 글자를 쓸 안쪽 너비 (여백과 색 띠 제외)
        horizontal_chrome = 2 * BUBBLE_PADDING + _ACCENT_WIDTH
        outer_width = preferred_width(
            text_rect, direction, area, BUBBLE_GAP, BUBBLE_MIN_WIDTH, BUBBLE_MAX_WIDTH
        )
        text_width = max(1, outer_width - horizontal_chrome)
        # 줄마다 한 줄로 충분히 들어가면 말풍선을 가장 긴 줄의 길이에 맞춰 줄인다.
        single_line = max(metrics.horizontalAdvance(line) for line in text.splitlines() or [""])
        if single_line < text_width:
            text_width = max(single_line, BUBBLE_MIN_WIDTH - horizontal_chrome, 1)
        max_text_height = max(
            metrics.height(), int(area.height() * BUBBLE_MAX_HEIGHT_RATIO) - 2 * BUBBLE_PADDING
        )
        self._display_text = fit_text(text, metrics, text_width, max_text_height)
        text_height = _wrapped_height(self._display_text, metrics, text_width)

        size = QSize(text_width + horizontal_chrome, text_height + 2 * BUBBLE_PADDING)
        geometry, self._placed_direction = place_bubble(text_rect, size, direction, area, BUBBLE_GAP)
        self.setGeometry(geometry)
        self.update()
        if not self.isVisible():
            self.show()

    def showEvent(self, event) -> None:  # noqa: N802 (Qt 이름 규칙)
        """표시될 때마다 캡처 제외를 다시 적용한다 (네이티브 창이 다시 만들어질 수 있음)."""
        super().showEvent(event)
        exclude_from_screen_capture(self)

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        """반투명 배경, 박스 색 띠, 번역문을 그린다."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(18, 18, 22, round(self._opacity * 255)))
        painter.drawRoundedRect(QRectF(self.rect()), 8, 8)
        painter.fillRect(QRect(0, 6, _ACCENT_WIDTH, self.height() - 12), self._accent)
        painter.setPen(QColor(255, 255, 255))
        text_rect = self.rect().adjusted(
            BUBBLE_PADDING + _ACCENT_WIDTH, BUBBLE_PADDING, -BUBBLE_PADDING, -BUBBLE_PADDING
        )
        painter.drawText(text_rect, int(Qt.TextFlag.TextWordWrap), self._display_text)


def fit_text(text: str, metrics: QFontMetrics, width: int, max_height: int) -> str:
    """줄바꿈했을 때 `max_height` 안에 들어가도록 필요하면 뒤를 잘라 말줄임표를 붙인다."""
    if _wrapped_height(text, metrics, width) <= max_height:
        return text
    # 들어가는 가장 긴 앞부분을 이분 탐색으로 찾는다.
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if _wrapped_height(text[:middle].rstrip() + _ELLIPSIS, metrics, width) <= max_height:
            low = middle
        else:
            high = middle - 1
    return text[:low].rstrip() + _ELLIPSIS


def _wrapped_height(text: str, metrics: QFontMetrics, width: int) -> int:
    """`width` 너비로 줄바꿈했을 때의 글자 높이."""
    bounds = metrics.boundingRect(QRect(0, 0, width, 100_000), int(Qt.TextFlag.TextWordWrap), text)
    return max(metrics.height(), bounds.height())
