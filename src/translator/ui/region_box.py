"""화면에 남아 있는 번역 영역(바운딩 박스) 창.

- 편집 모드: 테두리·손잡이·이름표(ID, 출력 방향, 자동 감지)가 보이고, 드래그로 이동,
  가장자리/모서리로 크기 조절, 우클릭 메뉴로 방향·자동 감지·번역·삭제를 고른다.
- 잠금 모드(편집이 아닐 때): 흐린 점선 테두리만 남고 클릭은 아래 프로그램으로 통과한다.

박스 창 자체는 캡처에서 제외되며, 테두리 여백 안쪽(내부)만 캡처 영역으로 쓴다.
이 창은 화면 표시와 입력만 담당하고, 설정 저장·번역 요청 같은 처리 로직은
`RegionSession`이 맡는다.
"""

from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QColor,
    QContextMenuEvent,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
)
from PySide6.QtWidgets import QMenu, QPushButton, QWidget

from translator.core.models import (
    DEFAULT_TRANSLATION_DIRECTION,
    ScreenRegion,
    TranslationDirection,
)
from translator.tuning import (
    REGION_BOX_IDLE_ALPHA,
    REGION_BOX_MARGIN,
    REGION_BOX_MIN_HEIGHT,
    REGION_BOX_MIN_WIDTH,
)
from translator.ui.screen_geometry import global_logical_rect_to_physical
from translator.ui.window_utils import exclude_from_screen_capture, set_click_through

# 테두리 안쪽으로 이만큼까지는 크기 조절 영역으로 본다 (논리 픽셀).
_RESIZE_GRIP = 6

# 박스와 그 번역 말풍선을 같은 색으로 연결해 보여 주기 위한 색 목록 (차례로 돌려 쓴다)
REGION_COLORS = (
    QColor(0, 170, 255),
    QColor(255, 140, 0),
    QColor(80, 200, 120),
    QColor(220, 90, 220),
    QColor(240, 200, 40),
    QColor(255, 90, 90),
)

# 출력 방향을 사용자에게 보여 줄 이름 (화살표 + 한국어)
DIRECTION_LABELS = {
    TranslationDirection.TOP: "↑ 위",
    TranslationDirection.BOTTOM: "↓ 아래",
    TranslationDirection.LEFT: "← 왼쪽",
    TranslationDirection.RIGHT: "→ 오른쪽",
}

_STYLE = """
QPushButton {
    color: white; background: rgba(20, 20, 24, 200);
    border: none; border-radius: 4px; padding: 0px 5px;
}
QPushButton:hover { background: rgba(220, 60, 60, 220); }
"""


def color_for_index(index: int) -> QColor:
    """박스 순번에 맞는 색."""
    return QColor(REGION_COLORS[index % len(REGION_COLORS)])


class RegionBox(QWidget):
    """번역 영역을 표시하고 이동/크기 조절/우클릭 메뉴 입력을 받는 테두리 창."""

    geometry_changed = Signal()
    remove_requested = Signal()
    direction_selected = Signal(object)  # TranslationDirection
    auto_toggled = Signal(bool)
    translate_requested = Signal()

    def __init__(
        self,
        region_id: str = "",
        accent: QColor | None = None,
        parent: QWidget | None = None,
    ) -> None:
        """테두리 창과 삭제 버튼을 만든다. `accent`는 박스와 말풍선을 잇는 색이다."""
        super().__init__(parent)
        self.setWindowTitle(f"번역 영역 {region_id}".strip())
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setMouseTracking(True)
        self.setStyleSheet(_STYLE)
        self.setMinimumSize(
            REGION_BOX_MIN_WIDTH + 2 * REGION_BOX_MARGIN,
            REGION_BOX_MIN_HEIGHT + 2 * REGION_BOX_MARGIN,
        )
        # 이름표에 보여 줄 박스 ID
        self._region_id = region_id
        # 테두리·이름표·말풍선 색 띠에 쓰는 박스 색
        self._accent = QColor(accent) if accent is not None else color_for_index(0)
        # True: 편집 모드(이동/크기 조절/메뉴 가능), False: 잠금 모드(흐린 점선, 클릭 통과)
        self._interactive = True
        # 이름표에 보여 줄 현재 설정 (실제 설정은 RegionSession이 가진다)
        self._direction = DEFAULT_TRANSLATION_DIRECTION
        self._auto_enabled = False

        self.remove_button = QPushButton("✕", self)
        self.remove_button.setToolTip("번역 영역 삭제")
        self.remove_button.setCursor(Qt.CursorShape.ArrowCursor)
        self.remove_button.clicked.connect(self.remove_requested)
        self.remove_button.adjustSize()

    @property
    def region_id(self) -> str:
        """박스 ID."""
        return self._region_id

    @property
    def accent(self) -> QColor:
        """박스 색."""
        return QColor(self._accent)

    @property
    def is_interactive(self) -> bool:
        """편집 모드(클릭을 받는 상태)인지 여부."""
        return self._interactive

    def set_capture_rect(self, rect: QRect) -> None:
        """캡처할 내부 영역(가상 데스크톱 논리 좌표)에 맞춰 박스를 배치한다."""
        m = REGION_BOX_MARGIN
        self.setGeometry(rect.adjusted(-m, -m, m, m))

    def capture_rect(self) -> QRect:
        """테두리 여백을 뺀 내부 영역 (가상 데스크톱 논리 좌표)."""
        m = REGION_BOX_MARGIN
        return self.geometry().adjusted(m, m, -m, -m)

    def capture_region(self) -> ScreenRegion | None:
        """캡처할 물리 픽셀 영역. 박스가 흐리게(잠금) 표시 중이어도 계산한다."""
        return global_logical_rect_to_physical(self.capture_rect())

    def set_direction(self, direction: TranslationDirection) -> None:
        """이름표에 보여 줄 출력 방향을 바꾼다."""
        self._direction = direction
        self.update()

    def set_auto_enabled(self, enabled: bool) -> None:
        """이름표에 보여 줄 자동 감지 상태를 바꾼다."""
        self._auto_enabled = enabled
        self.update()

    def label_text(self) -> str:
        """편집 모드 이름표 문구 (예: "box-1 · ↓ 아래 · 자동")."""
        parts = [self._region_id, DIRECTION_LABELS[self._direction]]
        if self._auto_enabled:
            parts.append("자동")
        return " · ".join(part for part in parts if part)

    def set_interactive(self, interactive: bool) -> None:
        """False이면 삭제 버튼을 숨기고 흐린 점선으로 바꾼 뒤 클릭을 통과시킨다."""
        self._interactive = interactive
        self.remove_button.setVisible(interactive)
        if not interactive:
            self.unsetCursor()
        set_click_through(self, not interactive)
        self.update()

    def build_context_menu(self) -> QMenu:
        """우클릭 메뉴: 출력 방향, 자동 감지, 지금 번역, 삭제."""
        menu = QMenu(self)
        direction_menu = menu.addMenu("번역 출력 방향")
        group = QActionGroup(direction_menu)
        group.setExclusive(True)
        for direction, label in DIRECTION_LABELS.items():
            action = QAction(label, direction_menu)
            action.setCheckable(True)
            action.setChecked(direction is self._direction)
            # 기본 인자로 방향을 고정해야 반복문의 마지막 값으로 묶이지 않는다.
            action.triggered.connect(
                lambda _checked=False, d=direction: self.direction_selected.emit(d)
            )
            group.addAction(action)
            direction_menu.addAction(action)

        auto_action = menu.addAction("자동 감지")
        auto_action.setCheckable(True)
        auto_action.setChecked(self._auto_enabled)
        auto_action.toggled.connect(self.auto_toggled)

        menu.addAction("지금 번역").triggered.connect(self.translate_requested)
        menu.addSeparator()
        menu.addAction("영역 삭제").triggered.connect(self.remove_requested)
        return menu

    def edges_at(self, pos: QPoint) -> Qt.Edge:
        """창 내부 좌표 `pos`에서 크기 조절할 가장자리 조합. 해당 없으면 빈 값."""
        grip = REGION_BOX_MARGIN + _RESIZE_GRIP
        edges = Qt.Edge(0)
        if pos.x() < grip:
            edges |= Qt.Edge.LeftEdge
        elif pos.x() >= self.width() - grip:
            edges |= Qt.Edge.RightEdge
        if pos.y() < grip:
            edges |= Qt.Edge.TopEdge
        elif pos.y() >= self.height() - grip:
            edges |= Qt.Edge.BottomEdge
        return edges

    def showEvent(self, event) -> None:  # noqa: N802
        """표시될 때마다 캡처 제외를 다시 적용한다 (네이티브 창이 다시 만들어질 수 있음)."""
        super().showEvent(event)
        exclude_from_screen_capture(self)

    def moveEvent(self, event) -> None:  # noqa: N802
        """이동하면 감시 영역이 바뀌었음을 알린다."""
        super().moveEvent(event)
        self.geometry_changed.emit()

    def resizeEvent(self, event) -> None:  # noqa: N802
        """크기가 바뀌면 삭제 버튼 위치를 맞추고 감시 영역이 바뀌었음을 알린다."""
        super().resizeEvent(event)
        m = REGION_BOX_MARGIN
        self.remove_button.move(self.width() - m - self.remove_button.width() - 2, m + 2)
        self.geometry_changed.emit()

    def contextMenuEvent(self, event: QContextMenuEvent) -> None:  # noqa: N802
        """편집 모드에서 우클릭하면 메뉴를 띄운다."""
        if not self._interactive:
            return
        menu = self.build_context_menu()
        menu.exec(event.globalPos())
        menu.deleteLater()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        """편집 모드에서 가장자리를 누르면 크기 조절, 안쪽을 누르면 이동을 시작한다."""
        handle = self.windowHandle()
        if not self._interactive or handle is None:
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        edges = self.edges_at(event.position().toPoint())
        if edges:
            handle.startSystemResize(edges)
        else:
            handle.startSystemMove()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        """마우스 위치에 맞춰 크기 조절/이동 커서를 보여 준다."""
        if self._interactive:
            self.setCursor(_cursor_for(self.edges_at(event.position().toPoint())))
        super().mouseMoveEvent(event)

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        """편집 모드에서는 테두리·손잡이·이름표를, 잠금 모드에서는 흐린 점선만 그린다."""
        painter = QPainter(self)
        m = REGION_BOX_MARGIN
        inner = self.rect().adjusted(m, m, -m, -m)
        if not self._interactive:
            faint = QColor(self._accent)
            faint.setAlpha(REGION_BOX_IDLE_ALPHA)
            painter.setPen(QPen(faint, 1, Qt.PenStyle.DashLine))
            painter.drawRect(inner.adjusted(-1, -1, 0, 0))
            return

        # 완전히 투명한 픽셀은 클릭이 통과하므로, 편집 중에는 아주 옅게 채워
        # 박스 어디를 눌러도 이동/크기 조절이 되게 한다.
        painter.fillRect(self.rect(), QColor(0, 0, 0, 1))
        tint = QColor(self._accent)
        tint.setAlpha(18)
        painter.fillRect(inner, tint)
        painter.setPen(QPen(self._accent, 2))
        painter.drawRect(inner.adjusted(-1, -1, 0, 0))
        for corner in (inner.topLeft(), inner.topRight(), inner.bottomLeft(), inner.bottomRight()):
            painter.fillRect(QRect(corner.x() - 4, corner.y() - 4, 8, 8), self._accent)

        # 이름표: 박스 ID, 출력 방향, 자동 감지 여부를 왼쪽 위에 표시한다.
        label = self.label_text()
        metrics = painter.fontMetrics()
        tag = QRect(inner.left() + 2, inner.top() + 2, metrics.horizontalAdvance(label) + 10, metrics.height() + 4)
        painter.fillRect(tag, QColor(20, 20, 24, 200))
        painter.setPen(QColor(self._accent).lighter(130))
        painter.drawText(tag, int(Qt.AlignmentFlag.AlignCenter), label)


def _cursor_for(edges: Qt.Edge) -> Qt.CursorShape:
    """크기 조절 가장자리 조합에 맞는 마우스 커서 모양을 고른다."""
    horizontal = edges & (Qt.Edge.LeftEdge | Qt.Edge.RightEdge)
    vertical = edges & (Qt.Edge.TopEdge | Qt.Edge.BottomEdge)
    if horizontal and vertical:
        main_diagonal = edges in (
            Qt.Edge.LeftEdge | Qt.Edge.TopEdge,
            Qt.Edge.RightEdge | Qt.Edge.BottomEdge,
        )
        return Qt.CursorShape.SizeFDiagCursor if main_diagonal else Qt.CursorShape.SizeBDiagCursor
    if horizontal:
        return Qt.CursorShape.SizeHorCursor
    if vertical:
        return Qt.CursorShape.SizeVerCursor
    return Qt.CursorShape.SizeAllCursor
