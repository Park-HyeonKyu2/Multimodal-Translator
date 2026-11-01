"""최소화된 항상 위(always-on-top) 오버레이 창.

잠금(편집이 꺼진) 상태에서도 오버레이는 항상 클릭할 수 있어야 하므로, 모든 조작 버튼을
이곳에 모은다.
"""

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QGuiApplication, QMouseEvent, QPainter, QPaintEvent
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

from translator.ui.window_utils import exclude_from_screen_capture

_STYLE = """
QLabel#dragHandle { color: rgba(255, 255, 255, 150); padding: 0 2px; }
QPushButton {
    color: white; background: rgba(255, 255, 255, 30);
    border: none; border-radius: 6px; padding: 4px 10px;
}
QPushButton:hover { background: rgba(255, 255, 255, 70); }
QPushButton:pressed { background: rgba(255, 255, 255, 110); }
QPushButton:checked { background: rgba(0, 170, 255, 170); }
QPushButton:disabled { color: rgba(255, 255, 255, 80); background: rgba(255, 255, 255, 12); }
QPushButton#closeButton { padding: 4px 8px; }
QPushButton#closeButton:hover { background: rgba(220, 60, 60, 200); }
"""


class MiniOverlay(QWidget):
    """빈 공간을 드래그하면 이동하는 작은 도구 막대.

    작업표시줄에 나타나지 않고 포커스를 빼앗지 않도록 Tool 창으로 만든다.
    """

    region_requested = Signal()
    full_screen_toggled = Signal(bool)
    region_translate_requested = Signal()
    auto_toggled = Signal(bool)
    edit_toggled = Signal(bool)
    close_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """버튼과 레이아웃을 만들고 시그널을 연결한다."""
        super().__init__(parent)
        self.setWindowTitle("Multimodal Translator")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet(_STYLE)

        handle = QLabel("⋮⋮", self)
        handle.setObjectName("dragHandle")
        handle.setToolTip("드래그하여 이동")
        # 핸들 위에서 누른 마우스도 창 이동으로 처리되도록 이벤트를 통과시킨다.
        handle.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        self.region_button = QPushButton("영역 추가", self)
        self.region_button.setToolTip("드래그로 번역 영역(박스)을 하나 더 추가합니다")
        self.full_screen_button = QPushButton("전체 화면", self)
        # 켜져 있는 동안 화면 전체의 텍스트를 추적하며 번역한다.
        self.full_screen_button.setCheckable(True)
        self.full_screen_button.setToolTip(
            "켜면 모든 모니터의 텍스트를 번역하고, 화면이 바뀌면 위치와 번역을 갱신합니다. "
            "끄면 전체 화면 번역을 지웁니다 (박스 안 텍스트는 각 박스가 맡습니다)"
        )
        self.translate_region_button = QPushButton("영역 번역", self)
        self.translate_region_button.setToolTip("모든 박스 안의 텍스트를 지금 번역합니다")
        self.auto_button = QPushButton("자동", self)
        self.auto_button.setCheckable(True)
        self.auto_button.setToolTip(
            "모든 박스의 자동 감지를 켜거나 끕니다 (박스별 설정은 편집 모드에서 우클릭)"
        )
        # 편집이 꺼진 상태가 예전의 "잠금"이다: 박스가 흐려지고 클릭이 통과한다.
        self.edit_button = QPushButton("편집", self)
        self.edit_button.setCheckable(True)
        self.edit_button.setToolTip(
            "켜면 박스를 옮기고 크기·출력 방향을 바꿀 수 있습니다. 끄면 박스가 흐려지고 "
            "박스와 번역문이 클릭을 통과시켜 게임 조작을 방해하지 않습니다(잠금)"
        )
        self.close_button = QPushButton("✕", self)
        self.close_button.setObjectName("closeButton")
        self.close_button.setToolTip("프로그램 종료")

        self.region_button.clicked.connect(self.region_requested)
        self.full_screen_button.toggled.connect(self.full_screen_toggled)
        self.translate_region_button.clicked.connect(self.region_translate_requested)
        self.auto_button.toggled.connect(self.auto_toggled)
        self.edit_button.toggled.connect(self.edit_toggled)
        self.close_button.clicked.connect(self.close)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 6, 6)
        layout.setSpacing(4)
        layout.addWidget(handle)
        for button in (
            self.region_button,
            self.full_screen_button,
            self.translate_region_button,
            self.auto_button,
            self.edit_button,
            self.close_button,
        ):
            layout.addWidget(button)
        self.set_region_available(False)
        self.adjustSize()

    def set_region_available(self, available: bool) -> None:
        """번역 영역(박스)이 하나라도 있을 때만 영역 번역/자동 버튼을 쓸 수 있게 한다."""
        self.translate_region_button.setEnabled(available)
        self.auto_button.setEnabled(available)
        if not available:
            self.set_auto_checked(False)

    def set_auto_checked(self, checked: bool) -> None:
        """시그널을 내보내지 않고 자동 버튼 상태만 맞춘다."""
        self.auto_button.blockSignals(True)
        self.auto_button.setChecked(checked)
        self.auto_button.blockSignals(False)

    def set_full_screen_checked(self, checked: bool) -> None:
        """시그널을 내보내지 않고 전체 화면 버튼 상태만 맞춘다."""
        self.full_screen_button.blockSignals(True)
        self.full_screen_button.setChecked(checked)
        self.full_screen_button.blockSignals(False)

    def set_edit_checked(self, checked: bool) -> None:
        """시그널을 내보내지 않고 편집 버튼 상태만 맞춘다."""
        self.edit_button.blockSignals(True)
        self.edit_button.setChecked(checked)
        self.edit_button.blockSignals(False)

    def move_to_default_position(self) -> None:
        """주 모니터 작업 영역의 위쪽 가운데에 배치한다."""
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return
        area = screen.availableGeometry()
        self.move(area.center().x() - self.width() // 2, area.top() + 12)

    def showEvent(self, event) -> None:  # noqa: N802 (Qt 이름 규칙)
        """표시될 때마다 캡처 제외를 다시 적용한다 (네이티브 창이 다시 만들어질 수 있음)."""
        super().showEvent(event)
        exclude_from_screen_capture(self)

    def closeEvent(self, event) -> None:  # noqa: N802
        """창이 닫히면 프로그램 종료를 요청한다."""
        self.close_requested.emit()
        super().closeEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        """빈 공간을 누르면 창 이동을 시작한다."""
        if event.button() == Qt.MouseButton.LeftButton and self.windowHandle() is not None:
            # 시스템 이동은 다중 모니터/DPI가 다른 모니터 사이에서도 올바르게 동작한다.
            self.windowHandle().startSystemMove()
        super().mousePressEvent(event)

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        """둥근 반투명 배경을 그린다."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(20, 20, 24, 200))
        painter.drawRoundedRect(QRectF(self.rect()), 10, 10)
