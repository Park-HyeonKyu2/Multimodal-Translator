"""상태·설정 창 (예전 "번역 결과 창").

번역문은 이제 원문 옆 말풍선(`TranslationBubble`)에 표시하므로, 이 창은 두 가지 역할만 한다.

- 알림: 오류나 "인식된 텍스트가 없습니다" 같은 안내를 잠깐 보여 준다. 잠금 상태에서는
  클릭을 통과시키므로 일정 시간 뒤 스스로 사라진다.
- 설정: 편집 모드에서는 말풍선 배경 불투명도와 전체 화면 번역의 출력 방향을 조절한다.

이동·크기 조절·불투명도·클릭 통과 등 기존 창 기능과 저장되는 설정(위치, 크기,
불투명도)은 그대로 유지한다. 클래스 이름과 저장 키도 호환을 위해 유지한다.
"""

from PySide6.QtCore import QRect, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QGuiApplication, QMouseEvent, QPainter, QPaintEvent
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizeGrip,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from translator.core.models import DEFAULT_TRANSLATION_DIRECTION, TranslationDirection
from translator.tuning import (
    RESULT_WINDOW_DEFAULT_OPACITY,
    RESULT_WINDOW_MIN_OPACITY,
    STATUS_ERROR_HIDE_MS,
    STATUS_NOTICE_HIDE_MS,
)
from translator.ui.region_box import DIRECTION_LABELS
from translator.ui.screen_geometry import is_rect_on_any_screen
from translator.ui.window_utils import exclude_from_screen_capture, set_click_through

_STYLE = """
QLabel { color: white; }
QLabel#titleLabel { color: rgba(255, 255, 255, 160); font-size: 11px; }
QLabel#statusLabel { color: rgba(255, 220, 120, 230); }
QLabel#statusLabel[error="true"] { color: rgba(255, 130, 130, 240); }
QLabel#hintLabel, QLabel#opacityLabel, QLabel#directionLabel {
    color: rgba(255, 255, 255, 150); font-size: 11px;
}
QPushButton {
    color: white; background: transparent; border: none;
    border-radius: 4px; padding: 2px 6px;
}
QPushButton:hover { background: rgba(220, 60, 60, 200); }
"""

EDIT_HINT = (
    "편집 모드: 박스를 드래그해 옮기거나 크기를 바꾸고, 우클릭으로 출력 방향·자동 감지를 "
    "정할 수 있습니다. '편집'을 끄면 박스가 흐려지고 클릭이 통과됩니다."
)


class ResultWindow(QWidget):
    """알림과 표시 설정을 담는 반투명 창.

    편집 모드에서는 드래그로 이동하고 오른쪽 아래 모서리로 크기를 조절하며, 설정
    항목이 보인다. 잠금 모드에서는 조작 요소를 숨기고 클릭을 아래 프로그램으로
    통과시킨다. 표시될 때 포커스를 빼앗지 않는다.
    """

    closed = Signal()
    opacity_changed = Signal(float)
    fullscreen_direction_changed = Signal(object)  # TranslationDirection

    def __init__(self, parent: QWidget | None = None) -> None:
        """알림 문구, 설정 항목(말풍선 배경 불투명도, 전체 화면 번역 위치)과 레이아웃을 만든다."""
        super().__init__(parent)
        self.setWindowTitle("상태")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setStyleSheet(_STYLE)
        self.setMinimumSize(280, 90)
        self.resize(420, 150)
        # 위치가 정해졌는지(사용자 이동, 설정 복원, 기본 위치 배치) 여부.
        # 처음 표시할 때 기본 위치로 옮길지, 종료할 때 위치를 저장할지 판단한다.
        self._has_position = False
        # True: 편집 모드(이동/크기/설정 조절 가능), False: 잠금 모드(클릭 통과).
        # 잠금 상태로 시작한다 (편집 모드는 사용자가 켠다).
        self._interactive = False
        # 배경 불투명도 (0~1). 이 창과 번역 말풍선에 함께 적용한다. 글자에는 적용하지 않는다.
        self._background_opacity = RESULT_WINDOW_DEFAULT_OPACITY
        # 전체 화면 번역 말풍선의 출력 방향
        self._fullscreen_direction = DEFAULT_TRANSLATION_DIRECTION
        # 잠금 상태에서 알림을 자동으로 숨기는 타이머
        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self._on_hide_timer)

        title = QLabel("Multimodal Translator", self)
        title.setObjectName("titleLabel")
        self.close_button = QPushButton("✕", self)
        self.close_button.setToolTip("창 닫기")
        self.close_button.clicked.connect(self.close)

        header = QHBoxLayout()
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(self.close_button)

        # OCR 결과 등에 섞인 태그가 해석되지 않도록 일반 텍스트로 표시한다.
        self.status_label = self._make_label("statusLabel")
        self.hint_label = self._make_label("hintLabel")
        self.hint_label.setText(EDIT_HINT)

        min_percent = round(RESULT_WINDOW_MIN_OPACITY * 100)
        self.opacity_label = QLabel(self)
        self.opacity_label.setObjectName("opacityLabel")
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.opacity_slider.setRange(min_percent, 100)
        self.opacity_slider.setFixedWidth(110)
        self.opacity_slider.setToolTip("번역 말풍선과 이 창의 배경 불투명도 (글자는 항상 선명하게 표시)")
        self.opacity_slider.valueChanged.connect(
            lambda percent: self.set_background_opacity(percent / 100)
        )

        self.direction_label = QLabel("전체 화면 번역 위치", self)
        self.direction_label.setObjectName("directionLabel")
        self.direction_combo = QComboBox(self)
        for direction, label in DIRECTION_LABELS.items():
            self.direction_combo.addItem(label, direction)
        self.direction_combo.setToolTip(
            "전체 화면 번역의 말풍선을 원문의 어느 쪽에 띄울지 (박스는 박스를 우클릭해 따로 정합니다)"
        )
        self.direction_combo.currentIndexChanged.connect(self._on_direction_combo)

        self.size_grip = QSizeGrip(self)

        # 설정 이름이 길어 한 줄에 두면 창이 넓어지므로 항목마다 줄을 나눈다.
        opacity_row = QHBoxLayout()
        opacity_row.setSpacing(8)
        opacity_row.addWidget(self.opacity_label)
        opacity_row.addWidget(self.opacity_slider)
        opacity_row.addStretch(1)
        direction_row = QHBoxLayout()
        direction_row.setSpacing(8)
        direction_row.addWidget(self.direction_label)
        direction_row.addWidget(self.direction_combo)
        direction_row.addStretch(1)
        direction_row.addWidget(self.size_grip, 0, Qt.AlignmentFlag.AlignBottom)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 6, 4)
        layout.setSpacing(6)
        layout.addLayout(header)
        layout.addWidget(self.status_label)
        layout.addWidget(self.hint_label)
        layout.addStretch(1)
        layout.addLayout(opacity_row)
        layout.addLayout(direction_row)
        self.set_background_opacity(self._background_opacity)
        self.set_fullscreen_direction(self._fullscreen_direction)
        self._settings_widgets = (
            self.close_button,
            self.hint_label,
            self.opacity_label,
            self.opacity_slider,
            self.direction_label,
            self.direction_combo,
            self.size_grip,
        )
        # 잠금 상태에 맞춰 설정 항목을 숨기고 클릭을 통과시킨다 (창은 아직 띄우지 않는다).
        for widget in self._settings_widgets:
            widget.setVisible(False)
        set_click_through(self, True)

    @property
    def background_opacity(self) -> float:
        """현재 배경 불투명도 (0~1)."""
        return self._background_opacity

    @property
    def fullscreen_direction(self) -> TranslationDirection:
        """전체 화면 번역의 출력 방향."""
        return self._fullscreen_direction

    @property
    def is_interactive(self) -> bool:
        """편집 모드(클릭을 받는 상태)인지 여부."""
        return self._interactive

    @property
    def has_position(self) -> bool:
        """위치가 정해져 있어 저장할 가치가 있는지 여부."""
        return self._has_position

    def restore_geometry(self, geometry: QRect) -> bool:
        """저장된 위치와 크기를 적용한다. 화면 밖이면 적용하지 않고 False를 반환한다."""
        if not is_rect_on_any_screen(geometry):
            return False
        # 최소 크기보다 작게 저장된 값은 resize가 최소 크기로 맞춘다.
        self.resize(geometry.size())
        self.move(geometry.topLeft())
        self._has_position = True
        return True

    def set_background_opacity(self, opacity: float) -> None:
        """배경 불투명도(0~1)를 바꾼다. 클릭을 받을 수 있도록 최솟값 아래로는 내리지 않는다."""
        opacity = min(max(opacity, RESULT_WINDOW_MIN_OPACITY), 1.0)
        changed = opacity != self._background_opacity
        self._background_opacity = opacity
        percent = round(opacity * 100)
        self.opacity_label.setText(f"말풍선 배경 불투명도 {percent}%")
        if self.opacity_slider.value() != percent:
            self.opacity_slider.setValue(percent)
        self.update()
        if changed:
            self.opacity_changed.emit(opacity)

    def set_fullscreen_direction(self, direction: TranslationDirection) -> None:
        """전체 화면 번역의 출력 방향을 바꾼다."""
        changed = direction is not self._fullscreen_direction
        self._fullscreen_direction = direction
        index = self.direction_combo.findData(direction)
        if index >= 0 and self.direction_combo.currentIndex() != index:
            self.direction_combo.setCurrentIndex(index)
        if changed:
            self.fullscreen_direction_changed.emit(direction)

    def set_interactive(self, interactive: bool) -> None:
        """True면 설정 창으로 보여 주고, False면 조작 요소를 숨기고 클릭을 통과시킨다."""
        self._interactive = interactive
        for widget in self._settings_widgets:
            widget.setVisible(interactive)
        set_click_through(self, not interactive)
        if interactive:
            # 편집 중에는 설정 창으로 계속 보여 준다.
            self._hide_timer.stop()
            self._present()
        elif not self.status_label.text():
            # 보여 줄 알림이 없으면 편집이 끝날 때 숨긴다.
            self.hide()
        else:
            self._hide_timer.start(STATUS_NOTICE_HIDE_MS)

    def show_no_text(self) -> None:
        """텍스트가 없는 것은 오류가 아니므로 일반 안내로 잠깐 표시한다."""
        self.show_notice("인식된 텍스트가 없습니다.", is_error=False)

    def show_error(self, message: str) -> None:
        """오류 메시지를 표시한다."""
        self.show_notice(f"오류: {message}", is_error=True)

    def show_notice(self, text: str, *, is_error: bool) -> None:
        """알림을 보여 준다. 잠금 상태에서는 일정 시간 뒤 스스로 숨긴다."""
        self._set_status(text, is_error=is_error)
        self._present()
        if not self._interactive:
            self._hide_timer.start(STATUS_ERROR_HIDE_MS if is_error else STATUS_NOTICE_HIDE_MS)

    def ensure_on_screen(self) -> None:
        """모니터가 분리되어 창이 화면 밖에 남았으면 기본 위치로 옮긴다."""
        if not is_rect_on_any_screen(self.frameGeometry()):
            self.move_to_default_position()

    def move_to_default_position(self) -> None:
        """주 모니터 작업 영역의 아래쪽 가운데에 배치한다."""
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return
        area = screen.availableGeometry()
        self.move(area.center().x() - self.width() // 2, area.bottom() - self.height() - 40)
        self._has_position = True

    def _on_hide_timer(self) -> None:
        """알림 시간이 끝나면 문구를 지우고, 편집 중이 아니면 창을 숨긴다."""
        self._set_status("", is_error=False)
        if not self._interactive:
            self.hide()

    def _on_direction_combo(self, index: int) -> None:
        """콤보 상자에서 방향을 고르면 설정을 바꾼다."""
        direction = self.direction_combo.itemData(index)
        if isinstance(direction, TranslationDirection):
            self.set_fullscreen_direction(direction)

    def _present(self) -> None:
        """처음이면 기본 위치로 옮긴 뒤 창을 보여 주고 맨 앞으로 올린다."""
        if not self._has_position:
            self.move_to_default_position()
        self.show()
        self.raise_()

    def _set_status(self, text: str, *, is_error: bool) -> None:
        """상태 문구와 오류 여부(색)를 바꾼다. 빈 문구면 숨긴다."""
        self.status_label.setText(text)
        self.status_label.setVisible(bool(text))
        self.status_label.setProperty("error", is_error)
        # 동적 속성 변경을 스타일시트에 반영한다.
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)

    def _make_label(self, object_name: str) -> QLabel:
        """일반 텍스트만 표시하는 줄바꿈 레이블을 만든다."""
        label = QLabel(self)
        label.setObjectName(object_name)
        label.setTextFormat(Qt.TextFormat.PlainText)
        label.setWordWrap(True)
        label.setVisible(False)
        return label

    def showEvent(self, event) -> None:  # noqa: N802
        """표시될 때마다 캡처 제외를 다시 적용한다 (네이티브 창이 다시 만들어질 수 있음)."""
        super().showEvent(event)
        exclude_from_screen_capture(self)

    def closeEvent(self, event) -> None:  # noqa: N802
        """창이 닫히면 `closed` 시그널을 보낸다."""
        self.closed.emit()
        super().closeEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        """편집 모드에서 빈 곳을 누르면 창 이동을 시작한다."""
        if (
            self._interactive
            and event.button() == Qt.MouseButton.LeftButton
            and self.windowHandle() is not None
        ):
            self.windowHandle().startSystemMove()
        super().mousePressEvent(event)

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        """설정된 불투명도로 둥근 배경을 그린다."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(18, 18, 22, round(self._background_opacity * 255)))
        painter.drawRoundedRect(QRectF(self.rect()), 12, 12)
