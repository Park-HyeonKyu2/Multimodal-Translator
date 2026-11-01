"""화면 영역 선택 UI.

모니터마다 DPI 배율이 다를 수 있으므로 가상 데스크톱 전체를 덮는 창 하나 대신
모니터별로 반투명 창을 하나씩 띄운다. 선택 결과는 가상 데스크톱 논리 좌표
`QRect`로 전달하며, 물리 픽셀 변환은 캡처 직전에 `screen_geometry`에서 한다.
"""

import logging

from PySide6.QtCore import QObject, QPoint, QRect, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QCursor,
    QGuiApplication,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QScreen,
)
from PySide6.QtWidgets import QWidget

from translator.tuning import MIN_SELECTION_SIZE

logger = logging.getLogger(__name__)

class ScreenSelectionWindow(QWidget):
    """한 모니터를 덮고 마우스 드래그로 사각형을 선택받는 창."""

    selection_made = Signal(QRect)  # 가상 데스크톱 논리 좌표
    cancelled = Signal()

    def __init__(self, screen: QScreen) -> None:
        """한 모니터를 덮도록 창을 배치한다."""
        super().__init__()
        # 이 창이 덮는 모니터 (선택 결과를 가상 데스크톱 좌표로 바꿀 때 원점으로 쓴다)
        self._screen = screen
        # 드래그 시작점과 현재 점 (창 내부 논리 좌표). 드래그 중이 아니면 None.
        self._origin: QPoint | None = None
        self._current: QPoint | None = None

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setScreen(screen)
        self.setGeometry(screen.geometry())

    def selection_rect(self) -> QRect | None:
        """현재 드래그 중인 사각형(창 내부 논리 좌표)."""
        if self._origin is None or self._current is None:
            return None
        current = QPoint(
            min(max(self._current.x(), 0), self.width()),
            min(max(self._current.y(), 0), self.height()),
        )
        left = min(self._origin.x(), current.x())
        top = min(self._origin.y(), current.y())
        return QRect(
            left, top, abs(current.x() - self._origin.x()), abs(current.y() - self._origin.y())
        )

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        """왼쪽 버튼은 드래그 시작, 오른쪽 버튼은 취소."""
        if event.button() == Qt.MouseButton.RightButton:
            self.cancelled.emit()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._origin = event.position().toPoint()
            self._current = self._origin
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        """드래그 중이면 선택 사각형을 갱신한다."""
        if self._origin is not None:
            self._current = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        """드래그를 끝내고, 충분히 크면 가상 데스크톱 논리 좌표로 선택 결과를 보낸다."""
        if event.button() != Qt.MouseButton.LeftButton or self._origin is None:
            return
        self._current = event.position().toPoint()
        rect = self.selection_rect()
        self._origin = self._current = None
        self.update()
        if rect is None or rect.width() < MIN_SELECTION_SIZE or rect.height() < MIN_SELECTION_SIZE:
            logger.debug("선택 영역이 너무 작아 무시합니다.")
            return
        origin = self._screen.geometry().topLeft()
        self.selection_made.emit(rect.translated(origin.x(), origin.y()))

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        """Esc를 누르면 선택을 취소한다."""
        if event.key() == Qt.Key.Key_Escape:
            self.cancelled.emit()
            return
        super().keyPressEvent(event)

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        """화면을 어둡게 덮고 선택 사각형이나 안내 문구를 그린다."""
        painter = QPainter(self)
        # 완전히 투명한 픽셀은 클릭이 아래 창으로 통과하므로 알파를 0으로 두지 않는다.
        painter.fillRect(self.rect(), QColor(0, 0, 0, 90))
        rect = self.selection_rect()
        if rect is None:
            painter.setPen(QColor(255, 255, 255, 220))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "드래그하여 번역할 영역을 선택하세요  (Esc 또는 우클릭: 취소)",
            )
            return
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        painter.fillRect(rect, QColor(255, 255, 255, 12))
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        painter.setPen(QPen(QColor(0, 170, 255), 2))
        painter.drawRect(rect.adjusted(0, 0, -1, -1))


class RegionSelector(QObject):
    """모든 모니터에 선택 창을 띄우고 결과를 하나로 모아 전달한다."""

    region_selected = Signal(QRect)  # 가상 데스크톱 논리 좌표
    cancelled = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        """모니터 구성 변경을 감시하도록 연결한다."""
        super().__init__(parent)
        # 모니터마다 하나씩 띄운 선택 창. 비어 있으면 선택 중이 아니다.
        self._windows: list[ScreenSelectionWindow] = []
        app = QGuiApplication.instance()
        if app is not None:
            # 선택 도중 모니터 구성이 바뀌면 좌표가 무효해지므로 취소한다.
            app.screenAdded.connect(self._on_screens_changed)
            app.screenRemoved.connect(self._on_screens_changed)

    @property
    def is_active(self) -> bool:
        """선택 창이 떠 있는지 여부."""
        return bool(self._windows)

    @property
    def windows(self) -> list[ScreenSelectionWindow]:
        """현재 떠 있는 모니터별 선택 창 목록 (복사본)."""
        return list(self._windows)

    def start(self) -> None:
        """모든 모니터에 선택 창을 띄운다. 이미 선택 중이면 무시한다."""
        if self._windows:
            return
        screens = QGuiApplication.screens()
        if not screens:
            logger.error("사용 가능한 모니터가 없어 영역 선택을 시작할 수 없습니다.")
            self.cancelled.emit()
            return
        for screen in screens:
            window = ScreenSelectionWindow(screen)
            window.selection_made.connect(self._on_selection_made)
            window.cancelled.connect(self.cancel)
            self._windows.append(window)
            window.show()

        # Esc 키를 받을 수 있도록 커서가 있는 모니터의 창을 활성화한다.
        cursor_screen = QGuiApplication.screenAt(QCursor.pos()) or screens[0]
        for window in self._windows:
            if window.screen() is cursor_screen:
                window.activateWindow()
                window.setFocus()
        logger.debug("영역 선택 시작: 모니터 %d개", len(screens))

    def cancel(self) -> None:
        """선택 창을 닫고 취소를 알린다."""
        if not self._windows:
            return
        self._close_windows()
        logger.debug("영역 선택 취소")
        self.cancelled.emit()

    def _on_selection_made(self, rect: QRect) -> None:
        """한 모니터에서 선택이 끝나면 모든 선택 창을 닫고 결과를 전달한다."""
        self._close_windows()
        logger.info("영역 선택 완료: %s", rect)
        self.region_selected.emit(rect)

    def _on_screens_changed(self, _screen: QScreen) -> None:
        """선택 도중 모니터 구성이 바뀌면 좌표가 무효해지므로 취소한다."""
        if self._windows:
            logger.info("모니터 구성이 바뀌어 영역 선택을 취소합니다.")
            self.cancel()

    def _close_windows(self) -> None:
        """모든 선택 창을 닫고 정리한다."""
        windows, self._windows = self._windows, []
        for window in windows:
            window.close()
            window.deleteLater()
