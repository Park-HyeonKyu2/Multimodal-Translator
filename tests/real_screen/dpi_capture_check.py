"""실제 화면에서 논리 좌표 → 물리 좌표 변환이 MSS 캡처와 맞는지 확인하는 스크립트.

`tests/test_dpi_real_screen.py`가 `QT_SCALE_FACTOR`를 바꿔 가며 별도 프로세스로 실행한다.
모니터마다 단색 창을 띄우고, 변환한 물리 영역을 캡처해 다음을 확인한다.

- 영역 안쪽(가장자리 1픽셀 제외)은 모두 창 색이다.
- 영역을 2픽셀 넓힌 바깥 테두리에는 창 색이 아닌 픽셀이 있다.

즉, 변환 결과가 실제 창과 1픽셀 이내로 일치하는지 본다. Qt 배율(QT_SCALE_FACTOR)은
Windows 디스플레이 배율 설정과 다르므로, 이 검사를 통과해도 실제 Windows 배율
환경이 검증된 것은 아니다.

화면 복사(GDI BitBlt)는 부하가 클 때 일시적으로 실패할 수 있어 앱의 캡처 구현과 같은
횟수만큼 다시 시도하고, 다시 시도한 횟수를 결과(`capture_retries`)에 남긴다.
"""

import json
import sys
import time

import mss
from PySide6.QtCore import QRect, Qt, QTimer
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QApplication, QWidget

from translator.tuning import CAPTURE_RETRY_DELAY_MS, CAPTURE_TRANSIENT_RETRIES
from translator.ui.screen_geometry import global_logical_rect_to_physical

# 배경과 겹칠 가능성이 낮은 표시 색 (RGB)
MARKER = (255, 0, 255)
# 창이 화면에 그려질 때까지 기다리는 시간
SETTLE_MS = 700


class MarkerWindow(QWidget):
    """항상 위에 떠 있는 단색 창."""

    def __init__(self) -> None:
        """테두리 없는 최상위 창을 만든다."""
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt 이름 규칙)
        """창 전체를 표시 색으로 칠한다."""
        QPainter(self).fillRect(self.rect(), QColor(*MARKER))


def grab_with_retry(sct, monitor: dict):
    """영역을 캡처한다. 일시 실패면 정해진 횟수만큼 다시 시도한다. (스크린샷, 재시도 횟수)를 반환한다."""
    for attempt in range(CAPTURE_TRANSIENT_RETRIES + 1):
        try:
            return sct.grab(monitor), attempt
        except mss.exception.ScreenShotError:
            if attempt == CAPTURE_TRANSIENT_RETRIES:
                raise
            time.sleep(CAPTURE_RETRY_DELAY_MS / 1000)
    raise AssertionError("도달할 수 없는 코드")  # pragma: no cover


def check_window(sct, window: QWidget) -> dict:
    """창 하나의 변환 결과를 캡처해 검사하고 결과를 사전으로 돌려준다."""
    region = global_logical_rect_to_physical(window.geometry())
    # 바깥 2픽셀까지 함께 캡처한다.
    pad = 2
    shot, retries = grab_with_retry(
        sct,
        {
            "left": region.left - pad,
            "top": region.top - pad,
            "width": region.width + 2 * pad,
            "height": region.height + 2 * pad,
        },
    )

    def is_marker(x: int, y: int) -> bool:
        """캡처 이미지의 (x, y) 픽셀이 표시 색인지 확인한다."""
        return shot.pixel(x, y) == MARKER

    width, height = shot.size
    # 안쪽: 변환 영역에서 가장자리 1픽셀을 뺀 부분
    inner_ok = all(
        is_marker(x, y)
        for y in range(pad + 1, height - pad - 1)
        for x in range(pad + 1, width - pad - 1)
    )
    # 바깥 테두리: 캡처 이미지의 맨 바깥 한 줄
    ring = [(x, 0) for x in range(width)] + [(x, height - 1) for x in range(width)]
    ring += [(0, y) for y in range(height)] + [(width - 1, y) for y in range(height)]
    outer_ok = not all(is_marker(x, y) for x, y in ring)
    return {
        "screen": window.screen().name(),
        "ratio": window.screen().devicePixelRatio(),
        "logical": window.geometry().getRect(),
        "physical": [region.left, region.top, region.width, region.height],
        "inner_ok": inner_ok,
        "outer_ok": outer_ok,
        "capture_retries": retries,
    }


def main() -> int:
    """모니터마다 창을 띄워 검사하고, 모두 통과하면 0을 반환한다."""
    app = QApplication(sys.argv)
    windows = []
    for screen in app.screens():
        geometry = screen.geometry()
        window = MarkerWindow()
        window.setScreen(screen)
        # 소수 배율에서 반올림이 일어나도록 홀수 위치와 크기를 쓴다.
        window.setGeometry(QRect(geometry.x() + 37, geometry.y() + 41, 123, 57))
        window.show()
        windows.append(window)

    results: list[dict] = []

    def run_checks() -> None:
        """창이 그려진 뒤 검사하고, 예외가 나도 이벤트 루프를 끝낸다."""
        try:
            with mss.MSS() as sct:
                results.extend(check_window(sct, window) for window in windows)
        except Exception as exc:  # 검사 실패 원인을 결과로 남긴다
            results.append({"error": repr(exc), "inner_ok": False, "outer_ok": False})
        finally:
            app.quit()

    QTimer.singleShot(SETTLE_MS, run_checks)
    app.exec()
    print(json.dumps(results, ensure_ascii=False))
    passed = bool(results) and all(r["inner_ok"] and r["outer_ok"] for r in results)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
