"""실제 화면에서 OCR 물리 좌표 → 번역 말풍선 배치가 맞는지 확인하는 스크립트.

`tests/test_dpi_real_screen.py`가 `QT_SCALE_FACTOR`를 바꿔 가며 별도 프로세스로 실행한다.
모니터마다 원문 위치(물리 픽셀)를 하나 정해 말풍선을 아래쪽에 띄운 뒤, Windows가 알려 주는
실제 창 위치(물리 픽셀, GetWindowRect)가 다음을 만족하는지 본다.

- 말풍선 위쪽 = 원문 아래쪽 + 간격 × 배율 (±2픽셀)
- 말풍선 가로 가운데 = 원문 가로 가운데 (±2픽셀)

Qt 배율(QT_SCALE_FACTOR)은 Windows 디스플레이 배율 설정과 다르므로, 이 검사를 통과해도
실제 Windows 배율 환경이 검증된 것은 아니다.
"""

import ctypes
import json
import sys
from ctypes import wintypes

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from translator.application.text_display import TextDisplayModel
from translator.core.models import ScreenRegion, TranslatedUnit, TranslationBatch
from translator.tuning import BUBBLE_GAP
from translator.ui.screen_geometry import round_like_qt
from translator.ui.translation_overlay import TranslationOverlay

# 창이 화면에 놓일 때까지 기다리는 시간
SETTLE_MS = 700
# 허용 오차 (물리 픽셀). 논리↔물리 반올림이 두 번 일어날 수 있다.
TOLERANCE = 2


def window_rect(widget) -> tuple[int, int, int, int]:
    """Windows가 알려 주는 창의 물리 픽셀 사각형 (왼쪽, 위, 오른쪽, 아래)."""
    rect = wintypes.RECT()
    ctypes.windll.user32.GetWindowRect(int(widget.winId()), ctypes.byref(rect))
    return rect.left, rect.top, rect.right, rect.bottom


def main() -> int:
    """모니터마다 말풍선을 띄워 검사하고, 모두 통과하면 0을 반환한다."""
    app = QApplication(sys.argv)
    overlay = TranslationOverlay(TextDisplayModel())
    checks: list[tuple[str, float, ScreenRegion]] = []
    for index, screen in enumerate(app.screens()):
        geometry = screen.geometry()
        ratio = screen.devicePixelRatio()
        # 모니터 원점 기준 논리 (200, 150) 위치, 논리 300×30 크기의 원문을 물리 좌표로 만든다.
        text = ScreenRegion(
            geometry.x() + round_like_qt(200 * ratio),
            geometry.y() + round_like_qt(150 * ratio),
            round_like_qt(300 * ratio),
            round_like_qt(30 * ratio),
        )
        unit = TranslatedUnit(f"text {index}", f"번역 {index}", text, "check")
        overlay.show_batch(TranslationBatch(f"screen-{index}", text, (unit,)))
        checks.append((screen.name(), ratio, text))

    results: list[dict] = []

    def run_checks() -> None:
        """말풍선이 놓인 뒤 실제 창 위치를 검사하고, 예외가 나도 이벤트 루프를 끝낸다."""
        try:
            check_bubbles()
        except Exception as exc:  # 검사 실패 원인을 결과로 남긴다
            results.append({"error": repr(exc), "top_ok": False, "center_ok": False})
        finally:
            overlay.clear()
            app.quit()

    def check_bubbles() -> None:
        """모니터마다 말풍선의 실제 창 위치를 원문 위치와 비교한다."""
        bubbles = {item.source_id: overlay.bubble_for(item.item_id) for item in overlay.model.items()}
        for index, (name, ratio, text) in enumerate(checks):
            left, top, right, bottom = window_rect(bubbles[f"screen-{index}"])
            expected_top = text.bottom + round_like_qt(BUBBLE_GAP * ratio)
            center_error = abs((left + right) / 2 - text.center[0])
            results.append(
                {
                    "screen": name,
                    "ratio": ratio,
                    "text": [text.left, text.top, text.width, text.height],
                    "bubble": [left, top, right - left, bottom - top],
                    "top_ok": abs(top - expected_top) <= TOLERANCE,
                    "center_ok": center_error <= TOLERANCE,
                }
            )

    QTimer.singleShot(SETTLE_MS, run_checks)
    app.exec()
    print(json.dumps(results, ensure_ascii=False))
    passed = bool(results) and all(r["top_ok"] and r["center_ok"] for r in results)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
