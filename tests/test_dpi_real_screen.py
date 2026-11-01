"""실제 화면에서 좌표 변환, MSS 캡처, 말풍선 배치가 맞는지 확인하는 선택 실행 테스트.

화면에 창을 잠깐 띄우므로 기본 실행에서는 건너뛴다. 실행 방법:

    TRANSLATOR_REAL_SCREEN_TESTS=1 pytest tests/test_dpi_real_screen.py

`QT_SCALE_FACTOR`로 Qt 배율만 흉내 낸다. Windows 디스플레이 배율 설정을 바꾼
환경이나 모니터마다 배율이 다른 환경은 이 테스트로 검증되지 않는다.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent / "real_screen" / "dpi_capture_check.py"
BUBBLE_SCRIPT = Path(__file__).parent / "real_screen" / "bubble_placement_check.py"
SCALE_FACTORS = ["1", "1.25", "1.5", "1.75", "2"]

pytestmark = pytest.mark.skipif(
    sys.platform != "win32" or os.environ.get("TRANSLATOR_REAL_SCREEN_TESTS") != "1",
    reason="실제 Windows 화면이 필요한 테스트 (TRANSLATOR_REAL_SCREEN_TESTS=1로 실행)",
)


def run_check(script: Path, scale_factor: str) -> list[dict]:
    """검사 스크립트를 Qt 배율을 바꿔 별도 프로세스로 실행하고 결과를 돌려준다."""
    env = os.environ.copy()
    env["QT_SCALE_FACTOR"] = scale_factor
    # conftest가 설정한 offscreen 플랫폼을 쓰면 실제 화면에 그려지지 않는다.
    env.pop("QT_QPA_PLATFORM", None)
    # 자식 프로세스 출력(JSON)을 UTF-8로 주고받는다.
    env["PYTHONIOENCODING"] = "utf-8"

    completed = subprocess.run(
        [sys.executable, str(script)],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )

    output = completed.stdout.strip().splitlines()
    results = json.loads(output[-1]) if output else []
    assert completed.returncode == 0, f"{results}\n{completed.stderr}"
    assert results, "검사한 모니터가 없습니다."
    return results


@pytest.mark.parametrize("scale_factor", SCALE_FACTORS)
def test_capture_region_matches_window_under_qt_scale(scale_factor: str) -> None:
    """Qt 배율별로 모든 모니터에서 변환 영역이 실제 창과 1픽셀 이내로 맞는다."""
    run_check(SCRIPT, scale_factor)


@pytest.mark.parametrize("scale_factor", SCALE_FACTORS)
def test_bubble_is_placed_below_ocr_text_under_qt_scale(scale_factor: str) -> None:
    """Qt 배율별로 모든 모니터에서 말풍선이 OCR 원문 바로 아래(간격×배율)에 놓인다."""
    run_check(BUBBLE_SCRIPT, scale_factor)
