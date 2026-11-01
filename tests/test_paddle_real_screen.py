"""실제 화면 + 실제 PaddleOCR로 앱 흐름을 확인하는 선택 실행 테스트 (Phase 2 2단계).

화면에 검사용 창을 띄우고 CPU로 OCR하므로(전체 화면은 수십 초) 기본 실행에서는 건너뛴다. 실행 방법:

    TRANSLATOR_REAL_SCREEN_TESTS=1 TRANSLATOR_PADDLE_OCR_TESTS=1 pytest tests/test_paddle_real_screen.py -s

동작 결과만 판정한다. 지연·CPU 같은 성능 값은 기록만 하고 판정하지 않는다 (3단계 성능 평가 대상).
직접 그린 가짜 대사 창으로 하는 시험이며, 실제 게임·영상 화면 검증이 아니다.
"""

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from translator.tuning import SHUTDOWN_TIMEOUT_MS

SCRIPT = Path(__file__).parent / "real_screen" / "paddle_app_check.py"
# 결과를 출력한 뒤 프로세스가 끝나기까지 허용하는 시간 (초). 종료가 멈추지 않는지 확인한다.
EXIT_AFTER_PRINT_LIMIT_SECONDS = 5.0

pytestmark = pytest.mark.skipif(
    sys.platform != "win32"
    or os.environ.get("TRANSLATOR_REAL_SCREEN_TESTS") != "1"
    or os.environ.get("TRANSLATOR_PADDLE_OCR_TESTS") != "1"
    or importlib.util.find_spec("paddleocr") is None,
    reason="실제 화면과 PaddleOCR가 필요한 테스트 "
    "(TRANSLATOR_REAL_SCREEN_TESTS=1, TRANSLATOR_PADDLE_OCR_TESTS=1로 실행)",
)


def run_check(*arguments: str) -> tuple[dict, float]:
    """검사 스크립트를 실제 화면에서 실행하고 (결과, 출력 뒤 프로세스가 끝나기까지 걸린 초)를 돌려준다."""
    env = os.environ.copy()
    # conftest가 설정한 offscreen 플랫폼을 쓰면 실제 화면에 그려지지 않는다.
    env.pop("QT_QPA_PLATFORM", None)
    env["PYTHONIOENCODING"] = "utf-8"
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=600,
    )
    ended_at = time.time()
    lines = completed.stdout.strip().splitlines()
    assert lines, completed.stderr[-3000:]
    results = json.loads(lines[-1])
    print(json.dumps(results, ensure_ascii=False, indent=1))
    assert completed.returncode == 0, f"{results}\n{completed.stderr[-3000:]}"
    return results, ended_at - results["printed_at"]


def test_app_flows_with_paddle_ocr_on_real_screen() -> None:
    """박스 수동 번역, 타자 효과, 대사 교체, 원문 이동, 전체 화면이 실제 OCR로 동작하고 프로세스가 바로 끝난다."""
    results, exit_delay = run_check()

    manual = results["scene1_manual"]
    assert manual["bubble_shown"]
    assert manual["dialogue_ratio"] >= 0.9
    assert manual["speaker_separate"]
    assert manual["bubble_below_text"]

    typing = results["scene2_typing"]
    assert typing["partial_translations"] == []
    assert typing["final_translations"] == 1

    replace = results["scene3_replace"]
    assert replace["new_translated"]
    assert not replace["stale_translation_shown"]

    move = results["scene4_move"]
    assert move["bubble_followed"]
    assert move["retranslated"] == []

    fullscreen = results["scene5_fullscreen"]
    assert fullscreen["sign_found"]
    assert not fullscreen["box_dialogue_in_fullscreen"]

    assert results["failures"] == []
    assert exit_delay < EXIT_AFTER_PRINT_LIMIT_SECONDS


def test_shutdown_while_model_is_loading_does_not_hang() -> None:
    """모델을 불러오는 도중 종료해도 대기 시간 안팎으로 끝나고 프로세스가 멈추지 않는다."""
    results, exit_delay = run_check("--shutdown-during-load")

    assert results["ocr_started"]
    assert results["shutdown_seconds"] < SHUTDOWN_TIMEOUT_MS / 1000 + 2
    assert exit_delay < EXIT_AFTER_PRINT_LIMIT_SECONDS
