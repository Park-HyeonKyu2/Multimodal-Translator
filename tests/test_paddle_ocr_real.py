"""실제 PaddleOCR 모델로 인식이 되는지 확인하는 선택 실행 테스트.

모델을 불러오고(처음이면 내려받고) CPU로 추론하므로 기본 실행에서는 건너뛴다. 실행 방법:

    TRANSLATOR_PADDLE_OCR_TESTS=1 pytest tests/test_paddle_ocr_real.py

글자는 Windows 글꼴(MS Gothic)로 그린 합성 이미지다. 실제 게임·영상 화면 검증이 아니다.
"""

import os
import sys
import threading
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("TRANSLATOR_PADDLE_OCR_TESTS") != "1",
    reason="실제 PaddleOCR 모델이 필요한 테스트 (TRANSLATOR_PADDLE_OCR_TESTS=1로 실행)",
)

# 영어·일본어를 함께 그릴 수 있는 Windows 기본 글꼴
FONT_PATH = Path("C:/Windows/Fonts/msgothic.ttc")
# 합성 이미지에 그릴 두 줄 (영어, 일본어)
LINES = ("Hello, adventurer!", "こんにちは、勇者よ。")


def _devices() -> list[str]:
    """검사할 장치: CPU와, GPU를 쓸 수 있으면 GPU."""
    from translator.adapters.ocr.factory import cuda_gpu_available, paddle_gpu_build_installed

    devices = ["cpu"]
    if paddle_gpu_build_installed() and cuda_gpu_available():
        devices.append("gpu:0")
    return devices


@pytest.fixture(scope="module", params=_devices())
def recognizer(request):
    """장치마다 실제 PaddleOCR 인식기 하나를 모듈 안에서 함께 쓴다 (모델 로딩은 장치마다 한 번)."""
    pytest.importorskip("paddleocr")
    from translator.adapters.ocr.paddle import PaddleOcrSettings, PaddleTextRecognizer

    return PaddleTextRecognizer(PaddleOcrSettings(device=request.param))


def render(origin: tuple[int, int] = (0, 0)):
    """어두운 배경에 두 줄을 그린 640 × 120 캡처 이미지를 만든다."""
    image_module = pytest.importorskip("PIL.Image")
    draw_module = pytest.importorskip("PIL.ImageDraw")
    font_module = pytest.importorskip("PIL.ImageFont")
    if sys.platform != "win32" or not FONT_PATH.exists():
        pytest.skip("Windows 글꼴이 필요합니다.")
    from translator.core.models import CapturedImage, ScreenRegion

    canvas = image_module.new("RGB", (640, 120), (20, 20, 30))
    pen = draw_module.Draw(canvas)
    font = font_module.truetype(str(FONT_PATH), 28)
    pen.text((20, 15), LINES[0], font=font, fill=(240, 240, 240))
    pen.text((20, 65), LINES[1], font=font, fill=(240, 240, 240))
    # RGB → BGRA (캡처 어댑터와 같은 형식)
    bgra = canvas.convert("RGBA").tobytes("raw", "BGRA")
    return CapturedImage(ScreenRegion(origin[0], origin[1], 640, 120), 640, 120, bgra)


def test_real_engine_reads_english_and_japanese_lines(recognizer) -> None:
    """실제 모델이 영어·일본어 두 줄을 읽고, 위치를 화면 좌표로 돌려준다."""
    from translator.core.cancellation import CancellationToken

    result = recognizer.recognize(render(origin=(500, 300)), CancellationToken())

    assert [segment.text for segment in result.segments] == list(LINES)
    first = result.segments[0].bounds
    assert 500 <= first.left < 560 and 300 <= first.top < 340
    assert all(segment.confidence is not None for segment in result.segments)


def test_real_engine_runs_on_another_worker_thread(recognizer) -> None:
    """모델을 만든 스레드와 다른 워커 스레드에서도 추론이 된다 (공용 추론 큐의 스레드 풀 상황)."""
    from translator.core.cancellation import CancellationToken

    # 워커 스레드의 결과나 예외
    outcome: list = []

    def work() -> None:
        """다른 스레드에서 인식한다."""
        try:
            outcome.append(recognizer.recognize(render(), CancellationToken()))
        except BaseException as exc:  # noqa: BLE001 (테스트에서 예외를 그대로 확인한다)
            outcome.append(exc)

    thread = threading.Thread(target=work)
    thread.start()
    thread.join(120)

    assert not isinstance(outcome[0], BaseException), outcome[0]
    assert [segment.text for segment in outcome[0].segments] == list(LINES)


def test_real_engine_result_groups_into_one_dialogue(recognizer) -> None:
    """실제 줄 결과가 기존 묶기 규칙으로 한 대사(왼쪽 정렬된 붙은 두 줄)로 묶인다."""
    from translator.application.text_grouping import group_segments
    from translator.core.cancellation import CancellationToken

    result = recognizer.recognize(render(), CancellationToken())
    units = group_segments(result.segments)

    assert [unit.text for unit in units] == [f"{LINES[0]} {LINES[1]}"]
