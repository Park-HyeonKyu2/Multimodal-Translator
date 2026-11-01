"""PaddleOCR 어댑터 테스트: 가짜 엔진으로 입력 변환, 출력 검증, 오류·취소·동시 호출을 확인한다.

실제 PaddleOCR 모델 없이 실행한다 (numpy만 필요). 실제 엔진 확인은 `test_paddle_ocr_real.py`.
"""

import logging
import re
import threading
import time

import pytest

np = pytest.importorskip("numpy")

from translator.adapters.ocr.paddle import (  # noqa: E402 (numpy 확인 뒤 가져온다)
    INFERENCE_FAILED_MESSAGE,
    INVALID_OUTPUT_MESSAGE,
    LOAD_FAILED_MESSAGE,
    OUT_OF_MEMORY_MESSAGE,
    PaddleOcrSettings,
    PaddleTextRecognizer,
    bgra_to_bgr_array,
    parse_paddle_output,
)
from translator.adapters.ocr.unavailable import PADDLE_NOT_INSTALLED_MESSAGE  # noqa: E402
from translator.adapters.translation.mock import MockTranslator  # noqa: E402
from translator.application.pipeline import PipelineRequest, TranslationPipeline  # noqa: E402
from translator.core.cancellation import CancellationToken  # noqa: E402
from translator.core.errors import OperationCancelledError, TextRecognitionError  # noqa: E402
from translator.core.interfaces import ScreenCapturer  # noqa: E402
from translator.core.models import CapturedImage, ScreenRegion, TextGranularity  # noqa: E402

# 비밀로 다뤄야 하는 예시 원문 (로그에 남지 않아야 한다)
SECRET_TEXT = "비밀 대사"


def make_image(width: int = 200, height: int = 60, origin: tuple[int, int] = (0, 0)) -> CapturedImage:
    """픽셀마다 (B, G, R, A) = (1, 2, 3, 255)인 캡처 이미지를 만든다."""
    pixels = bytes((1, 2, 3, 255)) * (width * height)
    return CapturedImage(ScreenRegion(origin[0], origin[1], width, height), width, height, pixels)


def page(texts, scores, boxes=None, polys=None) -> dict:
    """PaddleOCR 3.x 페이지 결과 모양의 사전을 만든다."""
    result = {"rec_texts": texts, "rec_scores": scores}
    if boxes is not None:
        result["rec_boxes"] = np.array(boxes, dtype=np.int16)
    if polys is not None:
        result["rec_polys"] = [np.array(p, dtype=np.int16) for p in polys]
    return result


class FakeEngine:
    """정해진 페이지 결과를 돌려주는 가짜 PaddleOCR 엔진. 받은 입력과 동시 실행 수를 기록한다."""

    def __init__(self, pages: list[dict] | None = None, error: BaseException | None = None) -> None:
        """돌려줄 결과나 던질 예외를 받는다."""
        # predict가 돌려줄 페이지 결과
        self.pages = pages if pages is not None else []
        # 설정하면 predict가 이 예외를 던진다.
        self.error = error
        # predict에 들어온 입력 배열
        self.inputs: list = []
        # 동시에 predict 안에 있는 호출 수와 그 최댓값
        self.active = 0
        self.max_active = 0
        # 설정하면 predict가 이 이벤트가 열릴 때까지 기다린다.
        self.gate: threading.Event | None = None
        # predict가 시작되면 알린다.
        self.started = threading.Event()
        self._lock = threading.Lock()

    def predict(self, input):  # noqa: A002 (PaddleOCR 인자 이름)
        """입력을 기록하고 결과를 돌려준다."""
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        self.started.set()
        try:
            self.inputs.append(input)
            if self.gate is not None:
                self.gate.wait(5)
            if self.error is not None:
                raise self.error
            return list(self.pages)
        finally:
            with self._lock:
                self.active -= 1


def recognizer_with(engine: FakeEngine, **kwargs) -> tuple[PaddleTextRecognizer, list]:
    """가짜 엔진을 쓰는 인식기와, 엔진 생성 호출 기록을 돌려준다."""
    # 엔진 생성 함수가 받은 설정 목록
    created: list = []

    def factory(settings: PaddleOcrSettings) -> FakeEngine:
        """받은 설정을 기록하고 가짜 엔진을 돌려준다."""
        created.append(settings)
        return engine

    return PaddleTextRecognizer(engine_factory=factory, **kwargs), created


def test_bgra_is_converted_to_bgr_array() -> None:
    """BGRA 바이트를 높이 × 너비 × 3 BGR 배열로 바꾼다 (알파 제거, 채널 순서 유지)."""
    array = bgra_to_bgr_array(make_image(4, 2))

    assert array.shape == (2, 4, 3)
    assert array.dtype == np.uint8
    assert array.flags["C_CONTIGUOUS"]
    assert array[0, 0].tolist() == [1, 2, 3]


def test_wrong_byte_count_is_rejected() -> None:
    """픽셀 바이트 수가 크기와 맞지 않으면 인식 오류로 처리한다."""
    image = CapturedImage(ScreenRegion(0, 0, 4, 2), 4, 2, b"\x00" * 10)

    with pytest.raises(TextRecognitionError):
        bgra_to_bgr_array(image)


def test_lines_are_returned_with_screen_coordinates_and_scores() -> None:
    """줄마다 화면 물리 좌표(캡처 원점 포함), 줄 단위, 엔진 점수를 담아 읽는 순서로 돌려준다."""
    engine = FakeEngine([page(["아래 줄", "윗줄"], [0.9, 0.8], boxes=[[10, 30, 90, 50], [5, 2, 60, 20]])])
    recognizer, _ = recognizer_with(engine)

    result = recognizer.recognize(make_image(origin=(1000, -200)), CancellationToken())

    assert [s.text for s in result.segments] == ["윗줄", "아래 줄"]
    assert result.segments[0].bounds == ScreenRegion(1005, -198, 55, 18)
    assert result.segments[1].bounds == ScreenRegion(1010, -170, 80, 20)
    assert all(s.granularity is TextGranularity.LINE for s in result.segments)
    assert [s.confidence for s in result.segments] == pytest.approx([0.8, 0.9])
    assert result.text == "윗줄\n아래 줄"
    # 엔진에는 캡처와 같은 크기의 BGR 배열이 들어간다.
    assert engine.inputs[0].shape == (60, 200, 3)


def test_polygons_are_used_when_boxes_are_missing() -> None:
    """`rec_boxes`가 없으면 꼭짓점(`rec_polys`)을 감싸는 사각형을 위치로 쓴다."""
    pages = [page(["글자"], [0.95], polys=[[[10, 5], [50, 7], [52, 25], [9, 22]]])]

    [segment] = parse_paddle_output(pages, make_image(), min_score=0.5)

    assert segment.bounds == ScreenRegion(9, 5, 43, 20)


def test_low_score_and_invalid_items_are_dropped() -> None:
    """점수 미달, 빈 텍스트, 문자열이 아닌 텍스트, 잘못된 점수, 이미지 밖·넓이 없는 위치는 버린다."""
    pages = [
        page(
            ["좋음", "낮음", "  ", 123, "nan", "범위 밖 점수", "밖", "납작"],
            [0.9, 0.3, 0.9, 0.9, float("nan"), 1.5, 0.9, 0.9],
            boxes=[
                [0, 0, 20, 10],
                [0, 20, 20, 30],
                [0, 0, 20, 10],
                [0, 0, 20, 10],
                [0, 0, 20, 10],
                [0, 0, 20, 10],
                [300, 0, 400, 10],  # 이미지(200 × 60) 오른쪽 밖
                [0, 40, 20, 40],  # 높이 0
            ],
        )
    ]

    segments = parse_paddle_output(pages, make_image(), min_score=0.5)

    assert [s.text for s in segments] == ["좋음"]


def test_box_partly_outside_image_is_clipped() -> None:
    """이미지 경계를 넘는 위치는 이미지 안으로 잘라낸다."""
    pages = [page(["가장자리"], [0.9], boxes=[[-5, -3, 210, 20]])]

    [segment] = parse_paddle_output(pages, make_image(), min_score=0.5)

    assert segment.bounds == ScreenRegion(0, 0, 200, 20)


@pytest.mark.parametrize(
    "broken",
    [
        {"rec_scores": [0.9], "rec_boxes": [[0, 0, 1, 1]]},  # 텍스트 필드 없음
        {"rec_texts": ["a"], "rec_scores": [0.9]},  # 위치 필드 없음
        {"rec_texts": ["a", "b"], "rec_scores": [0.9], "rec_boxes": [[0, 0, 1, 1], [0, 0, 1, 1]]},
    ],
)
def test_unexpected_output_format_is_an_error(broken: dict) -> None:
    """필드가 없거나 개수가 맞지 않으면 엔진 출력을 믿지 않고 인식 오류로 처리한다."""
    with pytest.raises(TextRecognitionError, match=re.escape(INVALID_OUTPUT_MESSAGE)):
        parse_paddle_output([broken], make_image(), min_score=0.5)


def test_no_text_is_an_empty_result() -> None:
    """텍스트가 없는 화면은 오류가 아니라 빈 결과다."""
    recognizer, _ = recognizer_with(FakeEngine([page([], [], boxes=np.zeros((0, 4)))]))

    result = recognizer.recognize(make_image(), CancellationToken())

    assert result.text == ""
    assert result.segments == ()


def test_engine_is_created_lazily_once_with_settings() -> None:
    """엔진은 생성자에서 만들지 않고 처음 인식할 때 한 번만 만든다 (UI 스레드에서 모델 로딩 방지)."""
    settings = PaddleOcrSettings(device="gpu:0")
    engine = FakeEngine([page([], [], boxes=np.zeros((0, 4)))])
    created: list = []
    recognizer = PaddleTextRecognizer(
        settings, engine_factory=lambda s: created.append(s) or engine
    )

    assert created == []
    recognizer.recognize(make_image(), CancellationToken())
    recognizer.recognize(make_image(), CancellationToken())

    assert created == [settings]


def test_missing_paddle_gives_install_hint_without_retrying() -> None:
    """PaddleOCR를 가져오지 못하면 설치 안내 오류를 내고, 다음 호출에서도 다시 시도하지 않는다."""
    calls: list = []

    def factory(settings: PaddleOcrSettings):
        """설치되지 않은 상황을 흉내 낸다."""
        calls.append(settings)
        raise ImportError("No module named 'paddleocr'")

    recognizer = PaddleTextRecognizer(engine_factory=factory)

    for _ in range(2):
        with pytest.raises(TextRecognitionError, match="pip install"):
            recognizer.recognize(make_image(), CancellationToken())
    assert len(calls) == 1
    assert PADDLE_NOT_INSTALLED_MESSAGE.startswith("PaddleOCR")


def test_model_load_failure_is_reported_once() -> None:
    """모델을 불러오지 못하면(다운로드 실패 등) 안내 오류를 내고 반복해서 다시 불러오지 않는다."""
    calls: list = []

    def factory(settings: PaddleOcrSettings):
        """모델 로딩 실패를 흉내 낸다."""
        calls.append(settings)
        raise RuntimeError("download failed")

    recognizer = PaddleTextRecognizer(engine_factory=factory)

    for _ in range(3):
        with pytest.raises(TextRecognitionError, match=re.escape(LOAD_FAILED_MESSAGE)):
            recognizer.recognize(make_image(), CancellationToken())
    assert len(calls) == 1


def test_inference_errors_become_recognition_errors(caplog: pytest.LogCaptureFixture) -> None:
    """추론 중 예외는 사용자용 인식 오류로 바꾸고, 로그에는 인식 텍스트를 남기지 않는다."""
    recognizer, _ = recognizer_with(FakeEngine(error=RuntimeError("kernel failed")))

    with caplog.at_level(logging.DEBUG, logger="translator"):
        with pytest.raises(TextRecognitionError, match=re.escape(INFERENCE_FAILED_MESSAGE)):
            recognizer.recognize(make_image(), CancellationToken())

    assert "PaddleOCR 추론 실패" in caplog.text


def test_out_of_memory_has_its_own_message() -> None:
    """메모리 부족은 따로 안내한다 (UI는 계속 동작)."""
    recognizer, _ = recognizer_with(FakeEngine(error=MemoryError()))

    with pytest.raises(TextRecognitionError, match=re.escape(OUT_OF_MEMORY_MESSAGE)):
        recognizer.recognize(make_image(), CancellationToken())


def test_recognized_text_is_not_logged(caplog: pytest.LogCaptureFixture) -> None:
    """인식에 성공해도 원문 텍스트는 로그에 남기지 않는다 (개인정보·저작물 보호)."""
    engine = FakeEngine([page([SECRET_TEXT, "낮은 점수"], [0.9, 0.1], boxes=[[0, 0, 50, 20], [0, 30, 50, 50]])])
    recognizer, _ = recognizer_with(engine)

    with caplog.at_level(logging.DEBUG, logger="translator"):
        recognizer.recognize(make_image(), CancellationToken())

    assert SECRET_TEXT not in caplog.text
    assert "낮은 점수" not in caplog.text


def test_cancelled_before_inference_does_not_call_engine() -> None:
    """이미 취소된 요청은 엔진을 호출하지 않는다."""
    engine = FakeEngine()
    recognizer, created = recognizer_with(engine)
    token = CancellationToken()
    token.cancel()

    with pytest.raises(OperationCancelledError):
        recognizer.recognize(make_image(), token)

    assert created == []
    assert engine.inputs == []


def test_cancelled_during_inference_discards_result() -> None:
    """추론 중에 취소되면 추론이 끝난 뒤 결과를 버린다 (오래된 결과 방지)."""
    engine = FakeEngine([page(["대사"], [0.9], boxes=[[0, 0, 50, 20]])])
    engine.gate = threading.Event()
    recognizer, _ = recognizer_with(engine)
    token = CancellationToken()
    # 워커 스레드의 결과나 예외
    outcome: list = []

    def work() -> None:
        """다른 스레드에서 인식한다."""
        try:
            outcome.append(recognizer.recognize(make_image(), token))
        except BaseException as exc:  # noqa: BLE001 (테스트에서 예외를 그대로 확인한다)
            outcome.append(exc)

    thread = threading.Thread(target=work)
    thread.start()
    assert engine.started.wait(5)
    token.cancel()
    engine.gate.set()
    thread.join(5)

    assert isinstance(outcome[0], OperationCancelledError)


def test_concurrent_calls_do_not_enter_engine_together() -> None:
    """여러 스레드에서 동시에 호출해도 엔진은 한 번에 하나씩만 실행한다."""
    engine = FakeEngine([page([], [], boxes=np.zeros((0, 4)))])
    engine.gate = threading.Event()
    recognizer, _ = recognizer_with(engine)
    threads = [
        threading.Thread(target=recognizer.recognize, args=(make_image(), CancellationToken()))
        for _ in range(3)
    ]
    for thread in threads:
        thread.start()
    assert engine.started.wait(5)
    engine.gate.set()
    for thread in threads:
        thread.join(5)

    assert len(engine.inputs) == 3
    assert engine.max_active == 1


def test_waiting_for_busy_engine_can_be_cancelled() -> None:
    """다른 요청이 엔진을 쓰는 동안 기다리던 요청은 취소되면 바로 빠져나온다."""
    engine = FakeEngine([page([], [], boxes=np.zeros((0, 4)))])
    engine.gate = threading.Event()
    recognizer, _ = recognizer_with(engine)
    busy = threading.Thread(target=recognizer.recognize, args=(make_image(), CancellationToken()))
    busy.start()
    assert engine.started.wait(5)
    token = CancellationToken()
    # 기다리던 요청의 결과나 예외
    outcome: list = []

    def wait_then_cancel() -> None:
        """엔진을 기다리다 취소되는 요청."""
        try:
            outcome.append(recognizer.recognize(make_image(), token))
        except BaseException as exc:  # noqa: BLE001
            outcome.append(exc)

    waiter = threading.Thread(target=wait_then_cancel)
    waiter.start()
    # 기다리던 요청이 엔진 잠금 대기에 들어갈 시간을 준다.
    time.sleep(0.2)
    token.cancel()
    waiter.join(2)
    engine.gate.set()
    busy.join(5)

    assert isinstance(outcome[0], OperationCancelledError)
    assert len(engine.inputs) == 1


class OneImageCapturer(ScreenCapturer):
    """항상 같은 이미지를 돌려주는 가짜 캡처."""

    def __init__(self, image: CapturedImage) -> None:
        """돌려줄 이미지를 받는다."""
        # 캡처 결과로 돌려줄 이미지
        self.image = image

    def capture_region(self, region: ScreenRegion) -> CapturedImage:
        """영역과 관계없이 정해진 이미지를 돌려준다."""
        return self.image

    def capture_full_screen(self) -> CapturedImage:
        """정해진 이미지를 돌려준다."""
        return self.image


def test_paddle_lines_flow_through_pipeline_grouping() -> None:
    """PaddleOCR 줄 결과가 기존 파이프라인에서 대사 단위로 묶이고 번역된다 (Mock 번역기와 호환)."""
    image = make_image(400, 120, origin=(100, 100))
    engine = FakeEngine(
        [
            page(
                ["첫째 줄 대사", "둘째 줄 대사", "멀리 있는 메뉴"],
                [0.9, 0.9, 0.9],
                # 앞 두 줄은 붙어 있고 왼쪽 정렬 → 한 대사. 셋째는 멀리 떨어져 따로.
                boxes=[[10, 10, 200, 30], [10, 34, 180, 54], [300, 100, 390, 116]],
            )
        ]
    )
    recognizer, _ = recognizer_with(engine)
    pipeline = TranslationPipeline(OneImageCapturer(image), recognizer, MockTranslator(delay_seconds=0))

    batch = pipeline.run(PipelineRequest("box-1", image.region), CancellationToken())

    assert [unit.source_text for unit in batch.units] == ["첫째 줄 대사 둘째 줄 대사", "멀리 있는 메뉴"]
    assert batch.units[0].bounds == ScreenRegion(110, 110, 190, 44)
    assert all(unit.translated_text for unit in batch.units)


def test_failed_device_falls_back_once_to_cpu() -> None:
    """자동으로 고른 GPU에서 모델을 불러오지 못하면 대체 장치(CPU)로 한 번만 다시 시도한다."""
    engine = FakeEngine([page([], [], boxes=np.zeros((0, 4)))])
    # 엔진 생성 함수가 받은 장치 목록
    devices: list[str] = []

    def factory(settings: PaddleOcrSettings) -> FakeEngine:
        """GPU에서는 실패하고 CPU에서는 성공한다."""
        devices.append(settings.device)
        if settings.device.startswith("gpu"):
            raise RuntimeError("CUDA init failed")
        return engine

    recognizer = PaddleTextRecognizer(
        PaddleOcrSettings(device="gpu:0", fallback_device="cpu"), engine_factory=factory
    )
    recognizer.recognize(make_image(), CancellationToken())
    recognizer.recognize(make_image(), CancellationToken())

    assert devices == ["gpu:0", "cpu"]
    assert recognizer.settings.device == "cpu"


def test_no_fallback_without_fallback_device() -> None:
    """대체 장치가 없으면(장치를 직접 정함) 다시 시도하지 않고 로딩 오류를 알린다."""
    devices: list[str] = []

    def factory(settings: PaddleOcrSettings):
        """항상 실패한다."""
        devices.append(settings.device)
        raise RuntimeError("CUDA init failed")

    recognizer = PaddleTextRecognizer(PaddleOcrSettings(device="gpu:0"), engine_factory=factory)

    with pytest.raises(TextRecognitionError, match=re.escape(LOAD_FAILED_MESSAGE)):
        recognizer.recognize(make_image(), CancellationToken())
    assert devices == ["gpu:0"]


def test_missing_install_is_not_retried_on_fallback_device() -> None:
    """설치 문제(ImportError)는 장치를 바꿔도 해결되지 않으므로 대체 장치로 다시 시도하지 않는다."""
    devices: list[str] = []

    def factory(settings: PaddleOcrSettings):
        """설치되지 않은 상황."""
        devices.append(settings.device)
        raise ImportError("No module named 'paddle'")

    recognizer = PaddleTextRecognizer(
        PaddleOcrSettings(device="gpu:0", fallback_device="cpu"), engine_factory=factory
    )

    with pytest.raises(TextRecognitionError, match="pip install"):
        recognizer.recognize(make_image(), CancellationToken())
    assert devices == ["gpu:0"]


@pytest.mark.parametrize(
    ("requested", "version", "expected"),
    [(True, "3.2.2", True), (True, "3.3.1", False), (True, "3.3.0", False), (False, "3.2.2", False)],
)
def test_onednn_is_disabled_on_broken_versions(requested: bool, version: str, expected: bool) -> None:
    """oneDNN 오류가 알려진 PaddlePaddle 3.3.x에서는 설정과 관계없이 oneDNN을 켜지 않는다."""
    from translator.adapters.ocr.paddle import effective_onednn

    assert effective_onednn(requested, version) is expected


def test_default_settings_enable_onednn_with_tuned_threads() -> None:
    """기본 설정은 CPU 가속을 켜고, 스레드 수는 tuning.py 값을 쓴다."""
    from translator.tuning import PADDLE_OCR_CPU_THREADS

    settings = PaddleOcrSettings()

    assert settings.enable_onednn is True
    assert settings.cpu_threads == PADDLE_OCR_CPU_THREADS
    assert settings.fallback_device is None
