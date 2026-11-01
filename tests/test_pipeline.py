"""번역 작업 흐름 테스트 (Qt 없이 실행): 인식 단계, 번역 단계, 두 단계를 잇는 호환 경로."""

import pytest

from translator.adapters.ocr.mock import MockTextRecognizer
from translator.adapters.translation.mock import MockTranslator
from translator.application.change_detection import compute_fingerprint
from translator.application.frames import Frame
from translator.application.ocr_service import OcrService
from translator.application.pipeline import PipelineRequest, TranslationPipeline
from translator.core.cancellation import CancellationToken
from translator.core.errors import OperationCancelledError, TranslationError
from translator.core.interfaces import ScreenCapturer, TextRecognizer, Translator
from translator.core.models import (
    CapturedImage,
    RecognizedText,
    ScreenRegion,
    TextGranularity,
    TextSegment,
    TranslationRequest,
    TranslationResult,
)


class FakeCapturer(ScreenCapturer):
    """캡처 요청을 기록하고 빈 이미지를 돌려주는 가짜 캡처."""

    def __init__(self) -> None:
        """요청 기록을 비워 둔다."""
        self.calls: list[ScreenRegion | None] = []

    def capture_region(self, region: ScreenRegion) -> CapturedImage:
        """영역을 기록하고 빈 이미지를 반환한다."""
        self.calls.append(region)
        return CapturedImage(region, region.width, region.height, b"")

    def capture_full_screen(self) -> CapturedImage:
        """전체 화면 요청을 기록하고 빈 이미지를 반환한다."""
        self.calls.append(None)
        return CapturedImage(ScreenRegion(0, 0, 1920, 1080), 1920, 1080, b"")


class ScriptedRecognizer(TextRecognizer):
    """미리 정한 조각을 돌려주는 가짜 OCR."""

    def __init__(self, *segments: TextSegment, text: str | None = None) -> None:
        """돌려줄 조각과 (선택) 전체 텍스트를 받는다."""
        self.segments = segments
        self.text = text if text is not None else " ".join(s.text for s in segments)

    def recognize(self, image: CapturedImage, cancellation: CancellationToken) -> RecognizedText:
        """정해 둔 결과를 반환한다."""
        return RecognizedText(text=self.text, segments=self.segments)


class RecordingTranslator(Translator):
    """번역 요청을 기록하는 가짜 번역 엔진."""

    def __init__(self) -> None:
        """요청 기록을 비워 둔다."""
        self.requests: list[TranslationRequest] = []

    def translate(self, request, cancellation) -> TranslationResult:
        """요청을 기록하고 원문을 대문자로 바꾼 결과를 반환한다."""
        self.requests.append(request)
        return TranslationResult(request.source_text, request.source_text.upper(), "recording")


class WrongCountTranslator(Translator):
    """요청보다 적은 결과를 돌려주는 잘못된 엔진 (출력 검증 테스트용)."""

    def translate(self, request, cancellation) -> TranslationResult:
        """사용되지 않는다."""
        raise AssertionError

    def translate_batch(self, requests, cancellation):
        """결과를 하나 빼고 돌려준다."""
        return [TranslationResult(r.source_text, "x", "bad") for r in requests][:-1]


def segment(text: str, left: int, top: int, width: int = 100, height: int = 20) -> TextSegment:
    """줄 단위 조각을 만든다."""
    return TextSegment(text, ScreenRegion(left, top, width, height), TextGranularity.LINE)


def run(pipeline: TranslationPipeline, region=None, exclude=(), token=None):
    """파이프라인을 한 번 실행한다."""
    return pipeline.run(PipelineRequest("box-1", region, exclude), token or CancellationToken())


def test_region_translation_returns_text_positions() -> None:
    """영역 번역 결과에는 원문 위치(물리 픽셀)와 번역문, 출처가 담긴다."""
    capturer = FakeCapturer()
    region = ScreenRegion(10, 20, 300, 100)
    pipeline = TranslationPipeline(
        capturer, MockTextRecognizer(delay_seconds=0), MockTranslator(delay_seconds=0)
    )

    batch = run(pipeline, region)

    assert capturer.calls == [region]
    assert batch.source_id == "box-1"
    assert batch.capture_region == region
    [unit] = batch.units
    assert "300x100" in unit.source_text
    assert unit.translated_text.startswith("[Mock 번역 → ko]")
    # Mock OCR은 영역 가운데 띠를 원문 위치로 돌려준다 (영역 안의 화면 좌표).
    assert region.contains_point(*unit.bounds.center)


def test_none_region_captures_full_screen() -> None:
    """영역이 None이면 전체 화면을 캡처한다."""
    capturer = FakeCapturer()
    pipeline = TranslationPipeline(
        capturer, MockTextRecognizer(delay_seconds=0), MockTranslator(delay_seconds=0)
    )

    batch = run(pipeline)

    assert capturer.calls == [None]
    assert "1920x1080" in batch.units[0].source_text


def test_each_text_unit_is_translated_separately_with_its_own_position() -> None:
    """떨어진 텍스트는 각자 번역되고 각자의 위치를 가진다 (서로 섞이지 않음)."""
    recognizer = ScriptedRecognizer(segment("hello", 0, 0), segment("bye", 500, 400))
    pipeline = TranslationPipeline(FakeCapturer(), recognizer, RecordingTranslator())

    batch = run(pipeline)

    assert [(u.source_text, u.translated_text) for u in batch.units] == [
        ("hello", "HELLO"),
        ("bye", "BYE"),
    ]
    assert batch.units[1].bounds == ScreenRegion(500, 400, 100, 20)


def test_excluded_areas_are_not_translated() -> None:
    """다른 출처(박스)가 맡은 영역 안의 텍스트는 번역하지 않는다 (중복 처리 방지)."""
    recognizer = ScriptedRecognizer(segment("inside", 100, 100), segment("outside", 900, 100))
    translator = RecordingTranslator()
    pipeline = TranslationPipeline(FakeCapturer(), recognizer, translator)

    batch = run(pipeline, exclude=(ScreenRegion(50, 50, 400, 200),))

    assert [u.source_text for u in batch.units] == ["outside"]
    assert [r.source_text for r in translator.requests] == ["outside"]


def test_repeated_text_uses_cache() -> None:
    """같은 원문은 캐시를 써서 번역 엔진을 다시 부르지 않는다."""
    recognizer = ScriptedRecognizer(segment("same", 0, 0), segment("same", 0, 500))
    translator = RecordingTranslator()
    pipeline = TranslationPipeline(FakeCapturer(), recognizer, translator)

    first = run(pipeline)
    run(pipeline)

    assert [r.source_text for r in translator.requests] == ["same"]
    assert len(first.units) == 2  # 위치가 다른 같은 원문은 각자 표시된다


def test_cache_size_is_limited() -> None:
    """캐시는 정해진 개수만 기억하고 오래된 것부터 버린다."""
    translator = RecordingTranslator()
    capturer = FakeCapturer()

    def run_with(text: str) -> None:
        """한 원문을 번역한다 (같은 번역 캐시를 공유)."""
        pipeline.translate_texts([text], CancellationToken())

    pipeline = TranslationPipeline(capturer, ScriptedRecognizer(), translator, cache_size=1)
    for text in ("a", "b", "a"):
        run_with(text)

    assert [r.source_text for r in translator.requests] == ["a", "b", "a"]


def test_engine_without_positions_uses_capture_region() -> None:
    """위치를 주지 않는 OCR 엔진이면 캡처 영역 전체를 원문 위치로 쓴다."""
    region = ScreenRegion(10, 10, 200, 50)
    pipeline = TranslationPipeline(
        FakeCapturer(), ScriptedRecognizer(text="plain text"), RecordingTranslator()
    )

    batch = run(pipeline, region)

    assert batch.units[0].bounds == region


def test_cancelled_token_stops_before_capture() -> None:
    """이미 취소된 요청은 캡처 전에 중단한다."""
    capturer = FakeCapturer()
    token = CancellationToken()
    token.cancel()
    pipeline = TranslationPipeline(
        capturer, MockTextRecognizer(delay_seconds=0), MockTranslator(delay_seconds=0)
    )

    with pytest.raises(OperationCancelledError):
        run(pipeline, token=token)
    assert capturer.calls == []


def test_mock_translator_wait_is_interrupted_by_cancellation() -> None:
    """번역 대기 중 취소되면 대기 시간을 다 기다리지 않고 중단한다."""
    token = CancellationToken()
    token.cancel()
    pipeline = TranslationPipeline(
        FakeCapturer(), MockTextRecognizer(delay_seconds=0), MockTranslator(delay_seconds=10)
    )

    with pytest.raises(OperationCancelledError):
        run(pipeline, token=token)


def test_empty_recognition_is_normal_result_without_translation() -> None:
    """인식된 텍스트가 없으면 오류 없이 빈 결과를 반환하고 번역 엔진을 호출하지 않는다."""
    translator = RecordingTranslator()
    pipeline = TranslationPipeline(FakeCapturer(), ScriptedRecognizer(text="   "), translator)

    batch = run(pipeline)

    assert batch.units == ()
    assert translator.requests == []


def test_engine_returning_wrong_count_is_rejected() -> None:
    """번역 결과 개수가 요청과 다르면 짝을 맞출 수 없으므로 오류로 처리한다."""
    recognizer = ScriptedRecognizer(segment("a", 0, 0), segment("b", 0, 500))
    pipeline = TranslationPipeline(FakeCapturer(), recognizer, WrongCountTranslator())

    with pytest.raises(TranslationError):
        run(pipeline)


class CountingRecognizer(ScriptedRecognizer):
    """호출 횟수를 세는 가짜 OCR."""

    def __init__(self, *segments: TextSegment) -> None:
        """돌려줄 조각을 받고 호출 횟수를 0으로 둔다."""
        super().__init__(*segments)
        self.calls = 0

    def recognize(self, image: CapturedImage, cancellation: CancellationToken) -> RecognizedText:
        """호출을 센 뒤 정해 둔 결과를 반환한다."""
        self.calls += 1
        return super().recognize(image, cancellation)


def gray_image(region: ScreenRegion, gray: int) -> CapturedImage:
    """단색 이미지를 만든다."""
    pixel = bytes((gray, gray, gray, 255))
    return CapturedImage(region, region.width, region.height, pixel * (region.width * region.height))


def recognize_frame(pipeline: TranslationPipeline, source_id: str, frame: Frame):
    """프레임 하나를 인식한다."""
    request = PipelineRequest(source_id, frame.region, frame=frame)
    return pipeline.recognize(request, CancellationToken())


def test_recognize_with_frame_does_not_capture_again() -> None:
    """자동 감지가 넘긴 프레임으로 인식하면 다시 캡처하지 않고 프레임 ID·캡처 시각을 그대로 쓴다."""
    capturer = FakeCapturer()
    region = ScreenRegion(0, 0, 200, 100)
    pipeline = TranslationPipeline(capturer, ScriptedRecognizer(segment("hi", 10, 10)), MockTranslator())
    frame = Frame(42, gray_image(region, 0), captured_at=12.5)

    result = recognize_frame(pipeline, "box-1", frame)

    assert capturer.calls == []
    assert (result.frame_id, result.captured_at, result.capture_region) == (42, 12.5, region)
    assert [unit.text for unit in result.units] == ["hi"]
    assert len(result.unit_fingerprints) == 1


def test_recognize_without_frame_captures_with_new_frame_id() -> None:
    """프레임이 없으면(수동 번역) 새로 캡처하고 요청마다 다른 프레임 ID를 붙인다."""
    capturer = FakeCapturer()
    region = ScreenRegion(0, 0, 100, 50)
    pipeline = TranslationPipeline(capturer, ScriptedRecognizer(), MockTranslator())

    first = pipeline.recognize(PipelineRequest("box-1", region), CancellationToken())
    second = pipeline.recognize(PipelineRequest("box-1", region), CancellationToken())

    assert capturer.calls == [region, region]
    assert first.frame_id != second.frame_id
    assert first.captured_at <= second.captured_at


def test_recognize_reports_probe_fingerprints_only_inside_capture() -> None:
    """표시 중인 텍스트 자리의 현재 픽셀 지문을 돌려준다. 캡처 밖의 자리는 건너뛴다."""
    region = ScreenRegion(0, 0, 100, 50)
    pipeline = TranslationPipeline(FakeCapturer(), ScriptedRecognizer(), MockTranslator())
    inside = ScreenRegion(10, 10, 30, 10)
    outside = ScreenRegion(500, 500, 30, 10)
    frame = Frame(1, gray_image(region, 80), 0.0)

    result = pipeline.recognize(
        PipelineRequest("box-1", region, frame=frame, probe_areas=(inside, outside)),
        CancellationToken(),
    )

    assert [area for area, _ in result.probes] == [inside]


def test_manual_ocr_fingerprint_uses_requested_grid() -> None:
    """프레임 없이 인식한 화면의 지문은 요청한 격자로 만든다 (출처의 감지기와 같은 격자)."""
    region = ScreenRegion(0, 0, 100, 50)
    capturer = FakeCapturer()
    capturer.capture_region = lambda r: gray_image(r, 30)  # type: ignore[method-assign]
    pipeline = TranslationPipeline(capturer, ScriptedRecognizer(), MockTranslator())

    result = pipeline.recognize(
        PipelineRequest("box-1", region, fingerprint_grid=(8, 4)), CancellationToken()
    )

    assert result.fingerprint == compute_fingerprint(gray_image(region, 30), 8, 4)


def test_same_capture_is_recognized_once_across_sources() -> None:
    """같은 화면 내용은 출처가 달라도 OCR을 한 번만 실행하고 결과를 재사용한다."""
    region = ScreenRegion(0, 0, 100, 50)
    recognizer = CountingRecognizer(segment("hi", 0, 0))
    pipeline = TranslationPipeline(FakeCapturer(), recognizer, MockTranslator())

    first = recognize_frame(pipeline, "box-1", Frame(1, gray_image(region, 0), 0.0))
    second = recognize_frame(pipeline, "box-2", Frame(2, gray_image(region, 0), 1.0))

    assert recognizer.calls == 1
    assert (first.ocr_reused, second.ocr_reused) == (False, True)
    assert second.source_id == "box-2" and second.frame_id == 2


def test_changed_screen_is_recognized_again() -> None:
    """화면 픽셀이 바뀌면 이전 OCR 결과를 재사용하지 않는다."""
    region = ScreenRegion(0, 0, 100, 50)
    recognizer = CountingRecognizer(segment("hi", 0, 0))
    pipeline = TranslationPipeline(FakeCapturer(), recognizer, MockTranslator())

    recognize_frame(pipeline, "box-1", Frame(1, gray_image(region, 0), 0.0))
    result = recognize_frame(pipeline, "box-1", Frame(2, gray_image(region, 1), 1.0))

    assert recognizer.calls == 2
    assert not result.ocr_reused


def test_pipeline_accepts_shared_ocr_service() -> None:
    """여러 파이프라인이 OCR 서비스 하나를 공유하면 캐시도 공유한다."""
    region = ScreenRegion(0, 0, 100, 50)
    recognizer = CountingRecognizer(segment("hi", 0, 0))
    service = OcrService(recognizer)
    first = TranslationPipeline(FakeCapturer(), service, MockTranslator())
    second = TranslationPipeline(FakeCapturer(), service, MockTranslator())

    recognize_frame(first, "a", Frame(1, gray_image(region, 0), 0.0))
    recognize_frame(second, "b", Frame(2, gray_image(region, 0), 0.0))

    assert first.ocr_service is service and recognizer.calls == 1
