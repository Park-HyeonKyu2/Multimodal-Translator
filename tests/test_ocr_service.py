"""OCR 공유 서비스 테스트: 같은 캡처의 중복 OCR 합치기, 캐시 재사용, 바뀐 화면 재인식 (Qt 없이 실행)."""

import threading

import pytest

from translator.application.ocr_service import OcrService
from translator.core.cancellation import CancellationToken
from translator.core.errors import TextRecognitionError, OperationCancelledError
from translator.core.interfaces import TextRecognizer
from translator.core.models import CapturedImage, RecognizedText, ScreenRegion

REGION = ScreenRegion(0, 0, 4, 2)


def image(gray: int, region: ScreenRegion = REGION) -> CapturedImage:
    """단색 이미지를 만든다."""
    pixel = bytes((gray, gray, gray, 255))
    return CapturedImage(region, region.width, region.height, pixel * (region.width * region.height))


class GatedRecognizer(TextRecognizer):
    """`gate`가 열릴 때까지 막히는 가짜 OCR. 호출 횟수와 시작 알림을 기록한다."""

    def __init__(self) -> None:
        """막힘 없이 시작한다 (`gate`를 비우면 막힌다)."""
        self.gate = threading.Event()
        self.gate.set()
        self.started = threading.Event()
        self.calls = 0
        # 설정하면 결과 대신 이 예외를 던진다.
        self.error: Exception | None = None

    def recognize(self, image: CapturedImage, cancellation: CancellationToken) -> RecognizedText:
        """호출을 세고 gate가 열리면 밝기를 텍스트로 돌려준다. 막혀 있는 동안 취소를 확인한다."""
        self.calls += 1
        self.started.set()
        while not self.gate.wait(0.01):
            cancellation.raise_if_cancelled()
        if self.error is not None:
            raise self.error
        return RecognizedText(text=f"gray-{image.bgra[0]}")


def run_in_thread(service: OcrService, img: CapturedImage, token: CancellationToken, out: list):
    """다른 스레드에서 OCR을 실행하고 결과나 예외를 out에 넣는다."""

    def target() -> None:
        """OCR 실행."""
        try:
            out.append(service.recognize(img, token))
        except BaseException as exc:  # noqa: BLE001 (테스트에서 예외를 그대로 확인한다)
            out.append(exc)

    thread = threading.Thread(target=target)
    thread.start()
    return thread


def test_same_capture_uses_cache() -> None:
    """같은 캡처 내용은 엔진을 다시 호출하지 않고 이전 결과를 재사용한다."""
    recognizer = GatedRecognizer()
    service = OcrService(recognizer)

    first = service.recognize(image(10), CancellationToken())
    second = service.recognize(image(10), CancellationToken())

    assert recognizer.calls == service.engine_calls == 1
    assert (first.reused, second.reused) == (False, True)
    assert second.recognized.text == "gray-10"


def test_changed_pixels_or_region_are_recognized_again() -> None:
    """픽셀이나 영역이 다르면 같은 캐시를 쓰지 않는다 (새로 바뀐 화면은 다시 OCR)."""
    recognizer = GatedRecognizer()
    service = OcrService(recognizer)

    service.recognize(image(10), CancellationToken())
    changed = service.recognize(image(11), CancellationToken())
    moved = service.recognize(image(10, ScreenRegion(5, 0, 4, 2)), CancellationToken())

    assert recognizer.calls == 3
    assert not changed.reused and not moved.reused
    assert changed.recognized.text == "gray-11"


def test_concurrent_requests_for_same_capture_are_merged() -> None:
    """같은 내용의 OCR이 실행 중이면 두 번째 요청은 새로 실행하지 않고 그 결과를 기다린다."""
    recognizer = GatedRecognizer()
    recognizer.gate.clear()
    service = OcrService(recognizer)
    results: list = []

    first = run_in_thread(service, image(10), CancellationToken(), results)
    assert recognizer.started.wait(2)
    second = run_in_thread(service, image(10), CancellationToken(), results)
    recognizer.gate.set()
    first.join(2)
    second.join(2)

    assert recognizer.calls == 1
    assert sorted(r.reused for r in results) == [False, True]
    assert {r.recognized.text for r in results} == {"gray-10"}


def test_waiter_runs_itself_when_owner_is_cancelled() -> None:
    """먼저 실행하던 요청이 취소되면 기다리던 요청이 직접 OCR을 실행한다 (결과를 잃지 않음)."""
    recognizer = GatedRecognizer()
    recognizer.gate.clear()
    service = OcrService(recognizer)
    owner_token = CancellationToken()
    owner_out: list = []
    waiter_out: list = []

    owner = run_in_thread(service, image(10), owner_token, owner_out)
    assert recognizer.started.wait(2)
    waiter = run_in_thread(service, image(10), CancellationToken(), waiter_out)
    owner_token.cancel()
    owner.join(2)
    recognizer.gate.set()
    waiter.join(2)

    assert isinstance(owner_out[0], OperationCancelledError)
    assert waiter_out[0].recognized.text == "gray-10"
    assert recognizer.calls == 2


def test_engine_error_is_shared_with_waiter_and_not_cached() -> None:
    """엔진 오류는 기다리던 요청에도 전달되고, 캐시에 남지 않아 다음 요청은 다시 실행한다."""
    recognizer = GatedRecognizer()
    recognizer.gate.clear()
    recognizer.error = TextRecognitionError("OCR 실패")
    service = OcrService(recognizer)
    results: list = []

    first = run_in_thread(service, image(10), CancellationToken(), results)
    assert recognizer.started.wait(2)
    second = run_in_thread(service, image(10), CancellationToken(), results)
    recognizer.gate.set()
    first.join(2)
    second.join(2)

    assert all(isinstance(r, TextRecognitionError) for r in results) and len(results) == 2
    recognizer.error = None
    assert service.recognize(image(10), CancellationToken()).recognized.text == "gray-10"


def test_waiting_request_can_be_cancelled() -> None:
    """다른 요청의 OCR을 기다리는 중에도 자기 취소에는 바로 반응한다."""
    recognizer = GatedRecognizer()
    recognizer.gate.clear()
    service = OcrService(recognizer)
    waiter_token = CancellationToken()
    owner_out: list = []
    waiter_out: list = []

    owner = run_in_thread(service, image(10), CancellationToken(), owner_out)
    assert recognizer.started.wait(2)
    waiter = run_in_thread(service, image(10), waiter_token, waiter_out)
    waiter_token.cancel()
    waiter.join(2)
    recognizer.gate.set()
    owner.join(2)

    assert isinstance(waiter_out[0], OperationCancelledError)
    assert owner_out[0].recognized.text == "gray-10"


@pytest.mark.parametrize("cache_size", [1, 2])
def test_cache_size_is_limited(cache_size) -> None:
    """캐시는 정해진 개수만 기억하고 오래된 것부터 버린다."""
    recognizer = GatedRecognizer()
    service = OcrService(recognizer, cache_size=cache_size)

    for gray in (1, 2, 1):
        service.recognize(image(gray), CancellationToken())

    # 크기 1이면 1이 밀려나 다시 실행, 크기 2면 1을 재사용한다.
    assert recognizer.calls == (3 if cache_size == 1 else 2)
