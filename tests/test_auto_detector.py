"""자동 감지기 테스트: 가짜 캡처로 화면 변화를 흉내 낸다.

감지기는 OCR이 필요한 표본을 `Frame`(캡처 이미지 포함)으로 넘긴다. 테스트에서는 그 영역을
"요청"으로 기록한다.
"""

import threading

import pytest

from translator.application.auto_detector import AutoDetector
from translator.application.change_detection import Fingerprint, compute_fingerprint
from translator.application.frames import Frame
from translator.core.errors import CaptureError
from translator.core.interfaces import ScreenCapturer
from translator.core.models import CapturedImage, ScreenRegion

REGION = ScreenRegion(0, 0, 8, 4)


class ScriptedCapturer(ScreenCapturer):
    """설정한 밝기의 단색 이미지를 돌려주거나 오류를 던진다."""

    def __init__(self, gray: int = 0) -> None:
        """기본 밝기와 실패 여부, 호출 기록을 준비한다."""
        self.gray = gray
        self.fail = False
        # True면 캡처할 때마다 밝기를 뒤집는다 (계속 움직이는 배경 흉내)
        self.flicker = False
        self.calls = 0
        self.threads: set[str] = set()

    def capture_region(self, region: ScreenRegion) -> CapturedImage:
        """호출을 기록하고 설정한 밝기의 단색 이미지를 반환한다 (fail이면 오류)."""
        self.calls += 1
        self.threads.add(threading.current_thread().name)
        if self.fail:
            raise CaptureError("선택한 영역을 캡처하지 못했습니다.")
        if self.flicker:
            self.gray = 255 - self.gray
        pixel = bytes((self.gray, self.gray, self.gray, 255))
        return CapturedImage(region, region.width, region.height, pixel * (region.width * region.height))

    def capture_full_screen(self) -> CapturedImage:
        """이 테스트에서는 쓰지 않는다."""
        raise NotImplementedError


@pytest.fixture
def make_detector(qapp):
    """짧은 간격의 자동 감지기를 만들고 테스트가 끝나면 정리한다."""
    detectors: list[AutoDetector] = []

    def factory(capturer, region_provider=lambda: REGION, **kwargs) -> AutoDetector:
        """가짜 캡처와 영역 제공 함수로 자동 감지기를 만든다."""
        detector = AutoDetector(capturer, region_provider, interval_ms=20, **kwargs)
        detectors.append(detector)
        return detector

    yield factory
    for detector in detectors:
        detector.shutdown()


def test_requests_translation_once_when_screen_is_stable(make_detector, qtbot) -> None:
    """화면이 멈춰 있으면 번역을 한 번만 요청하고, 표본 수집은 워커 스레드에서 한다."""
    capturer = ScriptedCapturer(gray=0)
    detector = make_detector(capturer)
    requests: list[ScreenRegion] = []
    detector.frame_ready.connect(lambda frame: requests.append(frame.region))

    detector.start()
    qtbot.waitUntil(lambda: len(requests) == 1, timeout=2000)
    calls_after_first = capturer.calls
    qtbot.waitUntil(lambda: capturer.calls >= calls_after_first + 3, timeout=2000)

    assert requests == [REGION]
    assert threading.main_thread().name not in capturer.threads


def test_requests_again_after_screen_changes(make_detector, qtbot) -> None:
    """번역한 뒤 화면이 바뀌면 다시 번역을 요청한다."""
    capturer = ScriptedCapturer(gray=0)
    detector = make_detector(capturer)
    requests: list[ScreenRegion] = []
    detector.frame_ready.connect(lambda frame: requests.append(frame.region))

    detector.start()
    qtbot.waitUntil(lambda: len(requests) == 1, timeout=2000)
    capturer.gray = 255  # 새 자막이 나타남

    qtbot.waitUntil(lambda: len(requests) == 2, timeout=2000)


def test_skips_when_no_region(make_detector, qtbot) -> None:
    """감시할 영역이 없으면 캡처하지 않고 계속 대기한다."""
    capturer = ScriptedCapturer()
    detector = make_detector(capturer, region_provider=lambda: None)

    detector.start()
    qtbot.wait(100)

    assert capturer.calls == 0
    assert detector.is_running


def test_stop_prevents_further_requests(make_detector, qtbot) -> None:
    """자동 감지를 끄면 진행 중인 표본 결과로도 번역을 요청하지 않는다."""
    capturer = ScriptedCapturer()
    detector = make_detector(capturer)
    requests: list[ScreenRegion] = []
    detector.frame_ready.connect(lambda frame: requests.append(frame.region))

    detector.start()
    detector.stop()
    qtbot.wait(150)

    assert requests == []
    assert not detector.is_running


def test_repeated_capture_failures_stop_detection(make_detector, qtbot) -> None:
    """캡처가 연속으로 실패하면 자동 감지를 끄고 더 재시도하지 않는다."""
    capturer = ScriptedCapturer()
    capturer.fail = True
    detector = make_detector(capturer, max_consecutive_failures=3)

    with qtbot.waitSignal(detector.stopped_by_error, timeout=2000) as blocker:
        detector.start()

    assert blocker.args == ["선택한 영역을 캡처하지 못했습니다."]
    assert not detector.is_running
    calls = capturer.calls
    qtbot.wait(100)
    assert capturer.calls == calls  # 무한 재시도하지 않음


def test_mark_translated_before_start_skips_same_screen(make_detector, qtbot) -> None:
    """수동 번역 기록 후 자동 감지를 켜면 같은 화면은 번역하지 않고, 바뀐 화면만 번역한다."""
    capturer = ScriptedCapturer(gray=0)
    detector = make_detector(capturer)
    requests: list[ScreenRegion] = []
    detector.frame_ready.connect(lambda frame: requests.append(frame.region))

    detector.mark_translated(REGION)
    qtbot.waitUntil(lambda: capturer.calls == 1, timeout=2000)  # 기록용 표본
    detector.start()
    qtbot.waitUntil(lambda: capturer.calls >= 5, timeout=2000)
    assert requests == []

    capturer.gray = 255
    qtbot.waitUntil(lambda: len(requests) == 1, timeout=2000)


def test_reset_discards_manual_translation_record(make_detector, qtbot) -> None:
    """박스가 바뀌면(reset) 이전 기록을 버리고 새 영역을 번역한다."""
    capturer = ScriptedCapturer(gray=0)
    detector = make_detector(capturer)
    requests: list[ScreenRegion] = []
    detector.frame_ready.connect(lambda frame: requests.append(frame.region))

    detector.mark_translated(REGION)
    qtbot.waitUntil(lambda: capturer.calls == 1, timeout=2000)
    detector.reset()
    detector.start()

    qtbot.waitUntil(lambda: len(requests) == 1, timeout=2000)


def test_baseline_failure_does_not_stop_detection(make_detector, qtbot) -> None:
    """기록용 표본 실패는 자동 감지를 멈추거나 실패 횟수에 넣지 않는다."""
    capturer = ScriptedCapturer()
    capturer.fail = True
    detector = make_detector(capturer, max_consecutive_failures=1)
    stopped: list[str] = []
    detector.stopped_by_error.connect(stopped.append)

    detector.mark_translated(REGION)
    qtbot.waitUntil(lambda: capturer.calls == 1, timeout=2000)
    qtbot.wait(50)

    assert stopped == []


class TextPolicy:
    """OCR 텍스트 기반 감지를 흉내 낸 판단 정책: 텍스트가 바뀌면 바로 번역."""

    def __init__(self) -> None:
        """마지막으로 번역한 텍스트를 비워 둔다."""
        self.last_translated: str | None = None

    def observe(self, observation: str) -> bool:
        """처음 보는 텍스트면 번역을 요청한다."""
        if observation == self.last_translated:
            return False
        self.last_translated = observation
        return True

    def mark_translated(self, observation: str) -> None:
        """이미 번역한 텍스트를 기록한다."""
        self.last_translated = observation

    def restart(self) -> None:
        """기록은 유지한다."""

    def reset(self) -> None:
        """기록을 비운다."""
        self.last_translated = None


def test_observer_and_policy_can_be_replaced(make_detector, qtbot) -> None:
    """관찰 방식과 판단 정책을 바꿔 끼워도 자동 감지기가 그대로 동작한다 (OCR 기반 교체 대비)."""
    capturer = ScriptedCapturer(gray=0)
    observed_threads: list[str] = []

    def fake_ocr(image) -> str:
        """밝기를 "인식 텍스트"로 바꾸는 가짜 OCR (백그라운드 스레드에서 실행됨)."""
        observed_threads.append(threading.current_thread().name)
        return f"text-{image.bgra[0]}"

    detector = make_detector(capturer, tracker=TextPolicy(), observer=fake_ocr)
    requests: list[ScreenRegion] = []
    detector.frame_ready.connect(lambda frame: requests.append(frame.region))

    detector.start()
    qtbot.waitUntil(lambda: len(requests) == 1, timeout=2000)
    capturer.gray = 255
    qtbot.waitUntil(lambda: len(requests) == 2, timeout=2000)

    assert threading.main_thread().name not in observed_threads


def test_frame_carries_sampled_image_so_ocr_does_not_recapture(make_detector, qtbot) -> None:
    """OCR이 필요한 표본은 캡처 이미지·프레임 ID·캡처 시각·지문을 담은 프레임으로 넘어온다."""
    capturer = ScriptedCapturer(gray=0)
    detector = make_detector(capturer)
    frames: list[Frame] = []
    detector.frame_ready.connect(frames.append)

    detector.start()
    qtbot.waitUntil(lambda: len(frames) == 1, timeout=2000)
    capturer.gray = 255
    qtbot.waitUntil(lambda: len(frames) == 2, timeout=2000)

    first, second = frames
    assert first.region == REGION and first.image.width == REGION.width
    assert isinstance(first.fingerprint, Fingerprint)
    assert first.frame_id != second.frame_id
    assert first.captured_at <= second.captured_at
    assert second.image.bgra[0] == 255  # 바뀐 화면의 캡처 그대로


def test_mark_fingerprint_records_screen_without_extra_capture(make_detector, qtbot) -> None:
    """수동 OCR 결과의 지문을 기록하면 추가 캡처 없이 같은 화면을 다시 요청하지 않는다."""
    capturer = ScriptedCapturer(gray=0)
    detector = make_detector(capturer)
    requests: list[ScreenRegion] = []
    detector.frame_ready.connect(lambda frame: requests.append(frame.region))
    same_screen = compute_fingerprint(capturer.capture_region(REGION))
    capturer.calls = 0

    detector.mark_fingerprint(same_screen, REGION)
    assert capturer.calls == 0  # 기록에 캡처가 필요 없다
    detector.start()
    qtbot.waitUntil(lambda: capturer.calls >= 4, timeout=2000)
    assert requests == []

    capturer.gray = 255
    qtbot.waitUntil(lambda: len(requests) == 1, timeout=2000)


def test_mark_fingerprint_for_other_region_is_ignored(make_detector, qtbot) -> None:
    """그사이 박스가 옮겨져 영역이 다르면 지문을 기록하지 않는다 (다른 화면을 기준으로 삼지 않음)."""
    capturer = ScriptedCapturer(gray=0)
    detector = make_detector(capturer)
    requests: list[ScreenRegion] = []
    detector.frame_ready.connect(lambda frame: requests.append(frame.region))
    fingerprint = compute_fingerprint(capturer.capture_region(REGION))

    detector.mark_fingerprint(fingerprint, ScreenRegion(100, 100, 8, 4))
    detector.start()

    qtbot.waitUntil(lambda: len(requests) == 1, timeout=2000)


def test_constantly_moving_background_does_not_postpone_ocr_forever(make_detector, qtbot) -> None:
    """화면이 멈추지 않아도 OCR을 계속 요청하되, 최소 간격으로 횟수를 제한한다."""
    capturer = ScriptedCapturer(gray=0)
    capturer.flicker = True
    detector = make_detector(capturer)
    requests: list[ScreenRegion] = []
    detector.frame_ready.connect(lambda frame: requests.append(frame.region))

    detector.start()
    qtbot.waitUntil(lambda: len(requests) >= 3, timeout=3000)

    # AUTO_OCR_MIN_SAMPLE_GAP(2) 때문에 표본마다 요청하지는 않는다.
    assert len(requests) < capturer.calls
