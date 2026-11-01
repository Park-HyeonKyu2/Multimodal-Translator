"""화면 캡처 어댑터 테스트. 실제 화면 대신 가짜 MSS 객체를 사용한다."""

from types import SimpleNamespace

import pytest

from translator.adapters.capture.windows import MssScreenCapturer
from translator.core.errors import CaptureError
from translator.core.models import ScreenRegion


class FakeMss:
    """MSS 대신 쓰는 가짜 객체. 두 모니터 구성과 캡처 요청 기록을 가진다."""

    monitors = [
        {"left": -1920, "top": 0, "width": 3840, "height": 1080},
        {"left": 0, "top": 0, "width": 1920, "height": 1080},
        {"left": -1920, "top": 0, "width": 1920, "height": 1080},
    ]

    def __init__(self, fail: bool = False) -> None:
        """fail이면 grab이 오류를 던지게 한다."""
        self.fail = fail
        self.grabbed: list[dict] = []

    def __enter__(self) -> "FakeMss":
        """with 문 진입."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """with 문 종료."""
        return None

    def grab(self, monitor: dict) -> SimpleNamespace:
        """요청한 영역 크기의 빈 스크린샷을 돌려준다."""
        if self.fail:
            raise RuntimeError("BitBlt failed")
        self.grabbed.append(monitor)
        width, height = monitor["width"], monitor["height"]
        return SimpleNamespace(size=(width, height), bgra=b"\x00" * (width * height * 4))


def test_capture_region_grabs_requested_physical_area() -> None:
    """영역 캡처는 요청한 물리 픽셀 영역을 그대로 MSS에 넘긴다."""
    fake = FakeMss()
    capturer = MssScreenCapturer(mss_factory=lambda: fake)

    image = capturer.capture_region(ScreenRegion(-100, 20, 40, 30))

    assert fake.grabbed == [{"left": -100, "top": 20, "width": 40, "height": 30}]
    assert (image.width, image.height) == (40, 30)
    assert len(image.bgra) == 40 * 30 * 4
    assert image.region == ScreenRegion(-100, 20, 40, 30)


def test_capture_full_screen_uses_virtual_desktop_bounds() -> None:
    """전체 화면 캡처는 가상 데스크톱 전체(monitors[0])를 사용한다."""
    fake = FakeMss()
    capturer = MssScreenCapturer(mss_factory=lambda: fake)

    image = capturer.capture_full_screen()

    assert fake.grabbed == [{"left": -1920, "top": 0, "width": 3840, "height": 1080}]
    assert image.region == ScreenRegion(-1920, 0, 3840, 1080)


@pytest.mark.parametrize("full_screen", [True, False])
def test_capture_failure_is_wrapped_in_capture_error(full_screen: bool) -> None:
    """MSS 오류는 사용자 메시지가 있는 `CaptureError`로 바뀐다."""
    capturer = MssScreenCapturer(mss_factory=lambda: FakeMss(fail=True))

    with pytest.raises(CaptureError):
        if full_screen:
            capturer.capture_full_screen()
        else:
            capturer.capture_region(ScreenRegion(0, 0, 10, 10))


@pytest.mark.parametrize("width, height", [(0, 10), (10, 0), (-5, 10)])
def test_screen_region_rejects_non_positive_size(width: int, height: int) -> None:
    """크기가 0 이하인 영역은 만들 수 없다."""
    with pytest.raises(ValueError):
        ScreenRegion(0, 0, width, height)


class FlakyMss(FakeMss):
    """처음 몇 번만 실패하는 가짜 MSS (부하 중 BitBlt 일시 실패 흉내)."""

    def __init__(self, failures: int) -> None:
        """실패할 횟수와 시도 횟수 기록을 준비한다."""
        super().__init__()
        self.failures = failures
        self.attempts = 0

    def grab(self, monitor: dict) -> SimpleNamespace:
        """정해진 횟수만큼 실패한 뒤 정상 결과를 돌려준다."""
        self.attempts += 1
        if self.attempts <= self.failures:
            raise RuntimeError("Windows graphics function failed: BitBlt")
        return super().grab(monitor)


def test_transient_capture_failure_is_retried() -> None:
    """화면 복사가 한 번 일시 실패하면 다시 시도해 정상 이미지를 돌려준다."""
    fake = FlakyMss(failures=1)
    capturer = MssScreenCapturer(mss_factory=lambda: fake, transient_retries=1, retry_delay_ms=0)

    image = capturer.capture_region(ScreenRegion(0, 0, 10, 10))

    assert fake.attempts == 2
    assert (image.width, image.height) == (10, 10)


def test_capture_retries_are_limited() -> None:
    """계속 실패하면 정해진 횟수만 다시 시도하고 `CaptureError`를 던진다 (무한 재시도 없음)."""
    fake = FlakyMss(failures=100)
    capturer = MssScreenCapturer(mss_factory=lambda: fake, transient_retries=2, retry_delay_ms=0)

    with pytest.raises(CaptureError):
        capturer.capture_full_screen()

    assert fake.attempts == 3
