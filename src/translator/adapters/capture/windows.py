"""Windows 화면 캡처 구현 (MSS 기반)."""

import logging
import time
from collections.abc import Callable
from typing import Any

import mss

from translator.core.errors import CaptureError
from translator.core.interfaces import ScreenCapturer
from translator.core.models import CapturedImage, ScreenRegion
from translator.tuning import CAPTURE_RETRY_DELAY_MS, CAPTURE_TRANSIENT_RETRIES

logger = logging.getLogger(__name__)


class MssScreenCapturer(ScreenCapturer):
    """MSS로 물리 픽셀 단위 화면을 캡처한다.

    MSS 인스턴스는 생성한 스레드에서만 안전하게 쓸 수 있으므로 호출마다 새로 만든다.
    캡처 이미지는 디스크에 저장하지 않는다.

    Windows의 화면 복사(GDI BitBlt)는 부하가 클 때 이유 없이 한 번씩 실패할 수 있다
    (실제 화면 시험에서 40회 중 1회 관찰). 그래서 정해진 횟수만큼 다시 시도한 뒤에도
    실패할 때만 `CaptureError`를 던진다.
    """

    def __init__(
        self,
        mss_factory: Callable[[], Any] = mss.MSS,
        transient_retries: int = CAPTURE_TRANSIENT_RETRIES,
        retry_delay_ms: int = CAPTURE_RETRY_DELAY_MS,
    ) -> None:
        """MSS 인스턴스를 만드는 함수(테스트에서는 가짜 객체)와 재시도 횟수·간격을 받는다."""
        self._mss_factory = mss_factory
        self._retries = transient_retries
        self._retry_delay_seconds = retry_delay_ms / 1000

    def capture_region(self, region: ScreenRegion) -> CapturedImage:
        """지정한 물리 픽셀 영역을 캡처한다. 실패하면 `CaptureError`를 던진다."""
        return self._with_retries(
            lambda sct: self._grab(sct, region),
            "선택한 영역을 캡처하지 못했습니다.",
            f"영역 캡처 실패: {region}",
        )

    def capture_full_screen(self) -> CapturedImage:
        """모든 모니터를 감싸는 가상 데스크톱 전체를 캡처한다."""

        def grab_virtual(sct: Any) -> CapturedImage:
            """가상 데스크톱 영역을 구해 캡처한다."""
            # monitors[0]은 모든 모니터를 감싸는 가상 데스크톱 영역이다.
            virtual = sct.monitors[0]
            region = ScreenRegion(
                virtual["left"], virtual["top"], virtual["width"], virtual["height"]
            )
            return self._grab(sct, region)

        return self._with_retries(grab_virtual, "전체 화면을 캡처하지 못했습니다.", "전체 화면 캡처 실패")

    def _with_retries(
        self, action: Callable[[Any], CapturedImage], user_message: str, log_message: str
    ) -> CapturedImage:
        """캡처를 실행하고, 일시 실패면 정해진 횟수만큼 다시 시도한다."""
        attempts = self._retries + 1
        for attempt in range(1, attempts + 1):
            try:
                with self._mss_factory() as sct:
                    return action(sct)
            except Exception as exc:
                if attempt < attempts:
                    logger.warning("%s (%d/%d회): 다시 시도합니다.", log_message, attempt, attempts)
                    time.sleep(self._retry_delay_seconds)
                    continue
                logger.exception("%s (%d회 모두 실패)", log_message, attempts)
                raise CaptureError(user_message) from exc
        raise AssertionError("도달할 수 없는 코드")  # pragma: no cover

    @staticmethod
    def _grab(sct: Any, region: ScreenRegion) -> CapturedImage:
        """열린 MSS 인스턴스로 영역을 캡처해 `CapturedImage`로 바꾼다."""
        shot = sct.grab(
            {
                "left": region.left,
                "top": region.top,
                "width": region.width,
                "height": region.height,
            }
        )
        width, height = shot.size
        logger.debug("캡처 완료: %s -> %dx%d", region, width, height)
        return CapturedImage(region=region, width=width, height=height, bgra=bytes(shot.bgra))
