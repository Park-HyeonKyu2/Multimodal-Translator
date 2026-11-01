"""실제 OCR 모델 대신 고정된 예시 텍스트를 반환하는 Mock 구현 (테스트용)."""

import logging

from translator.core.cancellation import CancellationToken
from translator.core.interfaces import TextRecognizer
from translator.core.models import CapturedImage, RecognizedText, TextGranularity, TextSegment
from translator.tuning import MOCK_OCR_DELAY_SECONDS

logger = logging.getLogger(__name__)


class MockTextRecognizer(TextRecognizer):
    """추론 지연을 흉내 내고, 이미지 가운데에 한 줄짜리 예시 문장이 있다고 돌려준다.

    화면 내용과 관계없이 항상 같은 결과를 내므로 결과를 예측해야 하는 테스트에 쓴다.
    실제 화면처럼 동작하는 Mock은 `MockLayoutTextRecognizer`다.
    """

    def __init__(self, delay_seconds: float = MOCK_OCR_DELAY_SECONDS) -> None:
        """흉내 낼 추론 지연 시간(초)을 정한다."""
        self._delay_seconds = delay_seconds

    def recognize(
        self, image: CapturedImage, cancellation: CancellationToken
    ) -> RecognizedText:
        """지연 후 캡처 크기를 담은 예시 문장과 그 위치(이미지 가운데 띠)를 반환한다."""
        cancellation.wait(self._delay_seconds)
        logger.debug("Mock OCR 실행: %dx%d", image.width, image.height)
        text = f"[Mock OCR] {image.width}x{image.height} 영역에서 인식된 예시 문장입니다."
        # 이미지 가로 80%, 세로 30%를 차지하는 가운데 띠를 글자 위치로 가정한다.
        bounds = image.to_screen_region(
            image.width // 10,
            image.height * 35 // 100,
            max(1, image.width * 8 // 10),
            max(1, image.height * 3 // 10),
        )
        segment = TextSegment(text=text, bounds=bounds, granularity=TextGranularity.LINE)
        return RecognizedText(text=text, segments=(segment,))
