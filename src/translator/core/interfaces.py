"""번역 엔진, 화면 캡처 및 기타 어댑터를 위한 공유 인터페이스.

애플리케이션/UI 계층은 이 인터페이스에만 의존하고, 실제 구현(MSS, PaddleOCR,
Qwen 등)은 adapters 패키지에서 주입한다. 모든 메서드는 백그라운드 스레드에서
호출될 수 있다.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from translator.core.cancellation import CancellationToken
from translator.core.content import ContentIdentity
from translator.core.models import (
    CapturedImage,
    RecognizedText,
    ScreenRegion,
    TranslationRequest,
    TranslationResult,
)


class ScreenCapturer(ABC):
    """화면 캡처. 실패 시 `CaptureError`를 던진다."""

    @abstractmethod
    def capture_region(self, region: ScreenRegion) -> CapturedImage:
        """지정한 물리 픽셀 영역을 캡처한다."""

    @abstractmethod
    def capture_full_screen(self) -> CapturedImage:
        """모든 모니터를 포함한 가상 데스크톱 전체를 캡처한다."""


class ContentDetector(ABC):
    """지금 사용자가 보고 있는 콘텐츠를 식별한다.

    식별하지 못하는 것은 예상된 상황이므로 예외 대신 `None`을 돌려준다 (창이 없음, 접근 권한
    없음, 우리 앱 자신의 창 등). `None`을 받은 쪽은 현재 세션을 바꾸지 않는다.
    """

    @abstractmethod
    def detect(self) -> ContentIdentity | None:
        """현재 콘텐츠를 식별한다. 식별하지 못하면 None."""


class TextRecognizer(ABC):
    """이미지에서 텍스트를 추출한다. 실패 시 `TextRecognitionError`를 던진다."""

    @abstractmethod
    def recognize(
        self, image: CapturedImage, cancellation: CancellationToken
    ) -> RecognizedText:
        """이미지의 텍스트를 인식한다."""


class Translator(ABC):
    """번역 엔진 공통 인터페이스. 실패 시 `TranslationError`를 던진다."""

    @abstractmethod
    def translate(
        self, request: TranslationRequest, cancellation: CancellationToken
    ) -> TranslationResult:
        """요청을 번역한다."""

    def translate_batch(
        self, requests: Sequence[TranslationRequest], cancellation: CancellationToken
    ) -> list[TranslationResult]:
        """여러 원문을 번역해 같은 순서로 돌려준다.

        기본 구현은 하나씩 번역한다. 한 번의 추론으로 여러 줄을 처리할 수 있는 엔진
        (예: 여러 대사를 한 프롬프트로 번역)은 재정의해 화면에 텍스트가 많을 때의
        지연을 줄인다.
        """
        results = []
        for request in requests:
            cancellation.raise_if_cancelled()
            results.append(self.translate(request, cancellation))
        return results
