"""실제 AI 모델 대신 Mock 번역 결과를 반환하는 구현."""

import logging
from collections.abc import Sequence

from translator.core.cancellation import CancellationToken
from translator.core.interfaces import Translator
from translator.core.models import TranslationRequest, TranslationResult
from translator.tuning import MOCK_TRANSLATION_DELAY_SECONDS

logger = logging.getLogger(__name__)


class MockTranslator(Translator):
    """추론 지연을 흉내 내고 원문에 표식을 붙여 반환한다."""

    engine_name = "mock"

    def __init__(self, delay_seconds: float = MOCK_TRANSLATION_DELAY_SECONDS) -> None:
        """흉내 낼 추론 지연 시간(초)을 정한다."""
        self._delay_seconds = delay_seconds

    def translate(
        self, request: TranslationRequest, cancellation: CancellationToken
    ) -> TranslationResult:
        """지연 후 원문 앞에 표식을 붙인 Mock 번역 결과를 반환한다."""
        cancellation.wait(self._delay_seconds)
        logger.debug(
            "Mock 번역 실행: 원문 %d자, 문맥 %d개", len(request.source_text), len(request.context)
        )
        return self._mock_result(request)

    def translate_batch(
        self, requests: Sequence[TranslationRequest], cancellation: CancellationToken
    ) -> list[TranslationResult]:
        """여러 원문을 한 번의 지연으로 번역한다 (한 번의 추론으로 여러 줄을 처리하는 엔진을 흉내 냄)."""
        if not requests:
            return []
        cancellation.wait(self._delay_seconds)
        logger.debug("Mock 일괄 번역 실행: %d개", len(requests))
        return [self._mock_result(request) for request in requests]

    def _mock_result(self, request: TranslationRequest) -> TranslationResult:
        """원문 앞에 표식을 붙인 Mock 결과를 만든다."""
        return TranslationResult(
            source_text=request.source_text,
            translated_text=f"[Mock 번역 → {request.target_language}] {request.source_text}",
            engine=self.engine_name,
        )
