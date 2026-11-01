"""OCR 결과 공유: 같은 캡처 내용의 OCR은 한 번만 실행한다.

- 캐시: 캡처 내용(영역 + 픽셀 해시)이 같으면 이전 OCR 결과를 재사용한다. 픽셀이 하나라도
  다르면 해시가 달라지므로, 새로 바뀐 화면은 캐시를 쓰지 않고 다시 OCR한다.
- 합치기: 같은 내용의 OCR이 이미 실행 중이면 새로 실행하지 않고 그 결과를 기다린다.

여러 워커 스레드에서 동시에 호출해도 안전하다. 픽셀은 보관하지 않고 해시와 OCR 결과만 기억한다.
"""

import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass, field

from translator.application.frames import content_key
from translator.core.cancellation import CancellationToken
from translator.core.errors import OperationCancelledError
from translator.core.interfaces import TextRecognizer
from translator.core.models import CapturedImage, RecognizedText
from translator.tuning import OCR_CACHE_SIZE

logger = logging.getLogger(__name__)

# 실행 중인 OCR을 기다릴 때 취소 여부를 확인하는 간격 (초)
_WAIT_POLL_SECONDS = 0.05


@dataclass
class _Pending:
    """실행 중인 OCR 하나. 끝나면 결과나 예외를 채우고 이벤트를 알린다."""

    done: threading.Event = field(default_factory=threading.Event)
    result: RecognizedText | None = None
    error: BaseException | None = None


@dataclass(frozen=True)
class OcrOutcome:
    """OCR 결과와 재사용 여부."""

    recognized: RecognizedText
    # True면 캐시나 다른 요청의 결과를 재사용했다 (OCR 엔진을 새로 호출하지 않음).
    reused: bool


class OcrService:
    """OCR 엔진 앞단에서 결과를 캐시하고 중복 요청을 합친다."""

    def __init__(self, recognizer: TextRecognizer, cache_size: int = OCR_CACHE_SIZE) -> None:
        """실제 OCR 엔진과 캐시 크기를 받는다."""
        self._recognizer = recognizer
        self._cache_size = cache_size
        # 캡처 내용 식별값 → OCR 결과 (오래된 것부터 버린다)
        self._cache: OrderedDict[tuple, RecognizedText] = OrderedDict()
        # 캡처 내용 식별값 → 실행 중인 OCR
        self._inflight: dict[tuple, _Pending] = {}
        self._lock = threading.Lock()
        # 엔진을 실제로 호출한 횟수 (측정·테스트용)
        self.engine_calls = 0

    def recognize(self, image: CapturedImage, cancellation: CancellationToken) -> OcrOutcome:
        """캡처 이미지를 OCR한다. 같은 내용이면 캐시나 실행 중인 결과를 재사용한다."""
        key = content_key(image)
        while True:
            with self._lock:
                cached = self._cache.get(key)
                if cached is not None:
                    self._cache.move_to_end(key)
                    return OcrOutcome(cached, reused=True)
                pending = self._inflight.get(key)
                owner = pending is None
                if owner:
                    pending = _Pending()
                    self._inflight[key] = pending
            if owner:
                return OcrOutcome(self._run(key, image, pending, cancellation), reused=False)
            result = self._wait(pending, cancellation)
            if result is not None:
                return OcrOutcome(result, reused=True)
            # 먼저 실행하던 요청이 취소되었다: 이번 요청이 직접 실행한다.
            logger.debug("함께 기다리던 OCR이 취소되어 다시 실행합니다.")

    def _run(
        self, key: tuple, image: CapturedImage, pending: _Pending, cancellation: CancellationToken
    ) -> RecognizedText:
        """OCR 엔진을 실행하고 결과를 캐시와 기다리는 요청에 알린다."""
        try:
            with self._lock:
                self.engine_calls += 1
            result = self._recognizer.recognize(image, cancellation)
        except BaseException as exc:
            with self._lock:
                self._inflight.pop(key, None)
            pending.error = exc
            pending.done.set()
            raise
        with self._lock:
            self._cache[key] = result
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
            self._inflight.pop(key, None)
        pending.result = result
        pending.done.set()
        return result

    @staticmethod
    def _wait(pending: _Pending, cancellation: CancellationToken) -> RecognizedText | None:
        """실행 중인 OCR을 기다린다. 그 요청이 취소되었으면 None, 다른 오류면 그대로 던진다."""
        while not pending.done.wait(_WAIT_POLL_SECONDS):
            cancellation.raise_if_cancelled()
        if pending.error is not None:
            if isinstance(pending.error, OperationCancelledError):
                return None
            raise pending.error
        return pending.result
