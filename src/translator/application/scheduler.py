"""GPU를 쓰는 추론 작업(OCR, 번역, 향후 장면 분석)의 공용 작업 큐.

- 동시에 실행되는 추론 작업 수를 `max_concurrency`로 제한한다 (기본 1: GPU를 한 작업만 점유).
- 큐에서 기다리는 작업은 우선순위가 높은 것부터 실행한다. 번역이 가장 높고, 향후 장면
  분석이 가장 낮다. 이미 실행 중인 작업을 중단하지는 않는다(협력적 취소만 지원).

자동 감지의 표본 캡처처럼 GPU를 쓰지 않는 가벼운 작업은 이 큐를 쓰지 않는다.
"""

import logging
from enum import IntEnum

from PySide6.QtCore import QObject, QRunnable, QThreadPool

from translator.tuning import INFERENCE_MAX_CONCURRENCY, SHUTDOWN_TIMEOUT_MS

logger = logging.getLogger(__name__)


class TaskPriority(IntEnum):
    """추론 작업 우선순위 (값이 클수록 먼저 실행). 값 자체는 순서만 의미가 있다."""

    # 향후 장면 분석 (VLM). 가장 낮다.
    SCENE_ANALYSIS = 0
    # 화면 변화가 없을 때의 OCR 재검사, 안정화를 위한 후속 OCR
    OCR_RECHECK = 10
    # 자동 감지·전체 화면 추적이 요청한 OCR
    OCR_AUTO = 20
    # 사용자가 직접 요청한 OCR (영역 번역, 전체 화면 켜기)
    OCR_USER = 30
    # 안정된 텍스트의 번역. 가장 높다.
    TRANSLATION = 40


class InferenceScheduler(QObject):
    """추론 작업을 우선순위 큐로 실행한다. 공개 메서드는 UI 스레드에서 호출한다."""

    def __init__(
        self,
        max_concurrency: int = INFERENCE_MAX_CONCURRENCY,
        parent: QObject | None = None,
    ) -> None:
        """동시에 실행할 추론 작업 수를 정한다."""
        super().__init__(parent)
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(max(1, max_concurrency))

    @property
    def max_concurrency(self) -> int:
        """동시에 실행할 수 있는 추론 작업 수."""
        return self._pool.maxThreadCount()

    def set_max_concurrency(self, value: int) -> None:
        """동시 실행 수를 바꾼다 (VRAM 여유에 맞춰 조정)."""
        self._pool.setMaxThreadCount(max(1, value))

    def submit(self, task: QRunnable, priority: TaskPriority) -> None:
        """작업을 큐에 넣는다. 자리가 나면 우선순위가 높은 작업부터 실행된다."""
        self._pool.start(task, int(priority))

    def clear_pending(self) -> None:
        """아직 시작하지 않은 작업을 모두 버린다 (종료할 때)."""
        self._pool.clear()

    def wait(self, timeout_ms: int = SHUTDOWN_TIMEOUT_MS) -> bool:
        """실행 중인 작업이 끝날 때까지 기다린다. 시간 안에 끝나면 True."""
        finished = self._pool.waitForDone(timeout_ms)
        if not finished:
            logger.warning("추론 작업이 %dms 안에 끝나지 않았습니다.", timeout_ms)
        return finished
