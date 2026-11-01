"""감시 영역(바운딩 박스, 전체 화면)을 주기적으로 관찰해 화면이 바뀌면 OCR을 요청한다.

- 캡처와 관찰값 계산(observer)은 백그라운드 스레드에서 한다 (GPU를 쓰지 않는 가벼운 작업).
- 판단 정책(`ChangePolicy`)과 타이머는 UI 스레드에서 처리한다.
- OCR이 필요하다고 판단한 표본은 캡처 이미지를 담은 `Frame`으로 넘긴다. 번역 단계가 같은
  화면을 다시 캡처하지 않게 하기 위해서다. 필요 없는 표본의 이미지는 바로 버린다.

관찰 방식과 판단 정책은 주입받는다. 기본값은 밝기 지문 + `ChangeTriggerPolicy`(바뀌면
간격을 두고 OCR 요청)이다. 텍스트가 다 나왔는지는 OCR 텍스트로 따로 판단한다.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Signal, Slot

from translator.application.change_detection import (
    ChangePolicy,
    ChangeTriggerPolicy,
    compute_fingerprint,
)
from translator.application.frames import Frame, next_frame_id, now
from translator.core.errors import UNEXPECTED_ERROR_MESSAGE, TranslatorError
from translator.core.interfaces import ScreenCapturer
from translator.core.models import CapturedImage, ScreenRegion
from translator.tuning import (
    AUTO_DETECT_INTERVAL_MS,
    AUTO_DETECT_MAX_CONSECUTIVE_FAILURES,
    SHUTDOWN_TIMEOUT_MS,
)

logger = logging.getLogger(__name__)

# 캡처 이미지 → 관찰값. 백그라운드 스레드에서 호출되므로 Qt 위젯에 접근하면 안 된다.
Observer = Callable[[CapturedImage], Any]


class _SampleSignals(QObject):
    """워커 스레드 → UI 스레드 전달용 시그널 (UI 스레드에서 생성되어 큐 연결로 전달됨)."""

    sampled = Signal(int, object, object, float)  # 작업 ID, 관찰값, 캡처 이미지, 캡처 시각
    failed = Signal(int, str)  # 작업 ID, 사용자에게 보여 줄 메시지


class _SampleTask(QRunnable):
    """영역을 한 번 캡처해 관찰값을 만드는 백그라운드 작업."""

    def __init__(
        self,
        task_id: int,
        capturer: ScreenCapturer,
        observer: Observer,
        region: ScreenRegion,
        signals: _SampleSignals,
    ) -> None:
        """작업 ID와 캡처 대상, 결과를 보낼 시그널을 받는다."""
        super().__init__()
        self._task_id = task_id
        self._capturer = capturer
        self._observer = observer
        self._region = region
        self._signals = signals

    def run(self) -> None:
        """캡처 → 관찰값 계산 후 결과나 오류를 시그널로 보낸다."""
        try:
            captured_at = now()
            image = self._capturer.capture_region(self._region)
            observation = self._observer(image)
        except TranslatorError as exc:
            self._signals.failed.emit(self._task_id, str(exc))
        except Exception:
            # 워커 스레드의 예외가 사라지지 않도록 기록하고 UI에는 일반 메시지만 전달한다.
            logger.exception("자동 감지 표본 수집 중 예기치 못한 오류")
            self._signals.failed.emit(self._task_id, UNEXPECTED_ERROR_MESSAGE)
        else:
            self._signals.sampled.emit(self._task_id, observation, image, captured_at)


@dataclass
class _PendingSample:
    """결과를 기다리는 표본 작업과, 결과가 아직 유효한지 판단할 정보."""

    task: _SampleTask
    signals: _SampleSignals
    region: ScreenRegion
    # 표본을 요청한 시점의 감시 영역 번호. 박스가 바뀌면 결과를 버린다.
    epoch: int
    # 표본을 요청한 시점의 감시 루프 번호. 자동 감지를 끄거나 다시 켜면 결과를 버린다.
    generation: int
    # True면 수동 번역한 화면을 기록하기 위한 표본이다.
    is_baseline: bool


class AutoDetector(QObject):
    """자동 감지 루프. 모든 공개 메서드는 UI 스레드에서 호출한다."""

    frame_ready = Signal(object)  # Frame: OCR이 필요한 화면
    stopped_by_error = Signal(str)

    def __init__(
        self,
        capturer: ScreenCapturer,
        region_provider: Callable[[], ScreenRegion | None],
        tracker: ChangePolicy[Any] | None = None,
        observer: Observer = compute_fingerprint,
        interval_ms: int = AUTO_DETECT_INTERVAL_MS,
        max_consecutive_failures: int = AUTO_DETECT_MAX_CONSECUTIVE_FAILURES,
        thread_pool: QThreadPool | None = None,
        parent: QObject | None = None,
    ) -> None:
        """캡처 구현, 감시 영역 제공 함수, 판단 정책과 관찰 방식을 주입받는다.

        `region_provider`는 UI 스레드에서 호출되며, 감시할 영역이 없으면 None을 반환한다.
        `tracker`와 `observer`는 짝이 맞아야 한다 (예: 밝기 지문 + `ChangeTracker`).
        """
        super().__init__(parent)
        self._capturer = capturer
        self._region_provider = region_provider
        self._tracker: ChangePolicy[Any] = tracker or ChangeTriggerPolicy()
        self._observer = observer
        self._max_failures = max_consecutive_failures
        if thread_pool is None:
            thread_pool = QThreadPool(self)
            # 표본 수집은 하나씩 처리한다. 순서가 보장되어 수동 번역 기록이 먼저 반영된다.
            thread_pool.setMaxThreadCount(1)
        self._pool = thread_pool
        self._timer = QTimer(self)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self._tick)
        # 감시 영역 번호: reset()(박스 이동/크기 변경)마다 증가한다.
        self._epoch = 0
        # 감시 루프 번호: start/stop/reset마다 증가해 이전 루프의 결과를 버린다.
        self._generation = 0
        # 감시 루프의 표본 작업이 진행 중인지 여부 (동시에 하나만 보낸다)
        self._in_flight = False
        # 연속 캡처 실패 횟수
        self._failures = 0
        self._pending: dict[int, _PendingSample] = {}
        self._next_task_id = 0

    @property
    def is_running(self) -> bool:
        """자동 감지 루프가 켜져 있는지 여부."""
        return self._timer.isActive()

    def start(self) -> None:
        """자동 감지를 켠다. 수동 번역으로 기록된 화면은 그대로 유지한다."""
        if self.is_running:
            return
        self._tracker.restart()
        self._new_generation()
        self._timer.start()
        logger.info("자동 감지 시작 (간격 %dms)", self._timer.interval())
        self._tick()

    def stop(self) -> None:
        """자동 감지를 끈다. 진행 중인 표본 결과는 버린다."""
        if not self.is_running:
            return
        self._timer.stop()
        self._new_generation()
        logger.info("자동 감지 중지")

    def reset(self) -> None:
        """박스 위치/크기가 바뀌면 이전 화면 기록을 모두 버리고 새로 관찰한다."""
        self._tracker.reset()
        self._epoch += 1
        self._new_generation()

    def mark_translated(self, region: ScreenRegion) -> None:
        """수동 번역한 영역의 현재 화면을 기록한다.

        자동 감지를 나중에 켜거나 이미 켜져 있을 때, 같은 화면을 다시 번역하지 않게
        한다. 자동 감지가 꺼져 있어도 기록한다. 기록용 표본은 감시 루프와 같은
        스레드 풀에서 순서대로 처리되므로, 이후 관찰보다 먼저 반영된다.
        """
        self._submit(region, is_baseline=True)

    def mark_fingerprint(self, fingerprint: object, region: ScreenRegion) -> None:
        """이미 OCR한 화면의 지문을 추가 캡처 없이 기록한다 (수동 OCR 결과의 지문).

        `region`이 지금 감시 영역과 다르면(그사이 박스가 옮겨졌으면) 기록하지 않는다.
        """
        if self._region_provider() == region:
            self._tracker.mark_translated(fingerprint)

    def shutdown(self, timeout_ms: int = SHUTDOWN_TIMEOUT_MS) -> None:
        """자동 감지를 끄고 백그라운드 작업이 끝날 때까지 기다린다."""
        self.stop()
        self._pool.clear()
        if not self._pool.waitForDone(timeout_ms):
            logger.warning("종료 시 자동 감지 작업이 %dms 안에 끝나지 않았습니다.", timeout_ms)

    def _new_generation(self) -> None:
        """감시 루프 번호를 올려 이전 루프의 결과를 무효로 만든다."""
        self._generation += 1
        self._in_flight = False
        self._failures = 0

    def _tick(self) -> None:
        """타이머마다 호출: 이전 표본이 끝났으면 새 표본을 요청한다."""
        if self._in_flight:
            return
        region = self._region_provider()
        if region is None:
            return
        self._in_flight = True
        self._submit(region, is_baseline=False)

    def _submit(self, region: ScreenRegion, *, is_baseline: bool) -> None:
        """표본 작업을 만들어 스레드 풀에 넣는다."""
        self._next_task_id += 1
        task_id = self._next_task_id
        signals = _SampleSignals()
        # 람다가 아닌 이 객체의 메서드에 연결해야 UI 스레드로 큐 전달된다.
        signals.sampled.connect(self._on_sampled)
        signals.failed.connect(self._on_failed)
        task = _SampleTask(task_id, self._capturer, self._observer, region, signals)
        # 작업이 끝날 때까지 Python 참조를 유지한다.
        self._pending[task_id] = _PendingSample(
            task, signals, region, self._epoch, self._generation, is_baseline
        )
        self._pool.start(task)

    def _take(self, task_id: int) -> _PendingSample | None:
        """끝난 작업의 정보를 꺼내고 시그널 객체를 정리한다."""
        pending = self._pending.pop(task_id, None)
        if pending is not None:
            pending.signals.deleteLater()
        return pending

    @Slot(int, object, object, float)
    def _on_sampled(
        self, task_id: int, observation: object, image: CapturedImage, captured_at: float
    ) -> None:
        """표본 결과를 판단 정책에 넘기고, 필요하면 그 캡처를 프레임으로 넘겨 OCR을 요청한다."""
        pending = self._take(task_id)
        if pending is None or pending.epoch != self._epoch:
            # 박스가 바뀐 뒤 도착한 결과는 다른 영역의 화면이므로 버린다.
            return
        if pending.is_baseline:
            self._tracker.mark_translated(observation)
            return
        if pending.generation != self._generation:
            return
        self._in_flight = False
        self._failures = 0
        if self._tracker.observe(observation):
            logger.debug("감시 영역 화면 변화 감지: OCR 요청")
            # 같은 화면을 다시 캡처하지 않도록 이 표본의 캡처를 그대로 넘긴다.
            self.frame_ready.emit(Frame(next_frame_id(), image, captured_at, observation))
        # OCR이 필요 없는 표본의 이미지는 여기서 버린다 (참조가 사라지면 해제됨).

    @Slot(int, str)
    def _on_failed(self, task_id: int, message: str) -> None:
        """표본 실패를 세고, 연속 실패가 한도에 이르면 자동 감지를 끈다."""
        pending = self._take(task_id)
        if pending is None:
            return
        if pending.is_baseline:
            # 기록 실패는 자동 감지를 멈출 이유가 아니다. 같은 화면이 한 번 더 번역될 수 있을 뿐이다.
            logger.warning("수동 번역 화면 기록 실패: 자동 감지가 같은 화면을 다시 번역할 수 있습니다.")
            return
        if pending.epoch != self._epoch or pending.generation != self._generation:
            return
        self._in_flight = False
        self._failures += 1
        logger.warning("자동 감지 캡처 실패 (%d/%d)", self._failures, self._max_failures)
        if self._failures >= self._max_failures:
            self.stop()
            self.stopped_by_error.emit(message)
