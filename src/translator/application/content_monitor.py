"""활성 창을 주기적으로 확인해 콘텐츠가 바뀌면 알린다.

- 콘텐츠 식별(`ContentDetector.detect`)은 백그라운드 스레드에서 한다. 식별 구현이 느려져도(예: 나중에
  브라우저 확장과 통신) UI가 멈추지 않게 하기 위해서다.
- 바뀌었는지 판단(`ContentSwitchPolicy`)과 타이머는 UI 스레드에서 처리한다.
- 이 객체는 "바뀌었다"고 알리기만 한다. 세션을 바꾸고 이전 작업을 정리하는 일은 번역 흐름이 맡는다.

모든 공개 메서드는 UI 스레드에서 호출한다.
"""

import logging

from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Signal, Slot

from translator.application.content_switch import ContentSwitchPolicy
from translator.core.content import ContentIdentity
from translator.core.interfaces import ContentDetector
from translator.tuning import (
    CONTENT_DETECT_INTERVAL_MS,
    CONTENT_DETECT_MAX_CONSECUTIVE_FAILURES,
    SHUTDOWN_TIMEOUT_MS,
)

logger = logging.getLogger(__name__)

# 연속 실패로 콘텐츠 자동 식별을 껐을 때 사용자에게 보여 줄 메시지
DETECTION_STOPPED_MESSAGE = (
    "활성 창을 확인하지 못해 콘텐츠 자동 식별을 껐습니다. 프로그램을 다시 시작하거나 로그 파일을 확인해 주세요."
)


class _DetectSignals(QObject):
    """워커 스레드 → UI 스레드 전달용 시그널 (UI 스레드에서 생성되어 큐 연결로 전달됨)."""

    detected = Signal(int, object)  # 작업 ID, 식별 결과 (ContentIdentity 또는 None)
    failed = Signal(int)  # 작업 ID


class _DetectTask(QRunnable):
    """콘텐츠를 한 번 식별하는 백그라운드 작업."""

    def __init__(self, task_id: int, detector: ContentDetector, signals: _DetectSignals) -> None:
        """작업 ID, 식별기, 결과를 보낼 시그널을 받는다."""
        super().__init__()
        # 이 작업의 번호 (결과가 어느 요청의 것인지 알아보는 데 쓴다)
        self._task_id = task_id
        # 콘텐츠를 식별하는 구현
        self._detector = detector
        # 결과를 UI 스레드로 보낼 시그널
        self._signals = signals

    def run(self) -> None:
        """콘텐츠를 식별해 결과나 실패를 시그널로 보낸다."""
        try:
            identity = self._detector.detect()
        except Exception:
            # 워커 스레드의 예외가 사라지지 않도록 기록한다. 활성 창 정보는 로그에 남기지 않는다.
            logger.exception("콘텐츠 식별 중 예기치 못한 오류")
            self._signals.failed.emit(self._task_id)
        else:
            self._signals.detected.emit(self._task_id, identity)


class ContentMonitor(QObject):
    """콘텐츠 감시 루프. 새 콘텐츠가 확정되면 `content_changed`를 보낸다."""

    content_changed = Signal(object)  # ContentIdentity: 확정된 새 콘텐츠
    stopped_by_error = Signal(str)  # 연속 실패로 감시를 멈췄다 (사용자에게 보여 줄 메시지)

    def __init__(
        self,
        detector: ContentDetector,
        policy: ContentSwitchPolicy | None = None,
        interval_ms: int = CONTENT_DETECT_INTERVAL_MS,
        max_consecutive_failures: int = CONTENT_DETECT_MAX_CONSECUTIVE_FAILURES,
        thread_pool: QThreadPool | None = None,
        parent: QObject | None = None,
    ) -> None:
        """식별기, 바뀜 판단 정책, 확인 간격, 허용 연속 실패 횟수를 받는다."""
        super().__init__(parent)
        # 콘텐츠를 식별하는 구현 (백그라운드 스레드에서 호출된다)
        self._detector = detector
        # 관찰 결과로 콘텐츠가 바뀌었는지 판단하는 정책
        self._policy = policy or ContentSwitchPolicy()
        # 이 횟수만큼 연속으로 예외로 실패하면 감시를 멈춘다
        self._max_failures = max_consecutive_failures
        if thread_pool is None:
            thread_pool = QThreadPool(self)
            # 식별은 하나씩 처리한다. 관찰 순서가 보장되어야 연속 관찰 횟수가 맞다.
            thread_pool.setMaxThreadCount(1)
        # 식별 작업을 실행하는 스레드 풀
        self._pool = thread_pool
        # 일정 간격으로 식별을 요청하는 타이머
        self._timer = QTimer(self)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self._tick)
        # 감시 루프 번호: start/stop마다 증가해 이전 루프의 결과를 버린다.
        self._generation = 0
        # 식별 작업이 진행 중인지 여부 (동시에 하나만 보낸다)
        self._in_flight = False
        # 연속 실패 횟수
        self._failures = 0
        # 작업 ID → (시그널 객체, 요청 시점의 감시 루프 번호). 끝날 때까지 참조를 유지한다.
        self._pending: dict[int, tuple[_DetectSignals, _DetectTask, int]] = {}
        # 다음에 발급할 작업 번호
        self._next_task_id = 0

    @property
    def is_running(self) -> bool:
        """감시 루프가 켜져 있는지 여부."""
        return self._timer.isActive()

    @property
    def policy(self) -> ContentSwitchPolicy:
        """바뀜 판단 정책 (현재 콘텐츠 조회·재설정용)."""
        return self._policy

    def start(self) -> None:
        """감시를 시작하고 바로 한 번 확인한다."""
        if self.is_running:
            return
        self._new_generation()
        self._timer.start()
        logger.info("콘텐츠 감시 시작 (간격 %dms)", self._timer.interval())
        self._tick()

    def stop(self) -> None:
        """감시를 멈춘다. 진행 중인 식별 결과는 버린다."""
        if not self.is_running:
            return
        self._timer.stop()
        self._new_generation()
        logger.info("콘텐츠 감시 중지")

    def shutdown(self, timeout_ms: int = SHUTDOWN_TIMEOUT_MS) -> None:
        """감시를 멈추고 백그라운드 작업이 끝날 때까지 기다린다."""
        self.stop()
        self._pool.clear()
        if not self._pool.waitForDone(timeout_ms):
            logger.warning("종료 시 콘텐츠 식별 작업이 %dms 안에 끝나지 않았습니다.", timeout_ms)

    def _new_generation(self) -> None:
        """감시 루프 번호를 올려 이전 루프의 결과를 무효로 만든다."""
        self._generation += 1
        self._in_flight = False
        self._failures = 0

    def _tick(self) -> None:
        """타이머마다 호출: 이전 식별이 끝났으면 새 식별을 요청한다."""
        if self._in_flight:
            return
        self._in_flight = True
        self._next_task_id += 1
        task_id = self._next_task_id
        signals = _DetectSignals()
        # 람다가 아닌 이 객체의 메서드에 연결해야 UI 스레드로 큐 전달된다.
        signals.detected.connect(self._on_detected)
        signals.failed.connect(self._on_failed)
        task = _DetectTask(task_id, self._detector, signals)
        self._pending[task_id] = (signals, task, self._generation)
        self._pool.start(task)

    def _take(self, task_id: int) -> int | None:
        """끝난 작업을 꺼내고 시그널 객체를 정리한다.

        반환값:
            작업을 요청한 시점의 감시 루프 번호. 알 수 없는 작업이면 None.
        """
        entry = self._pending.pop(task_id, None)
        if entry is None:
            return None
        entry[0].deleteLater()
        return entry[2]

    @Slot(int, object)
    def _on_detected(self, task_id: int, identity: object) -> None:
        """식별 결과를 정책에 넘기고, 콘텐츠가 바뀐 것으로 확정되면 알린다."""
        generation = self._take(task_id)
        if generation is None or generation != self._generation:
            return
        self._in_flight = False
        self._failures = 0
        changed = self._policy.observe(identity if isinstance(identity, ContentIdentity) else None)
        if changed is not None:
            self.content_changed.emit(changed)

    @Slot(int)
    def _on_failed(self, task_id: int) -> None:
        """식별 실패를 세고, 연속 실패가 한도에 이르면 감시를 멈춘다 (무한 재시도 방지)."""
        generation = self._take(task_id)
        if generation is None or generation != self._generation:
            return
        self._in_flight = False
        self._failures += 1
        logger.warning("콘텐츠 식별 실패 (%d/%d)", self._failures, self._max_failures)
        if self._failures >= self._max_failures:
            self.stop()
            self.stopped_by_error.emit(DETECTION_STOPPED_MESSAGE)
