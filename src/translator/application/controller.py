"""인식(OCR)과 번역 작업을 공용 추론 큐에서 실행하고 결과를 UI 스레드로 전달한다.

출처(바운딩 박스 ID, 전체 화면)마다, 작업 종류(OCR, 번역)마다 "현재 유효한 요청"을 하나씩
관리한다.

- 같은 출처·같은 종류의 새 요청이 오면 이전 요청을 취소한다. 큐에서 기다리던 작업은
  시작하자마자 끝나므로 불필요한 중복 작업이 쌓이지 않는다.
- 다른 출처의 요청은 서로 취소하지 않고, 큐에서 우선순위 순으로 처리된다.
- 결과는 출처·종류·요청 번호가 모두 현재 값과 맞을 때만 전달한다. 늦게 끝난 오래된
  요청의 결과가 최신 결과를 덮어쓰지 않는다.
- 요청마다 요청한 시점의 콘텐츠·세션(`SessionContext`)을 결과에 실어 보낸다. 세션이 바뀌면
  (`begin_session`) 진행 중인 모든 요청을 취소하고 늦게 도착하는 결과를 버린다.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, replace

from PySide6.QtCore import QObject, QRunnable, Signal, Slot

from translator.application.frames import Frame
from translator.application.pipeline import OcrResult, PipelineRequest, TranslationPipeline
from translator.application.scheduler import InferenceScheduler, TaskPriority
from translator.core.cancellation import CancellationToken
from translator.core.errors import (
    UNEXPECTED_ERROR_MESSAGE,
    OperationCancelledError,
    TranslatorError,
)
from translator.core.models import ScreenRegion
from translator.core.session import SessionContext, new_session
from translator.tuning import AUTO_DETECT_GRID_HEIGHT, AUTO_DETECT_GRID_WIDTH, SHUTDOWN_TIMEOUT_MS

logger = logging.getLogger(__name__)

# 작업 종류
KIND_OCR = "ocr"
KIND_TRANSLATE = "translate"


@dataclass(frozen=True)
class TranslationOutcome:
    """번역 단계의 결과: 원문 → (번역문, 엔진). `session`은 요청한 시점의 콘텐츠·세션이다."""

    source_id: str
    request_id: int
    translations: dict[str, tuple[str, str]]
    session: SessionContext | None = None


class _TaskSignals(QObject):
    """워커 스레드 → UI 스레드 전달용 시그널 (UI 스레드에서 생성되어 큐 연결로 전달됨)."""

    succeeded = Signal(int, object)  # 요청 번호, 결과
    failed = Signal(int, str)  # 요청 번호, 사용자에게 보여 줄 메시지
    cancelled = Signal(int)  # 요청 번호


class _Task(QRunnable):
    """추론 작업 하나 (OCR 또는 번역)."""

    def __init__(
        self,
        request_id: int,
        work: Callable[[CancellationToken], object],
        cancellation: CancellationToken,
        signals: _TaskSignals,
    ) -> None:
        """요청 번호, 실행할 함수, 취소 토큰, 결과 시그널을 받는다."""
        super().__init__()
        self._request_id = request_id
        self._work = work
        self._cancellation = cancellation
        self._signals = signals

    def run(self) -> None:
        """작업을 실행하고 성공/실패/취소 결과를 시그널로 보낸다."""
        request_id = self._request_id
        try:
            # 큐에서 기다리는 동안 더 새로운 요청으로 대체되었으면 바로 끝낸다.
            self._cancellation.raise_if_cancelled()
            result = self._work(self._cancellation)
        except OperationCancelledError:
            logger.debug("작업 %d 취소됨", request_id)
            self._emit(self._signals.cancelled, request_id)
        except TranslatorError as exc:
            logger.warning("작업 %d 실패 (%s): %s", request_id, type(exc).__name__, exc)
            self._emit(self._signals.failed, request_id, str(exc))
        except Exception:
            # 워커 스레드의 예외가 사라지지 않도록 기록하고 UI에는 일반 메시지만 전달한다.
            logger.exception("작업 %d 처리 중 예기치 못한 오류", request_id)
            self._emit(self._signals.failed, request_id, UNEXPECTED_ERROR_MESSAGE)
        else:
            self._emit(self._signals.succeeded, request_id, result)

    def _emit(self, signal, *args: object) -> None:
        """결과 시그널을 보낸다. 받을 객체가 이미 정리되었으면(종료 뒤에 끝난 작업) 결과를 버린다.

        매개변수:
            signal: 보낼 시그널 (`_TaskSignals`의 시그널).
            args: 시그널 인자.
        """
        try:
            signal.emit(*args)
        except RuntimeError:
            # 멈출 수 없는 추론(예: 긴 OCR)이 종료 대기 시간을 넘겨 끝나면, 컨트롤러와 시그널 객체가
            # 이미 삭제되어 있다. 그 결과는 쓸 곳이 없으므로 버린다.
            logger.debug("작업 %d의 결과를 받을 객체가 이미 정리되어 결과를 버립니다.", self._request_id)


@dataclass
class _Active:
    """출처·종류별로 현재 유효한 요청."""

    request_id: int
    token: CancellationToken


@dataclass
class _Running:
    """결과를 기다리는 작업. 끝날 때까지 Python 참조를 유지한다."""

    task: _Task
    signals: _TaskSignals
    source_id: str
    kind: str
    # 요청한 시점의 콘텐츠·세션
    session: SessionContext


class TranslationController(QObject):
    """UI가 사용하는 인식·번역 요청 진입점. 모든 공개 메서드는 UI 스레드에서 호출한다."""

    ocr_ready = Signal(object)  # OcrResult (request_id 포함)
    translations_ready = Signal(object)  # TranslationOutcome
    failed = Signal(str, str)  # 출처 ID, 사용자에게 보여 줄 메시지

    def __init__(
        self,
        pipeline: TranslationPipeline,
        scheduler: InferenceScheduler | None = None,
        parent: QObject | None = None,
        session: SessionContext | None = None,
    ) -> None:
        """파이프라인, 공용 추론 큐, 처음 세션을 받는다.

        큐가 없으면 새로 만들고, 세션이 없으면 기본 콘텐츠의 새 세션으로 시작한다.
        """
        super().__init__(parent)
        self._pipeline = pipeline
        self._scheduler = scheduler or InferenceScheduler(parent=self)
        # 지금 유효한 콘텐츠·세션. 요청은 이 값을 달고 나가고, 다른 세션의 결과는 버려진다.
        self._session = session or new_session()
        # 마지막으로 발급한 요청 번호 (출처·종류와 관계없이 전체에서 1씩 증가)
        self._last_request_id = 0
        # (출처 ID, 종류) → 결과를 화면에 보여 줄 유일한 요청
        self._active: dict[tuple[str, str], _Active] = {}
        # 요청 번호 → 실행 중이거나 큐에서 기다리는 작업
        self._running: dict[int, _Running] = {}
        # 종료된 뒤에는 새 요청을 받지 않는다.
        self._is_shut_down = False
        # 종료할 때 실행 중인 작업이 대기 시간 안에 모두 끝났는지
        self._work_finished_at_shutdown = True

    @property
    def scheduler(self) -> InferenceScheduler:
        """공용 추론 큐."""
        return self._scheduler

    @property
    def session(self) -> SessionContext:
        """지금 유효한 콘텐츠·세션."""
        return self._session

    def begin_session(self, session: SessionContext) -> None:
        """새 세션을 시작한다. 이전 세션의 요청은 모두 취소하고, 늦게 도착하는 결과는 버린다.

        매개변수:
            session: 새 콘텐츠·세션.
        """
        if session == self._session:
            return
        logger.info(
            "세션 변경: %s → %s (진행 중 요청 %d개 취소)",
            self._session.session_id, session.session_id, len(self._active),
        )
        self.cancel_all()
        self._session = session

    @property
    def is_busy(self) -> bool:
        """결과를 기다리는 요청이 하나라도 있는지 여부."""
        return bool(self._active)

    def is_pending(self, source_id: str, kind: str | None = None) -> bool:
        """출처의 요청(종류를 주면 그 종류만)이 결과를 기다리는 중인지 여부."""
        kinds = (kind,) if kind else (KIND_OCR, KIND_TRANSLATE)
        return any((source_id, k) in self._active for k in kinds)

    def request_ocr(
        self,
        source_id: str,
        region: ScreenRegion | None,
        *,
        frame: Frame | None = None,
        exclude: tuple[ScreenRegion, ...] = (),
        probe_areas: tuple[ScreenRegion, ...] = (),
        fingerprint_grid: tuple[int, int] = (AUTO_DETECT_GRID_WIDTH, AUTO_DETECT_GRID_HEIGHT),
        priority: TaskPriority = TaskPriority.OCR_USER,
    ) -> int | None:
        """인식(캡처·OCR·묶기)을 요청한다. 같은 출처의 이전 OCR 요청은 취소된다.

        `frame`이 있으면 다시 캡처하지 않고 그 프레임을 쓴다. `region`이 None이고 프레임도
        없으면 전체 화면을 캡처한다.
        """
        request = PipelineRequest(
            source_id, region, exclude, frame, probe_areas, fingerprint_grid
        )
        return self._submit(
            source_id, KIND_OCR, lambda token: self._pipeline.recognize(request, token), priority
        )

    def request_translation(self, source_id: str, texts: tuple[str, ...]) -> int | None:
        """안정된 원문들의 번역을 요청한다. 같은 출처의 이전 번역 요청은 취소된다.

        호출하는 쪽은 아직 번역이 필요한 원문을 모두 넘기므로, 이전 요청을 취소해도 빠지는
        원문이 없다. 번역 결과는 캐시에 남아 같은 원문을 다시 번역하지 않는다.
        """
        return self._submit(
            source_id,
            KIND_TRANSLATE,
            lambda token: self._pipeline.translate_texts(list(texts), token),
            TaskPriority.TRANSLATION,
        )

    def cancel(self, source_id: str) -> None:
        """출처의 모든 요청을 취소하고, 늦게 도착하는 결과도 무시하게 한다."""
        for kind in (KIND_OCR, KIND_TRANSLATE):
            active = self._active.pop((source_id, kind), None)
            if active is not None:
                active.token.cancel()

    def cancel_all(self) -> None:
        """모든 출처의 요청을 취소한다."""
        for source_id, _kind in list(self._active):
            self.cancel(source_id)

    def shutdown(self, timeout_ms: int = SHUTDOWN_TIMEOUT_MS) -> bool:
        """새 요청을 막고, 기다리는 작업을 버린 뒤 실행 중인 작업이 끝날 때까지 기다린다.

        매개변수:
            timeout_ms: 실행 중인 작업을 기다리는 최대 시간 (밀리초).
        반환값:
            실행 중인 작업이 시간 안에 모두 끝났으면 True. 이미 종료했으면 그때의 결과.
        """
        if self._is_shut_down:
            return self._work_finished_at_shutdown
        self._is_shut_down = True
        self.cancel_all()
        self._scheduler.clear_pending()
        self._work_finished_at_shutdown = self._scheduler.wait(timeout_ms)
        return self._work_finished_at_shutdown

    def _submit(
        self,
        source_id: str,
        kind: str,
        work: Callable[[CancellationToken], object],
        priority: TaskPriority,
    ) -> int | None:
        """요청 번호와 취소 토큰을 만들어 작업을 공용 추론 큐에 넣는다."""
        if self._is_shut_down:
            logger.warning("종료된 컨트롤러에 요청이 들어와 무시합니다.")
            return None
        previous = self._active.pop((source_id, kind), None)
        if previous is not None:
            previous.token.cancel()

        self._last_request_id += 1
        request_id = self._last_request_id
        token = CancellationToken()
        self._active[(source_id, kind)] = _Active(request_id, token)

        signals = _TaskSignals()
        signals.succeeded.connect(self._on_task_succeeded)
        signals.failed.connect(self._on_task_failed)
        signals.cancelled.connect(self._on_task_cancelled)
        task = _Task(request_id, work, token, signals)
        self._running[request_id] = _Running(task, signals, source_id, kind, self._session)
        logger.debug("작업 %d 요청 (%s, %s, 우선순위 %s)", request_id, source_id, kind, priority.name)
        self._scheduler.submit(task, priority)
        return request_id

    def _finish(self, request_id: int) -> _Running | None:
        """작업 참조를 정리하고, 현재 유효한 요청이면 그 정보를 반환한다."""
        running = self._running.pop(request_id, None)
        if running is None:
            return None
        running.signals.deleteLater()
        key = (running.source_id, running.kind)
        active = self._active.get(key)
        if active is None or active.request_id != request_id:
            logger.debug("오래된 작업 %d(%s)의 결과를 무시합니다.", request_id, running.source_id)
            return None
        del self._active[key]
        return running

    @Slot(int, object)
    def _on_task_succeeded(self, request_id: int, result: object) -> None:
        """현재 유효한 요청의 결과만 UI에 전달한다."""
        running = self._finish(request_id)
        if running is None:
            return
        if running.kind == KIND_OCR:
            assert isinstance(result, OcrResult)
            self.ocr_ready.emit(replace(result, request_id=request_id, session=running.session))
        else:
            self.translations_ready.emit(
                TranslationOutcome(running.source_id, request_id, dict(result), running.session)
            )

    @Slot(int, str)
    def _on_task_failed(self, request_id: int, message: str) -> None:
        """현재 유효한 요청의 실패 메시지만 출처와 함께 UI에 전달한다."""
        running = self._finish(request_id)
        if running is not None:
            self.failed.emit(running.source_id, message)

    @Slot(int)
    def _on_task_cancelled(self, request_id: int) -> None:
        """취소된 작업의 참조를 정리한다."""
        self._finish(request_id)
