"""출처(바운딩 박스, 전체 화면)별 번역 처리 흐름.

화면(UI)과 무관한 흐름 로직을 한곳에 모은다. 화면 쪽은 `BoxPort`·`DisplayPort` 구현을 주입하고,
이 객체가 보내는 시그널(알림)을 받아 창을 갱신한다.

처리 흐름 (출처마다 같다):
1. OCR 요청: 수동 번역은 새로 캡처하고, 자동 감지는 감지기가 이미 찍은 프레임을 넘긴다.
2. 인식 결과(`OcrResult`)를 표시 모델에 반영한다: 위치 갱신(이동은 재번역 없음), 텍스트
   안정화, OCR 누락/실제 사라짐 판단.
3. 안정됐지만 아직 번역하지 않은 원문만 번역을 요청한다 (번역 큐가 OCR보다 우선).
4. 안정되지 않은 텍스트가 있거나 OCR이 텍스트를 놓쳤으면 `ObservationScheduler`가 정한
   시각에 다시 캡처해 OCR한다 (자동 감지 박스와 전체 화면 추적만).

전체 화면 번역은 켜고 끄는 방식이다. 켜면 한 번 인식한 뒤 전체 화면 감지기가 화면 변화를
지켜보며 원문 위치와 번역을 갱신하고, 끄면 전체 화면 번역을 지운다.

세션: 콘텐츠·세션이 바뀌면(`change_session`) 이전 세션의 요청을 모두 취소하고 표시를 지우고 박스를
새 콘텐츠의 것으로 바꾼 뒤, 자동으로 추적하는 출처는 현재 화면을 새 세션에서 다시 인식한다. 이전 세션의 결과가 늦게
도착해도 표시하지 않는다.

UI 스레드에서만 사용한다.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager

from PySide6.QtCore import QObject, QTimer, Signal

from translator.application.auto_detector import AutoDetector
from translator.application.content_monitor import ContentMonitor
from translator.application.controller import (
    KIND_OCR,
    KIND_TRANSLATE,
    TranslationController,
    TranslationOutcome,
)
from translator.application.frames import Frame
from translator.application.pipeline import OcrResult
from translator.application.recheck import ObservationScheduler
from translator.application.scheduler import TaskPriority
from translator.application.workflow_ports import BoxPort, DisplayPort
from translator.core.content import ContentIdentity
from translator.core.models import ScreenRegion
from translator.core.session import SessionContext, new_session
from translator.tuning import (
    AUTO_DETECT_GRID_HEIGHT,
    AUTO_DETECT_GRID_WIDTH,
    FULLSCREEN_DETECT_GRID_HEIGHT,
    FULLSCREEN_DETECT_GRID_WIDTH,
    OBSERVATION_TICK_MS,
)

logger = logging.getLogger(__name__)

# 전체 화면 번역 결과의 출처 ID
FULLSCREEN_SOURCE_ID = "fullscreen"

# OCR 요청 방식
# 사용자가 직접 요청: 안정화를 기다리지 않고, 결과가 비면 "텍스트 없음"을 알린다.
MODE_MANUAL = "manual"
# 전체 화면 번역에 딸린 박스 요청: 수동처럼 처리하되 "텍스트 없음"을 따로 알리지 않는다.
MODE_QUIET = "quiet"
# 자동 감지·후속 OCR·재검사: 텍스트가 안정된 뒤 번역한다.
MODE_AUTO = "auto"


class TranslationWorkflow(QObject):
    """출처별 OCR 요청, 안정화, 번역 요청, 재검사 예약, 전체 화면 추적, 세션 전환을 맡는다."""

    # 전체 화면 번역이 켜지거나(True) 꺼졌다(False). 버튼 상태를 맞추는 데 쓴다.
    full_screen_changed = Signal(bool)
    # 전체 화면 추적이 연속 실패로 멈췄다 (사용자에게 보여 줄 메시지)
    full_screen_error = Signal(str)
    # 출처의 인식·번역이 실패했다 (출처 ID, 사용자에게 보여 줄 메시지)
    source_failed = Signal(str, str)
    # 사용자가 요청한 인식에서 텍스트를 찾지 못했다 (OCR이 놓친 것도 아님)
    no_text_found = Signal()
    # 콘텐츠 자동 식별이 연속 실패로 멈췄다 (사용자에게 보여 줄 메시지)
    content_detection_error = Signal(str)

    def __init__(
        self,
        controller: TranslationController,
        display: DisplayPort,
        boxes: BoxPort,
        fullscreen_detector: AutoDetector | None = None,
        observation: ObservationScheduler | None = None,
        observation_tick_ms: int = OBSERVATION_TICK_MS,
        content_monitor: ContentMonitor | None = None,
        parent: QObject | None = None,
    ) -> None:
        """작업 컨트롤러, 화면 쪽 인터페이스, 전체 화면 감지기, 재검사 예약을 받는다.

        - `fullscreen_detector`가 없으면 전체 화면 번역은 켤 때 한 번만 인식한다 (추적 없음).
        - `content_monitor`가 있으면 콘텐츠가 바뀔 때 세션을 바꾼다 (`start`/`stop`에서 함께 시작·종료).
          없으면 콘텐츠를 식별하지 않고 세션은 호출하는 쪽이 바꾼다.
        - `observation`은 후속 OCR·재검사 시각을 정한다 (없으면 기본값으로 만든다).
        - 예약 시각은 `observation_tick_ms`마다 확인한다 (`start`로 시작).
        """
        super().__init__(parent)
        self._controller = controller
        self._display = display
        self._boxes = boxes
        # 전체 화면 추적 감지기 (없으면 켤 때 한 번만 인식)
        self._fullscreen_detector = fullscreen_detector
        # 후속 OCR·재검사 예약
        self._observation = observation or ObservationScheduler()
        # 활성 창으로 콘텐츠 변경을 알려 주는 감시기 (없으면 콘텐츠를 식별하지 않음)
        self._content_monitor = content_monitor
        # 출처 → (현재 OCR 요청 번호, 요청 방식). 결과가 그 요청의 것일 때만 방식을 적용한다.
        self._ocr_modes: dict[str, tuple[int, str]] = {}
        # 출처 → 진행 중인 번역 요청의 원문들 (같은 원문의 번역을 다시 요청해 취소·재시작하지 않게)
        self._translating: dict[str, frozenset[str]] = {}
        # 전체 화면 번역이 켜져 있는지 여부
        self._full_screen_on = False
        # 영역 선택 화면이 떠 있는지 여부 (그동안 전체 화면 감지를 멈춘다)
        self._selection_active = False
        # 종료가 시작되면 늦게 도착한 타이머·시그널이 새 작업을 만들지 않게 막는다.
        self._stopped = False
        # 예약 시각을 확인하는 타이머
        self._timer = QTimer(self)
        self._timer.setInterval(observation_tick_ms)
        self._timer.timeout.connect(self._run_due_observations)

        controller.ocr_ready.connect(self.handle_ocr_ready)
        controller.translations_ready.connect(self.handle_translations_ready)
        controller.failed.connect(self.handle_failed)
        if fullscreen_detector is not None:
            fullscreen_detector.frame_ready.connect(self._on_fullscreen_frame)
            fullscreen_detector.stopped_by_error.connect(self._on_fullscreen_detector_stopped)
        if content_monitor is not None:
            content_monitor.content_changed.connect(self.handle_content_identified)
            content_monitor.stopped_by_error.connect(self.content_detection_error)

    # --- 상태 조회 ------------------------------------------------------------------

    @property
    def controller(self) -> TranslationController:
        """OCR·번역 작업 컨트롤러."""
        return self._controller

    @property
    def observation(self) -> ObservationScheduler:
        """후속 OCR·재검사 예약."""
        return self._observation

    @property
    def fullscreen_detector(self) -> AutoDetector | None:
        """전체 화면 추적 감지기 (없으면 None)."""
        return self._fullscreen_detector

    @property
    def full_screen_on(self) -> bool:
        """전체 화면 번역이 켜져 있는지 여부."""
        return self._full_screen_on

    @property
    def session(self) -> SessionContext:
        """지금 유효한 콘텐츠·세션."""
        return self._controller.session

    @property
    def is_observing(self) -> bool:
        """예약 확인 타이머가 동작 중인지 여부."""
        return self._timer.isActive()

    # --- 시작·종료 ------------------------------------------------------------------

    def start(self) -> None:
        """후속 OCR·재검사 예약 확인과 콘텐츠 감시를 시작한다."""
        self._timer.start()
        if self._content_monitor is not None:
            self._content_monitor.start()

    def stop(self) -> None:
        """새 작업이 생기지 않도록 예약 확인과 전체 화면 감지를 멈춘다 (종료할 때).

        작업 컨트롤러는 멈추지 않는다. 호출하는 쪽이 박스를 정리한 뒤 `controller.shutdown()`을 부른다.
        """
        self._stopped = True
        self._timer.stop()
        if self._content_monitor is not None:
            self._content_monitor.shutdown()
        if self._fullscreen_detector is not None:
            self._fullscreen_detector.shutdown()

    def use_observation(self, observation: ObservationScheduler, tick_ms: int | None = None) -> None:
        """재검사 예약을 바꾼다 (테스트에서 간격을 짧게 하는 용도). `tick_ms`를 주면 확인 간격도 바꾼다."""
        self._observation = observation
        if tick_ms is not None:
            self._timer.setInterval(tick_ms)

    # --- 세션 ---------------------------------------------------------------------

    def handle_content_identified(self, identity: ContentIdentity) -> None:
        """콘텐츠 감시기가 새 콘텐츠를 확정했을 때 새 세션을 시작한다.

        매개변수:
            identity: 확정된 새 콘텐츠.
        """
        if self._stopped:
            return
        self.change_content(identity.content_id)

    def change_content(self, content_id: str) -> SessionContext:
        """다른 콘텐츠로 바뀌었음을 알리고 새 세션을 시작한다.

        매개변수:
            content_id: 새 콘텐츠 ID.
        반환값:
            새로 시작한 세션.
        """
        session = new_session(content_id)
        self.change_session(session)
        return session

    def change_session(self, session: SessionContext) -> None:
        """새 세션을 시작한다. 이전 세션의 작업·예약·표시를 모두 정리한다.

        자동으로 추적하는 출처(자동 감지 박스, 전체 화면)는 현재 화면을 새 세션에서 바로 다시
        인식한다. 수동으로 번역한 박스는 사용자가 다시 요청해야 한다.

        매개변수:
            session: 새 콘텐츠·세션. 지금 세션과 같으면 아무것도 하지 않는다.
        """
        if session == self._controller.session:
            return
        if self._content_monitor is not None:
            # 다른 경로로 콘텐츠가 바뀌었어도 감시기가 같은 콘텐츠를 또 "바뀜"으로 알리지 않게 맞춘다.
            self._content_monitor.policy.reset(session.content_id)
        self._controller.begin_session(session)
        self._observation.clear()
        self._ocr_modes.clear()
        self._translating.clear()
        self._display.clear()
        # 박스를 새 콘텐츠의 것으로 바꾼 뒤에 다시 인식해야 새 박스가 대상이 된다 (순서 중요).
        self._boxes.switch_content(session.content_id)
        self._restart_tracked_sources()

    def _restart_tracked_sources(self) -> None:
        """자동으로 추적하는 출처의 현재 화면을 새 세션에서 다시 인식한다."""
        if self._stopped:
            return
        if self._full_screen_on and self._fullscreen_detector is not None:
            self._request_ocr(FULLSCREEN_SOURCE_ID, None, mode=MODE_AUTO, priority=TaskPriority.OCR_AUTO)
        for box_id in self._boxes.box_ids():
            region = self._boxes.capture_region(box_id)
            if self._boxes.is_auto(box_id) and region is not None:
                self._request_ocr(box_id, region, mode=MODE_AUTO, priority=TaskPriority.OCR_AUTO)

    # --- 박스 ---------------------------------------------------------------------

    def on_box_ocr_requested(
        self, box_id: str, region: ScreenRegion, frame: Frame | None, manual: bool
    ) -> None:
        """박스의 OCR 요청(수동: 새 캡처, 자동: 감지기 프레임)을 추론 큐에 넣는다."""
        if self._stopped or not self._boxes.exists(box_id):
            return
        if manual:
            self._request_ocr(box_id, region, mode=MODE_MANUAL, priority=TaskPriority.OCR_USER)
            return
        # 화면이 바뀌었으므로 같은 화면 재검사 횟수를 처음부터 센다.
        self._observation.on_screen_changed(box_id)
        if self._is_manual_pending(box_id):
            # 사용자가 요청한 OCR이 곧 새 화면을 캡처하므로 자동 요청으로 취소하지 않는다.
            return
        self._request_ocr(box_id, region, frame=frame, mode=MODE_AUTO, priority=TaskPriority.OCR_AUTO)

    def forget_source(self, source_id: str) -> None:
        """출처의 진행 중인 요청과 예약, 요청 기록을 지운다 (표시는 그대로)."""
        self._controller.cancel(source_id)
        self._observation.cancel(source_id)
        self._ocr_modes.pop(source_id, None)
        self._translating.pop(source_id, None)

    # --- 요청 ---------------------------------------------------------------------

    def _request_ocr(
        self,
        source_id: str,
        region: ScreenRegion | None,
        *,
        mode: str,
        priority: TaskPriority,
        frame: Frame | None = None,
    ) -> int | None:
        """출처의 OCR을 요청한다. 표시 중인 텍스트 자리를 함께 넘겨 OCR 누락 판단에 쓴다."""
        is_full_screen = source_id == FULLSCREEN_SOURCE_ID
        # 박스 안의 텍스트는 그 박스가 맡으므로 전체 화면 인식에서는 제외한다.
        exclude = self._boxes.all_capture_regions() if is_full_screen else ()
        # 화면 지문 격자는 그 출처의 자동 감지기와 같아야 수동 OCR 화면을 기준으로 기록할 수 있다.
        grid = (
            (FULLSCREEN_DETECT_GRID_WIDTH, FULLSCREEN_DETECT_GRID_HEIGHT)
            if is_full_screen
            else (AUTO_DETECT_GRID_WIDTH, AUTO_DETECT_GRID_HEIGHT)
        )
        request_id = self._controller.request_ocr(
            source_id,
            region,
            frame=frame,
            exclude=exclude,
            probe_areas=self._display.item_bounds(source_id),
            fingerprint_grid=grid,
            priority=priority,
        )
        if request_id is None:
            return None
        self._ocr_modes[source_id] = (request_id, mode)
        if not is_full_screen and self._boxes.exists(source_id):
            self._boxes.mark_pending(source_id, request_id)
        return request_id

    def _request_translations(self, source_id: str) -> None:
        """출처에서 안정됐지만 아직 번역하지 않은 원문의 번역을 요청한다 (중복 요청 방지)."""
        texts = self._display.texts_to_translate(source_id)
        if not texts:
            return
        pending = self._translating.get(source_id)
        if (
            pending is not None
            and self._controller.is_pending(source_id, KIND_TRANSLATE)
            and pending.issuperset(texts)
        ):
            # 같은 원문의 번역이 이미 진행 중이다: 다시 요청하면 진행 중인 작업이 취소된다.
            return
        if self._controller.request_translation(source_id, texts) is not None:
            self._translating[source_id] = frozenset(texts)

    def _is_manual_pending(self, source_id: str) -> bool:
        """사용자가 요청한 OCR이 아직 결과를 기다리는 중인지 여부."""
        entry = self._ocr_modes.get(source_id)
        return (
            entry is not None
            and entry[1] != MODE_AUTO
            and self._controller.is_pending(source_id, KIND_OCR)
        )

    def _is_tracked(self, source_id: str) -> bool:
        """화면 변화가 없어도 후속 OCR·재검사를 하는 출처인지 (자동 감지 박스, 전체 화면 추적)."""
        if source_id == FULLSCREEN_SOURCE_ID:
            return self._full_screen_on and self._fullscreen_detector is not None
        return self._boxes.exists(source_id) and self._boxes.is_auto(source_id)

    def _is_source_alive(self, source_id: str) -> bool:
        """결과를 표시할 출처가 아직 있는지 (박스가 지워졌거나 전체 화면이 꺼졌으면 False)."""
        if source_id == FULLSCREEN_SOURCE_ID:
            return self._full_screen_on
        return self._boxes.exists(source_id)

    def _run_due_observations(self) -> None:
        """예약 시각이 된 출처를 다시 캡처해 OCR한다 (후속 OCR, 재검사)."""
        if self._stopped:
            return
        for source_id in self._observation.pop_due():
            if not self._is_tracked(source_id):
                # 그사이 자동 감지가 꺼졌거나 박스가 지워졌다.
                self._observation.cancel(source_id)
                continue
            if self._controller.is_pending(source_id, KIND_OCR):
                # 이미 진행 중인 OCR의 결과가 다음 예약을 다시 정한다.
                continue
            if source_id == FULLSCREEN_SOURCE_ID:
                region = None
            else:
                region = self._boxes.capture_region(source_id)
                if region is None:
                    continue
            logger.debug("%s: 예약된 OCR 재확인", source_id)
            self._request_ocr(source_id, region, mode=MODE_AUTO, priority=TaskPriority.OCR_RECHECK)

    # --- 전체 화면 번역 -----------------------------------------------------------

    @contextmanager
    def quiet_box_requests(self) -> Iterator[None]:
        """이 블록 안에서 새로 생긴 박스의 수동 요청은 "텍스트 없음"을 따로 알리지 않게 한다.

        전체 화면을 켜면서 일어나는 박스 요청(편집을 끌 때의 재번역 포함)은 전체 화면 번역에 딸린
        요청이라, 박스가 비어 있어도 알리지 않는다. 전체 화면 자신의 요청은 그대로 둔다.
        """
        # 블록에 들어가기 전의 요청 기록 (블록 안에서 새로 생기거나 바뀐 요청만 고른다)
        previous = dict(self._ocr_modes)
        try:
            yield
        finally:
            for source_id, entry in list(self._ocr_modes.items()):
                if source_id == FULLSCREEN_SOURCE_ID or previous.get(source_id) == entry:
                    continue
                if entry[1] == MODE_MANUAL:
                    self._ocr_modes[source_id] = (entry[0], MODE_QUIET)

    def start_full_screen(self) -> None:
        """전체 화면 번역을 켠다: 지금 화면을 인식하고, 결과가 오면 위치 추적을 시작한다.

        박스 안의 텍스트는 그 박스가 맡으므로 전체 화면 인식에서는 제외한다. 아직 번역
        결과가 없는 박스는 함께 인식해, 화면 전체가 빠짐없이 번역되게 한다.
        """
        self._full_screen_on = True
        self.full_screen_changed.emit(True)
        logger.info("전체 화면 번역 켜기")
        self._request_ocr(FULLSCREEN_SOURCE_ID, None, mode=MODE_MANUAL, priority=TaskPriority.OCR_USER)
        for box_id in self._boxes.box_ids():
            if self._display.has_items(box_id) or self._controller.is_pending(box_id):
                continue
            region = self._boxes.capture_region(box_id)
            if region is not None:
                self._request_ocr(box_id, region, mode=MODE_QUIET, priority=TaskPriority.OCR_USER)

    def stop_full_screen(self) -> None:
        """전체 화면 번역을 끈다: 추적을 멈추고 요청·예약·표시를 지운다."""
        self._full_screen_on = False
        self.full_screen_changed.emit(False)
        logger.info("전체 화면 번역 끄기")
        if self._fullscreen_detector is not None:
            self._fullscreen_detector.stop()
            # 다음에 켤 때 이전 화면 기록과 비교하지 않도록 지운다.
            self._fullscreen_detector.reset()
        self.forget_source(FULLSCREEN_SOURCE_ID)
        self._display.remove_source(FULLSCREEN_SOURCE_ID)

    def pause_tracking_for_selection(self) -> None:
        """영역 선택이 시작되면 전체 화면 감지를 멈춘다 (선택 화면을 화면 변화로 OCR하지 않게)."""
        if self._fullscreen_detector is not None:
            self._fullscreen_detector.stop()
        self._selection_active = True

    def resume_tracking_after_selection(self) -> None:
        """영역 선택이 끝나면(선택·취소) 전체 화면 감지를 다시 켠다."""
        self._selection_active = False
        self._resume_fullscreen_tracking()

    def _resume_fullscreen_tracking(self) -> None:
        """전체 화면 번역이 켜져 있으면 전체 화면 감지기를 (다시) 켠다.

        켤 때의 수동 인식이 아직 진행 중이면 기다린다 (감지기의 첫 요청이 그 인식을 취소하지
        않도록). 영역 선택 중에는 켜지 않는다.
        """
        detector = self._fullscreen_detector
        if (
            detector is None
            or not self._full_screen_on
            or self._stopped
            or self._selection_active
            or detector.is_running
            or self._is_manual_pending(FULLSCREEN_SOURCE_ID)
        ):
            return
        detector.start()

    def _on_fullscreen_frame(self, frame: Frame) -> None:
        """전체 화면 감지기가 화면 변화를 알리면 그 프레임으로 OCR한다 (원문 위치 갱신)."""
        if not self._full_screen_on or self._stopped:
            return
        self._observation.on_screen_changed(FULLSCREEN_SOURCE_ID)
        if self._is_manual_pending(FULLSCREEN_SOURCE_ID):
            return
        self._request_ocr(
            FULLSCREEN_SOURCE_ID,
            frame.region,
            frame=frame,
            mode=MODE_AUTO,
            priority=TaskPriority.OCR_AUTO,
        )

    def _on_fullscreen_detector_stopped(self, message: str) -> None:
        """전체 화면 감지기가 연속 캡처 실패로 멈추면 전체 화면 번역을 끄고 알린다."""
        if self._stopped:
            return
        self.stop_full_screen()
        self.full_screen_error.emit(message)

    # --- 인식·번역 결과 ------------------------------------------------------------

    def _is_other_session(self, session: SessionContext | None) -> bool:
        """결과의 세션이 지금 세션과 다른지 여부 (세션을 모르는 결과는 다르다고 보지 않는다)."""
        return session is not None and session != self._controller.session

    def handle_ocr_ready(self, result: OcrResult) -> None:
        """인식 결과로 원문 위치·안정화 상태를 갱신하고, 필요한 번역과 다음 OCR을 예약한다."""
        source_id = result.source_id
        if self._is_other_session(result.session):
            logger.debug("%s: 다른 세션의 인식 결과를 버립니다.", source_id)
            return
        entry = self._ocr_modes.get(source_id)
        mode = MODE_AUTO
        if entry is not None and entry[0] == result.request_id:
            mode = entry[1]
            del self._ocr_modes[source_id]
        if self._stopped or not self._is_source_alive(source_id):
            logger.debug("없어진 출처 %s의 인식 결과를 버립니다.", source_id)
            return
        manual = mode != MODE_AUTO
        outcome = self._display.show_ocr(result, force_stable=manual)

        if source_id == FULLSCREEN_SOURCE_ID:
            if manual and self._fullscreen_detector is not None:
                # 자동 감지가 같은 화면을 다시 OCR하지 않도록 이 화면의 지문을 기록한다 (추가 캡처 없음).
                self._fullscreen_detector.mark_fingerprint(result.fingerprint, result.capture_region)
            self._resume_fullscreen_tracking()
        else:
            self._boxes.on_ocr(source_id, result, manual)

        if not outcome.accepted:
            # 이미 더 새로운 화면의 결과를 반영했다.
            return
        self._request_translations(source_id)
        if mode == MODE_MANUAL and not result.units and outcome.ocr_misses == 0:
            self.no_text_found.emit()
        if self._is_tracked(source_id):
            self._observation.after_result(
                source_id,
                unstable=outcome.unstable,
                # 결과가 비었거나 OCR이 텍스트를 놓쳤으면 화면이 그대로여도 다시 확인한다.
                needs_recheck=outcome.recognized == 0 or outcome.ocr_misses > 0,
            )

    def handle_translations_ready(self, outcome: TranslationOutcome) -> None:
        """번역 결과를 표시에 반영하고, 그사이 안정된 다른 원문이 있으면 이어서 번역한다."""
        source_id = outcome.source_id
        if self._is_other_session(outcome.session):
            logger.debug("%s: 다른 세션의 번역 결과를 버립니다.", source_id)
            return
        self._translating.pop(source_id, None)
        if self._stopped or not self._is_source_alive(source_id):
            return
        self._display.show_translations(source_id, outcome.translations)
        remaining = self._display.texts_to_translate(source_id)
        if remaining and set(remaining).issubset(outcome.translations):
            # 방금 번역한 원문이 반영되지 않았다: 다시 요청하면 끝없이 반복되므로 멈춘다.
            logger.warning("%s: 번역 결과를 반영하지 못한 원문이 있습니다.", source_id)
            return
        self._request_translations(source_id)

    def handle_failed(self, source_id: str, message: str) -> None:
        """인식·번역 실패를 해당 박스에 기록하고 알린다."""
        entry = self._ocr_modes.get(source_id)
        if entry is not None and not self._controller.is_pending(source_id, KIND_OCR):
            del self._ocr_modes[source_id]
        if self._stopped or not self._is_source_alive(source_id):
            return
        if source_id == FULLSCREEN_SOURCE_ID:
            # 처음 인식이 실패해도 추적은 시작한다 (화면이 바뀌면 다시 인식).
            self._resume_fullscreen_tracking()
        else:
            self._boxes.on_failed(source_id, message)
        self.source_failed.emit(source_id, message)
