"""번역 흐름(`TranslationWorkflow`) 단위 테스트: 가짜 컨트롤러·박스·감지기로 위젯과 스레드 없이 검증한다.

요청 방식(수동·자동·조용히), 안정화 뒤 번역 요청과 중복 방지, 없어진 출처 결과 무시, 전체 화면
켜기·끄기·추적, 재검사 예약, 종료, 세션 전환을 확인한다. 화면과 합친 전체 흐름은
`test_coordinator.py`가 맡는다.
"""

import itertools
from dataclasses import dataclass

import pytest
from PySide6.QtCore import QObject, Signal

from translator.application.change_detection import Fingerprint
from translator.application.controller import KIND_OCR, KIND_TRANSLATE, TranslationOutcome
from translator.application.frames import Frame
from translator.application.pipeline import OcrResult
from translator.application.recheck import ObservationScheduler
from translator.application.scheduler import TaskPriority
from translator.application.text_display import TextDisplayModel
from translator.application.workflow import FULLSCREEN_SOURCE_ID, TranslationWorkflow
from translator.core.content import ContentIdentity, IdentificationMethod
from translator.core.models import CapturedImage, ScreenRegion, TextUnit
from translator.core.session import new_session

# 박스 캡처 영역과 화면 전체 영역
BOX_REGION = ScreenRegion(100, 100, 400, 200)
SCREEN = ScreenRegion(0, 0, 1920, 1080)
# 테스트용 작은 화면 지문
FINGERPRINT = Fingerprint(1, 1, b"\x00")
# 프레임 번호와 캡처 시각을 차례로 발급한다 (표시 모델은 오래된 캡처의 결과를 버린다).
_frame_numbers = itertools.count(1)
_capture_times = itertools.count(1)


@dataclass
class OcrRequest:
    """가짜 컨트롤러가 받은 OCR 요청."""

    request_id: int
    source_id: str
    region: ScreenRegion | None
    frame: Frame | None
    exclude: tuple[ScreenRegion, ...]
    probe_areas: tuple[ScreenRegion, ...]
    priority: TaskPriority


class FakeController(QObject):
    """요청을 기록하고 결과는 테스트가 시그널로 직접 보내는 가짜 컨트롤러."""

    ocr_ready = Signal(object)
    translations_ready = Signal(object)
    failed = Signal(str, str)

    def __init__(self) -> None:
        """기록 목록과 세션을 준비한다."""
        super().__init__()
        # 지금 세션 (결과의 세션과 비교한다)
        self.session = new_session("game-a", "s1")
        # 받은 OCR 요청과 번역 요청 (출처, 원문들)
        self.ocr_requests: list[OcrRequest] = []
        self.translation_requests: list[tuple[str, tuple[str, ...]]] = []
        # 결과를 기다리는 (출처, 종류)
        self.pending: set[tuple[str, str]] = set()
        # 취소된 출처 목록과 시작한 세션 목록
        self.cancelled: list[str] = []
        self.begun_sessions: list = []
        self._next_id = 0

    def request_ocr(self, source_id, region, *, frame=None, exclude=(), probe_areas=(), fingerprint_grid=None,
                    priority=TaskPriority.OCR_USER) -> int:
        """요청을 기록하고 요청 번호를 돌려준다."""
        self._next_id += 1
        self.ocr_requests.append(
            OcrRequest(self._next_id, source_id, region, frame, tuple(exclude), tuple(probe_areas), priority)
        )
        self.pending.add((source_id, KIND_OCR))
        return self._next_id

    def request_translation(self, source_id, texts) -> int:
        """번역 요청을 기록하고 요청 번호를 돌려준다."""
        self._next_id += 1
        self.translation_requests.append((source_id, tuple(texts)))
        self.pending.add((source_id, KIND_TRANSLATE))
        return self._next_id

    def is_pending(self, source_id: str, kind: str | None = None) -> bool:
        """출처의 요청(종류를 주면 그 종류만)이 결과를 기다리는 중인지."""
        kinds = (kind,) if kind else (KIND_OCR, KIND_TRANSLATE)
        return any((source_id, k) in self.pending for k in kinds)

    def cancel(self, source_id: str) -> None:
        """출처의 요청을 취소한다."""
        self.cancelled.append(source_id)
        self.pending = {p for p in self.pending if p[0] != source_id}

    def begin_session(self, session) -> None:
        """새 세션을 시작한다 (진행 중 요청은 모두 취소)."""
        self.begun_sessions.append(session)
        self.session = session
        self.pending.clear()

    def finish(self, source_id: str, kind: str) -> None:
        """출처의 요청이 끝났음을 기록한다 (결과를 보내기 전에 부른다)."""
        self.pending.discard((source_id, kind))

    def shutdown(self) -> bool:
        """종료한다 (테스트에서는 쓰지 않는다)."""
        return True


class FakeDisplay:
    """실제 `TextDisplayModel`을 `DisplayPort` 모양으로 감싼 표시 상태."""

    def __init__(self) -> None:
        """실제 표시 모델을 만든다."""
        self.model = TextDisplayModel()

    def show_ocr(self, result, force_stable=False):
        """인식 결과를 모델에 반영한다."""
        return self.model.apply_ocr(result, force_stable)

    def show_translations(self, source_id, translations) -> None:
        """번역 결과를 모델에 반영한다."""
        self.model.apply_translations(source_id, translations)

    def texts_to_translate(self, source_id):
        """안정됐지만 번역하지 않은 원문들."""
        return self.model.texts_to_translate(source_id)

    def item_bounds(self, source_id):
        """표시 중인 텍스트 자리."""
        return tuple(item.bounds for item in self.model.items_for(source_id))

    def has_items(self, source_id) -> bool:
        """표시 중인 텍스트가 있는지."""
        return self.model.has_items(source_id)

    def remove_source(self, source_id) -> None:
        """출처의 표시를 지운다."""
        self.model.remove_source(source_id)

    def clear(self) -> None:
        """모든 표시를 지운다."""
        self.model.clear()


class FakeBoxPort:
    """박스 몇 개를 흉내 내고, 흐름이 알려 준 것을 기록하는 가짜 `BoxPort`."""

    def __init__(self) -> None:
        """박스가 없는 상태로 시작한다."""
        # 박스 ID → (자동 감지 여부, 캡처 영역)
        self.boxes: dict[str, tuple[bool, ScreenRegion | None]] = {}
        # 알려 준 내용
        self.pending: list[tuple[str, int]] = []
        self.ocr_calls: list[tuple[str, bool, tuple[str, ...]]] = []
        self.failures: list[tuple[str, str]] = []
        # 박스 목록을 바꾸라고 알려 준 콘텐츠 ID들 (세션이 바뀐 순서)
        self.switched_contents: list[str] = []

    def switch_content(self, content_id: str) -> None:
        """박스 목록을 바꾸라는 알림을 기록한다."""
        self.switched_contents.append(content_id)

    def add(self, box_id: str, auto: bool = False, region: ScreenRegion | None = BOX_REGION) -> None:
        """박스를 추가한다."""
        self.boxes[box_id] = (auto, region)

    def box_ids(self):
        """모든 박스의 ID."""
        return tuple(self.boxes)

    def exists(self, box_id: str) -> bool:
        """박스가 있는지."""
        return box_id in self.boxes

    def is_auto(self, box_id: str) -> bool:
        """자동 감지 중인지."""
        return self.boxes[box_id][0] if box_id in self.boxes else False

    def capture_region(self, box_id: str):
        """캡처 영역."""
        return self.boxes[box_id][1] if box_id in self.boxes else None

    def all_capture_regions(self):
        """모든 박스의 캡처 영역."""
        return tuple(region for _, region in self.boxes.values() if region is not None)

    def mark_pending(self, box_id: str, request_id: int) -> None:
        """요청이 들어갔음을 기록한다."""
        self.pending.append((box_id, request_id))

    def on_ocr(self, box_id: str, result, manual: bool) -> None:
        """인식 결과를 기록한다."""
        self.ocr_calls.append((box_id, manual, tuple(unit.text for unit in result.units)))

    def on_failed(self, box_id: str, message: str) -> None:
        """실패를 기록한다."""
        self.failures.append((box_id, message))


class FakeDetector(QObject):
    """전체 화면 감지기를 흉내 내는 가짜 (시작·정지 호출을 기록한다)."""

    frame_ready = Signal(object)
    stopped_by_error = Signal(str)

    def __init__(self) -> None:
        """정지 상태로 시작한다."""
        super().__init__()
        self.is_running = False
        self.calls: list = []

    def start(self) -> None:
        """감지를 시작한다."""
        self.is_running = True
        self.calls.append("start")

    def stop(self) -> None:
        """감지를 멈춘다."""
        self.is_running = False
        self.calls.append("stop")

    def reset(self) -> None:
        """이전 화면 기록을 지운다."""
        self.calls.append("reset")

    def shutdown(self) -> None:
        """감지기를 끝낸다."""
        self.is_running = False
        self.calls.append("shutdown")

    def mark_fingerprint(self, fingerprint, region) -> None:
        """기준 화면을 기록한다."""
        self.calls.append(("mark", region))


class Setup:
    """테스트 한 건에 필요한 흐름과 가짜 의존성 묶음, 알림 수집 목록."""

    def __init__(self, with_detector: bool = False) -> None:
        """가짜 의존성으로 흐름을 만들고 알림 시그널을 수집한다."""
        self.controller = FakeController()
        self.display = FakeDisplay()
        self.boxes = FakeBoxPort()
        self.detector = FakeDetector() if with_detector else None
        self.workflow = TranslationWorkflow(
            self.controller, self.display, self.boxes, fullscreen_detector=self.detector
        )
        # 수집한 알림
        self.no_text: list[bool] = []
        self.failures: list[tuple[str, str]] = []
        self.full_screen_states: list[bool] = []
        self.full_screen_errors: list[str] = []
        self.workflow.no_text_found.connect(lambda: self.no_text.append(True))
        self.workflow.source_failed.connect(lambda s, m: self.failures.append((s, m)))
        self.workflow.full_screen_changed.connect(self.full_screen_states.append)
        self.workflow.full_screen_error.connect(self.full_screen_errors.append)

    def deliver(self, source_id: str, texts: list[str], request_id: int | None = None, session=None) -> None:
        """OCR 결과를 컨트롤러 시그널로 보낸다 (요청 번호를 주지 않으면 가장 최근 요청의 번호)."""
        if request_id is None:
            matching = [r for r in self.controller.ocr_requests if r.source_id == source_id]
            request_id = matching[-1].request_id if matching else 0
        self.controller.finish(source_id, KIND_OCR)
        self.controller.ocr_ready.emit(make_result(source_id, texts, request_id, session))


def make_result(source_id: str, texts: list[str], request_id: int, session=None) -> OcrResult:
    """텍스트마다 세로로 떨어진 위치를 가진 OCR 결과를 만든다."""
    units = tuple(TextUnit(text, ScreenRegion(120, 120 + 60 * i, 300, 20)) for i, text in enumerate(texts))
    return OcrResult(
        source_id=source_id,
        frame_id=next(_frame_numbers),
        captured_at=float(next(_capture_times)),
        capture_region=BOX_REGION,
        units=units,
        unit_fingerprints=(FINGERPRINT,) * len(units),
        probes=(),
        fingerprint=FINGERPRINT,
        ocr_reused=False,
        request_id=request_id,
        session=session,
    )


@pytest.fixture
def setup(qapp) -> Setup:
    """박스가 하나(box-1) 있는 흐름 (감지기 없음)."""
    prepared = Setup()
    prepared.boxes.add("box-1")
    return prepared


@pytest.fixture
def setup_with_detector(qapp) -> Setup:
    """박스가 하나(box-1) 있고 전체 화면 감지기가 있는 흐름."""
    prepared = Setup(with_detector=True)
    prepared.boxes.add("box-1")
    return prepared


# --- 수동·자동 요청 ------------------------------------------------------------------


def test_manual_request_runs_at_user_priority_and_translates_without_waiting(setup: Setup) -> None:
    """수동 요청은 사용자 우선순위로 새로 캡처하고, 안정화를 기다리지 않고 번역을 요청한다."""
    setup.workflow.on_box_ocr_requested("box-1", BOX_REGION, None, True)

    [request] = setup.controller.ocr_requests
    assert (request.source_id, request.region, request.frame) == ("box-1", BOX_REGION, None)
    assert request.priority is TaskPriority.OCR_USER
    assert setup.boxes.pending == [("box-1", request.request_id)]

    setup.deliver("box-1", ["Hello there"])

    assert setup.controller.translation_requests == [("box-1", ("Hello there",))]
    assert setup.boxes.ocr_calls == [("box-1", True, ("Hello there",))]


def test_auto_result_waits_until_text_is_stable(setup: Setup) -> None:
    """자동 요청의 결과는 같은 텍스트가 두 번 연속 인식된 뒤에 번역한다 (타자 효과 중 부분 번역 방지)."""
    setup.boxes.add("box-1", auto=True)
    frame = Frame(1, CapturedImage(BOX_REGION, 1, 1, b"\0" * 4), 1.0)

    setup.workflow.on_box_ocr_requested("box-1", BOX_REGION, frame, False)
    assert setup.controller.ocr_requests[0].frame is frame
    assert setup.controller.ocr_requests[0].priority is TaskPriority.OCR_AUTO
    setup.deliver("box-1", ["Typing in prog"])
    assert setup.controller.translation_requests == []
    assert setup.workflow.observation.due_at("box-1") is not None  # 후속 OCR이 예약된다.

    setup.workflow.on_box_ocr_requested("box-1", BOX_REGION, frame, False)
    setup.deliver("box-1", ["Typing in prog"])
    assert setup.controller.translation_requests == [("box-1", ("Typing in prog",))]


def test_auto_frame_does_not_cancel_a_pending_manual_request(setup: Setup) -> None:
    """사용자가 요청한 OCR이 진행 중이면 자동 프레임 요청은 그것을 취소하지 않는다."""
    setup.workflow.on_box_ocr_requested("box-1", BOX_REGION, None, True)

    setup.workflow.on_box_ocr_requested("box-1", BOX_REGION, None, False)

    assert [r.priority for r in setup.controller.ocr_requests] == [TaskPriority.OCR_USER]


def test_requests_for_missing_boxes_or_after_stop_are_ignored(setup: Setup) -> None:
    """없는 박스의 요청과 종료 뒤의 요청은 무시한다."""
    setup.workflow.on_box_ocr_requested("box-9", BOX_REGION, None, True)
    setup.workflow.stop()
    setup.workflow.on_box_ocr_requested("box-1", BOX_REGION, None, True)

    assert setup.controller.ocr_requests == []


def test_translation_is_not_requested_twice_while_pending(setup: Setup) -> None:
    """같은 원문의 번역이 진행 중이면 다시 요청하지 않는다 (진행 중인 작업이 취소되지 않게)."""
    setup.workflow.on_box_ocr_requested("box-1", BOX_REGION, None, True)
    setup.deliver("box-1", ["Hello"])
    setup.workflow.on_box_ocr_requested("box-1", BOX_REGION, None, True)
    setup.deliver("box-1", ["Hello"])

    assert setup.controller.translation_requests == [("box-1", ("Hello",))]


def test_translation_result_is_shown_and_remaining_texts_are_requested(setup: Setup) -> None:
    """번역 결과를 표시에 반영하고, 그사이 안정된 다른 원문이 있으면 이어서 번역을 요청한다."""
    setup.workflow.on_box_ocr_requested("box-1", BOX_REGION, None, True)
    setup.deliver("box-1", ["First"])
    setup.workflow.on_box_ocr_requested("box-1", BOX_REGION, None, True)
    setup.deliver("box-1", ["First", "Second"])  # 번역 진행 중에 새 원문이 안정됨
    setup.controller.finish("box-1", KIND_TRANSLATE)

    setup.controller.translations_ready.emit(
        TranslationOutcome("box-1", 5, {"First": ("첫째", "fake")}, setup.controller.session)
    )

    [item_first] = [i for i in setup.display.model.items_for("box-1") if i.source_text == "First"]
    assert item_first.translated_text == "첫째"
    assert setup.controller.translation_requests[-1] == ("box-1", ("Second",))


def test_results_for_removed_boxes_are_dropped(setup: Setup) -> None:
    """지워진 박스의 늦은 결과는 표시에도, 번역 요청에도 반영하지 않는다."""
    setup.workflow.on_box_ocr_requested("box-1", BOX_REGION, None, True)
    del setup.boxes.boxes["box-1"]

    setup.deliver("box-1", ["Late"])
    setup.controller.translations_ready.emit(TranslationOutcome("box-1", 1, {"x": ("y", "fake")}))

    assert setup.display.model.items() == []
    assert setup.controller.translation_requests == []
    assert setup.boxes.ocr_calls == []


# --- 알림 -----------------------------------------------------------------------------


def test_manual_empty_result_reports_no_text(setup: Setup) -> None:
    """사용자가 요청한 인식에서 텍스트가 없으면 알린다. 자동 요청은 알리지 않는다."""
    setup.workflow.on_box_ocr_requested("box-1", BOX_REGION, None, True)
    setup.deliver("box-1", [])
    setup.workflow.on_box_ocr_requested("box-1", BOX_REGION, None, False)
    setup.deliver("box-1", [])

    assert setup.no_text == [True]


def test_failure_is_recorded_on_the_box_and_reported(setup: Setup) -> None:
    """박스의 실패는 박스에 기록하고 알림으로 보낸다. 없는 박스의 실패는 알리지 않는다."""
    setup.workflow.on_box_ocr_requested("box-1", BOX_REGION, None, True)

    setup.controller.failed.emit("box-1", "OCR 실패")
    setup.controller.failed.emit("box-9", "없는 박스")

    assert setup.boxes.failures == [("box-1", "OCR 실패")]
    assert setup.failures == [("box-1", "OCR 실패")]


# --- 전체 화면 -----------------------------------------------------------------------


def test_full_screen_start_requests_screen_and_boxes_without_results(setup: Setup) -> None:
    """전체 화면을 켜면 박스를 뺀 화면 전체를 인식하고, 결과가 없는 박스도 함께 인식한다 (이미 결과가 있는 박스는 제외)."""
    setup.boxes.add("box-2", region=ScreenRegion(700, 100, 300, 100))
    setup.workflow.on_box_ocr_requested("box-1", BOX_REGION, None, True)
    setup.deliver("box-1", ["Has result"])
    setup.controller.ocr_requests.clear()

    setup.workflow.start_full_screen()

    assert setup.full_screen_states == [True]
    by_source = {r.source_id: r for r in setup.controller.ocr_requests}
    assert set(by_source) == {FULLSCREEN_SOURCE_ID, "box-2"}
    assert by_source[FULLSCREEN_SOURCE_ID].region is None
    assert by_source[FULLSCREEN_SOURCE_ID].exclude == (BOX_REGION, ScreenRegion(700, 100, 300, 100))
    assert by_source["box-2"].region == ScreenRegion(700, 100, 300, 100)


def test_full_screen_box_requests_do_not_report_no_text(setup: Setup) -> None:
    """전체 화면에 딸린 박스 요청은 박스가 비어 있어도 "텍스트 없음"을 알리지 않는다. 전체 화면 자신은 알린다."""
    setup.workflow.start_full_screen()

    setup.deliver("box-1", [])
    assert setup.no_text == []
    setup.deliver(FULLSCREEN_SOURCE_ID, [])
    assert setup.no_text == [True]


def test_quiet_block_relabels_new_manual_box_requests(setup: Setup) -> None:
    """quiet_box_requests 안에서 새로 생긴 박스의 수동 요청은 조용히 처리한다 (편집 종료 재번역 등). 블록 밖은 그대로."""
    with setup.workflow.quiet_box_requests():
        setup.workflow.on_box_ocr_requested("box-1", BOX_REGION, None, True)
    setup.deliver("box-1", [])
    assert setup.no_text == []

    setup.workflow.on_box_ocr_requested("box-1", BOX_REGION, None, True)
    setup.deliver("box-1", [])
    assert setup.no_text == [True]


def test_full_screen_stop_clears_everything_of_the_source(setup_with_detector: Setup) -> None:
    """전체 화면을 끄면 감지를 멈추고 기록을 지우며, 요청을 취소하고 표시를 지운다."""
    setup = setup_with_detector
    setup.workflow.start_full_screen()
    setup.deliver(FULLSCREEN_SOURCE_ID, ["Menu"])
    assert setup.display.has_items(FULLSCREEN_SOURCE_ID)

    setup.workflow.stop_full_screen()

    assert setup.full_screen_states == [True, False]
    assert not setup.workflow.full_screen_on
    assert setup.detector.calls[-2:] == ["stop", "reset"]
    assert FULLSCREEN_SOURCE_ID in setup.controller.cancelled
    assert not setup.display.has_items(FULLSCREEN_SOURCE_ID)


def test_tracking_starts_after_first_result_and_follows_frames(setup_with_detector: Setup) -> None:
    """첫 인식 결과가 오면 감지기를 켜고, 화면이 바뀌었다는 프레임이 오면 그 프레임으로 OCR한다 (수동 요청이 끝난 뒤에만)."""
    setup = setup_with_detector
    setup.workflow.start_full_screen()
    frame = Frame(7, CapturedImage(SCREEN, 1, 1, b"\0" * 4), 1.0)
    setup.detector.frame_ready.emit(frame)  # 수동 인식이 진행 중이므로 무시된다.
    assert len([r for r in setup.controller.ocr_requests if r.frame is not None]) == 0
    assert "start" not in setup.detector.calls

    setup.deliver(FULLSCREEN_SOURCE_ID, ["Menu"])
    assert ("mark", BOX_REGION) in setup.detector.calls  # 수동 인식 화면을 기준으로 기록
    assert setup.detector.calls.count("start") == 1

    setup.detector.frame_ready.emit(frame)
    [auto_request] = [r for r in setup.controller.ocr_requests if r.frame is not None]
    assert (auto_request.source_id, auto_request.region) == (FULLSCREEN_SOURCE_ID, SCREEN)
    assert auto_request.priority is TaskPriority.OCR_AUTO


def test_selection_pauses_and_resumes_tracking(setup_with_detector: Setup) -> None:
    """영역 선택 중에는 전체 화면 감지를 멈추고, 선택이 끝나면 (전체 화면이 켜져 있을 때) 다시 켠다."""
    setup = setup_with_detector
    setup.workflow.start_full_screen()
    setup.deliver(FULLSCREEN_SOURCE_ID, ["Menu"])

    setup.workflow.pause_tracking_for_selection()
    assert not setup.detector.is_running
    setup.workflow.resume_tracking_after_selection()

    assert setup.detector.is_running


def test_detector_error_turns_full_screen_off_and_reports(setup_with_detector: Setup) -> None:
    """감지기가 연속 실패로 멈추면 전체 화면 번역을 끄고 메시지를 알린다."""
    setup = setup_with_detector
    setup.workflow.start_full_screen()

    setup.detector.stopped_by_error.emit("캡처 실패")

    assert not setup.workflow.full_screen_on
    assert setup.full_screen_states == [True, False]
    assert setup.full_screen_errors == ["캡처 실패"]


def test_full_screen_failure_still_starts_tracking(setup_with_detector: Setup) -> None:
    """처음 전체 화면 인식이 실패해도 추적은 시작한다 (화면이 바뀌면 다시 인식)."""
    setup = setup_with_detector
    setup.workflow.start_full_screen()
    setup.controller.finish(FULLSCREEN_SOURCE_ID, KIND_OCR)

    setup.controller.failed.emit(FULLSCREEN_SOURCE_ID, "OCR 실패")

    assert setup.detector.is_running
    assert setup.failures == [(FULLSCREEN_SOURCE_ID, "OCR 실패")]


# --- 재검사 예약 ------------------------------------------------------------------------


def test_due_observation_rechecks_only_tracked_sources(setup: Setup, qtbot) -> None:
    """예약 시각이 되면 자동 감지 중인 박스만 낮은 우선순위로 다시 OCR한다. 더 이상 추적하지 않는 출처는 예약을 지운다."""
    setup.boxes.add("auto-box", auto=True)
    scheduler = ObservationScheduler(followup_seconds=0.0)
    setup.workflow.use_observation(scheduler, tick_ms=10)
    scheduler.after_result("auto-box", unstable=True, needs_recheck=False)
    scheduler.after_result("box-1", unstable=True, needs_recheck=False)  # 수동 박스: 추적 대상이 아님

    setup.workflow.start()
    qtbot.waitUntil(lambda: bool(setup.controller.ocr_requests), timeout=2000)
    setup.workflow.stop()

    assert [(r.source_id, r.priority) for r in setup.controller.ocr_requests] == [
        ("auto-box", TaskPriority.OCR_RECHECK)
    ]
    assert scheduler.due_at("box-1") is None


# --- 종료 -------------------------------------------------------------------------------


def test_stop_halts_timer_detector_and_late_results(setup_with_detector: Setup) -> None:
    """종료하면 예약 확인과 감지를 멈추고, 늦게 도착한 결과로 새 작업을 만들지 않는다."""
    setup = setup_with_detector
    setup.workflow.start()
    setup.workflow.on_box_ocr_requested("box-1", BOX_REGION, None, True)
    request_count = len(setup.controller.ocr_requests)

    setup.workflow.stop()
    setup.deliver("box-1", ["Late"])

    assert not setup.workflow.is_observing
    assert "shutdown" in setup.detector.calls
    assert setup.controller.translation_requests == []
    assert len(setup.controller.ocr_requests) == request_count


# --- 세션 -------------------------------------------------------------------------------


def test_session_change_clears_work_and_display_and_restarts_tracked_sources(setup_with_detector: Setup) -> None:
    """세션이 바뀌면 이전 작업·예약·표시를 정리하고, 자동으로 추적하는 출처(자동 박스, 전체 화면)만 새 세션에서 다시 인식한다."""
    setup = setup_with_detector
    setup.boxes.add("auto-box", auto=True)
    setup.workflow.on_box_ocr_requested("box-1", BOX_REGION, None, True)
    setup.deliver("box-1", ["Old session text"])
    setup.workflow.start_full_screen()
    setup.workflow.observation.after_result("auto-box", unstable=True, needs_recheck=False)
    setup.controller.ocr_requests.clear()
    new_context = new_session("game-b", "s2")

    setup.workflow.change_session(new_context)

    assert setup.controller.begun_sessions == [new_context]
    assert setup.workflow.session == new_context
    assert setup.display.model.items() == []
    assert setup.workflow.observation.due_at("auto-box") is None
    assert {r.source_id for r in setup.controller.ocr_requests} == {FULLSCREEN_SOURCE_ID, "auto-box"}
    assert all(r.priority is TaskPriority.OCR_AUTO for r in setup.controller.ocr_requests)


def test_results_of_the_old_session_are_dropped(setup: Setup) -> None:
    """이전 세션에서 시작한 인식·번역 결과가 늦게 도착해도 표시하거나 번역을 요청하지 않는다 (회귀)."""
    old_session = setup.controller.session
    setup.workflow.on_box_ocr_requested("box-1", BOX_REGION, None, True)
    old_request_id = setup.controller.ocr_requests[0].request_id
    setup.workflow.change_session(new_session("game-b", "s2"))

    setup.controller.ocr_ready.emit(make_result("box-1", ["Old text"], old_request_id, old_session))
    setup.controller.translations_ready.emit(
        TranslationOutcome("box-1", 9, {"Old text": ("옛 번역", "fake")}, old_session)
    )

    assert setup.display.model.items() == []
    assert setup.controller.translation_requests == []
    assert setup.boxes.ocr_calls == []


def test_results_of_the_current_session_are_shown(setup: Setup) -> None:
    """현재 세션을 단 결과는 그대로 처리한다."""
    setup.workflow.on_box_ocr_requested("box-1", BOX_REGION, None, True)
    request_id = setup.controller.ocr_requests[0].request_id

    setup.controller.ocr_ready.emit(make_result("box-1", ["Now"], request_id, setup.controller.session))

    assert setup.controller.translation_requests == [("box-1", ("Now",))]


def test_same_session_change_is_a_no_op(setup: Setup) -> None:
    """같은 세션으로 바꾸려 하면 아무것도 하지 않는다 (표시와 요청 유지)."""
    setup.workflow.on_box_ocr_requested("box-1", BOX_REGION, None, True)
    setup.deliver("box-1", ["Keep"])

    setup.workflow.change_session(setup.controller.session)

    assert setup.controller.begun_sessions == []
    assert setup.display.has_items("box-1")


def test_change_content_starts_a_fresh_session_for_that_content(setup: Setup) -> None:
    """콘텐츠를 바꾸면 그 콘텐츠의 새 세션이 시작된다. 같은 콘텐츠로 다시 바꿔도 새 세션이다."""
    first = setup.workflow.change_content("game-b")
    second = setup.workflow.change_content("game-b")

    assert first.content_id == second.content_id == "game-b"
    assert first != second
    assert setup.controller.begun_sessions == [first, second]


# --- 콘텐츠 감시기 연결 -------------------------------------------------------------------


class FakePolicy:
    """감시기의 바뀜 판단 정책을 흉내 내는 가짜 (재설정 호출을 기록한다)."""

    def __init__(self) -> None:
        """호출 기록을 비운 채 시작한다."""
        # `reset`에 넘어온 콘텐츠 ID들
        self.resets: list[str] = []

    def reset(self, content_id: str) -> None:
        """재설정 호출을 기록한다."""
        self.resets.append(content_id)


class FakeContentMonitor(QObject):
    """콘텐츠 감시기를 흉내 내는 가짜 (시작·종료 호출을 기록하고, 알림은 테스트가 직접 보낸다)."""

    content_changed = Signal(object)
    stopped_by_error = Signal(str)

    def __init__(self) -> None:
        """호출 기록과 가짜 정책을 준비한다."""
        super().__init__()
        self.calls: list[str] = []
        self.policy = FakePolicy()

    def start(self) -> None:
        """감시를 시작한다."""
        self.calls.append("start")

    def shutdown(self) -> None:
        """감시를 끝낸다."""
        self.calls.append("shutdown")


def make_monitored_setup() -> tuple[Setup, FakeContentMonitor]:
    """가짜 콘텐츠 감시기가 연결된 흐름을 만든다."""
    setup = Setup()
    monitor = FakeContentMonitor()
    setup.workflow = TranslationWorkflow(
        setup.controller, setup.display, setup.boxes, content_monitor=monitor
    )
    return setup, monitor


def test_identified_content_starts_a_new_session_for_it(qapp) -> None:
    """감시기가 새 콘텐츠를 확정해 알리면 그 콘텐츠의 새 세션이 시작되고 이전 표시는 지워진다."""
    setup, monitor = make_monitored_setup()
    setup.boxes.add("box-1")
    setup.workflow.on_box_ocr_requested("box-1", BOX_REGION, None, True)
    setup.deliver("box-1", ["Old content"])

    monitor.content_changed.emit(ContentIdentity("app:game.exe", "game", IdentificationMethod.PROCESS_NAME))

    assert setup.workflow.session.content_id == "app:game.exe"
    assert len(setup.controller.begun_sessions) == 1
    assert setup.display.model.items() == []


def test_content_monitor_starts_and_shuts_down_with_the_workflow(qapp) -> None:
    """흐름을 시작하면 감시기도 시작하고, 종료하면 감시기도 끝낸다."""
    setup, monitor = make_monitored_setup()

    setup.workflow.start()
    setup.workflow.stop()

    assert monitor.calls == ["start", "shutdown"]


def test_content_identified_after_stop_is_ignored(qapp) -> None:
    """종료가 시작된 뒤 늦게 도착한 콘텐츠 변경 알림은 새 세션을 만들지 않는다."""
    setup, monitor = make_monitored_setup()
    setup.workflow.stop()

    monitor.content_changed.emit(ContentIdentity("app:game.exe", "game", IdentificationMethod.PROCESS_NAME))

    assert setup.controller.begun_sessions == []


def test_session_change_from_elsewhere_resyncs_the_monitor_policy(qapp) -> None:
    """다른 경로로 세션이 바뀌면 감시기 정책의 현재 콘텐츠도 맞춘다 (같은 콘텐츠를 또 바뀜으로 알리지 않게)."""
    setup, monitor = make_monitored_setup()

    setup.workflow.change_content("game-b")

    assert monitor.policy.resets == ["game-b"]


def test_content_detection_error_is_forwarded_for_the_status_window(qapp) -> None:
    """감시기가 연속 실패로 멈추면 그 메시지를 상태 창이 받을 수 있는 시그널로 전달한다."""
    setup, monitor = make_monitored_setup()
    messages: list[str] = []
    setup.workflow.content_detection_error.connect(messages.append)

    monitor.stopped_by_error.emit("식별 중지")

    assert messages == ["식별 중지"]


# --- 콘텐츠별 박스 교체 -------------------------------------------------------------------


def test_boxes_are_switched_before_tracked_sources_restart(setup: Setup) -> None:
    """세션이 바뀌면 박스를 새 콘텐츠의 것으로 먼저 바꾼 뒤 다시 인식한다 (이전 콘텐츠의 박스를 인식하면 안 된다)."""
    setup.boxes.add("old-auto", auto=True)
    profiles = {"game-b": ["new-auto"]}

    def switch(content_id: str) -> None:
        """박스 목록을 새 콘텐츠의 것으로 바꾸는 가짜 동작."""
        setup.boxes.switched_contents.append(content_id)
        setup.boxes.boxes.clear()
        for box_id in profiles[content_id]:
            setup.boxes.add(box_id, auto=True)

    setup.boxes.switch_content = switch
    setup.controller.ocr_requests.clear()

    setup.workflow.change_session(new_session("game-b", "s2"))

    assert setup.boxes.switched_contents == ["game-b"]
    assert {r.source_id for r in setup.controller.ocr_requests} == {"new-auto"}


def test_content_identified_switches_boxes_to_that_content(qapp) -> None:
    """감시기가 새 콘텐츠를 확정하면 그 콘텐츠의 박스로 바뀐다."""
    setup, monitor = make_monitored_setup()

    monitor.content_changed.emit(ContentIdentity("app:game.exe", "game", IdentificationMethod.PROCESS_NAME))

    assert setup.boxes.switched_contents == ["app:game.exe"]


def test_same_session_does_not_switch_boxes(setup: Setup) -> None:
    """세션이 그대로면 박스도 바꾸지 않는다."""
    setup.workflow.change_session(setup.controller.session)

    assert setup.boxes.switched_contents == []
