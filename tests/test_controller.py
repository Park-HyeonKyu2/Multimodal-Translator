"""인식·번역 작업 큐 테스트: 출처별 요청 관리, 오래된 결과 무시, 우선순위, 동시 실행 수, 오류 전달.

실제 OCR·번역 모델 없이 가짜 파이프라인(Mock 작업)으로 동시성을 검증한다.
"""

import threading
import time

import pytest
from PySide6.QtCore import QTimer

from translator.application.change_detection import Fingerprint
from translator.application.controller import (
    KIND_OCR,
    KIND_TRANSLATE,
    TranslationController,
    TranslationOutcome,
)
from translator.application.frames import Frame
from translator.application.pipeline import OcrResult, PipelineRequest
from translator.application.scheduler import InferenceScheduler, TaskPriority
from translator.core.cancellation import CancellationToken
from translator.core.errors import UNEXPECTED_ERROR_MESSAGE, CaptureError
from translator.core.models import CapturedImage, ScreenRegion, TextUnit
from translator.core.session import new_session

REGION = ScreenRegion(0, 0, 10, 10)
# 테스트용 작은 지문
FINGERPRINT = Fingerprint(1, 1, b"\x00")


def ocr_result(source_id: str, text: str | None) -> OcrResult:
    """텍스트 하나(또는 없음)짜리 인식 결과를 만든다."""
    units = (TextUnit(text, REGION),) if text else ()
    return OcrResult(
        source_id=source_id,
        frame_id=1,
        captured_at=0.0,
        capture_region=REGION,
        units=units,
        unit_fingerprints=(FINGERPRINT,) * len(units),
        probes=(),
        fingerprint=FINGERPRINT,
        ocr_reused=False,
    )


class FakePipeline:
    """인식·번역을 흉내 내는 가짜 파이프라인. 실행 순서, 스레드, 동시 실행 수를 기록한다.

    `hooks[출처 ID]`가 있으면 인식할 때 그 함수를 먼저 실행한다 (막기, 예외 등).
    인식 결과 텍스트는 `texts[출처 ID]`(없으면 출처 ID)이다.
    """

    def __init__(self, work_seconds: float = 0.0) -> None:
        """기록 목록과 작업 하나에 걸리는 시간(초)을 준비한다."""
        self._lock = threading.Lock()
        self.work_seconds = work_seconds
        self.hooks: dict[str, object] = {}
        self.texts: dict[str, str | None] = {}
        self.order: list[str] = []
        self.thread_names: list[str] = []
        self.requests: list[PipelineRequest] = []
        # 지금 실행 중인 작업 수와 그 최댓값 (GPU 동시 점유 검증용)
        self.running = 0
        self.max_running = 0

    def _enter(self, label: str) -> None:
        """작업 시작을 기록한다."""
        with self._lock:
            self.order.append(label)
            self.thread_names.append(threading.current_thread().name)
            self.running += 1
            self.max_running = max(self.max_running, self.running)

    def _leave(self) -> None:
        """작업 끝을 기록한다."""
        with self._lock:
            self.running -= 1

    def recognize(self, request: PipelineRequest, cancellation: CancellationToken) -> OcrResult:
        """출처의 훅을 실행한 뒤 텍스트 하나짜리 결과를 돌려준다."""
        self._enter(request.source_id)
        try:
            self.requests.append(request)
            hook = self.hooks.get(request.source_id)
            if hook is not None:
                hook()
            time.sleep(self.work_seconds)
            return ocr_result(request.source_id, self.texts.get(request.source_id, request.source_id))
        finally:
            self._leave()

    def translate_texts(self, texts, cancellation: CancellationToken) -> dict[str, tuple[str, str]]:
        """원문마다 "번역:원문"을 돌려준다."""
        self._enter("T:" + ",".join(texts))
        try:
            time.sleep(self.work_seconds)
            return {text: (f"번역:{text}", "fake") for text in texts}
        finally:
            self._leave()


def blocker(release: threading.Event):
    """release될 때까지 막히는 훅 (취소를 무시하는 엔진 흉내)."""

    def hook() -> None:
        """release를 기다린다."""
        assert release.wait(5)

    return hook


def raiser(exc: Exception):
    """지정한 예외를 던지는 훅."""

    def hook() -> None:
        """예외를 던진다."""
        raise exc

    return hook


@pytest.fixture
def make_controller(qapp):
    """컨트롤러를 만들고 테스트가 끝나면 종료한다."""
    controllers: list[TranslationController] = []

    def factory(pipeline, max_concurrency: int = 1) -> TranslationController:
        """가짜 파이프라인과 동시 실행 수로 컨트롤러를 만든다."""
        controller = TranslationController(pipeline, InferenceScheduler(max_concurrency))
        controllers.append(controller)
        return controller

    yield factory
    for controller in controllers:
        controller.shutdown()


def test_ocr_result_is_delivered_from_background_thread(make_controller, qtbot) -> None:
    """인식은 워커 스레드에서 실행되고 결과는 요청 번호와 함께 UI 스레드로 전달된다."""
    pipeline = FakePipeline()
    controller = make_controller(pipeline)

    with qtbot.waitSignal(controller.ocr_ready, timeout=3000) as blocker_:
        request_id = controller.request_ocr("box-1", REGION)

    result = blocker_.args[0]
    assert result.source_id == "box-1" and result.request_id == request_id
    assert result.units[0].text == "box-1"
    assert pipeline.thread_names[0] != threading.main_thread().name
    assert not controller.is_busy


def test_request_options_are_passed_to_pipeline(make_controller, qtbot) -> None:
    """제외 영역, 프레임, 표시 중인 텍스트 자리, 지문 격자가 파이프라인에 그대로 전달된다."""
    pipeline = FakePipeline()
    controller = make_controller(pipeline)
    exclude = (ScreenRegion(1, 2, 3, 4),)
    probes = (ScreenRegion(5, 5, 2, 2),)
    frame = Frame(7, CapturedImage(REGION, 10, 10, b"\x00" * 400), 1.0)

    with qtbot.waitSignal(controller.ocr_ready, timeout=3000):
        controller.request_ocr(
            "fullscreen", None, frame=frame, exclude=exclude, probe_areas=probes,
            fingerprint_grid=(4, 2),
        )

    assert pipeline.requests[0] == PipelineRequest("fullscreen", None, exclude, frame, probes, (4, 2))


def test_translation_result_is_delivered_as_outcome(make_controller, qtbot) -> None:
    """번역 요청의 결과는 원문 → (번역문, 엔진)으로 출처와 함께 전달된다."""
    controller = make_controller(FakePipeline())

    with qtbot.waitSignal(controller.translations_ready, timeout=3000) as blocker_:
        controller.request_translation("box-1", ("hello", "bye"))

    outcome = blocker_.args[0]
    assert isinstance(outcome, TranslationOutcome)
    assert outcome.source_id == "box-1"
    assert outcome.translations == {"hello": ("번역:hello", "fake"), "bye": ("번역:bye", "fake")}


def test_stale_result_of_same_source_is_ignored(make_controller, qtbot) -> None:
    """같은 출처에 새 요청이 있으면 늦게 끝난 이전 요청의 결과는 전달하지 않는다."""
    release = threading.Event()
    pipeline = FakePipeline()
    calls = {"count": 0}

    def first_blocks() -> None:
        """첫 요청만 막는다."""
        calls["count"] += 1
        if calls["count"] == 1:
            pipeline.texts["box-1"] = "new"
            assert release.wait(5)

    pipeline.hooks["box-1"] = first_blocks
    controller = make_controller(pipeline, max_concurrency=2)
    received: list[OcrResult] = []
    controller.ocr_ready.connect(received.append)

    controller.request_ocr("box-1", REGION)
    qtbot.waitUntil(lambda: calls["count"] == 1, timeout=3000)
    latest = controller.request_ocr("box-1", REGION)
    qtbot.waitUntil(lambda: len(received) >= 1, timeout=3000)
    release.set()
    qtbot.wait(100)  # 오래된 결과가 뒤늦게 도착하지 않는지 확인

    assert [r.request_id for r in received] == [latest]


def test_ocr_and_translation_of_same_source_do_not_cancel_each_other(make_controller, qtbot) -> None:
    """같은 출처라도 인식과 번역은 다른 종류라 서로 취소하지 않는다."""
    controller = make_controller(FakePipeline())
    ocr: list[OcrResult] = []
    translations: list[TranslationOutcome] = []
    controller.ocr_ready.connect(ocr.append)
    controller.translations_ready.connect(translations.append)

    controller.request_ocr("box-1", REGION)
    controller.request_translation("box-1", ("hello",))
    assert controller.is_pending("box-1", KIND_OCR) and controller.is_pending("box-1", KIND_TRANSLATE)

    qtbot.waitUntil(lambda: len(ocr) == 1 and len(translations) == 1, timeout=3000)


def test_requests_of_different_sources_do_not_cancel_each_other(make_controller, qtbot) -> None:
    """박스 A와 박스 B의 요청은 서로 취소하지 않고, 결과는 각자의 출처로 전달된다."""
    release = threading.Event()
    pipeline = FakePipeline()
    pipeline.hooks["box-a"] = blocker(release)
    controller = make_controller(pipeline)
    received: list[OcrResult] = []
    controller.ocr_ready.connect(received.append)

    controller.request_ocr("box-a", REGION)
    controller.request_ocr("box-b", REGION)
    assert controller.is_pending("box-a") and controller.is_pending("box-b")
    release.set()

    qtbot.waitUntil(lambda: len(received) == 2, timeout=3000)
    assert sorted(r.source_id for r in received) == ["box-a", "box-b"]


def test_translation_runs_before_waiting_ocr_and_rechecks_run_last(make_controller, qtbot) -> None:
    """GPU 자리가 나면 번역 → 사용자 OCR → 자동 OCR → 재검사 순으로 실행한다."""
    release = threading.Event()
    pipeline = FakePipeline()
    pipeline.hooks["gate"] = blocker(release)
    controller = make_controller(pipeline, max_concurrency=1)
    done: list[object] = []
    controller.ocr_ready.connect(done.append)
    controller.translations_ready.connect(done.append)

    controller.request_ocr("gate", REGION)  # 유일한 자리를 차지한다
    qtbot.waitUntil(lambda: pipeline.order == ["gate"], timeout=3000)
    controller.request_ocr("recheck", REGION, priority=TaskPriority.OCR_RECHECK)
    controller.request_ocr("auto", REGION, priority=TaskPriority.OCR_AUTO)
    controller.request_translation("box-1", ("hello",))
    controller.request_ocr("user", REGION, priority=TaskPriority.OCR_USER)
    release.set()

    qtbot.waitUntil(lambda: len(done) == 5, timeout=3000)
    assert pipeline.order == ["gate", "T:hello", "user", "auto", "recheck"]


@pytest.mark.parametrize("limit", [1, 2])
def test_inference_concurrency_is_limited(make_controller, qtbot, limit) -> None:
    """자동 감지 OCR과 번역을 섞어 넣어도 동시에 실행되는 추론 작업 수는 한도를 넘지 않는다."""
    pipeline = FakePipeline(work_seconds=0.03)
    controller = make_controller(pipeline, max_concurrency=limit)
    done: list[object] = []
    controller.ocr_ready.connect(done.append)
    controller.translations_ready.connect(done.append)

    for index in range(4):
        controller.request_ocr(f"box-{index}", REGION, priority=TaskPriority.OCR_AUTO)
        controller.request_translation(f"box-{index}", (f"t{index}",))

    qtbot.waitUntil(lambda: len(done) == 8, timeout=5000)
    assert pipeline.max_running == limit


def test_concurrency_limit_can_be_changed(make_controller) -> None:
    """VRAM 여유에 맞춰 동시 실행 수를 바꿀 수 있다 (1 미만은 1로)."""
    controller = make_controller(FakePipeline())

    controller.scheduler.set_max_concurrency(3)
    assert controller.scheduler.max_concurrency == 3
    controller.scheduler.set_max_concurrency(0)
    assert controller.scheduler.max_concurrency == 1


def test_superseded_queued_request_does_not_run(make_controller, qtbot) -> None:
    """큐에서 기다리다 새 요청으로 대체된 작업은 엔진을 호출하지 않고 끝난다."""
    release = threading.Event()
    pipeline = FakePipeline()
    pipeline.hooks["gate"] = blocker(release)
    controller = make_controller(pipeline, max_concurrency=1)
    received: list[OcrResult] = []
    controller.ocr_ready.connect(received.append)

    controller.request_ocr("gate", REGION)
    qtbot.waitUntil(lambda: pipeline.order == ["gate"], timeout=3000)
    for _ in range(5):
        controller.request_ocr("box-1", REGION, priority=TaskPriority.OCR_AUTO)
    release.set()

    qtbot.waitUntil(lambda: len(received) == 2, timeout=3000)
    qtbot.wait(50)
    assert pipeline.order.count("box-1") == 1


def test_ui_stays_responsive_while_queue_is_busy(make_controller, qtbot) -> None:
    """여러 박스의 요청이 큐에 쌓여 작업이 막혀 있어도 UI 이벤트 루프는 계속 돈다."""
    release = threading.Event()
    pipeline = FakePipeline()
    pipeline.hooks["box-0"] = blocker(release)
    controller = make_controller(pipeline)
    ticks: list[int] = []

    timer = QTimer()
    timer.timeout.connect(lambda: ticks.append(1))
    timer.start(10)
    for index in range(5):
        controller.request_ocr(f"box-{index}", REGION)
    qtbot.wait(150)  # 첫 작업이 막혀 있는 동안
    timer.stop()
    release.set()

    assert len(ticks) >= 5  # 요청을 넣는 동안과 막혀 있는 동안에도 타이머가 계속 실행됨


def test_cancelled_source_does_not_emit_result(make_controller, qtbot) -> None:
    """취소한 출처(삭제된 박스 등)의 결과는 늦게 끝나도 전달하지 않는다."""
    release = threading.Event()
    pipeline = FakePipeline()
    pipeline.hooks["box-1"] = blocker(release)
    controller = make_controller(pipeline)
    received: list[object] = []
    controller.ocr_ready.connect(received.append)
    controller.translations_ready.connect(received.append)
    controller.failed.connect(lambda *args: received.append(args))

    controller.request_ocr("box-1", REGION)
    controller.request_translation("box-1", ("x",))
    controller.cancel("box-1")
    release.set()

    qtbot.wait(200)
    assert received == []
    assert not controller.is_pending("box-1")


def test_expected_error_message_is_forwarded_with_source(make_controller, qtbot) -> None:
    """예상된 오류의 사용자 메시지는 출처와 함께 그대로 전달한다."""
    pipeline = FakePipeline()
    pipeline.hooks["box-2"] = raiser(CaptureError("선택한 영역을 캡처하지 못했습니다."))
    controller = make_controller(pipeline)

    with qtbot.waitSignal(controller.failed, timeout=3000) as blocker_:
        controller.request_ocr("box-2", REGION)

    assert blocker_.args == ["box-2", "선택한 영역을 캡처하지 못했습니다."]


def test_unexpected_error_is_logged_and_reported_generically(make_controller, qtbot, caplog) -> None:
    """예상하지 못한 오류는 로그에 남기고 일반 메시지만 전달한다."""
    pipeline = FakePipeline()
    pipeline.hooks["box-1"] = raiser(RuntimeError("secret detail"))
    controller = make_controller(pipeline)

    with qtbot.waitSignal(controller.failed, timeout=3000) as blocker_:
        controller.request_ocr("box-1", REGION)

    assert blocker_.args == ["box-1", UNEXPECTED_ERROR_MESSAGE]
    assert any("예기치 못한 오류" in r.getMessage() for r in caplog.records)


def test_empty_result_is_delivered_as_normal_result(make_controller, qtbot) -> None:
    """인식된 텍스트가 없으면 실패가 아니라 빈 결과로 전달한다."""
    pipeline = FakePipeline()
    pipeline.texts["box-1"] = None
    controller = make_controller(pipeline)
    failures: list[object] = []
    controller.failed.connect(lambda *args: failures.append(args))

    with qtbot.waitSignal(controller.ocr_ready, timeout=3000) as blocker_:
        controller.request_ocr("box-1", REGION)
    assert blocker_.args[0].units == ()
    assert failures == []


def test_requests_after_shutdown_are_ignored(make_controller) -> None:
    """종료 후 들어온 요청은 무시한다."""
    controller = make_controller(FakePipeline())
    controller.shutdown()

    assert controller.request_ocr("box-1", REGION) is None
    assert controller.request_translation("box-1", ("x",)) is None


def test_shutdown_waits_for_running_task_and_drops_queued_ones(make_controller, qtbot) -> None:
    """종료하면 기다리던 작업은 실행하지 않고, 실행 중인 작업이 끝날 때까지 기다리며, 결과는 버린다."""
    pipeline = FakePipeline()
    started = threading.Event()

    def slow() -> None:
        """시작을 알리고 잠시 걸린다."""
        started.set()
        time.sleep(0.1)

    pipeline.hooks["running"] = slow
    controller = make_controller(pipeline)
    received: list[object] = []
    controller.ocr_ready.connect(received.append)

    controller.request_ocr("running", REGION)
    for index in range(3):
        controller.request_ocr(f"queued-{index}", REGION)
    assert started.wait(3)
    controller.shutdown()

    assert pipeline.running == 0  # 실행 중이던 작업이 끝난 뒤 반환됨
    qtbot.wait(50)
    assert pipeline.order == ["running"]
    assert received == []


def test_shutdown_reports_work_that_did_not_finish_in_time(make_controller) -> None:
    """실행 중인 작업이 종료 대기 시간을 넘기면 shutdown이 False를 돌려준다 (멈출 수 없는 긴 OCR)."""
    pipeline = FakePipeline()
    started = threading.Event()
    release = threading.Event()

    def blocked() -> None:
        """시작을 알리고 풀릴 때까지 막힌다 (취소를 확인하지 않는 추론을 흉내 냄)."""
        started.set()
        release.wait(5)

    pipeline.hooks["long"] = blocked
    controller = make_controller(pipeline)
    controller.request_ocr("long", REGION)
    assert started.wait(3)

    finished = controller.shutdown(timeout_ms=50)
    release.set()

    assert finished is False
    assert controller.shutdown() is False  # 두 번째 호출도 같은 결과
    assert controller.scheduler.wait(3000)


def test_shutdown_reports_finished_work(make_controller) -> None:
    """실행 중인 작업이 없거나 시간 안에 끝나면 shutdown이 True를 돌려준다."""
    controller = make_controller(FakePipeline())

    assert controller.shutdown() is True


def test_task_finishing_after_signals_are_deleted_does_not_raise() -> None:
    """종료 뒤 결과를 받을 시그널 객체가 이미 삭제되었으면 결과를 버리고 예외를 내지 않는다 (회귀)."""
    import shiboken6

    from translator.application.controller import _Task, _TaskSignals

    for work in (lambda token: "결과", lambda token: (_ for _ in ()).throw(CaptureError("실패"))):
        signals = _TaskSignals()
        task = _Task(1, work, CancellationToken(), signals)
        shiboken6.delete(signals)

        task.run()  # 예외가 나면 테스트 실패


def test_results_carry_the_session_they_were_requested_in(make_controller, qtbot) -> None:
    """인식·번역 결과에는 요청한 시점의 콘텐츠·세션이 실려 온다."""
    controller = make_controller(FakePipeline())
    session = controller.session

    with qtbot.waitSignal(controller.ocr_ready, timeout=3000) as ocr:
        controller.request_ocr("box-1", REGION)
    with qtbot.waitSignal(controller.translations_ready, timeout=3000) as translated:
        controller.request_translation("box-1", ("x",))

    assert ocr.args[0].session == session
    assert translated.args[0].session == session


def test_new_session_cancels_running_requests_and_drops_late_results(make_controller, qtbot) -> None:
    """세션이 바뀌면 진행 중인 요청이 취소되고, 이전 세션의 늦은 결과는 전달되지 않는다 (회귀)."""
    pipeline = FakePipeline()
    started = threading.Event()
    release = threading.Event()

    def blocked() -> None:
        """시작을 알리고 풀릴 때까지 막힌다 (취소를 무시하는 추론을 흉내 냄)."""
        started.set()
        release.wait(5)

    pipeline.hooks["old"] = blocked
    controller = make_controller(pipeline)
    received: list = []
    controller.ocr_ready.connect(received.append)
    old_session = controller.session

    controller.request_ocr("old", REGION)
    assert started.wait(3)
    new_session_context = new_session("other-game")
    controller.begin_session(new_session_context)
    release.set()
    qtbot.wait(200)

    assert controller.session == new_session_context != old_session
    assert received == []
    assert not controller.is_busy


def test_requests_after_session_change_use_the_new_session(make_controller, qtbot) -> None:
    """세션이 바뀐 뒤의 요청은 새 세션을 달고 나간다."""
    controller = make_controller(FakePipeline())
    new_context = new_session("other-game")
    controller.begin_session(new_context)

    with qtbot.waitSignal(controller.ocr_ready, timeout=3000) as blocker_:
        controller.request_ocr("box-1", REGION)

    assert blocker_.args[0].session == new_context


def test_beginning_the_same_session_again_keeps_running_requests(make_controller) -> None:
    """같은 세션을 다시 시작해도 진행 중인 요청을 취소하지 않는다."""
    pipeline = FakePipeline()
    controller = make_controller(pipeline)
    controller.request_ocr("box-1", REGION)

    controller.begin_session(controller.session)

    assert controller.is_pending("box-1", KIND_OCR)
