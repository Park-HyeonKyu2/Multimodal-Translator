"""여러 바운딩 박스 관리 테스트 (가짜 자동 감지기 사용)."""

import pytest
from PySide6.QtCore import QObject, QRect, Signal

from translator.application.regions import RegionConfig, RequestStatus
from translator.application.frames import Frame
from translator.core.models import CapturedImage, ScreenRegion, TranslationDirection
from translator.ui.region_box import RegionBox
from translator.ui.region_manager import RegionManager


class FakeDetector(QObject):
    """자동 감지기 대신 쓰는 가짜. 호출 기록만 남긴다."""

    frame_ready = Signal(object)
    stopped_by_error = Signal(str)

    def __init__(self, provider) -> None:
        """감시 영역 제공 함수를 받는다."""
        super().__init__()
        self.provider = provider
        self.is_running = False
        self.calls: list[str] = []
        self.marked: list[ScreenRegion] = []
        # (지문, 영역): 수동 OCR 결과로 기록된 화면
        self.fingerprints: list[tuple[object, ScreenRegion]] = []

    def start(self) -> None:
        """시작 기록."""
        self.is_running = True
        self.calls.append("start")

    def stop(self) -> None:
        """중지 기록."""
        self.is_running = False
        self.calls.append("stop")

    def reset(self) -> None:
        """기록 초기화 기록."""
        self.calls.append("reset")

    def mark_translated(self, region: ScreenRegion) -> None:
        """수동 번역 기록."""
        self.marked.append(region)

    def mark_fingerprint(self, fingerprint: object, region: ScreenRegion) -> None:
        """수동 OCR 화면 지문 기록."""
        self.fingerprints.append((fingerprint, region))

    def shutdown(self) -> None:
        """종료 기록."""
        self.is_running = False
        self.calls.append("shutdown")


@pytest.fixture
def manager(qapp):
    """가짜 감지기를 쓰는 박스 관리자."""
    detectors: list[FakeDetector] = []

    def make_detector(provider):
        """가짜 감지기를 만들고 기록한다."""
        detector = FakeDetector(provider)
        detectors.append(detector)
        return detector

    regions = RegionManager(
        box_factory=lambda region_id, accent: RegionBox(region_id, accent),
        detector_factory=make_detector,
    )
    regions.detectors = detectors
    yield regions
    regions.shutdown()


def test_multiple_boxes_get_unique_ids_and_colors(manager) -> None:
    """박스를 여러 개 만들 수 있고, 각자 고유 ID와 다른 색을 가진다."""
    first = manager.add(QRect(100, 100, 200, 50), interactive=False)
    second = manager.add(QRect(100, 400, 200, 50), interactive=False)
    third = manager.add(QRect(400, 400, 100, 50), interactive=False)

    assert [s.region_id for s in manager.sessions()] == ["box-1", "box-2", "box-3"]
    assert len({first.accent.name(), second.accent.name(), third.accent.name()}) == 3
    assert len(manager) == 3


def test_boxes_move_and_resize_independently(manager, qtbot) -> None:
    """한 박스를 옮기거나 크기를 바꿔도 다른 박스의 설정은 그대로다."""
    first = manager.add(QRect(100, 100, 200, 50), interactive=True)
    second = manager.add(QRect(100, 400, 200, 50), interactive=True)
    for detector in manager.detectors:
        detector.calls.clear()  # 처음 표시할 때 생기는 이동 이벤트 기록은 지운다

    first.box.move(first.box.x() + 50, first.box.y())
    first.box.resize(first.box.width() + 30, first.box.height())

    assert (first.config.left, first.config.width) == (150, 230)
    assert (second.config.left, second.config.top, second.config.width) == (100, 400, 200)
    assert "reset" in manager.detectors[0].calls
    assert "reset" not in manager.detectors[1].calls


def test_geometry_change_while_editing_marks_box_for_translation(manager) -> None:
    """편집 중에 바뀐 박스는 편집을 마칠 때 다시 번역하도록 표시된다."""
    editing = manager.add(QRect(100, 100, 200, 50), interactive=True)
    locked = manager.add(QRect(100, 400, 200, 50), interactive=False)
    editing.state.needs_translation = False

    editing.box.move(editing.box.x() + 10, editing.box.y())
    locked.box.move(locked.box.x() + 10, locked.box.y())

    assert editing.state.needs_translation
    assert not locked.state.needs_translation


def test_translate_now_requests_with_box_region(manager, qtbot) -> None:
    """박스의 수동 번역 요청에는 박스 ID, 캡처 영역, 수동 표시가 담긴다."""
    session = manager.add(QRect(100, 100, 200, 50), interactive=False)

    with qtbot.waitSignal(manager.ocr_requested, timeout=1000) as blocker:
        session.translate_now()

    # 수동 번역은 새로 캡처하므로 프레임이 없다.
    assert blocker.args == ["box-1", session.capture_region(), None, True]


def test_auto_detection_request_is_tagged_with_box_id(manager, qtbot) -> None:
    """박스의 자동 감지 요청은 그 박스 ID로, 자동(수동 아님)으로 전달된다."""
    manager.add(QRect(100, 100, 200, 50), interactive=False)
    second = manager.add(QRect(100, 400, 200, 50), interactive=False)
    region = second.capture_region()

    frame = Frame(5, CapturedImage(region, region.width, region.height, b""), 1.0)

    with qtbot.waitSignal(manager.ocr_requested, timeout=1000) as blocker:
        manager.detectors[1].frame_ready.emit(frame)

    # 자동 감지는 감지기가 찍은 프레임을 그대로 넘긴다 (다시 캡처하지 않음).
    assert blocker.args == ["box-2", region, frame, False]


def test_manual_ocr_fingerprint_is_forwarded_to_detector(manager) -> None:
    """수동 OCR 결과의 화면 지문을 추가 캡처 없이 그 박스의 자동 감지기에 기록한다."""
    session = manager.add(QRect(100, 100, 200, 50), interactive=False)
    region = session.capture_region()

    session.mark_ocr_fingerprint("fp", region)

    assert manager.detectors[0].fingerprints == [("fp", region)]
    assert manager.detectors[0].marked == []


def test_direction_and_auto_are_per_box(manager, qtbot) -> None:
    """출력 방향과 자동 감지는 박스마다 따로 설정된다."""
    first = manager.add(QRect(100, 100, 200, 50), interactive=True)
    second = manager.add(QRect(100, 400, 200, 50), interactive=True)

    with qtbot.waitSignal(manager.appearance_changed, timeout=1000) as blocker:
        first.box.direction_selected.emit(TranslationDirection.RIGHT)
    first.box.auto_toggled.emit(True)

    assert blocker.args == ["box-1"]
    assert first.config.direction is TranslationDirection.RIGHT
    assert second.config.direction is TranslationDirection.BOTTOM
    assert manager.detectors[0].is_running and not manager.detectors[1].is_running
    assert not manager.all_auto_enabled()
    manager.set_auto_all(True)
    assert manager.all_auto_enabled()


def test_removing_one_box_keeps_others_working(manager, qtbot) -> None:
    """박스 하나를 지워도 다른 박스는 그대로 번역을 요청할 수 있다."""
    manager.add(QRect(100, 100, 200, 50), interactive=False)
    second = manager.add(QRect(100, 400, 200, 50), interactive=False)

    manager.remove("box-1")

    assert manager.get("box-1") is None
    assert "shutdown" in manager.detectors[0].calls
    assert "shutdown" not in manager.detectors[1].calls
    with qtbot.waitSignal(manager.ocr_requested, timeout=1000):
        second.translate_now()


def test_runtime_state_tracks_last_texts_and_status(manager) -> None:
    """박스마다 마지막 요청 상태와 마지막으로 처리한 원문을 기록한다."""
    session = manager.add(QRect(100, 100, 200, 50), interactive=False)
    region = ScreenRegion(100, 100, 200, 50)

    session.mark_pending(7)
    assert (session.state.status, session.state.last_request_id) == (RequestStatus.PENDING, 7)
    session.on_ocr(("hello",))
    assert session.state.status is RequestStatus.DONE
    assert session.state.last_texts == ("hello",)
    session.on_ocr(())
    assert session.state.status is RequestStatus.NO_TEXT
    session.on_failed("오류")
    assert (session.state.status, session.state.last_error) == (RequestStatus.FAILED, "오류")


def test_restore_keeps_ids_and_settings_and_skips_offscreen(manager) -> None:
    """저장된 박스를 ID·방향·자동 감지 그대로 복원하고, 화면 밖 박스는 건너뛴다."""
    configs = [
        RegionConfig("box-4", 100, 100, 200, 50, TranslationDirection.LEFT, True),
        RegionConfig("box-7", 5000, 5000, 200, 50),  # 지금 모니터 구성에서 화면 밖
    ]

    restored = manager.restore(configs, interactive=False)

    assert [s.region_id for s in restored] == ["box-4"]
    assert restored[0].config == configs[0]
    assert manager.detectors[0].is_running  # 자동 감지 상태도 복원
    # 새 박스 ID는 복원된 ID는 물론, 설정만 보존한 화면 밖 박스(box-7)의 ID와도 겹치지 않는다.
    assert manager.add(QRect(10, 10, 50, 20), interactive=False).region_id == "box-8"


def test_set_interactive_applies_to_all_boxes(manager) -> None:
    """편집 모드 전환은 모든 박스에 적용된다."""
    manager.add(QRect(100, 100, 200, 50), interactive=False)
    manager.add(QRect(100, 400, 200, 50), interactive=False)

    manager.set_interactive(True)

    assert all(s.box.is_interactive for s in manager.sessions())
