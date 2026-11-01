"""콘텐츠별 박스 저장·교체 테스트 (`RegionManager` + 메모리 SQLite, 감지기는 가짜)."""

import pytest
from PySide6.QtCore import QObject, QRect, Signal

from translator.adapters.storage.sqlite_profiles import SqliteContentProfileStore
from translator.application.profile_store import ContentProfileStore
from translator.application.regions import RegionConfig
from translator.core.errors import StorageError
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

    def shutdown(self) -> None:
        """종료 기록."""
        self.is_running = False
        self.calls.append("shutdown")


class CountingStore(SqliteContentProfileStore):
    """저장 호출 횟수를 세는 메모리 SQLite 저장소."""

    def __init__(self) -> None:
        """메모리 DB를 열고 저장 기록을 비운다."""
        super().__init__()
        # `save_regions`에 넘어온 (콘텐츠 ID, 설정 목록) 기록
        self.writes: list[tuple[str, list[RegionConfig]]] = []

    def save_regions(self, content_id, configs) -> None:
        """호출을 기록한 뒤 실제로 저장한다."""
        self.writes.append((content_id, list(configs)))
        super().save_regions(content_id, configs)


@pytest.fixture
def manager(qapp, qtbot):
    """메모리 SQLite 저장소와 연결된 박스 관리자 (저장 지연은 짧게)."""
    store = CountingStore()
    detectors: list[FakeDetector] = []

    def make_detector(provider):
        """가짜 감지기를 만들고 기록한다."""
        detector = FakeDetector(provider)
        detectors.append(detector)
        return detector

    regions = RegionManager(
        box_factory=lambda region_id, accent: RegionBox(region_id, accent),
        detector_factory=make_detector,
        profile_store=store,
        save_delay_ms=20,
    )
    regions.store = store
    regions.detectors = detectors
    yield regions
    regions.shutdown()
    # 지운 박스의 지연 삭제를 관리자가 살아 있는 동안 처리한다. 관리자를 먼저 해제하면 이어지는 테스트의
    # 이벤트 처리에서 프로세스가 중단된다 (기존 `remove()`에도 있는 성질이라 테스트에서 피한다).
    qtbot.wait(30)
    store.close()


def test_switching_content_saves_old_boxes_and_restores_new_ones(manager) -> None:
    """콘텐츠를 바꾸면 지금 박스를 저장하고 닫은 뒤, 돌아왔을 때 같은 박스를 복원한다."""
    manager.switch_content("app:a.exe")
    manager.add(QRect(100, 100, 200, 50), interactive=False, auto_enabled=True)
    manager.add(QRect(100, 400, 200, 50), interactive=False)
    saved = manager.configs()

    manager.switch_content("app:b.exe")

    assert len(manager) == 0
    assert manager.content_id == "app:b.exe"
    assert manager.store.load_regions("app:a.exe") == saved

    manager.switch_content("app:a.exe")

    assert manager.configs() == saved
    assert [s.detector.is_running for s in manager.sessions()] == [True, False]


def test_boxes_of_one_content_never_appear_in_another(manager) -> None:
    """한 콘텐츠에서 만든 박스는 다른 콘텐츠에 나타나지 않는다 (콘텐츠 간 섞임 방지)."""
    manager.switch_content("app:a.exe")
    manager.add(QRect(100, 100, 200, 50), interactive=False)

    manager.switch_content("app:b.exe")
    manager.add(QRect(300, 300, 100, 40), interactive=False)

    assert [(c.region_id, c.left) for c in manager.configs()] == [("box-1", 300)]
    assert [c.left for c in manager.store.load_regions("app:a.exe")] == [100]


def test_switching_shuts_down_detectors_of_old_boxes(manager) -> None:
    """바꾸기 전 박스의 자동 감지기는 종료된다 (이전 콘텐츠의 감지가 계속 돌지 않는다)."""
    manager.switch_content("app:a.exe")
    manager.add(QRect(100, 100, 200, 50), interactive=False, auto_enabled=True)
    old_detector = manager.detectors[0]

    manager.switch_content("app:b.exe")

    assert "shutdown" in old_detector.calls


def test_switching_to_same_content_changes_nothing(manager) -> None:
    """같은 콘텐츠로 바꾸면 박스를 그대로 둔다."""
    manager.switch_content("app:a.exe")
    session = manager.add(QRect(100, 100, 200, 50), interactive=False)
    changes: list[bool] = []
    manager.regions_changed.connect(lambda: changes.append(True))

    manager.switch_content("app:a.exe")

    assert manager.sessions() == [session]
    assert changes == []


def test_switching_resets_box_ids_and_keeps_edit_mode(manager) -> None:
    """새 콘텐츠의 박스 ID는 다시 1번부터이고, 편집 모드는 복원한 박스에도 적용된다."""
    manager.switch_content("app:a.exe")
    manager.add(QRect(100, 100, 200, 50), interactive=False)
    manager.add(QRect(100, 200, 200, 50), interactive=False)
    manager.set_interactive(True)

    manager.switch_content("app:b.exe")
    first = manager.add(QRect(300, 300, 100, 40), interactive=True)
    manager.switch_content("app:a.exe")

    assert first.region_id == "box-1"
    assert len(manager) == 2
    assert all(session.box.is_interactive for session in manager.sessions())


def test_default_boxes_are_adopted_by_first_content_without_boxes(manager) -> None:
    """식별 전(default)의 박스는, 박스가 없는 콘텐츠가 처음 식별될 때 그 콘텐츠가 가져간다."""
    manager.add(QRect(100, 100, 200, 50), interactive=False)
    before = manager.configs()

    manager.switch_content("app:a.exe")

    assert manager.configs() == before
    assert manager.store.load_regions("default") == []
    assert manager.store.load_regions("app:a.exe") == before


def test_default_boxes_are_not_adopted_by_content_that_has_its_own(manager) -> None:
    """이미 자기 박스가 있는 콘텐츠는 default의 박스를 가져가지 않는다 (default에 그대로 남는다)."""
    own = [RegionConfig("box-1", 700, 700, 100, 40)]
    manager.store.save_regions("app:a.exe", own)
    session = manager.add(QRect(100, 100, 200, 50), interactive=False)

    manager.switch_content("app:a.exe")

    assert manager.configs() == own
    assert [c.region_id for c in manager.store.load_regions("default")] == [session.region_id]


def test_only_the_first_switch_from_default_adopts(manager) -> None:
    """default의 박스를 가져가는 것은 default에서 처음 바뀔 때뿐이다. 다음 콘텐츠는 빈 상태로 시작한다."""
    manager.add(QRect(100, 100, 200, 50), interactive=False)
    manager.switch_content("app:a.exe")

    manager.switch_content("app:b.exe")

    assert len(manager) == 0


def test_without_store_boxes_stay_when_content_changes(qapp) -> None:
    """저장소가 없으면 콘텐츠별로 나눌 수 없으므로 박스를 그대로 둔다 (기존 동작)."""
    plain = RegionManager(
        box_factory=lambda region_id, accent: RegionBox(region_id, accent),
        detector_factory=FakeDetector,
    )
    try:
        plain.add(QRect(100, 100, 200, 50), interactive=False)

        plain.switch_content("app:a.exe")

        assert len(plain) == 1
        assert plain.content_id == "app:a.exe"
    finally:
        plain.shutdown()


def test_load_profile_restores_boxes_of_current_content(manager) -> None:
    """시작할 때 `load_profile`은 지금 콘텐츠의 저장된 박스를 복원한다."""
    saved = [RegionConfig("box-3", 100, 100, 200, 50), RegionConfig("box-1", 100, 300, 200, 50)]
    manager.store.save_regions("default", saved)

    manager.load_profile()

    assert manager.configs() == saved
    assert manager.add(QRect(100, 500, 100, 40), interactive=False).region_id == "box-4"


def test_changes_are_saved_once_after_the_delay(manager, qtbot) -> None:
    """설정이 계속 바뀌어도 저장은 모아서 마지막 상태를 한 번 한다."""
    manager.add(QRect(100, 100, 200, 50), interactive=False)
    manager.store.writes.clear()

    for _ in range(5):
        manager.schedule_save()
    qtbot.waitUntil(lambda: len(manager.store.writes) >= 1, timeout=2000)
    qtbot.wait(80)

    assert len(manager.store.writes) == 1
    assert manager.store.writes[0] == ("default", manager.configs())


def test_save_profile_writes_immediately_and_cancels_scheduled_save(manager, qtbot) -> None:
    """바로 저장하면 예약된 저장은 취소된다 (종료할 때 기다리지 않고 저장)."""
    manager.add(QRect(100, 100, 200, 50), interactive=False)
    manager.schedule_save()
    manager.store.writes.clear()

    manager.save_profile()
    qtbot.wait(80)

    assert len(manager.store.writes) == 1
    assert manager.store.load_regions("default") == manager.configs()


def test_storage_failures_are_reported_once_and_do_not_raise(qapp) -> None:
    """저장소가 실패해도 예외가 새지 않고, 같은 실패는 한 번만 알린다. 성공하면 다시 알릴 수 있다."""

    class FlakyStore(ContentProfileStore):
        """실패 여부를 바꿀 수 있는 저장소."""

        def __init__(self) -> None:
            """처음에는 실패하는 상태로 시작한다."""
            self.failing = True

        def load_regions(self, content_id):
            """실패 상태면 예외, 아니면 빈 목록."""
            if self.failing:
                raise StorageError("read failed")
            return []

        def save_regions(self, content_id, configs):
            """실패 상태면 예외."""
            if self.failing:
                raise StorageError("write failed")

        def adopt_regions(self, source_content_id, target_content_id):
            """실패 상태면 예외, 아니면 옮길 것이 없다."""
            if self.failing:
                raise StorageError("adopt failed")
            return []

    store = FlakyStore()
    flaky = RegionManager(
        box_factory=lambda region_id, accent: RegionBox(region_id, accent),
        detector_factory=FakeDetector,
        profile_store=store,
    )
    messages: list[str] = []
    flaky.storage_failed.connect(messages.append)
    try:
        flaky.load_profile()
        flaky.save_profile()
        flaky.save_profile()
        assert len(messages) == 1  # 읽기·쓰기 연속 실패는 한 번만 알린다.

        store.failing = False
        flaky.save_profile()  # 성공하면 실패 상태가 풀린다.
        store.failing = True
        flaky.save_profile()
        assert len(messages) == 2
    finally:
        flaky.shutdown()


# --- 화면 밖 박스 보존 (회귀) --------------------------------------------------------------


def offscreen_and_onscreen() -> list[RegionConfig]:
    """지금 모니터 구성에서 화면 안인 박스 하나와 화면 밖인 박스 하나."""
    return [
        RegionConfig("box-1", 100, 100, 200, 50),
        RegionConfig("box-2", 9000, 9000, 200, 50),
    ]


def test_visiting_a_content_keeps_its_offscreen_boxes_in_the_store(manager, qtbot) -> None:
    """콘텐츠를 방문해 저장이 일어나도 복원하지 못한(화면 밖) 박스는 저장소에서 지워지지 않는다."""
    manager.store.save_regions("app:a.exe", offscreen_and_onscreen())

    manager.switch_content("app:a.exe")
    qtbot.wait(80)  # 전환 뒤 예약된 저장이 실행될 시간
    manager.switch_content("app:b.exe")

    assert [session.region_id for session in manager.sessions()] == []
    assert manager.store.load_regions("app:a.exe") == offscreen_and_onscreen()


def test_preserved_offscreen_box_returns_when_screen_is_available_again(manager, monkeypatch) -> None:
    """모니터를 다시 연결해(화면 안이 되면) 같은 콘텐츠로 돌아올 때 보존한 박스가 복원된다."""
    manager.store.save_regions("app:a.exe", offscreen_and_onscreen())
    manager.switch_content("app:a.exe")
    assert [s.region_id for s in manager.sessions()] == ["box-1"]
    manager.switch_content("app:b.exe")
    monkeypatch.setattr("translator.ui.region_manager.is_rect_on_any_screen", lambda rect: True)

    manager.switch_content("app:a.exe")

    assert [s.region_id for s in manager.sessions()] == ["box-1", "box-2"]


def test_new_box_does_not_reuse_id_of_preserved_offscreen_box(manager) -> None:
    """새 박스는 보존 중인 화면 밖 박스의 ID를 받지 않는다 (저장할 때 서로 덮어쓰지 않도록)."""
    manager.store.save_regions("app:a.exe", offscreen_and_onscreen())
    manager.switch_content("app:a.exe")

    added = manager.add(QRect(300, 300, 100, 40), interactive=False)
    manager.save_profile()

    assert added.region_id == "box-3"
    assert [c.region_id for c in manager.store.load_regions("app:a.exe")] == ["box-1", "box-3", "box-2"]


def test_removing_a_visible_box_keeps_the_preserved_one(manager) -> None:
    """화면 안 박스를 지워도 보존 중인 화면 밖 박스는 남는다."""
    manager.store.save_regions("app:a.exe", offscreen_and_onscreen())
    manager.switch_content("app:a.exe")

    manager.remove("box-1")
    manager.save_profile()

    assert [c.region_id for c in manager.store.load_regions("app:a.exe")] == ["box-2"]
