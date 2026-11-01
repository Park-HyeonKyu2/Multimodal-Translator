"""콘텐츠별 박스가 앱 전체(coordinator)에서 동작하는지 확인하는 테스트: 콘텐츠 전환 시 박스 교체,
처음 식별되는 콘텐츠의 기존 박스 인수, 종료 후 복원, 이전 형식(레지스트리) 박스 이전."""

import pytest
from PySide6.QtCore import QRect, QSettings

from translator.adapters.ocr.mock import MockTextRecognizer
from translator.adapters.storage.sqlite_profiles import SqliteContentProfileStore
from translator.adapters.translation.mock import MockTranslator
from translator.application.regions import RegionConfig
from translator.core.interfaces import ScreenCapturer
from translator.core.models import CapturedImage, ScreenRegion
from translator.core.session import DEFAULT_CONTENT_ID
from translator.main import build_coordinator
from translator.ui.ui_settings import UiSettingsStore


class BlankCapturer(ScreenCapturer):
    """검은 화면만 돌려주는 가짜 캡처 (이 테스트에서는 캡처 내용이 중요하지 않다)."""

    def capture_region(self, region: ScreenRegion) -> CapturedImage:
        """영역 크기의 검은 이미지를 반환한다."""
        return CapturedImage(region, region.width, region.height, bytes(region.width * region.height * 4))

    def capture_full_screen(self) -> CapturedImage:
        """작은 검은 이미지를 반환한다."""
        return CapturedImage(ScreenRegion(0, 0, 4, 4), 4, 4, bytes(64))


@pytest.fixture
def paths(tmp_path):
    """테스트마다 새 INI 파일과 SQLite 파일 경로 (같은 경로로 다시 시작하면 재실행을 흉내 낸다)."""
    return str(tmp_path / "ui.ini"), str(tmp_path / "translator.db")


def start_app(ini_path: str, db_path: str):
    """설정·콘텐츠별 저장소를 연결한 coordinator를 만들어 시작한다."""
    store = SqliteContentProfileStore(db_path)
    coordinator = build_coordinator(
        capturer=BlankCapturer(),
        recognizer=MockTextRecognizer(delay_seconds=0),
        translator=MockTranslator(delay_seconds=0),
        settings=UiSettingsStore(QSettings(ini_path, QSettings.Format.IniFormat)),
        profile_store=store,
    )
    coordinator.start()
    return coordinator, store


def add_box(coordinator, rect: QRect):
    """영역 추가 버튼을 누르고 `rect`가 선택된 것처럼 처리해 박스를 만든다."""
    coordinator._set_edit_mode(True)  # 편집 중 추가: 바로 번역하지 않는다
    coordinator._overlay.region_button.click()
    coordinator._selector._on_selection_made(rect)
    return coordinator._regions.sessions()[-1]


def test_changing_content_swaps_boxes_and_restores_them_on_return(qtbot, paths) -> None:
    """콘텐츠가 바뀌면 박스가 그 콘텐츠의 것으로 바뀌고, 이전 콘텐츠로 돌아오면 원래 박스가 다시 나타난다."""
    coordinator, store = start_app(*paths)
    try:
        coordinator.workflow.change_content("app:a.exe")
        add_box(coordinator, QRect(100, 100, 300, 80))
        a_configs = coordinator._regions.configs()

        coordinator.workflow.change_content("app:b.exe")
        assert coordinator._regions.sessions() == []
        assert not coordinator._overlay.translate_region_button.isEnabled()  # 박스가 없으면 버튼도 꺼진다

        add_box(coordinator, QRect(100, 400, 200, 60))
        coordinator.workflow.change_content("app:a.exe")

        assert coordinator._regions.configs() == a_configs
        assert coordinator._overlay.translate_region_button.isEnabled()
    finally:
        coordinator.shutdown()
        store.close()


def test_existing_boxes_are_adopted_by_the_first_identified_content(qtbot, paths) -> None:
    """식별 전에 있던 박스는 처음 식별되는 콘텐츠로 넘어가 계속 보인다 (박스가 사라진 것처럼 보이지 않는다)."""
    coordinator, store = start_app(*paths)
    try:
        add_box(coordinator, QRect(100, 100, 300, 80))
        before = coordinator._regions.configs()

        coordinator.workflow.change_content("app:a.exe")

        assert coordinator._regions.configs() == before
        assert store.load_regions(DEFAULT_CONTENT_ID) == []
    finally:
        coordinator.shutdown()
        store.close()


def test_boxes_per_content_are_restored_after_restart(qtbot, paths) -> None:
    """종료할 때 저장한 박스는 다음 실행에 같은 콘텐츠가 식별되면 복원된다."""
    first, first_store = start_app(*paths)
    first.workflow.change_content("app:a.exe")
    add_box(first, QRect(100, 100, 300, 80))
    expected = first._regions.configs()
    first.shutdown()
    first_store.close()

    second, second_store = start_app(*paths)
    try:
        assert second._regions.sessions() == []  # 식별 전(default)에는 박스가 없다.
        second.workflow.change_content("app:a.exe")
        assert second._regions.configs() == expected
    finally:
        second.shutdown()
        second_store.close()


def test_legacy_registry_boxes_move_to_sqlite_and_follow_the_first_content(qtbot, paths) -> None:
    """이전 버전이 레지스트리에 저장한 박스는 시작할 때 SQLite로 옮겨지고, 처음 식별되는 콘텐츠가 가져간다."""
    ini_path, db_path = paths
    legacy = UiSettingsStore(QSettings(ini_path, QSettings.Format.IniFormat))
    old_config = RegionConfig("box-1", 100, 100, 300, 80)
    legacy.save_regions([old_config])
    legacy.sync()

    coordinator, store = start_app(ini_path, db_path)
    try:
        assert coordinator._regions.configs() == [old_config]  # 시작할 때 default 박스로 복원
        assert UiSettingsStore(QSettings(ini_path, QSettings.Format.IniFormat)).load_regions() == []

        coordinator.workflow.change_content("app:a.exe")

        assert coordinator._regions.configs() == [old_config]
        assert store.load_regions("app:a.exe") == [old_config]
    finally:
        coordinator.shutdown()
        store.close()


def test_storage_failure_is_shown_in_status_window_and_app_keeps_working(qtbot, paths) -> None:
    """저장소가 닫혀 저장에 실패해도 앱은 계속 동작하고 상태 창에 오류가 표시된다."""
    coordinator, store = start_app(*paths)
    errors: list[str] = []
    coordinator._status_window.show_error = errors.append
    coordinator._regions.storage_failed.connect(errors.append)
    try:
        store.close()  # 저장소를 일부러 닫아 저장 실패를 만든다.

        add_box(coordinator, QRect(100, 100, 300, 80))
        coordinator._regions.save_profile()
        coordinator._regions.save_profile()

        assert len(errors) >= 1
        assert len(coordinator._regions.sessions()) == 1  # 화면은 계속 동작한다.
    finally:
        coordinator.shutdown()
