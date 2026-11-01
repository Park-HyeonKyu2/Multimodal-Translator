"""SQLite를 열지 못했을 때의 대체 저장소(레지스트리 기반) 테스트: 기존 박스 유지, 콘텐츠별 저장, 다시 SQLite로 이전."""

import pytest
from PySide6.QtCore import QRect, QSettings

from translator.adapters.ocr.mock import MockTextRecognizer
from translator.adapters.storage.sqlite_profiles import SqliteContentProfileStore
from translator.adapters.translation.mock import MockTranslator
from translator.application.profile_store import ContentProfileStore
from translator.application.regions import RegionConfig
from translator.core.interfaces import ScreenCapturer
from translator.core.models import CapturedImage, ScreenRegion
from translator.main import build_coordinator, open_profile_store
from translator.ui.region_migration import migrate_legacy_regions
from translator.ui.ui_settings import SettingsProfileStore, UiSettingsStore


class BlankCapturer(ScreenCapturer):
    """검은 화면만 돌려주는 가짜 캡처 (이 테스트에서는 캡처 내용이 중요하지 않다)."""

    def capture_region(self, region: ScreenRegion) -> CapturedImage:
        """영역 크기의 검은 이미지를 반환한다."""
        return CapturedImage(region, region.width, region.height, bytes(region.width * region.height * 4))

    def capture_full_screen(self) -> CapturedImage:
        """작은 검은 이미지를 반환한다."""
        return CapturedImage(ScreenRegion(0, 0, 4, 4), 4, 4, bytes(64))


def new_settings(path: str) -> UiSettingsStore:
    """INI 파일을 쓰는 UI 설정 저장소 (같은 경로로 다시 만들면 재실행을 흉내 낸다)."""
    return UiSettingsStore(QSettings(path, QSettings.Format.IniFormat))


def box(region_id: str = "box-1", left: int = 100) -> RegionConfig:
    """테스트용 박스 설정."""
    return RegionConfig(region_id, left, 100, 300, 80)


@pytest.fixture
def ini_path(tmp_path) -> str:
    """테스트마다 새 INI 파일 경로."""
    return str(tmp_path / "ui.ini")


@pytest.fixture
def corrupt_db(tmp_path) -> str:
    """SQLite 파일이 아닌 내용이 든 파일 경로 (열면 실패한다)."""
    path = tmp_path / "translator.db"
    path.write_bytes(b"not a database" * 50)
    return str(path)


def test_settings_profile_store_keeps_boxes_per_content(qapp, ini_path) -> None:
    """대체 저장소는 콘텐츠 ID(`app:...exe`처럼 특수문자가 든 값)별로 박스를 따로 저장하고 다시 읽는다."""
    store = SettingsProfileStore(new_settings(ini_path))

    store.save_regions("app:chrome.exe", [box("box-1", 100), box("box-2", 400)])
    store.save_regions("app:game.exe", [box("box-1", 700)])

    reopened = SettingsProfileStore(new_settings(ini_path))
    assert [c.left for c in reopened.load_regions("app:chrome.exe")] == [100, 400]
    assert [c.left for c in reopened.load_regions("app:game.exe")] == [700]
    assert reopened.load_regions("app:other.exe") == []


def test_settings_profile_store_adopt_follows_the_same_rules(qapp, ini_path) -> None:
    """옮기기: 대상이 비어 있을 때만 원본의 박스를 옮기고 원본은 비운다."""
    store = SettingsProfileStore(new_settings(ini_path))
    store.save_regions("default", [box()])

    assert store.adopt_regions("default", "app:a.exe") == [box()]
    assert store.load_regions("default") == []

    store.save_regions("default", [box("box-9")])
    assert store.adopt_regions("default", "app:a.exe") == []  # 대상에 이미 박스가 있다.
    assert [c.region_id for c in store.load_regions("default")] == ["box-9"]
    assert store.adopt_regions("app:a.exe", "app:a.exe") == []


def test_open_profile_store_returns_sqlite_when_it_opens(qapp, ini_path, tmp_path) -> None:
    """정상이면 SQLite 저장소와 안내 없음을 돌려준다."""
    store, notice = open_profile_store(new_settings(ini_path), tmp_path / "ok.db")
    try:
        assert isinstance(store, SqliteContentProfileStore)
        assert notice is None
    finally:
        store.close()


def test_open_profile_store_falls_back_to_registry_with_actionable_notice(qapp, ini_path, corrupt_db) -> None:
    """SQLite를 열 수 없으면 레지스트리 기반 저장소와, 파일 경로·기존 박스 유지·다음 조치가 담긴 안내를 돌려준다."""
    store, notice = open_profile_store(new_settings(ini_path), corrupt_db)

    assert isinstance(store, SettingsProfileStore)
    assert notice is not None
    assert corrupt_db in notice and "지워지지 않았습니다" in notice and "다시 시작" in notice


def test_existing_boxes_still_appear_when_sqlite_cannot_open(qtbot, ini_path, corrupt_db) -> None:
    """SQLite를 열 수 없어도 기존(레지스트리) 박스가 복원되고, 새 박스도 계속 저장된다 (회귀)."""
    legacy = new_settings(ini_path)
    legacy.save_regions([box()])
    legacy.sync()
    settings = new_settings(ini_path)
    store, _ = open_profile_store(settings, corrupt_db)
    coordinator = build_coordinator(
        capturer=BlankCapturer(),
        recognizer=MockTextRecognizer(delay_seconds=0),
        translator=MockTranslator(delay_seconds=0),
        settings=settings,
        profile_store=store,
    )
    coordinator.start()
    try:
        assert [c.left for c in coordinator._regions.configs()] == [100]  # 기존 박스가 보인다.
        assert new_settings(ini_path).load_regions() == [box()]  # 대체 저장소는 레지스트리를 비우지 않는다.

        coordinator._set_edit_mode(True)
        coordinator._overlay.region_button.click()
        coordinator._selector._on_selection_made(QRect(100, 400, 200, 60))
    finally:
        coordinator.shutdown()

    assert len(new_settings(ini_path).load_regions()) == 2  # 종료할 때 새 박스까지 저장됐다.


def test_boxes_saved_by_the_fallback_move_to_sqlite_when_it_opens_again(qapp, ini_path, tmp_path) -> None:
    """대체 저장소가 콘텐츠별로 레지스트리에 쌓은 박스는 다음에 SQLite를 열 수 있을 때 모두 SQLite로 옮겨진다."""
    settings = new_settings(ini_path)
    fallback = SettingsProfileStore(settings)
    fallback.save_regions("default", [box("box-1", 100)])
    fallback.save_regions("app:chrome.exe", [box("box-1", 300), box("box-2", 500)])
    sqlite_store = SqliteContentProfileStore(tmp_path / "translator.db")

    moved = migrate_legacy_regions(settings, sqlite_store)

    try:
        assert moved == 3
        assert [c.left for c in sqlite_store.load_regions("default")] == [100]
        assert [c.left for c in sqlite_store.load_regions("app:chrome.exe")] == [300, 500]
        assert settings.profiles() == [] or all(not settings.load_regions(p) for p in settings.profiles())
    finally:
        sqlite_store.close()


def test_fallback_store_is_not_migrated_onto_itself(qapp, ini_path) -> None:
    """대체 저장소를 쓸 때는 시작 시 이전을 하지 않아 레지스트리의 박스가 그대로 남는다."""
    settings = new_settings(ini_path)
    settings.save_regions([box()])
    settings.sync()

    coordinator = build_coordinator(
        capturer=BlankCapturer(),
        settings=settings,
        profile_store=SettingsProfileStore(settings),
    )
    try:
        assert new_settings(ini_path).load_regions() == [box()]
    finally:
        coordinator.shutdown()


def test_store_close_defaults_to_doing_nothing(qapp, ini_path) -> None:
    """`close()`는 기본적으로 아무것도 하지 않으므로 대체 저장소도 안전하게 닫을 수 있다."""
    store: ContentProfileStore = SettingsProfileStore(new_settings(ini_path))

    store.close()
    store.close()
