"""이전 형식(레지스트리) 박스를 SQLite로 옮기는 테스트."""

import pytest
from PySide6.QtCore import QSettings

from translator.adapters.storage.sqlite_profiles import SqliteContentProfileStore
from translator.application.profile_store import ContentProfileStore
from translator.application.regions import RegionConfig
from translator.core.errors import StorageError
from translator.core.models import TranslationDirection
from translator.core.session import DEFAULT_CONTENT_ID
from translator.ui.region_migration import migrate_legacy_regions
from translator.ui.ui_settings import UiSettingsStore


def legacy_configs() -> list[RegionConfig]:
    """이전 형식으로 저장해 둔 박스 두 개."""
    return [
        RegionConfig("box-1", 100, 100, 300, 80, TranslationDirection.LEFT, True),
        RegionConfig("box-2", 100, 400, 200, 60),
    ]


@pytest.fixture
def settings(qapp, tmp_path) -> UiSettingsStore:
    """임시 INI 파일을 쓰는 UI 설정 저장소 (이전 형식 박스가 들어 있다)."""
    created = UiSettingsStore(QSettings(str(tmp_path / "ui.ini"), QSettings.Format.IniFormat))
    created.save_regions(legacy_configs())
    created.sync()
    return created


@pytest.fixture
def store():
    """메모리 DB를 쓰는 저장소."""
    created = SqliteContentProfileStore()
    yield created
    created.close()


def test_legacy_regions_move_to_default_content_and_are_cleared(settings, store) -> None:
    """이전 형식의 박스는 default 콘텐츠로 옮겨지고 레지스트리 값은 지워진다."""
    assert migrate_legacy_regions(settings, store) == 2

    assert store.load_regions(DEFAULT_CONTENT_ID) == legacy_configs()
    assert settings.load_regions() == []


def test_migration_runs_only_once(settings, store) -> None:
    """두 번째 실행에서는 옮길 것이 없다 (SQLite 내용을 다시 덮어쓰지 않는다)."""
    migrate_legacy_regions(settings, store)
    store.save_regions(DEFAULT_CONTENT_ID, legacy_configs()[:1])

    assert migrate_legacy_regions(settings, store) == 0
    assert store.load_regions(DEFAULT_CONTENT_ID) == legacy_configs()[:1]


def test_existing_sqlite_regions_are_not_overwritten(settings, store) -> None:
    """default에 이미 박스가 있으면 덮어쓰지 않고, 사용자 데이터를 잃지 않도록 레지스트리 값도 남긴다."""
    existing = [RegionConfig("box-9", 1, 1, 50, 20)]
    store.save_regions(DEFAULT_CONTENT_ID, existing)

    assert migrate_legacy_regions(settings, store) == 0

    assert store.load_regions(DEFAULT_CONTENT_ID) == existing
    assert settings.load_regions() == legacy_configs()


def test_failed_migration_keeps_legacy_values_for_next_run(settings) -> None:
    """옮기는 데 실패하면 레지스트리 값을 지우지 않아 다음 실행에 다시 시도할 수 있다."""

    class FailingStore(ContentProfileStore):
        """항상 실패하는 저장소."""

        def load_regions(self, content_id):
            """읽기는 비어 있다고 답한다."""
            return []

        def save_regions(self, content_id, configs):
            """쓰기는 항상 실패한다."""
            raise StorageError("disk full")

        def adopt_regions(self, source_content_id, target_content_id):
            """쓰지 않는다."""
            return []

    assert migrate_legacy_regions(settings, FailingStore()) == 0
    assert settings.load_regions() == legacy_configs()


def test_nothing_to_migrate_when_there_are_no_legacy_regions(qapp, tmp_path, store) -> None:
    """이전 형식의 박스가 없으면 아무것도 하지 않는다."""
    empty = UiSettingsStore(QSettings(str(tmp_path / "empty.ini"), QSettings.Format.IniFormat))

    assert migrate_legacy_regions(empty, store) == 0
    assert store.load_regions(DEFAULT_CONTENT_ID) == []
