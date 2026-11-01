"""SQLite 콘텐츠별 프로필 저장소 테스트: 저장·복원, 콘텐츠 분리, 옮기기, 스키마 버전, 손상·실패 처리."""

import sqlite3
import time

import pytest

from translator.adapters.storage.sqlite_profiles import (
    DATABASE_FILE_NAME,
    SCHEMA_VERSION,
    SqliteContentProfileStore,
    default_database_path,
)
from translator.application.regions import RegionConfig
from translator.core.errors import StorageError
from translator.core.models import TranslationDirection
from translator.tuning import STORAGE_BUSY_TIMEOUT_SECONDS


def make_configs() -> list[RegionConfig]:
    """테스트용 박스 설정 두 개 (음수 좌표, 서로 다른 방향·자동 감지 여부)."""
    return [
        RegionConfig("box-2", -1500, 10, 300, 80, TranslationDirection.TOP, True),
        RegionConfig("box-1", 0, 0, 100, 40),
    ]


@pytest.fixture
def store():
    """메모리 DB를 쓰는 저장소."""
    created = SqliteContentProfileStore()
    yield created
    created.close()


def test_saved_regions_are_loaded_in_the_saved_order(store) -> None:
    """저장한 박스가 값과 순서(ID 순서가 아니라 저장한 순서) 그대로 복원된다."""
    store.save_regions("app:a.exe", make_configs())

    assert store.load_regions("app:a.exe") == make_configs()


def test_unknown_content_has_no_regions(store) -> None:
    """저장한 적 없는 콘텐츠는 빈 목록이다."""
    assert store.load_regions("app:unknown.exe") == []


def test_regions_are_isolated_per_content(store) -> None:
    """한 콘텐츠의 저장이 다른 콘텐츠의 박스에 영향을 주지 않는다 (콘텐츠 간 섞임 방지)."""
    store.save_regions("app:a.exe", make_configs())
    store.save_regions("app:b.exe", [RegionConfig("box-1", 5, 5, 50, 20)])

    store.save_regions("app:a.exe", [])

    assert store.load_regions("app:a.exe") == []
    assert [c.region_id for c in store.load_regions("app:b.exe")] == ["box-1"]


def test_saving_replaces_the_previous_list(store) -> None:
    """저장은 목록을 통째로 바꾼다 (줄어든 목록에 이전 항목이 남지 않는다)."""
    store.save_regions("app:a.exe", make_configs())

    store.save_regions("app:a.exe", make_configs()[:1])

    assert store.load_regions("app:a.exe") == make_configs()[:1]


def test_regions_survive_reopening_the_file(tmp_path) -> None:
    """파일에 저장한 박스는 다시 열어도 남아 있고, 폴더가 없으면 만든다."""
    path = tmp_path / "nested" / "folder" / DATABASE_FILE_NAME
    first = SqliteContentProfileStore(path)
    first.save_regions("app:a.exe", make_configs())
    first.close()

    second = SqliteContentProfileStore(path)
    try:
        assert second.load_regions("app:a.exe") == make_configs()
    finally:
        second.close()


def test_failed_save_keeps_the_previous_content(store) -> None:
    """저장 도중 실패하면(중복 ID) 되돌려서 이전 내용이 그대로 남는다 (트랜잭션)."""
    store.save_regions("app:a.exe", make_configs())
    duplicated = [RegionConfig("box-9", 0, 0, 10, 10), RegionConfig("box-9", 1, 1, 10, 10)]

    with pytest.raises(StorageError):
        store.save_regions("app:a.exe", duplicated)

    assert store.load_regions("app:a.exe") == make_configs()


def test_corrupted_rows_are_skipped(tmp_path) -> None:
    """알 수 없는 방향이나 0 크기처럼 손상된 행은 건너뛰고 나머지를 복원한다."""
    path = tmp_path / DATABASE_FILE_NAME
    first = SqliteContentProfileStore(path)
    first.save_regions("app:a.exe", make_configs())
    first.close()
    raw = sqlite3.connect(path)
    raw.execute("UPDATE region SET direction = 'sideways' WHERE region_id = 'box-2'")
    raw.execute("UPDATE region SET width = 0 WHERE region_id = 'box-1'")
    raw.commit()
    raw.close()

    second = SqliteContentProfileStore(path)
    try:
        assert second.load_regions("app:a.exe") == []
    finally:
        second.close()


# --- 옮기기 (기존 박스를 처음 식별되는 콘텐츠가 가져감) -------------------------------------


def test_adopt_moves_regions_when_target_is_empty(store) -> None:
    """대상 콘텐츠에 박스가 없으면 원본의 박스를 모두 옮기고, 원본은 비게 된다."""
    store.save_regions("default", make_configs())

    adopted = store.adopt_regions("default", "app:a.exe")

    assert adopted == make_configs()
    assert store.load_regions("app:a.exe") == make_configs()
    assert store.load_regions("default") == []


def test_adopt_does_nothing_when_target_already_has_regions(store) -> None:
    """대상에 이미 박스가 있으면 옮기지 않는다 (기존 박스를 덮어쓰지 않는다)."""
    store.save_regions("default", make_configs())
    own = [RegionConfig("box-1", 7, 7, 70, 30)]
    store.save_regions("app:a.exe", own)

    assert store.adopt_regions("default", "app:a.exe") == []
    assert store.load_regions("app:a.exe") == own
    assert store.load_regions("default") == make_configs()


def test_adopt_with_empty_source_or_same_content_returns_nothing(store) -> None:
    """원본이 비었거나 같은 콘텐츠면 아무것도 옮기지 않는다."""
    store.save_regions("app:a.exe", make_configs())

    assert store.adopt_regions("default", "app:b.exe") == []
    assert store.adopt_regions("app:a.exe", "app:a.exe") == []
    assert store.load_regions("app:a.exe") == make_configs()


# --- 스키마 버전과 손상 --------------------------------------------------------------------


def test_new_database_records_the_schema_version(tmp_path) -> None:
    """새 파일은 현재 스키마 버전을 기록한다."""
    path = tmp_path / DATABASE_FILE_NAME
    SqliteContentProfileStore(path).close()

    raw = sqlite3.connect(path)
    try:
        assert raw.execute("PRAGMA user_version").fetchone() == (SCHEMA_VERSION,)
    finally:
        raw.close()


def test_database_from_a_newer_program_is_rejected_untouched(tmp_path) -> None:
    """더 새로운 버전이 만든 파일은 열지 않고 업데이트를 안내한다 (데이터를 건드리지 않는다)."""
    path = tmp_path / DATABASE_FILE_NAME
    raw = sqlite3.connect(path)
    raw.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    raw.commit()
    raw.close()

    with pytest.raises(StorageError, match="업데이트"):
        SqliteContentProfileStore(path)

    check = sqlite3.connect(path)
    try:
        assert check.execute("PRAGMA user_version").fetchone() == (SCHEMA_VERSION + 1,)
    finally:
        check.close()


def test_corrupted_file_raises_storage_error(tmp_path) -> None:
    """SQLite 파일이 아닌 내용이면 `StorageError`로 알린다 (예외가 그대로 새지 않는다)."""
    path = tmp_path / DATABASE_FILE_NAME
    path.write_bytes(b"this is not a database" * 50)

    with pytest.raises(StorageError):
        SqliteContentProfileStore(path)


def test_closed_store_raises_storage_error(store) -> None:
    """닫은 뒤 쓰면 `StorageError`다."""
    store.close()
    store.close()  # 여러 번 닫아도 된다.

    with pytest.raises(StorageError):
        store.load_regions("app:a.exe")


class CommitFailingConnection(sqlite3.Connection):
    """`COMMIT`이 한 번 실패하게 만들 수 있는 연결 (디스크 오류 흉내). 실패해도 트랜잭션은 열린 채로 남는다."""

    # True면 다음 COMMIT을 실패시킨다
    fail_next_commit = False

    def execute(self, sql, *args, **kwargs):
        """COMMIT을 실패시킬 차례면 실행하지 않고 오류를 던진다 (SQLite도 실패 후 트랜잭션을 열어 둔다)."""
        if sql == "COMMIT" and self.fail_next_commit:
            self.fail_next_commit = False
            raise sqlite3.OperationalError("disk I/O error")
        return super().execute(sql, *args, **kwargs)


def test_failed_commit_is_rolled_back_so_later_saves_work(tmp_path) -> None:
    """커밋이 실패하면 트랜잭션을 되돌려, 이후 저장이 "트랜잭션 안에서 시작할 수 없음"으로 계속 실패하지 않는다 (회귀)."""
    path = tmp_path / DATABASE_FILE_NAME
    store = SqliteContentProfileStore(path)
    store._connection.close()
    flaky = sqlite3.connect(path, isolation_level=None, factory=CommitFailingConnection)
    store._connection = flaky
    store.save_regions("app:a.exe", make_configs())

    flaky.fail_next_commit = True
    with pytest.raises(StorageError):
        store.save_regions("app:a.exe", make_configs()[:1])

    assert not flaky.in_transaction
    # 실패한 저장은 반영되지 않고, 다음 저장은 정상으로 동작한다.
    assert store.load_regions("app:a.exe") == make_configs()
    store.save_regions("app:a.exe", make_configs()[:1])
    assert store.load_regions("app:a.exe") == make_configs()[:1]
    store.close()


def test_save_while_another_connection_holds_the_lock_fails_fast(tmp_path) -> None:
    """다른 연결이 쓰기를 잡고 있어도 정해진 시간 안에 `StorageError`로 끝난다 (UI 스레드가 오래 멈추지 않는다)."""
    path = tmp_path / DATABASE_FILE_NAME
    store = SqliteContentProfileStore(path, busy_timeout_seconds=0.1)
    store.save_regions("app:a.exe", make_configs())
    holder = sqlite3.connect(path, isolation_level=None)
    holder.execute("BEGIN IMMEDIATE")  # 다른 프로세스가 쓰는 중인 상황
    try:
        started = time.monotonic()
        with pytest.raises(StorageError):
            store.save_regions("app:a.exe", make_configs()[:1])
        elapsed = time.monotonic() - started
    finally:
        holder.execute("ROLLBACK")
        holder.close()

    assert elapsed < 2.0
    # 잠금이 풀리면 이전 내용이 그대로이고 다음 저장은 정상이다.
    assert store.load_regions("app:a.exe") == make_configs()
    store.save_regions("app:a.exe", make_configs()[:1])
    assert store.load_regions("app:a.exe") == make_configs()[:1]
    store.close()


def test_default_busy_timeout_is_short_enough_for_the_ui_thread() -> None:
    """기본 대기 시간은 SQLite 기본값(5초)보다 짧다."""
    assert STORAGE_BUSY_TIMEOUT_SECONDS < 5.0


def test_default_database_path_uses_local_app_data(tmp_path) -> None:
    """기본 경로는 %LOCALAPPDATA%\\MultimodalTranslator 아래이고, 없으면 홈 폴더 아래다."""
    assert default_database_path({"LOCALAPPDATA": str(tmp_path)}) == (
        tmp_path / "MultimodalTranslator" / DATABASE_FILE_NAME
    )
    assert default_database_path({}).name == DATABASE_FILE_NAME
