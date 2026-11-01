"""SQLite로 콘텐츠별 프로필(바운딩 박스 설정)을 저장한다.

- 표준 라이브러리 `sqlite3`만 쓰므로 새 의존성이 없다. 데이터는 사용자 PC의 로컬 파일에만 저장하며
  번역 내용이나 화면 정보는 저장하지 않는다.
- 스키마 버전은 `PRAGMA user_version`에 기록하고, `_MIGRATIONS`를 순서대로 적용해 올린다. 이
  프로그램보다 새로운 버전의 파일은 데이터를 망가뜨리지 않도록 열지 않는다.
- 연결은 만든 스레드에서만 쓸 수 있다 (UI 스레드에서만 사용한다).
- 모든 쓰기는 트랜잭션 하나로 처리해, 도중에 실패해도 이전 내용이 남는다.
"""

import logging
import os
import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path

from translator.application.profile_store import ContentProfileStore
from translator.application.regions import RegionConfig
from translator.core.errors import StorageError
from translator.core.models import TranslationDirection
from translator.tuning import STORAGE_BUSY_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)

# 데이터베이스 파일 이름
DATABASE_FILE_NAME = "translator.db"

# 스키마 변경 목록. 버전 N으로 올리는 SQL 문들이 N-1번째 항목에 들어 있다 (버전 0 → 1, 1 → 2, ...).
# 이미 배포한 항목은 고치지 않고 새 항목을 뒤에 추가한다.
_MIGRATIONS: tuple[tuple[str, ...], ...] = (
    (
        # 콘텐츠별 바운딩 박스. 위치·크기는 Qt 논리 좌표, position은 저장한 순서(색·복원 순서 유지)
        """
        CREATE TABLE region (
            content_id TEXT NOT NULL,
            region_id TEXT NOT NULL,
            position INTEGER NOT NULL,
            x INTEGER NOT NULL,
            y INTEGER NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            direction TEXT NOT NULL,
            auto_enabled INTEGER NOT NULL,
            PRIMARY KEY (content_id, region_id)
        )
        """,
    ),
)
# 이 프로그램이 이해하는 최신 스키마 버전
SCHEMA_VERSION = len(_MIGRATIONS)


def default_database_path(environ: Mapping[str, str] | None = None) -> Path:
    """기본 데이터베이스 경로 (%LOCALAPPDATA%\\MultimodalTranslator\\translator.db).

    매개변수:
        environ: 환경 변수 (None이면 `os.environ`).
    반환값:
        데이터베이스 파일 경로. LOCALAPPDATA가 없으면 사용자 홈 폴더 아래.
    """
    environ = os.environ if environ is None else environ
    base = environ.get("LOCALAPPDATA")
    root = Path(base) if base else Path.home()
    return root / "MultimodalTranslator" / DATABASE_FILE_NAME


class SqliteContentProfileStore(ContentProfileStore):
    """`ContentProfileStore`의 SQLite 구현. UI 스레드에서만 사용한다."""

    def __init__(
        self,
        path: str | Path = ":memory:",
        busy_timeout_seconds: float = STORAGE_BUSY_TIMEOUT_SECONDS,
    ) -> None:
        """데이터베이스를 열고 스키마를 최신으로 만든다.

        매개변수:
            path: 데이터베이스 파일 경로 (기본값은 저장되지 않는 메모리 DB, 테스트용).
            busy_timeout_seconds: 다른 연결이 쓰는 동안 쓰기를 기다리는 최대 시간(초).
        예외:
            StorageError: 파일을 열 수 없거나 손상되었거나 더 새로운 버전일 때.
        """
        # 열린 데이터베이스 연결
        self._connection: sqlite3.Connection | None = None
        try:
            if str(path) != ":memory:":
                Path(path).parent.mkdir(parents=True, exist_ok=True)
            # 자동 커밋 모드로 열고, 쓰기는 `_transaction`으로 명시적으로 묶는다 (DDL도 같은 트랜잭션에 넣기 위해).
            self._connection = sqlite3.connect(
                path, isolation_level=None, timeout=busy_timeout_seconds
            )
            if str(path) != ":memory:":
                # 쓰기 때마다 디스크 동기화를 기다리지 않게 한다 (박스를 끌 때 UI가 끊기지 않도록).
                # WAL에서는 NORMAL이어도 앱이 비정상 종료한 경우 데이터가 깨지지 않는다.
                self._connection.execute("PRAGMA journal_mode=WAL")
                self._connection.execute("PRAGMA synchronous=NORMAL")
            self._migrate()
        except (sqlite3.Error, OSError) as exc:
            self.close()
            logger.exception("데이터베이스를 열지 못했습니다.")
            raise StorageError(
                "저장 파일을 열 수 없습니다. 파일이 손상되었거나 접근 권한이 없을 수 있습니다."
            ) from exc
        except StorageError:
            self.close()
            raise

    def close(self) -> None:
        """연결을 닫는다. 여러 번 불러도 된다."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def load_regions(self, content_id: str) -> list[RegionConfig]:
        """콘텐츠에 저장된 박스 설정 목록 (저장한 순서). 손상된 행은 건너뛴다."""
        rows = self._query(
            "SELECT region_id, x, y, width, height, direction, auto_enabled "
            "FROM region WHERE content_id = ? ORDER BY position",
            (content_id,),
        )
        configs: list[RegionConfig] = []
        for region_id, x, y, width, height, direction, auto_enabled in rows:
            try:
                configs.append(
                    RegionConfig(region_id, x, y, width, height, TranslationDirection(direction), bool(auto_enabled))
                )
            except ValueError:
                logger.warning("저장된 박스 설정이 올바르지 않아 건너뜁니다 (%s).", region_id)
        return configs

    def save_regions(self, content_id: str, configs: Sequence[RegionConfig]) -> None:
        """콘텐츠의 박스 설정 목록을 통째로 바꿔 저장한다 (한 트랜잭션)."""
        try:
            with self._transaction() as connection:
                connection.execute("DELETE FROM region WHERE content_id = ?", (content_id,))
                connection.executemany(
                    "INSERT INTO region "
                    "(content_id, region_id, position, x, y, width, height, direction, auto_enabled) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        (
                            content_id, config.region_id, position, config.left, config.top,
                            config.width, config.height, config.direction.value, int(config.auto_enabled),
                        )
                        for position, config in enumerate(configs)
                    ],
                )
        except sqlite3.Error as exc:
            raise StorageError("박스 설정을 저장하지 못했습니다.") from exc

    def adopt_regions(self, source_content_id: str, target_content_id: str) -> list[RegionConfig]:
        """`source`의 박스를 `target`으로 옮긴다. `target`에 박스가 이미 있으면 옮기지 않는다 (한 트랜잭션)."""
        if source_content_id == target_content_id:
            return []
        try:
            with self._transaction() as connection:
                # 옮길 대상이 비어 있고 가져올 박스가 있을 때만 옮긴다.
                (target_count,) = connection.execute(
                    "SELECT COUNT(*) FROM region WHERE content_id = ?", (target_content_id,)
                ).fetchone()
                if target_count > 0:
                    return []
                connection.execute(
                    "UPDATE region SET content_id = ? WHERE content_id = ?",
                    (target_content_id, source_content_id),
                )
        except sqlite3.Error as exc:
            raise StorageError("박스 설정을 옮기지 못했습니다.") from exc
        return self.load_regions(target_content_id)

    def _migrate(self) -> None:
        """스키마를 최신 버전으로 올린다. 더 새로운 버전이면 `StorageError`."""
        connection = self._require_connection()
        # 파일에 기록된 스키마 버전 (새 파일은 0)
        (version,) = connection.execute("PRAGMA user_version").fetchone()
        if version > SCHEMA_VERSION:
            raise StorageError(
                "저장 파일이 더 새로운 버전의 프로그램에서 만들어졌습니다. 프로그램을 최신 버전으로 업데이트해 주세요."
            )
        for target_version in range(version + 1, SCHEMA_VERSION + 1):
            with self._transaction():
                for statement in _MIGRATIONS[target_version - 1]:
                    connection.execute(statement)
                # PRAGMA는 매개변수를 바인딩할 수 없어 정수를 직접 넣는다 (내부 상수라 안전하다).
                connection.execute(f"PRAGMA user_version = {int(target_version)}")
            logger.info("저장소 스키마를 버전 %d로 올렸습니다.", target_version)

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        """쓰기를 트랜잭션 하나로 묶는다. 블록이나 커밋에서 예외가 나면 되돌린다.

        커밋이 실패해도 SQLite는 트랜잭션을 열어 둔 채로 두므로, 되돌리지 않으면 이후의 모든 쓰기가
        "트랜잭션 안에서 시작할 수 없음"으로 실패한다.

        반환값:
            트랜잭션 안에서 쓸 연결.
        예외:
            StorageError: 저장소가 이미 닫혔을 때. 블록 안과 커밋의 예외는 되돌린 뒤 그대로 다시 던진다.
        """
        connection = self._require_connection()
        connection.execute("BEGIN")
        try:
            yield connection
            connection.execute("COMMIT")
        except BaseException:
            self._rollback(connection)
            raise

    @staticmethod
    def _rollback(connection: sqlite3.Connection) -> None:
        """열려 있는 트랜잭션을 되돌린다. 되돌리지 못해도 원래 예외를 가리지 않도록 기록만 한다."""
        if not connection.in_transaction:
            return
        try:
            connection.execute("ROLLBACK")
        except sqlite3.Error:
            logger.exception("트랜잭션을 되돌리지 못했습니다.")

    def _query(self, sql: str, parameters: tuple) -> list[tuple]:
        """읽기 전용 질의를 실행해 모든 행을 돌려준다. 실패하면 `StorageError`."""
        connection = self._require_connection()
        try:
            return connection.execute(sql, parameters).fetchall()
        except sqlite3.Error as exc:
            raise StorageError("저장된 설정을 읽지 못했습니다.") from exc

    def _require_connection(self) -> sqlite3.Connection:
        """열린 연결을 돌려준다. 이미 닫혔으면 `StorageError`."""
        if self._connection is None:
            raise StorageError("저장소가 이미 닫혔습니다.")
        return self._connection
