"""콘텐츠별 프로필(지금은 바운딩 박스 설정)의 저장소 인터페이스.

저장 방식(SQLite 등)은 adapters가 구현하고, 화면(UI)은 이 인터페이스에만 의존한다. 번역 엔진과는
무관하다 (문맥 기억도 같은 저장소 위에 얹을 수 있게 콘텐츠 ID를 기준으로 나눈다).
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from translator.application.regions import RegionConfig


class ContentProfileStore(ABC):
    """콘텐츠 ID별로 박스 설정 목록을 저장한다. 실패 시 `StorageError`를 던진다."""

    @abstractmethod
    def load_regions(self, content_id: str) -> list[RegionConfig]:
        """콘텐츠에 저장된 박스 설정 목록 (저장한 순서). 없으면 빈 목록."""

    @abstractmethod
    def save_regions(self, content_id: str, configs: Sequence[RegionConfig]) -> None:
        """콘텐츠의 박스 설정 목록을 통째로 바꿔 저장한다 (줄어든 만큼 이전 항목은 지운다)."""

    def close(self) -> None:
        """저장소가 쥔 자원을 놓는다 (기본 구현은 할 일이 없다). 여러 번 불러도 된다."""

    @abstractmethod
    def adopt_regions(self, source_content_id: str, target_content_id: str) -> list[RegionConfig]:
        """`source`의 박스를 `target`으로 옮긴다. `target`에 이미 박스가 있으면 아무것도 하지 않는다.

        반환값:
            옮겨진 박스 설정 목록. 옮기지 않았으면 빈 목록.
        """
