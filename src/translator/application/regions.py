"""바운딩 박스(번역 영역)별 설정과 처리 상태.

설정(`RegionConfig`)은 저장·복원 대상이고, 처리 상태(`RegionRuntimeState`)는 실행 중에만
쓰는 값이다. 둘 다 Qt에 의존하지 않으므로 설정 저장소나 향후 콘텐츠별 프로필(Phase 3)에서
그대로 쓸 수 있다.
"""

import re
from dataclasses import dataclass, field, replace
from enum import Enum

from translator.core.models import DEFAULT_TRANSLATION_DIRECTION, TranslationDirection

# 박스 ID 형식 (예: "box-3"). 저장된 ID와 새로 만드는 ID가 겹치지 않게 하는 데 쓴다.
_ID_PATTERN = re.compile(r"^box-(\d+)$")


@dataclass(frozen=True)
class RegionConfig:
    """바운딩 박스 하나의 저장되는 설정.

    위치와 크기는 캡처할 내부 영역의 **Qt 논리 좌표**다 (화면에 박스를 다시 놓는 데 쓴다).
    캡처할 때는 그때의 모니터 배율로 물리 좌표를 다시 계산한다.
    """

    region_id: str
    left: int
    top: int
    width: int
    height: int
    direction: TranslationDirection = DEFAULT_TRANSLATION_DIRECTION
    auto_enabled: bool = False

    def __post_init__(self) -> None:
        """크기가 0 이하인 설정을 거부한다."""
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"박스 크기는 양수여야 합니다: {self.width}x{self.height}")

    def with_rect(self, left: int, top: int, width: int, height: int) -> "RegionConfig":
        """위치와 크기만 바꾼 새 설정."""
        return replace(self, left=left, top=top, width=width, height=height)


class RequestStatus(Enum):
    """박스의 마지막 번역 요청 상태."""

    IDLE = "idle"
    PENDING = "pending"
    DONE = "done"
    NO_TEXT = "no_text"
    FAILED = "failed"


@dataclass
class RegionRuntimeState:
    """박스 하나의 실행 중 상태 (저장하지 않음)."""

    status: RequestStatus = RequestStatus.IDLE
    # 마지막으로 요청한 번역의 요청 ID (결과가 이 박스 것인지 확인할 때 참고)
    last_request_id: int | None = None
    # 마지막으로 처리한 원문 텍스트들 (화면 위치 순서)
    last_texts: tuple[str, ...] = field(default_factory=tuple)
    # 마지막 실패 메시지
    last_error: str | None = None
    # 편집 중 위치/크기가 바뀌어, 편집을 마치면 다시 번역해야 하는지 여부
    needs_translation: bool = False


class RegionIdAllocator:
    """박스 ID를 만든다. 저장된 박스를 복원한 뒤에도 ID가 겹치지 않게 한다."""

    def __init__(self) -> None:
        """1번부터 발급한다."""
        self._next = 1

    def reserve(self, region_id: str) -> None:
        """이미 쓰고 있는 ID(복원된 박스)를 알려 다음 ID가 그보다 크게 한다."""
        match = _ID_PATTERN.match(region_id)
        if match:
            self._next = max(self._next, int(match.group(1)) + 1)

    def allocate(self) -> str:
        """새 박스 ID를 발급한다."""
        region_id = f"box-{self._next}"
        self._next += 1
        return region_id
