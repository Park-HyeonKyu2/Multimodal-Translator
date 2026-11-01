"""식별한 콘텐츠 정보.

"지금 사용자가 무엇을 보고 있는가"에 대한 식별 결과를 담는다. 이 값의 `content_id`가
`SessionContext.content_id`가 되어 박스 설정과 문맥 기억을 콘텐츠별로 나누는 기준이 된다.

식별 근거(`IdentificationMethod`)와 확신 정도는 따로 둔다. 확신 정도(confidence)의 보정 방법과
임계값은 아직 정해지지 않았으므로(07 의사결정 기록의 미결정 항목) 지금은 근거만 기록하고,
임의의 점수를 만들어 넣지 않는다.
"""

from dataclasses import dataclass
from enum import Enum


class IdentificationMethod(Enum):
    """콘텐츠를 어떤 근거로 식별했는지."""

    # 활성 창을 소유한 프로세스(실행 파일 이름)로 식별
    PROCESS_NAME = "process_name"
    # 사용자가 직접 고름 (식별에 실패했을 때의 수동 선택)
    MANUAL = "manual"


@dataclass(frozen=True)
class ContentIdentity:
    """식별한 콘텐츠 하나.

    - `content_id`: 같은 게임·영상이면 같은 값. 박스 설정과 문맥 기억을 나누는 기준
    - `display_name`: 사용자에게 보여 줄 이름 (창 제목처럼 개인정보가 될 수 있는 값은 쓰지 않는다)
    - `method`: 식별 근거
    """

    content_id: str
    display_name: str
    method: IdentificationMethod

    def __post_init__(self) -> None:
        """빈 ID와 빈 이름을 거부한다 (식별되지 않은 콘텐츠가 조용히 섞이는 것을 막는다)."""
        if not self.content_id.strip() or not self.display_name.strip():
            raise ValueError("콘텐츠 ID와 표시 이름은 비어 있을 수 없습니다.")
