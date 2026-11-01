"""콘텐츠·세션 식별 정보.

세션은 "같은 콘텐츠를 보는 하나의 연속된 시간"이다 (게임 한 번 실행, 영상 하나 재생 등).
세션이 바뀌면(창 전환, 게임 종료, 다른 영상으로 이동) 이전 세션에서 시작한 작업의 결과는
새 세션에 섞이면 안 된다. 요청과 결과에 이 정보를 실어 보내, 늦게 도착한 결과를 알아보고
버리며, 이후 문맥 기억을 콘텐츠별로 나누는 기준으로도 쓴다.

콘텐츠를 실제로 식별하는 일(활성 창·프로세스 확인)은 이 모듈이 하지 않는다. 식별하지 못한
동안은 `DEFAULT_CONTENT_ID`를 쓴다.
"""

import uuid
from dataclasses import dataclass

# 콘텐츠를 식별하지 못했을 때(또는 식별 기능이 없는 동안) 쓰는 콘텐츠 ID
DEFAULT_CONTENT_ID = "default"
# 세션 ID의 길이 (uuid 16진수 앞부분). 한 번의 프로그램 실행 안에서 겹치지 않으면 충분하다.
_SESSION_ID_LENGTH = 12


@dataclass(frozen=True)
class SessionContext:
    """콘텐츠 하나를 보는 세션 하나의 식별 정보.

    - `content_id`: 무엇을 보는지 (같은 게임·영상이면 같은 값). 문맥 기억과 박스 설정을 나누는 기준
    - `session_id`: 언제부터 언제까지인지 (같은 콘텐츠라도 다시 시작하면 새 값). 늦게 도착한
      결과를 알아보는 기준
    """

    content_id: str
    session_id: str

    def __post_init__(self) -> None:
        """빈 ID를 거부한다 (식별되지 않는 세션이 조용히 섞이는 것을 막는다)."""
        if not self.content_id or not self.session_id:
            raise ValueError("콘텐츠 ID와 세션 ID는 비어 있을 수 없습니다.")


def new_session(content_id: str = DEFAULT_CONTENT_ID, session_id: str | None = None) -> SessionContext:
    """새 세션을 만든다.

    매개변수:
        content_id: 콘텐츠 ID.
        session_id: 세션 ID. None이면 새로 만든다 (테스트에서는 값을 정해 줄 수 있다).
    반환값:
        새 `SessionContext`.
    예외:
        ValueError: ID가 비어 있을 때.
    """
    return SessionContext(content_id, session_id or uuid.uuid4().hex[:_SESSION_ID_LENGTH])
