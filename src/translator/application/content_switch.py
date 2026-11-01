"""콘텐츠가 바뀌었는지 판단하는 정책 (창 전환 디바운스).

활성 창은 Alt-Tab 중간 화면, 작업 표시줄 클릭, 잠깐 다른 창을 확인하는 동작 때문에 짧게 바뀌었다
돌아온다. 관찰될 때마다 세션을 바꾸면 진행 중인 번역이 계속 취소되고 말풍선이 지워지므로, 새
콘텐츠가 여러 번 연속으로 관찰된 뒤에만 "바뀌었다"고 판단한다.

Qt에 의존하지 않는 순수 로직이라 창이나 타이머 없이 테스트할 수 있다.
"""

import logging

from translator.core.content import ContentIdentity
from translator.core.session import DEFAULT_CONTENT_ID
from translator.tuning import CONTENT_SWITCH_CONFIRM_SAMPLES

logger = logging.getLogger(__name__)


class ContentSwitchPolicy:
    """관찰 결과를 받아 콘텐츠가 바뀌었는지 판단한다.

    - 식별하지 못한 관찰(None)은 판단에 쓰지 않는다. 현재 콘텐츠도, 후보의 연속 횟수도 바꾸지 않는다.
    - 현재 콘텐츠와 같은 관찰이 나오면 후보를 버린다 (잠깐 다른 창을 보고 돌아온 경우).
    - 현재와 다른 같은 콘텐츠가 `confirm_samples`번 연속 관찰되면 그 콘텐츠로 바뀐 것으로 본다.
    """

    def __init__(
        self,
        confirm_samples: int = CONTENT_SWITCH_CONFIRM_SAMPLES,
        initial_content_id: str = DEFAULT_CONTENT_ID,
    ) -> None:
        """확인에 필요한 연속 관찰 횟수와 처음 콘텐츠 ID를 받는다.

        매개변수:
            confirm_samples: 새 콘텐츠를 확정하는 데 필요한 연속 관찰 횟수 (1 이상).
            initial_content_id: 처음 상태의 콘텐츠 ID (식별 전에는 기본 콘텐츠).
        예외:
            ValueError: 횟수가 1보다 작을 때.
        """
        if confirm_samples < 1:
            raise ValueError("확인에 필요한 연속 관찰 횟수는 1 이상이어야 합니다.")
        # 새 콘텐츠를 확정하는 데 필요한 연속 관찰 횟수
        self._confirm_samples = confirm_samples
        # 지금 유효한 콘텐츠 ID
        self._current_content_id = initial_content_id
        # 바뀔 가능성이 있는 후보 콘텐츠 (없으면 None)
        self._candidate: ContentIdentity | None = None
        # 후보가 연속으로 관찰된 횟수
        self._candidate_count = 0

    @property
    def current_content_id(self) -> str:
        """지금 유효한 콘텐츠 ID."""
        return self._current_content_id

    def observe(self, identity: ContentIdentity | None) -> ContentIdentity | None:
        """관찰 결과 하나를 반영한다.

        매개변수:
            identity: 이번에 식별한 콘텐츠. 식별하지 못했으면 None.
        반환값:
            콘텐츠가 바뀐 것으로 확정되면 새 콘텐츠. 아니면 None.
        """
        if identity is None:
            return None
        if identity.content_id == self._current_content_id:
            self._candidate = None
            self._candidate_count = 0
            return None
        if self._candidate is not None and self._candidate.content_id == identity.content_id:
            self._candidate_count += 1
        else:
            self._candidate_count = 1
        self._candidate = identity
        if self._candidate_count < self._confirm_samples:
            return None
        self._current_content_id = identity.content_id
        self._candidate = None
        self._candidate_count = 0
        logger.info("콘텐츠 변경 확정 (%d회 연속 관찰)", self._confirm_samples)
        return identity

    def reset(self, content_id: str = DEFAULT_CONTENT_ID) -> None:
        """현재 콘텐츠를 다시 정하고 후보를 버린다 (사용자가 세션을 직접 바꾼 경우 등).

        매개변수:
            content_id: 새로 유효한 콘텐츠 ID.
        """
        self._current_content_id = content_id
        self._candidate = None
        self._candidate_count = 0
