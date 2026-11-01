"""테스트와 화면 확인용 Mock 콘텐츠 식별기."""

from collections.abc import Iterable

from translator.core.content import ContentIdentity
from translator.core.interfaces import ContentDetector


class ScriptedContentDetector(ContentDetector):
    """미리 정한 식별 결과를 순서대로 돌려준다. 마지막 결과 뒤에는 마지막 값을 계속 돌려준다.

    창 전환, 식별 실패(None) 같은 상황을 실제 창 없이 흉내 낼 때 쓴다.
    """

    def __init__(self, results: Iterable[ContentIdentity | None]) -> None:
        """돌려줄 결과 목록을 받는다.

        매개변수:
            results: 호출 순서대로 돌려줄 식별 결과 (None은 식별 실패).
        예외:
            ValueError: 결과가 하나도 없을 때.
        """
        # 아직 돌려주지 않은 결과 목록 (앞에서부터 하나씩 꺼낸다)
        self._remaining = list(results)
        if not self._remaining:
            raise ValueError("식별 결과가 하나 이상 필요합니다.")
        # 마지막으로 돌려준 결과 (목록을 다 쓰면 이 값을 반복한다)
        self._last: ContentIdentity | None = None

    def detect(self) -> ContentIdentity | None:
        """다음 결과를 돌려준다. 목록을 다 썼으면 마지막 결과를 다시 돌려준다."""
        if self._remaining:
            self._last = self._remaining.pop(0)
        return self._last
