"""화면 변화가 없을 때의 OCR 재검사와 안정화 후속 OCR 예약.

화면 픽셀이 그대로면 자동 감지는 OCR을 다시 요청하지 않는다. 그래서 OCR이 한 번 텍스트를
놓치면 그 텍스트를 영영 인식하지 못할 수 있다. 이 모듈은 출처(박스, 전체 화면)별로
다음 OCR 시각을 정한다.

- 후속 OCR: 아직 안정되지 않은 텍스트가 있으면 짧은 간격 뒤 다시 OCR한다.
- 재검사: 결과가 비었거나 OCR이 텍스트를 놓쳤으면, 간격을 두 배씩 늘리며 최대 횟수까지만
  다시 OCR한다. 화면이 바뀌면 횟수를 처음부터 센다 (불필요한 반복 호출 제한).

Qt에 의존하지 않으며 시계를 주입할 수 있어 단독으로 테스트할 수 있다.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass

from translator.tuning import (
    OCR_RECHECK_BASE_MS,
    OCR_RECHECK_MAX_ATTEMPTS,
    OCR_RECHECK_MAX_MS,
    TEXT_FOLLOWUP_MS,
)


@dataclass
class _SourceState:
    """출처 하나의 예약 상태."""

    # 다음 OCR 예정 시각 (없으면 None)
    due_at: float | None = None
    # 같은 화면에서 지금까지 재검사한 횟수
    attempts: int = 0


class ObservationScheduler:
    """출처별 후속 OCR·재검사 시각을 관리한다. UI 스레드에서만 사용한다."""

    def __init__(
        self,
        clock: Callable[[], float] = time.monotonic,
        followup_seconds: float = TEXT_FOLLOWUP_MS / 1000,
        recheck_base_seconds: float = OCR_RECHECK_BASE_MS / 1000,
        recheck_max_seconds: float = OCR_RECHECK_MAX_MS / 1000,
        max_attempts: int = OCR_RECHECK_MAX_ATTEMPTS,
    ) -> None:
        """시계와 후속 OCR 간격, 재검사 간격(처음·최대), 최대 재검사 횟수를 정한다."""
        self._clock = clock
        self._followup = followup_seconds
        self._base = recheck_base_seconds
        self._max_interval = recheck_max_seconds
        self._max_attempts = max_attempts
        self._states: dict[str, _SourceState] = {}

    def after_result(self, source_id: str, *, unstable: bool, needs_recheck: bool) -> None:
        """OCR 결과를 반영해 다음 OCR 시각을 정한다.

        - `unstable`: 안정되지 않은 텍스트가 있음 → 짧은 후속 OCR
        - `needs_recheck`: 결과가 비었거나 OCR이 놓친 텍스트가 있음 → 간격을 늘리며 재검사
        - 둘 다 아니면 재검사 횟수를 0으로 되돌리고 예약을 지운다.
        """
        state = self._states.setdefault(source_id, _SourceState())
        now = self._clock()
        if unstable:
            state.due_at = now + self._followup
            return
        if needs_recheck:
            if state.attempts >= self._max_attempts:
                # 같은 화면에서 충분히 다시 확인했다: 화면이 바뀔 때까지 더 하지 않는다.
                state.due_at = None
                return
            delay = min(self._base * (2 ** state.attempts), self._max_interval)
            state.attempts += 1
            state.due_at = now + delay
            return
        state.attempts = 0
        state.due_at = None

    def on_screen_changed(self, source_id: str) -> None:
        """화면이 바뀌어 새로 OCR하게 되면 재검사 횟수를 처음부터 센다."""
        state = self._states.get(source_id)
        if state is not None:
            state.attempts = 0

    def cancel(self, source_id: str) -> None:
        """출처의 예약과 기록을 지운다 (박스 삭제, 전체 화면 끄기)."""
        self._states.pop(source_id, None)

    def clear(self) -> None:
        """모든 출처의 예약과 기록을 지운다 (세션이 바뀔 때)."""
        self._states.clear()

    def pop_due(self) -> list[str]:
        """예정 시각이 지난 출처 목록을 꺼낸다 (꺼낸 예약은 지워진다)."""
        now = self._clock()
        due = [s for s, state in self._states.items() if state.due_at is not None and state.due_at <= now]
        for source_id in due:
            self._states[source_id].due_at = None
        return due

    def due_at(self, source_id: str) -> float | None:
        """출처의 다음 OCR 예정 시각 (테스트·진단용)."""
        state = self._states.get(source_id)
        return state.due_at if state else None

    def attempts(self, source_id: str) -> int:
        """출처가 같은 화면에서 재검사한 횟수 (테스트·진단용)."""
        state = self._states.get(source_id)
        return state.attempts if state else 0
