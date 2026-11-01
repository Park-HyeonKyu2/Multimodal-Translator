"""스레드 간에 공유되는 협력적 취소 토큰."""

import threading

from translator.core.errors import OperationCancelledError


class CancellationToken:
    """백그라운드 작업이 주기적으로 확인하는 취소 신호."""

    def __init__(self) -> None:
        """취소되지 않은 상태로 시작한다."""
        self._event = threading.Event()

    def cancel(self) -> None:
        """취소를 알린다. 여러 번 호출해도 된다."""
        self._event.set()

    @property
    def is_cancelled(self) -> bool:
        """취소되었는지 여부."""
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        """취소되었으면 `OperationCancelledError`를 던진다."""
        if self._event.is_set():
            raise OperationCancelledError("작업이 취소되었습니다.")

    def wait(self, seconds: float) -> None:
        """최대 `seconds` 동안 대기하고, 그 사이 취소되면 즉시 예외를 던진다."""
        if self._event.wait(seconds):
            raise OperationCancelledError("작업이 취소되었습니다.")
