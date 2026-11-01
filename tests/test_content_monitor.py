"""콘텐츠 감시기(`ContentMonitor`) 테스트: 실제 Qt 타이머와 워커 스레드로 확정 알림, 일시 전환 무시,
연속 실패 중지, 멈춘 뒤 늦은 결과 무시를 확인한다 (식별기는 가짜)."""

import threading

from translator.adapters.content.mock import ScriptedContentDetector
from translator.application.content_monitor import DETECTION_STOPPED_MESSAGE, ContentMonitor
from translator.application.content_switch import ContentSwitchPolicy
from translator.core.content import ContentIdentity, IdentificationMethod
from translator.core.interfaces import ContentDetector

# 테스트에서 감시 간격 (밀리초). 실제 값보다 훨씬 짧게 해 테스트를 빠르게 한다.
FAST_INTERVAL_MS = 5
# 시그널을 기다리는 최대 시간 (밀리초)
WAIT_TIMEOUT_MS = 3000


def app(name: str) -> ContentIdentity:
    """테스트용 프로세스 기반 콘텐츠."""
    return ContentIdentity(f"app:{name}.exe", name, IdentificationMethod.PROCESS_NAME)


class RaisingDetector(ContentDetector):
    """호출할 때마다 예외를 던지는 식별기 (예상 밖 오류 흉내)."""

    def detect(self) -> ContentIdentity | None:
        """항상 RuntimeError를 던진다."""
        raise RuntimeError("detector failure")


class BlockingDetector(ContentDetector):
    """풀어 줄 때까지 식별을 끝내지 않는 식별기 (늦게 도착하는 결과 흉내)."""

    def __init__(self, result: ContentIdentity) -> None:
        """돌려줄 결과를 받는다."""
        # 식별이 시작되었음을 알리는 신호와, 끝내도 된다는 신호
        self.started = threading.Event()
        self.release = threading.Event()
        self._result = result

    def detect(self) -> ContentIdentity | None:
        """시작을 알린 뒤 풀어 줄 때까지 기다렸다가 결과를 돌려준다."""
        self.started.set()
        self.release.wait(timeout=5)
        return self._result


def make_monitor(detector: ContentDetector, confirm_samples: int = 2, **kwargs) -> ContentMonitor:
    """짧은 간격과 주어진 확인 횟수로 감시기를 만든다."""
    return ContentMonitor(
        detector,
        policy=ContentSwitchPolicy(confirm_samples=confirm_samples),
        interval_ms=FAST_INTERVAL_MS,
        **kwargs,
    )


def test_confirmed_content_is_reported(qtbot) -> None:
    """같은 콘텐츠가 연속 관찰되면 content_changed로 한 번 알린다."""
    monitor = make_monitor(ScriptedContentDetector([app("a")]))

    with qtbot.waitSignal(monitor.content_changed, timeout=WAIT_TIMEOUT_MS) as blocker:
        monitor.start()
    monitor.shutdown()

    assert blocker.args[0].content_id == "app:a.exe"


def test_content_is_reported_only_once_while_it_stays_the_same(qtbot) -> None:
    """콘텐츠가 그대로인 동안에는 다시 알리지 않는다."""
    monitor = make_monitor(ScriptedContentDetector([app("a")]))
    reported: list[ContentIdentity] = []
    monitor.content_changed.connect(reported.append)

    monitor.start()
    qtbot.waitUntil(lambda: len(reported) == 1, timeout=WAIT_TIMEOUT_MS)
    qtbot.wait(100)
    monitor.shutdown()

    assert [identity.content_id for identity in reported] == ["app:a.exe"]


def test_brief_switch_is_not_reported(qtbot) -> None:
    """잠깐 다른 창이 관찰되고 원래 콘텐츠로 돌아오면 알리지 않는다."""
    detector = ScriptedContentDetector([app("a"), app("a"), app("b"), app("a")])
    monitor = make_monitor(detector, confirm_samples=2)
    reported: list[ContentIdentity] = []
    monitor.content_changed.connect(reported.append)

    monitor.start()
    qtbot.waitUntil(lambda: len(reported) == 1, timeout=WAIT_TIMEOUT_MS)
    qtbot.wait(100)
    monitor.shutdown()

    # a가 먼저 확정된 뒤, b는 한 번만 관찰되고 a로 돌아왔으므로 더 알리지 않는다.
    assert [identity.content_id for identity in reported] == ["app:a.exe"]


def test_repeated_unexpected_errors_stop_monitoring_and_notify(qtbot) -> None:
    """식별이 예외로 연속 실패하면 감시를 멈추고 사용자에게 보여 줄 메시지를 보낸다 (무한 재시도 방지)."""
    monitor = make_monitor(RaisingDetector(), max_consecutive_failures=3)

    with qtbot.waitSignal(monitor.stopped_by_error, timeout=WAIT_TIMEOUT_MS) as blocker:
        monitor.start()
    monitor.shutdown()

    assert blocker.args == [DETECTION_STOPPED_MESSAGE]
    assert not monitor.is_running


def test_result_arriving_after_stop_is_ignored(qtbot) -> None:
    """멈춘 뒤에 끝난 식별 결과는 버린다 (이전 루프의 결과가 콘텐츠를 바꾸면 안 된다)."""
    detector = BlockingDetector(app("a"))
    monitor = make_monitor(detector, confirm_samples=1)
    reported: list[ContentIdentity] = []
    monitor.content_changed.connect(reported.append)

    monitor.start()
    assert detector.started.wait(timeout=3)
    monitor.stop()
    detector.release.set()
    monitor.shutdown()
    qtbot.wait(100)

    assert reported == []


def test_start_twice_and_stop_without_start_are_safe(qtbot) -> None:
    """이미 시작했으면 다시 시작해도, 시작하지 않았으면 멈춰도 문제가 없다."""
    monitor = make_monitor(ScriptedContentDetector([None]))

    monitor.stop()
    monitor.start()
    monitor.start()

    assert monitor.is_running
    monitor.shutdown()
    assert not monitor.is_running


def test_unidentified_results_never_report_change(qtbot) -> None:
    """식별하지 못하는 동안(None)에는 콘텐츠 변경을 알리지 않는다."""
    monitor = make_monitor(ScriptedContentDetector([None]), confirm_samples=1)
    reported: list[ContentIdentity] = []
    monitor.content_changed.connect(reported.append)

    monitor.start()
    qtbot.wait(100)
    monitor.shutdown()

    assert reported == []
