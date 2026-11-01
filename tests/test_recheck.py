"""OCR 후속 확인·재검사 예약 테스트 (Qt 없이, 가짜 시계 사용)."""

from translator.application.recheck import ObservationScheduler


class FakeClock:
    """테스트에서 직접 넘기는 시계."""

    def __init__(self) -> None:
        """0초에서 시작한다."""
        self.now = 0.0

    def __call__(self) -> float:
        """현재 시각."""
        return self.now


def make_scheduler(max_attempts: int = 3) -> tuple[ObservationScheduler, FakeClock]:
    """후속 0.4초, 재검사 1초부터 최대 4초까지 두 배씩 늘리는 예약기를 만든다."""
    clock = FakeClock()
    scheduler = ObservationScheduler(
        clock=clock,
        followup_seconds=0.4,
        recheck_base_seconds=1.0,
        recheck_max_seconds=4.0,
        max_attempts=max_attempts,
    )
    return scheduler, clock


def test_unstable_text_schedules_short_followup() -> None:
    """안정되지 않은 텍스트가 있으면 짧은 간격 뒤 다시 OCR한다."""
    scheduler, clock = make_scheduler()

    scheduler.after_result("box-1", unstable=True, needs_recheck=False)

    assert scheduler.due_at("box-1") == 0.4
    clock.now = 0.3
    assert scheduler.pop_due() == []
    clock.now = 0.4
    assert scheduler.pop_due() == ["box-1"]
    assert scheduler.pop_due() == []  # 꺼낸 예약은 지워진다


def test_recheck_backs_off_and_stops_after_max_attempts() -> None:
    """재검사 간격은 두 배씩 늘고(상한 있음), 최대 횟수 뒤에는 더 예약하지 않는다."""
    scheduler, clock = make_scheduler(max_attempts=4)
    delays = []
    for _ in range(5):
        scheduler.after_result("box-1", unstable=False, needs_recheck=True)
        due = scheduler.due_at("box-1")
        delays.append(None if due is None else due - clock.now)
        if due is not None:
            clock.now = due
            scheduler.pop_due()

    assert delays == [1.0, 2.0, 4.0, 4.0, None]
    assert scheduler.attempts("box-1") == 4


def test_screen_change_restarts_recheck_count() -> None:
    """화면이 바뀌면 재검사 횟수를 처음부터 센다."""
    scheduler, _ = make_scheduler(max_attempts=1)
    scheduler.after_result("box-1", unstable=False, needs_recheck=True)
    scheduler.after_result("box-1", unstable=False, needs_recheck=True)
    assert scheduler.due_at("box-1") is None  # 한도 도달

    scheduler.on_screen_changed("box-1")
    scheduler.after_result("box-1", unstable=False, needs_recheck=True)

    assert scheduler.due_at("box-1") is not None


def test_good_result_clears_schedule_and_count() -> None:
    """안정되고 놓친 텍스트도 없으면 예약과 재검사 횟수를 지운다."""
    scheduler, _ = make_scheduler()
    scheduler.after_result("box-1", unstable=False, needs_recheck=True)

    scheduler.after_result("box-1", unstable=False, needs_recheck=False)

    assert scheduler.due_at("box-1") is None and scheduler.attempts("box-1") == 0


def test_sources_are_independent_and_cancel_removes_one() -> None:
    """출처별 예약은 서로 독립이며, 취소하면 그 출처만 지운다."""
    scheduler, clock = make_scheduler()
    scheduler.after_result("box-1", unstable=True, needs_recheck=False)
    scheduler.after_result("fullscreen", unstable=True, needs_recheck=False)

    scheduler.cancel("box-1")
    clock.now = 1.0

    assert scheduler.pop_due() == ["fullscreen"]
