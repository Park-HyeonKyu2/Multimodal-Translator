"""화면 지문과 변화 판단 테스트 (Qt 없이 실행)."""

from translator.application.change_detection import (
    ChangeTracker,
    ChangeTriggerPolicy,
    Fingerprint,
    compute_fingerprint,
    difference,
    is_changed,
)
from translator.core.models import CapturedImage, ScreenRegion


def solid_image(width: int, height: int, gray: int) -> CapturedImage:
    """단색 이미지를 만든다."""
    pixel = bytes((gray, gray, gray, 255))
    return CapturedImage(ScreenRegion(0, 0, width, height), width, height, pixel * (width * height))


def image_with_bar(width: int, height: int, bar_rows: range) -> CapturedImage:
    """검은 배경에 흰 가로줄(자막을 흉내 냄)이 있는 이미지."""
    data = bytearray()
    for y in range(height):
        gray = 255 if y in bar_rows else 0
        data += bytes((gray, gray, gray, 255)) * width
    return CapturedImage(ScreenRegion(0, 0, width, height), width, height, bytes(data))


def fp(value: int, size: int = 4) -> Fingerprint:
    """모든 표본이 같은 값인 지문을 만든다."""
    return Fingerprint(size, 1, bytes([value] * size))


def test_fingerprint_reads_luminance_on_grid() -> None:
    """격자 지점의 밝기를 표본으로 쓴다."""
    fingerprint = compute_fingerprint(solid_image(20, 10, 200), grid_width=4, grid_height=2)

    assert (fingerprint.grid_width, fingerprint.grid_height) == (4, 2)
    assert fingerprint.samples == bytes([200] * 8)


def test_fingerprint_grid_is_capped_by_image_size() -> None:
    """이미지가 격자보다 작으면 픽셀 수만큼만 표본을 뽑는다."""
    fingerprint = compute_fingerprint(solid_image(3, 2, 0), grid_width=96, grid_height=32)

    assert (fingerprint.grid_width, fingerprint.grid_height) == (3, 2)


def test_new_text_bar_produces_large_difference() -> None:
    """자막 한 줄이 나타나면 차이가 판단 기준보다 크다."""
    blank = compute_fingerprint(image_with_bar(200, 40, range(0)))
    subtitle = compute_fingerprint(image_with_bar(200, 40, range(15, 25)))

    assert difference(blank, subtitle) > 6.0
    assert difference(blank, blank) == 0.0


def test_difference_of_mismatched_grids_is_maximal() -> None:
    """격자 크기가 다른 지문은 완전히 다른 화면으로 본다."""
    assert difference(fp(0, size=4), fp(0, size=5)) == 255.0


def test_tracker_translates_once_after_screen_settles() -> None:
    """화면이 멈추면 한 번만 번역하고, 잡음 수준 차이는 무시한다."""
    tracker = ChangeTracker(settle_samples=1)

    assert tracker.observe(fp(10)) is False  # 첫 관찰: 아직 멈췄는지 모름
    assert tracker.observe(fp(10)) is True  # 멈춤 → 번역
    assert tracker.observe(fp(10)) is False  # 같은 화면은 다시 번역하지 않음
    assert tracker.observe(fp(11)) is False  # 잡음 수준 차이


def test_tracker_waits_while_screen_keeps_changing() -> None:
    """화면이 계속 바뀌는 동안은 기다렸다가 멈춘 뒤 번역한다."""
    tracker = ChangeTracker(settle_samples=1)
    tracker.observe(fp(10))
    tracker.observe(fp(10))

    assert tracker.observe(fp(100)) is False  # 바뀌는 중
    assert tracker.observe(fp(200)) is False  # 계속 바뀌는 중
    assert tracker.observe(fp(200)) is True  # 멈춤 → 새 대사 번역


def test_tracker_skips_returning_to_already_translated_screen() -> None:
    """잠깐 다른 화면이 보였다가 원래 화면으로 돌아오면 다시 번역하지 않는다."""
    tracker = ChangeTracker(settle_samples=1)
    tracker.observe(fp(10))
    assert tracker.observe(fp(10)) is True

    tracker.observe(fp(100))  # 잠깐 다른 화면 (멈추기 전)
    tracker.observe(fp(10))  # 원래 화면으로 복귀

    assert tracker.observe(fp(10)) is False


def test_tracker_respects_settle_samples() -> None:
    """멈춤 판단 횟수만큼 같은 화면이 이어져야 번역한다."""
    tracker = ChangeTracker(settle_samples=3)
    results = [tracker.observe(fp(50)) for _ in range(5)]

    assert results == [False, False, False, True, False]


def test_reset_forgets_translated_screen() -> None:
    """reset 후에는 같은 화면도 새로 번역한다."""
    tracker = ChangeTracker(settle_samples=1)
    tracker.observe(fp(10))
    tracker.observe(fp(10))

    tracker.reset()

    assert tracker.observe(fp(10)) is False
    assert tracker.observe(fp(10)) is True


def test_mark_translated_prevents_translating_same_screen() -> None:
    """수동 번역으로 기록한 화면은 자동 감지가 다시 번역하지 않는다."""
    tracker = ChangeTracker(settle_samples=1)
    tracker.mark_translated(fp(10))

    assert tracker.observe(fp(10)) is False
    assert tracker.observe(fp(10)) is False  # 멈췄지만 이미 번역한 화면
    tracker.observe(fp(200))
    assert tracker.observe(fp(200)) is True  # 다른 화면은 번역


def test_restart_keeps_translated_screen_but_reset_clears_it() -> None:
    """자동 감지를 다시 켤 때(restart)는 기록을 유지하고, 박스가 바뀔 때(reset)는 비운다."""
    tracker = ChangeTracker(settle_samples=1)
    tracker.mark_translated(fp(10))

    tracker.restart()
    tracker.observe(fp(10))
    assert tracker.observe(fp(10)) is False

    tracker.reset()
    tracker.observe(fp(10))
    assert tracker.observe(fp(10)) is True


# --- ChangeTriggerPolicy: 화면이 멈추기를 기다리지 않고 간격을 두어 OCR 요청 -----------------


def trigger_policy(min_gap: int = 2) -> ChangeTriggerPolicy:
    """표본 하나라도 크게 바뀌면 "바뀜"으로 보는 정책."""
    return ChangeTriggerPolicy(pixel_delta=40, changed_ratio_threshold=0.01, min_sample_gap=min_gap)


def test_trigger_policy_requests_first_screen_and_skips_same_screen() -> None:
    """처음 화면은 바로 요청하고, 같은 화면이 이어지면 다시 요청하지 않는다."""
    policy = trigger_policy()

    assert policy.observe(fp(0)) is True
    assert [policy.observe(fp(0)) for _ in range(5)] == [False] * 5


def test_trigger_policy_keeps_requesting_while_screen_keeps_changing() -> None:
    """화면이 계속 바뀌어도(움직이는 배경) 최소 간격마다 요청한다 (영구 보류 없음)."""
    policy = trigger_policy(min_gap=2)

    # 매 표본이 직전에 요청한 화면과 다르다.
    decisions = [policy.observe(fp(v)) for v in (0, 100, 200, 50, 150, 250, 0, 100)]

    assert decisions == [True, False, True, False, True, False, True, False]


def test_trigger_policy_requests_final_screen_after_changes_stop() -> None:
    """바뀌는 도중 건너뛴 마지막 화면은 화면이 멈춘 뒤 한 번 더 요청한다."""
    policy = trigger_policy(min_gap=3)
    policy.observe(fp(0))
    assert policy.observe(fp(255)) is False  # 간격이 아직 안 됨

    later = [policy.observe(fp(255)) for _ in range(3)]

    # 최소 간격(표본 3개)이 지나면 마지막 화면을 한 번만 더 요청한다.
    assert later == [False, True, False]


def test_trigger_policy_mark_and_reset() -> None:
    """수동 OCR로 기록한 화면은 요청하지 않고, reset하면 기록을 버린다."""
    policy = trigger_policy()
    policy.mark_translated(fp(0))
    assert policy.observe(fp(0)) is False

    policy.reset()
    assert policy.observe(fp(0)) is True


def test_area_fingerprint_reads_only_that_area() -> None:
    """영역을 주면 이미지 중 그 부분만 표본으로 쓴다 (텍스트 자리 변화 판단용)."""
    image = image_with_bar(40, 20, range(10, 20))  # 아래 절반이 흰색

    top = compute_fingerprint(image, 4, 2, ScreenRegion(0, 0, 40, 10))
    bottom = compute_fingerprint(image, 4, 2, ScreenRegion(0, 10, 40, 10))

    assert set(top.samples) == {0} and set(bottom.samples) == {255}
    assert is_changed(top, bottom)
    assert not is_changed(top, compute_fingerprint(image, 4, 2, ScreenRegion(5, 0, 30, 9)))


def test_area_fingerprint_of_empty_image_is_safe() -> None:
    """픽셀 데이터가 없는 이미지도 오류 없이 지문을 만든다."""
    image = CapturedImage(ScreenRegion(0, 0, 10, 10), 10, 10, b"")

    assert compute_fingerprint(image, 4, 2, ScreenRegion(2, 2, 4, 4)).samples == bytes(8)
