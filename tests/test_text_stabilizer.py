"""텍스트 안정화(타자 효과) 판단 테스트 (Qt 없이 실행).

Mock OCR의 한계: 실제 OCR의 글자 흔들림(오인식)은 흉내 내지 않는다. 여기서는 같은 텍스트의
반복, 글자 늘어남, 다른 텍스트로의 교체만 검증한다.
"""

from translator.application.text_stabilizer import TextStabilizer, normalize_text


def test_normalize_collapses_spaces_and_trailing_markers() -> None:
    """공백을 하나로 줄이고 끝의 진행 표시·커서 문자를 지운다."""
    assert normalize_text("  안녕   하세요 ▼ ") == "안녕 하세요"
    assert normalize_text("Hello_") == "Hello"
    assert normalize_text("…") == ""


def test_text_is_stable_after_repeated_observations() -> None:
    """같은 텍스트가 정해진 횟수만큼 연속으로 인식되면 안정으로 본다."""
    stabilizer = TextStabilizer(settle_observations=2, max_wait_seconds=10)

    assert stabilizer.observe("안녕", 0.0) is None
    assert stabilizer.observe("안녕", 0.4) == "안녕"
    assert stabilizer.observe("안녕", 0.8) == "안녕"  # 이후에도 안정 상태 유지


def test_typing_text_is_not_stable_while_growing() -> None:
    """글자가 늘어나는 동안(타자 효과)은 안정으로 보지 않고, 멈추면 안정으로 본다."""
    stabilizer = TextStabilizer(settle_observations=2, max_wait_seconds=10)

    assert stabilizer.observe("안", 0.0) is None
    assert stabilizer.observe("안녕", 0.4) is None
    assert stabilizer.observe("안녕하", 0.8) is None
    assert stabilizer.observe("안녕하세요", 1.2) is None
    assert stabilizer.observe("안녕하세요", 1.6) == "안녕하세요"


def test_cursor_marker_does_not_break_stability() -> None:
    """끝의 커서 문자가 깜빡여도 같은 텍스트로 본다."""
    stabilizer = TextStabilizer(settle_observations=2, max_wait_seconds=10)

    stabilizer.observe("대사입니다▼", 0.0)
    assert stabilizer.observe("대사입니다", 0.4) == "대사입니다"


def test_endlessly_growing_text_is_stabilized_after_max_wait() -> None:
    """글자가 계속 늘어나도 최대 대기 시간이 지나면 현재 텍스트로 안정 처리한다 (영구 보류 방지)."""
    stabilizer = TextStabilizer(settle_observations=2, max_wait_seconds=1.0)
    text = ""
    result = None
    for step in range(10):
        text += "가"
        result = stabilizer.observe(text, step * 0.3)
        if result is not None:
            break

    assert result is not None and step * 0.3 >= 1.0


def test_replaced_text_restarts_waiting() -> None:
    """완전히 다른 텍스트로 바뀌면 기다림을 처음부터 다시 잰다."""
    stabilizer = TextStabilizer(settle_observations=3, max_wait_seconds=1.0)

    stabilizer.observe("첫 대사", 0.0)
    stabilizer.observe("첫 대사가", 0.5)
    # 다른 텍스트: 처음 본 시각이 0.9로 바뀌므로 1.2초에는 아직 최대 대기 시간이 지나지 않았다.
    assert stabilizer.observe("두 번째", 0.9) is None
    assert stabilizer.observe("두 번째 대사", 1.2) is None
    assert stabilizer.last_text == "두 번째 대사"


def test_force_marks_stable_immediately() -> None:
    """사용자가 직접 요청한 번역은 안정화를 기다리지 않는다."""
    stabilizer = TextStabilizer(settle_observations=5, max_wait_seconds=10)

    assert stabilizer.observe("바로 번역", 0.0, force=True) == "바로 번역"
