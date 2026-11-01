"""박스 하나의 번역문들을 말풍선 하나의 텍스트로 합치는 규칙 테스트 (Qt 없이 실행)."""

from translator.application.bubble_text import BubbleEntry, compose_bubble
from translator.core.models import ScreenRegion


def entry(text: str, left: int, top: int, width: int = 100, height: int = 20) -> BubbleEntry:
    """번역문 하나와 그 원문 위치."""
    return BubbleEntry(text, ScreenRegion(left, top, width, height))


def test_texts_on_different_lines_are_separated_by_line_breaks() -> None:
    """원문이 다른 줄에 있으면 번역문도 줄을 바꾼다 (위에서 아래 순서)."""
    composed = compose_bubble([entry("셋째 줄", 0, 100), entry("첫째 줄", 0, 0), entry("둘째 줄", 0, 50)])

    assert composed.text == "첫째 줄\n둘째 줄\n셋째 줄"


def test_texts_on_same_line_are_joined_left_to_right() -> None:
    """원문이 같은 줄에 있으면 왼쪽부터 이어서 쓴다."""
    composed = compose_bubble([entry("오른쪽", 300, 2), entry("왼쪽", 0, 0), entry("가운데", 150, 1)])

    assert composed.text == "왼쪽 가운데 오른쪽"


def test_mixed_lines_keep_reading_order() -> None:
    """같은 줄은 이어 쓰고 다른 줄은 줄을 바꿔 읽는 순서를 지킨다."""
    composed = compose_bubble(
        [
            entry("B", 200, 0),
            entry("C", 0, 40),
            entry("A", 0, 0),
            entry("D", 200, 41),
        ]
    )

    assert composed.text == "A B\nC D"


def test_bounds_cover_all_source_texts() -> None:
    """말풍선 위치 기준은 원문 전체를 감싸는 범위다."""
    composed = compose_bubble([entry("a", 10, 10, 50, 20), entry("b", 100, 200, 80, 30)])

    assert composed.bounds == ScreenRegion(10, 10, 170, 220)


def test_slightly_offset_text_on_same_line_is_not_broken() -> None:
    """같은 줄의 텍스트 높이가 조금 어긋나도(절반 이상 겹침) 줄을 바꾸지 않는다."""
    composed = compose_bubble([entry("앞", 0, 0, 100, 20), entry("뒤", 120, 9, 100, 20)])

    assert composed.text == "앞 뒤"


def test_chinese_and_japanese_are_joined_without_space() -> None:
    """번역문의 이어지는 글자가 한자·가나면 띄어쓰지 않는다 (텍스트 묶기 규칙과 같음)."""
    composed = compose_bubble([entry("你好", 0, 0), entry("世界", 120, 0)])

    assert composed.text == "你好世界"


def test_empty_entries_are_skipped() -> None:
    """빈 번역문은 넣지 않고, 넣을 것이 없으면 None이다."""
    assert compose_bubble([]) is None
    assert compose_bubble([entry("  ", 0, 0)]) is None
    assert compose_bubble([entry(" ", 0, 0), entry("내용", 0, 50)]).text == "내용"
