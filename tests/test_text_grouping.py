"""OCR 조각을 대사 단위로 묶는 규칙 테스트 (Qt 없이 실행)."""

from translator.application.text_grouping import group_segments, join_text
from translator.core.models import ScreenRegion, TextGranularity, TextSegment

WORD = TextGranularity.WORD
LINE = TextGranularity.LINE
CHAR = TextGranularity.CHARACTER


def seg(text: str, left: int, top: int, width: int, height: int = 20, kind=WORD) -> TextSegment:
    """조각 하나를 만든다."""
    return TextSegment(text, ScreenRegion(left, top, width, height), kind)


def texts(units) -> list[str]:
    """묶인 대사 텍스트 목록."""
    return [unit.text for unit in units]


def test_words_on_same_line_are_joined_in_reading_order() -> None:
    """같은 줄의 단어는 왼쪽부터 띄어 써서 이어 붙인다 (입력 순서와 무관)."""
    units = group_segments([seg("world", 70, 100, 50), seg("Hello", 10, 102, 55)])

    assert texts(units) == ["Hello world"]
    assert units[0].bounds == ScreenRegion(10, 100, 110, 22)
    assert len(units[0].segments) == 2


def test_wide_gap_splits_words_into_separate_units() -> None:
    """같은 높이라도 멀리 떨어진 단어(다른 UI 텍스트)는 묶지 않는다."""
    units = group_segments([seg("HP", 10, 100, 30), seg("MP", 600, 100, 30)])

    assert texts(units) == ["HP", "MP"]


def test_lines_of_one_dialogue_are_merged() -> None:
    """간격이 좁고 왼쪽이 맞는 두 줄은 하나의 대사로 묶는다."""
    units = group_segments(
        [seg("first line", 100, 100, 200, kind=LINE), seg("second line", 100, 126, 220, kind=LINE)]
    )

    assert texts(units) == ["first line second line"]


def test_centered_subtitle_lines_are_merged() -> None:
    """가운데 정렬 자막(왼쪽 끝은 다르지만 가운데가 맞음)도 하나의 대사로 묶는다."""
    units = group_segments(
        [seg("short", 400, 500, 100, kind=LINE), seg("a much longer line", 300, 525, 300, kind=LINE)]
    )

    assert texts(units) == ["short a much longer line"]


def test_lines_with_different_font_size_are_not_merged() -> None:
    """글자 크기가 많이 다르면(예: 화자 이름과 대사) 다른 텍스트로 본다."""
    units = group_segments(
        [seg("NAME", 100, 100, 80, height=40, kind=LINE), seg("dialogue", 100, 145, 200, kind=LINE)]
    )

    assert texts(units) == ["NAME", "dialogue"]


def test_far_apart_lines_are_not_merged() -> None:
    """위아래로 멀리 떨어진 줄은 묶지 않는다."""
    units = group_segments(
        [seg("top text", 100, 100, 200, kind=LINE), seg("bottom text", 100, 400, 200, kind=LINE)]
    )

    assert texts(units) == ["top text", "bottom text"]


def test_misaligned_lines_are_not_merged() -> None:
    """가까워도 정렬이 전혀 맞지 않는 줄(옆 열의 텍스트)은 묶지 않는다."""
    units = group_segments(
        [seg("left column", 100, 100, 150, kind=LINE), seg("right column", 700, 124, 150, kind=LINE)]
    )

    assert texts(units) == ["left column", "right column"]


def test_engine_blocks_are_kept_as_is() -> None:
    """엔진이 이미 블록으로 묶어 준 조각은 그대로 하나의 대사다."""
    block = seg("already grouped", 0, 0, 300, 60, kind=TextGranularity.BLOCK)

    assert texts(group_segments([block])) == ["already grouped"]


def test_blank_segments_are_ignored() -> None:
    """공백만 있는 조각은 버린다."""
    assert group_segments([seg("  ", 0, 0, 10)]) == []


def test_characters_are_joined_without_space_unless_gap_is_wide() -> None:
    """글자 단위 조각은 붙여 쓰고, 간격이 넓을 때만 띄어 쓴다."""
    units = group_segments(
        [
            seg("안", 0, 0, 18, kind=CHAR),
            seg("녕", 19, 0, 18, kind=CHAR),
            seg("하", 60, 0, 18, kind=CHAR),
        ]
    )

    assert texts(units) == ["안녕 하"]


def test_join_text_spacing_rules() -> None:
    """한자·가나끼리는 붙이고, 한글·영문은 띄어 쓴다."""
    assert join_text("こんにちは", "世界") == "こんにちは世界"
    assert join_text("안녕", "세상") == "안녕 세상"
    assert join_text("Hello", "world") == "Hello world"
    assert join_text("", "x") == "x"


def test_units_are_sorted_top_to_bottom_then_left_to_right() -> None:
    """결과는 위→아래, 왼쪽→오른쪽 순서다."""
    units = group_segments(
        [seg("C", 500, 300, 30), seg("B", 600, 100, 30), seg("A", 10, 100, 30)]
    )

    assert texts(units) == ["A", "B", "C"]
