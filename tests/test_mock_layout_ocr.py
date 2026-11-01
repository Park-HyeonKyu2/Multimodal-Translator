"""위치를 돌려주는 Mock OCR 테스트: 글자처럼 보이는 영역을 단어 조각으로 찾는다."""

from translator.adapters.ocr.mock import MockTextRecognizer
from translator.adapters.ocr.mock_layout import MockLayoutTextRecognizer, detect_word_segments
from translator.core.cancellation import CancellationToken
from translator.core.models import CapturedImage, ScreenRegion, TextGranularity


def draw(width: int, height: int, boxes: list[tuple[int, int, int, int]], origin=(0, 0)) -> CapturedImage:
    """어두운 배경에 흰 사각형(글자 덩어리)을 그린 이미지를 만든다."""
    pixels = bytearray(bytes((30, 30, 30, 255)) * (width * height))
    for left, top, w, h in boxes:
        for y in range(top, top + h):
            for x in range(left, left + w):
                offset = (y * width + x) * 4
                pixels[offset:offset + 4] = bytes((255, 255, 255, 255))
    region = ScreenRegion(origin[0], origin[1], width, height)
    return CapturedImage(region, width, height, bytes(pixels))


def test_words_on_a_line_are_found_with_screen_coordinates() -> None:
    """한 줄의 두 단어를 찾고, 위치를 화면 물리 좌표(캡처 원점 포함)로 돌려준다."""
    image = draw(200, 60, [(10, 20, 40, 16), (60, 20, 30, 16)], origin=(1000, 500))

    segments = detect_word_segments(image)

    assert [s.text for s in segments] == ["단어1", "단어2"]
    assert all(s.granularity is TextGranularity.WORD for s in segments)
    assert segments[0].bounds == ScreenRegion(1010, 520, 40, 16)
    assert segments[1].bounds == ScreenRegion(1060, 520, 30, 16)


def test_separate_lines_are_found_separately() -> None:
    """위아래로 떨어진 두 줄을 각각 찾는다."""
    image = draw(200, 120, [(10, 10, 50, 14), (10, 80, 70, 14)])

    tops = sorted({s.bounds.top for s in detect_word_segments(image)})

    assert tops == [10, 80]


def test_blank_image_has_no_text() -> None:
    """글자가 없는 화면은 빈 결과다."""
    assert detect_word_segments(draw(100, 40, [])) == []


def test_too_thin_or_too_tall_blobs_are_ignored() -> None:
    """선(너무 얇음)이나 큰 그림(너무 큼)은 글자로 보지 않는다."""
    image = draw(300, 300, [(0, 10, 200, 2), (10, 50, 100, 200)])

    assert detect_word_segments(image) == []


def test_moved_text_moves_segments() -> None:
    """글자 덩어리를 옮기면 조각 위치도 따라 바뀐다 (말풍선 이동 검증의 바탕)."""
    before = detect_word_segments(draw(300, 60, [(10, 20, 40, 16)]))
    after = detect_word_segments(draw(300, 60, [(160, 20, 40, 16)]))

    assert before[0].text == after[0].text
    assert after[0].bounds.left - before[0].bounds.left == 150


def test_recognizer_returns_segments_and_joined_text() -> None:
    """Mock OCR은 조각 목록과 이어 붙인 텍스트를 함께 돌려준다."""
    image = draw(200, 60, [(10, 20, 40, 16), (60, 20, 30, 16)])

    result = MockLayoutTextRecognizer(delay_seconds=0).recognize(image, CancellationToken())

    assert result.text == "단어1 단어2"
    assert len(result.segments) == 2


def test_fixed_mock_returns_center_band_segment() -> None:
    """고정 Mock OCR은 이미지 가운데 띠를 원문 위치로 돌려준다."""
    image = CapturedImage(ScreenRegion(100, 200, 300, 100), 300, 100, b"")

    result = MockTextRecognizer(delay_seconds=0).recognize(image, CancellationToken())

    [segment] = result.segments
    assert segment.bounds == ScreenRegion(130, 235, 240, 30)
    assert segment.granularity is TextGranularity.LINE
