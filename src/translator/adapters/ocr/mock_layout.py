"""화면에서 "글자처럼 보이는" 영역을 찾아 위치가 있는 단어 조각을 돌려주는 Mock OCR.

실제 OCR 모델 없이도 원문 위치 기준 번역 표시, 텍스트 이동·사라짐 추적을 실제 화면에서
확인하기 위한 구현이다. 글자를 읽지는 못하므로 텍스트는 "단어1", "단어2"처럼 순서만
나타낸다. 단어 단위(WORD)로 돌려주므로 줄·대사로 묶는 과정도 함께 검증된다.

방식: 표본 격자의 밝기에서 가장 흔한 값을 배경으로 보고, 배경과 차이가 큰 표본을
"잉크"로 표시한다. 잉크가 있는 행을 이어 줄을 찾고, 줄 안에서 빈칸으로 단어를 나눈다.
"""

import logging
import math

from translator.core.cancellation import CancellationToken
from translator.core.interfaces import TextRecognizer
from translator.core.models import CapturedImage, RecognizedText, TextGranularity, TextSegment
from translator.tuning import (
    MOCK_OCR_DELAY_SECONDS,
    MOCK_OCR_GRID_MAX_HEIGHT,
    MOCK_OCR_GRID_MAX_WIDTH,
    MOCK_OCR_INK_CONTRAST,
    MOCK_OCR_MAX_SEGMENTS,
    MOCK_OCR_MAX_TEXT_HEIGHT,
    MOCK_OCR_MIN_TEXT_HEIGHT,
    MOCK_OCR_WORD_GAP_RATIO,
)

logger = logging.getLogger(__name__)


class MockLayoutTextRecognizer(TextRecognizer):
    """배경과 대비가 큰 영역을 단어 조각으로 돌려주는 Mock OCR."""

    def __init__(self, delay_seconds: float = MOCK_OCR_DELAY_SECONDS) -> None:
        """흉내 낼 추론 지연 시간(초)을 정한다."""
        self._delay_seconds = delay_seconds

    def recognize(
        self, image: CapturedImage, cancellation: CancellationToken
    ) -> RecognizedText:
        """지연 후 글자처럼 보이는 단어 조각 목록을 반환한다."""
        cancellation.wait(self._delay_seconds)
        segments = detect_word_segments(image, cancellation)
        logger.debug("Mock 배치 OCR: %dx%d에서 조각 %d개", image.width, image.height, len(segments))
        return RecognizedText(
            text=" ".join(segment.text for segment in segments), segments=tuple(segments)
        )


def detect_word_segments(
    image: CapturedImage, cancellation: CancellationToken | None = None
) -> list[TextSegment]:
    """이미지에서 단어 조각을 찾는다. 읽는 순서(위→아래, 왼쪽→오른쪽)로 반환한다."""
    if image.width <= 0 or image.height <= 0 or not image.bgra:
        return []
    # 큰 이미지는 표본 간격을 넓혀 계산량을 제한한다.
    step = max(
        1,
        math.ceil(image.width / MOCK_OCR_GRID_MAX_WIDTH),
        math.ceil(image.height / MOCK_OCR_GRID_MAX_HEIGHT),
    )
    columns = image.width // step
    rows = image.height // step
    if columns == 0 or rows == 0:
        return []

    luminance = _sample_luminance(image, step, columns, rows)
    background = _most_common_value(luminance)
    # ink[r][c]: 배경과 충분히 다른 표본인지
    ink = [
        [abs(value - background) >= MOCK_OCR_INK_CONTRAST for value in row] for row in luminance
    ]
    if cancellation is not None:
        cancellation.raise_if_cancelled()

    segments: list[TextSegment] = []
    for row_start, row_end in _ink_runs([any(row) for row in ink], 0):
        height_px = (row_end - row_start) * step
        if not MOCK_OCR_MIN_TEXT_HEIGHT <= height_px <= MOCK_OCR_MAX_TEXT_HEIGHT:
            continue  # 너무 얇거나(선, 잡음) 너무 큰(그림, 창) 덩어리는 글자가 아니라고 본다.
        column_has_ink = [
            any(ink[r][c] for r in range(row_start, row_end)) for c in range(columns)
        ]
        # 줄 높이에 비례한 빈칸보다 넓게 떨어진 잉크 덩어리를 다른 단어로 나눈다.
        word_gap = max(1, round((row_end - row_start) * MOCK_OCR_WORD_GAP_RATIO))
        for word_index, (col_start, col_end) in enumerate(_ink_runs(column_has_ink, word_gap)):
            bounds = image.to_screen_region(
                col_start * step,
                row_start * step,
                (col_end - col_start) * step,
                (row_end - row_start) * step,
            )
            segments.append(
                TextSegment(
                    text=f"단어{word_index + 1}",
                    bounds=bounds,
                    granularity=TextGranularity.WORD,
                )
            )
            if len(segments) >= MOCK_OCR_MAX_SEGMENTS:
                logger.debug("Mock OCR 조각 수 상한(%d)에 도달했습니다.", MOCK_OCR_MAX_SEGMENTS)
                return segments
    return segments


def _sample_luminance(image: CapturedImage, step: int, columns: int, rows: int) -> list[list[int]]:
    """격자 표본의 밝기(0~255) 표를 만든다."""
    stride = image.width * 4
    data = image.bgra
    table = []
    for row in range(rows):
        offset_row = (row * step + step // 2) * stride
        values = []
        for column in range(columns):
            offset = offset_row + (column * step + step // 2) * 4
            blue, green, red = data[offset], data[offset + 1], data[offset + 2]
            values.append((299 * red + 587 * green + 114 * blue) // 1000)
        table.append(values)
    return table


def _most_common_value(table: list[list[int]]) -> int:
    """표에서 가장 자주 나오는 밝기 (배경색으로 본다)."""
    histogram = [0] * 256
    for row in table:
        for value in row:
            histogram[value] += 1
    return max(range(256), key=histogram.__getitem__)


def _ink_runs(flags: list[bool], max_gap: int) -> list[tuple[int, int]]:
    """True가 이어지는 구간 목록 [시작, 끝). `max_gap` 이하의 False 구간은 이어 붙인다."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    last_true = -1
    for index, flag in enumerate(flags):
        if not flag:
            continue
        if start is None:
            start = index
        elif index - last_true - 1 > max_gap:
            runs.append((start, last_true + 1))
            start = index
        last_true = index
    if start is not None:
        runs.append((start, last_true + 1))
    return runs
