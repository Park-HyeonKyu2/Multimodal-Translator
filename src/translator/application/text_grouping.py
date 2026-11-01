"""OCR 조각(글자·단어·줄)을 번역하기 좋은 대사 단위(`TextUnit`)로 묶는다.

규칙 요약:

1. 글자/단어 조각 → 줄: 세로로 충분히 겹치고, 가로 간격이 글자 높이에 비해 가까우면 같은 줄.
2. 줄 → 대사(블록): 위아래로 가깝고, 글자 높이가 비슷하고, 왼쪽 끝이나 가운데가 맞으면 같은 대사.
3. 엔진이 이미 블록 단위로 준 조각은 그대로 하나의 대사로 쓴다.

글꼴 크기가 다르거나 멀리 떨어진 텍스트는 묶지 않는다(관계없는 텍스트를 한 문장으로
합치지 않기 위함). Qt에 의존하지 않는 순수 함수라 단독으로 테스트할 수 있다.
"""

from dataclasses import dataclass, field

from translator.core.models import ScreenRegion, TextGranularity, TextSegment, TextUnit
from translator.tuning import (
    TEXT_CHAR_SPACE_RATIO,
    TEXT_LINE_ALIGN_RATIO,
    TEXT_LINE_GAP_RATIO,
    TEXT_LINE_HEIGHT_SIMILARITY,
    TEXT_LINE_VERTICAL_OVERLAP,
    TEXT_WORD_GAP_RATIO,
)


@dataclass
class _Line:
    """묶는 도중의 한 줄: 조각 목록, 이어 붙인 텍스트, 전체 범위."""

    segments: list[TextSegment]
    text: str
    bounds: ScreenRegion
    # 줄을 이루는 조각의 단위 (띄어쓰기 판단에 쓴다)
    granularity: TextGranularity = field(default=TextGranularity.LINE)


def group_segments(segments: tuple[TextSegment, ...] | list[TextSegment]) -> list[TextUnit]:
    """OCR 조각을 대사 단위로 묶어 읽는 순서(위→아래, 왼쪽→오른쪽)로 반환한다."""
    # 공백만 있는 조각은 번역할 내용이 없으므로 버린다.
    usable = [segment for segment in segments if segment.text.strip()]
    given_blocks = [s for s in usable if s.granularity is TextGranularity.BLOCK]
    given_lines = [s for s in usable if s.granularity is TextGranularity.LINE]
    pieces = [
        s
        for s in usable
        if s.granularity in (TextGranularity.CHARACTER, TextGranularity.WORD)
    ]

    lines = [_Line([s], s.text.strip(), s.bounds) for s in given_lines]
    lines += _merge_pieces_into_lines(pieces)

    units = [TextUnit(s.text.strip(), s.bounds, (s,)) for s in given_blocks]
    units += _merge_lines_into_units(lines)
    return sorted(units, key=lambda unit: (unit.bounds.top, unit.bounds.left))


def join_text(left: str, right: str, separator: str = " ") -> str:
    """두 텍스트를 잇는다. 경계의 두 글자가 모두 한자·가나면 띄어쓰지 않는다.

    한국어(한글)는 띄어쓰기를 하므로 한자·가나와 달리 공백을 넣는다.
    """
    if not left:
        return right
    if not right:
        return left
    if _is_cjk_no_space(left[-1]) and _is_cjk_no_space(right[0]):
        return left + right
    return left + separator + right


def _merge_pieces_into_lines(pieces: list[TextSegment]) -> list[_Line]:
    """글자/단어 조각을 왼쪽부터 보며 같은 줄에 이어 붙인다."""
    lines: list[_Line] = []
    for piece in sorted(pieces, key=lambda s: (s.bounds.left, s.bounds.top)):
        target = _best_line_for(piece, lines)
        if target is None:
            lines.append(_Line([piece], piece.text.strip(), piece.bounds, piece.granularity))
            continue
        last = target.segments[-1].bounds
        target.text = join_text(target.text, piece.text.strip(), _piece_separator(last, piece))
        target.segments.append(piece)
        target.bounds = target.bounds.union(piece.bounds)
    return lines


def _best_line_for(piece: TextSegment, lines: list[_Line]) -> _Line | None:
    """조각을 이어 붙일 줄을 찾는다. 후보가 여럿이면 가로 간격이 가장 작은 줄."""
    best: _Line | None = None
    best_gap = float("inf")
    bounds = piece.bounds
    for line in lines:
        last = line.segments[-1].bounds
        # 세로 겹침이 충분해야 같은 줄
        overlap = min(last.bottom, bounds.bottom) - max(last.top, bounds.top)
        if overlap < TEXT_LINE_VERTICAL_OVERLAP * min(last.height, bounds.height):
            continue
        # 가로 간격: 음수면 살짝 겹친 것. 많이 겹치면 다른 줄(겹쳐 그려진 텍스트)로 본다.
        gap = bounds.left - last.right
        height = max(last.height, bounds.height)
        if gap > TEXT_WORD_GAP_RATIO * height or gap < -0.5 * min(last.width, bounds.width):
            continue
        if gap < best_gap:
            best, best_gap = line, gap
    return best


def _piece_separator(previous: ScreenRegion, piece: TextSegment) -> str:
    """앞 조각과 이을 때 쓸 구분자. 글자 단위는 간격이 넓을 때만 띄어 쓴다."""
    if piece.granularity is TextGranularity.CHARACTER:
        gap = piece.bounds.left - previous.right
        return " " if gap > TEXT_CHAR_SPACE_RATIO * max(previous.height, piece.bounds.height) else ""
    return " "


def _merge_lines_into_units(lines: list[_Line]) -> list[TextUnit]:
    """위에서부터 줄을 보며, 바로 위 줄과 이어지는 대사면 같은 단위로 묶는다."""
    blocks: list[list[_Line]] = []
    for line in sorted(lines, key=lambda item: (item.bounds.top, item.bounds.left)):
        target = _best_block_for(line, blocks)
        if target is None:
            blocks.append([line])
        else:
            target.append(line)

    units = []
    for block in blocks:
        text = ""
        bounds = block[0].bounds
        segments: list[TextSegment] = []
        for line in block:
            text = join_text(text, line.text)
            bounds = bounds.union(line.bounds)
            segments.extend(line.segments)
        units.append(TextUnit(text=text, bounds=bounds, segments=tuple(segments)))
    return units


def _best_block_for(line: _Line, blocks: list[list[_Line]]) -> list[_Line] | None:
    """줄을 이어 붙일 대사를 찾는다. 후보가 여럿이면 세로 간격이 가장 작은 대사."""
    best: list[_Line] | None = None
    best_gap = float("inf")
    bounds = line.bounds
    for block in blocks:
        previous = block[-1].bounds
        small = min(previous.height, bounds.height)
        large = max(previous.height, bounds.height)
        # 글자 크기가 많이 다르면 다른 텍스트 (예: 화자 이름과 UI 라벨)
        if small / large < TEXT_LINE_HEIGHT_SIMILARITY:
            continue
        gap = bounds.top - previous.bottom
        if gap > TEXT_LINE_GAP_RATIO * small or gap < -0.5 * small:
            continue
        # 왼쪽 정렬 또는 가운데 정렬(자막)이어야 같은 대사
        tolerance = TEXT_LINE_ALIGN_RATIO * small
        left_aligned = abs(bounds.left - previous.left) <= tolerance
        center_aligned = abs(bounds.center[0] - previous.center[0]) <= tolerance
        if not (left_aligned or center_aligned):
            continue
        if gap < best_gap:
            best, best_gap = block, gap
    return best


def _is_cjk_no_space(character: str) -> bool:
    """띄어쓰기 없이 이어 쓰는 문자(한자, 히라가나, 가타카나, 전각 기호)인지 여부."""
    code = ord(character)
    return (
        0x3000 <= code <= 0x30FF  # 전각 기호, 히라가나, 가타카나
        or 0x3400 <= code <= 0x4DBF  # 한자 확장 A
        or 0x4E00 <= code <= 0x9FFF  # 한자
        or 0xFF00 <= code <= 0xFFEF  # 전각 영숫자·기호
    )
