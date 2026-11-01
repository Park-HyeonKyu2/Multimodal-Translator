"""한 출처(바운딩 박스)의 번역문들을 말풍선 하나에 넣을 텍스트로 합친다.

원문 위치를 읽는 순서(위 → 아래, 같은 줄은 왼쪽 → 오른쪽)로 정렬한다.
- 원문이 같은 줄에 있으면 번역문을 이어서 쓴다.
- 원문이 다른 줄에 있으면(줄넘김) 번역문도 줄을 바꾼다.

한 대사가 여러 줄에 걸쳐 있어 OCR 묶기에서 한 대사로 묶인 경우는 번역문 하나이므로,
말풍선 너비에 맞춰 자동으로 줄이 바뀔 뿐 강제로 줄을 나누지 않는다 (번역문은 원문과 어순이
달라 원문의 줄 위치를 그대로 옮길 수 없다). Qt에 의존하지 않는다.
"""

from dataclasses import dataclass

from translator.application.text_grouping import join_text
from translator.core.models import ScreenRegion
from translator.tuning import TEXT_LINE_VERTICAL_OVERLAP


@dataclass(frozen=True)
class BubbleEntry:
    """말풍선에 넣을 번역문 하나와 그 원문 위치 (물리 픽셀)."""

    text: str
    bounds: ScreenRegion


@dataclass(frozen=True)
class ComposedBubble:
    """합친 결과: 말풍선에 쓸 텍스트와, 말풍선 위치 기준인 원문 전체 범위."""

    text: str
    bounds: ScreenRegion


def compose_bubble(
    entries: list[BubbleEntry] | tuple[BubbleEntry, ...],
    line_overlap: float = TEXT_LINE_VERTICAL_OVERLAP,
) -> ComposedBubble | None:
    """번역문들을 읽는 순서로 합친다. 같은 줄은 이어 쓰고 다른 줄은 줄을 바꾼다.

    두 원문의 세로 겹침이 작은 쪽 높이의 `line_overlap` 이상이면 같은 줄로 본다.
    넣을 번역문이 없으면 None을 반환한다.
    """
    usable = [entry for entry in entries if entry.text.strip()]
    if not usable:
        return None

    # 읽는 순서로 줄을 만든다: 위에서부터 보며, 지금 줄과 세로로 충분히 겹치면 같은 줄에 넣는다.
    rows: list[list[BubbleEntry]] = []
    # 각 줄의 세로 범위 (줄에 들어간 원문들을 합친 범위)
    row_bounds: list[ScreenRegion] = []
    for entry in sorted(usable, key=lambda e: (e.bounds.top, e.bounds.left)):
        index = _row_for(entry.bounds, row_bounds, line_overlap)
        if index is None:
            rows.append([entry])
            row_bounds.append(entry.bounds)
        else:
            rows[index].append(entry)
            row_bounds[index] = row_bounds[index].union(entry.bounds)

    lines = []
    for row in rows:
        text = ""
        for entry in sorted(row, key=lambda e: e.bounds.left):
            text = join_text(text, entry.text.strip())
        lines.append(text)

    bounds = usable[0].bounds
    for entry in usable[1:]:
        bounds = bounds.union(entry.bounds)
    return ComposedBubble("\n".join(lines), bounds)


def _row_for(bounds: ScreenRegion, row_bounds: list[ScreenRegion], line_overlap: float) -> int | None:
    """원문이 들어갈 기존 줄의 순번 (같은 줄이 없으면 None)."""
    for index, row in enumerate(row_bounds):
        overlap = min(row.bottom, bounds.bottom) - max(row.top, bounds.top)
        if overlap >= line_overlap * min(row.height, bounds.height):
            return index
    return None
