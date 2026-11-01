"""화면에 표시할 번역 텍스트 항목을 추적한다 (출처별, 고유 ID, 안정화, 중복 번역 판단).

인식 결과(`OcrResult`)가 오면 같은 출처의 기존 항목과 짝을 맞춘다.

- 같은 원문이 가까운 위치에서 보이면 → 같은 항목이 **이동**한 것 (ID 유지, 위치만 갱신,
  다시 번역하지 않음). 멀리 떨어진 같은 문자열은 다른 문장으로 본다.
- 같은 자리(IoU 기준)이거나 같은 자리에서 글자가 늘어난 경우 → 같은 항목의 **내용이
  바뀐** 것 (ID 유지)
- 짝이 없는 새 텍스트 → 새 항목 (새 ID)
- 이번 결과에 없는 기존 항목:
  - 그 자리의 픽셀이 그대로면 → **OCR이 놓친 것**으로 보고 유지한다 (재검사 대상).
  - 픽셀이 바뀌었으면 → 텍스트가 **실제로 사라진** 것으로 보고 유예 시간 뒤 지운다.

각 항목의 텍스트가 안정됐는지는 `TextStabilizer`가 판단하고(타자 효과), 안정된 텍스트를
이미 번역했는지(중복 번역 판단)는 이 모델이 따로 판단한다. 다른 출처의 항목은 서로 영향을
주지 않는다. 늦게 도착한 오래된 프레임의 결과는 무시한다. Qt에 의존하지 않으며 시계를
주입할 수 있다. UI 스레드에서만 사용한다.
"""

import itertools
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from translator.application.change_detection import Fingerprint, is_changed
from translator.application.text_stabilizer import TextStabilizer, normalize_text
from translator.core.models import ScreenRegion, TranslationBatch
from translator.tuning import (
    TEXT_AREA_CHANGED_RATIO,
    TEXT_HIDE_GRACE_MS,
    TEXT_MOVE_MAX_DISTANCE_RATIO,
    TEXT_SAME_PLACE_IOU,
)


@dataclass
class DisplayItem:
    """화면에 표시 중인 번역 텍스트 하나."""

    # 출처 안에서 유일하며 항목이 살아 있는 동안 바뀌지 않는 ID (예: "box-1#3")
    item_id: str
    source_id: str
    # 가장 최근에 인식된 원문
    source_text: str
    # 원문 텍스트의 화면 위치 (물리 픽셀). 말풍선은 이 위치를 기준으로 놓인다.
    bounds: ScreenRegion
    # 마지막으로 인식된 시각
    last_seen: float
    # 마지막으로 인식됐을 때 텍스트 자리의 픽셀 지문 (OCR 누락 판단용)
    area_fingerprint: Fingerprint | None = None
    # 안정됐다고 판단한 원문 (정규화). 아직 안정되지 않았으면 None.
    stable_text: str | None = None
    # 번역문과, 그 번역문이 어떤 원문(정규화)의 번역인지
    translated_text: str | None = None
    translated_for: str | None = None
    # 인식 결과에서 빠지기 시작한 시각. 다시 인식되면 None으로 돌아간다.
    missing_since: float | None = None
    # 픽셀은 그대로인데 OCR이 놓친 연속 횟수
    ocr_misses: int = 0
    # 이 시각이 지나면 인식 여부와 관계없이 지운다 (사용하지 않으면 None).
    expires_at: float | None = None
    # 텍스트 안정화 판단기
    stabilizer: TextStabilizer = field(default_factory=TextStabilizer, repr=False)

    @property
    def display_text(self) -> str | None:
        """지금 말풍선에 보여 줄 번역문 (보여 줄 것이 없으면 None).

        - 안정된 원문의 번역이 있으면 그것을 보여 준다.
        - 글자가 늘어나는 중이면(이전 번역 원문 뒤에 글자가 붙는 중) 이전 번역을 유지한다.
        - 다른 텍스트로 바뀌어 새 번역을 기다리는 중이면 보여 주지 않는다 (오래된 번역 방지).
        """
        if self.translated_text is None or self.translated_for is None:
            return None
        if self.stable_text == self.translated_for:
            return self.translated_text
        current = normalize_text(self.source_text)
        if self.stable_text is None and current.startswith(self.translated_for):
            return self.translated_text
        return None


@dataclass(frozen=True)
class ApplyOutcome:
    """인식 결과를 반영한 뒤의 판단."""

    # False면 오래된 프레임이라 무시했다.
    accepted: bool = True
    # 안정됐지만 아직 번역하지 않은 원문들 (중복 제거)
    texts_to_translate: tuple[str, ...] = ()
    # 아직 안정되지 않은 텍스트가 있음 (후속 OCR 필요)
    unstable: bool = False
    # 픽셀은 그대로인데 OCR이 놓친 항목 수 (재검사 필요)
    ocr_misses: int = 0
    # 실제로 사라졌다고 판단한 항목 수
    disappeared: int = 0
    # 인식된 텍스트 수
    recognized: int = 0


@dataclass(frozen=True)
class _Observed:
    """짝을 맞출 새 인식 텍스트 하나."""

    text: str
    bounds: ScreenRegion
    area_fingerprint: Fingerprint | None


class TextDisplayModel:
    """출처별 번역 텍스트 항목 목록과 그 생명 주기를 관리한다."""

    def __init__(
        self,
        clock: Callable[[], float] = time.monotonic,
        hide_grace_seconds: float = TEXT_HIDE_GRACE_MS / 1000,
        same_place_iou: float = TEXT_SAME_PLACE_IOU,
        move_max_distance_ratio: float = TEXT_MOVE_MAX_DISTANCE_RATIO,
        area_changed_ratio: float = TEXT_AREA_CHANGED_RATIO,
        stabilizer_factory: Callable[[], TextStabilizer] = TextStabilizer,
    ) -> None:
        """시계(초)와 유예 시간, 짝 맞추기 기준, 텍스트 자리 변화 기준, 안정화 판단기를 정한다."""
        self._clock = clock
        self._grace = hide_grace_seconds
        self._same_place_iou = same_place_iou
        self._move_max_ratio = move_max_distance_ratio
        self._area_ratio = area_changed_ratio
        self._stabilizer_factory = stabilizer_factory
        # 항목 ID → 항목. 삽입 순서를 유지해 표시 순서가 흔들리지 않게 한다.
        self._items: dict[str, DisplayItem] = {}
        # 새 항목 ID 발급용 번호
        self._ids = itertools.count(1)
        # 출처 → 마지막으로 반영한 프레임의 캡처 시각 (오래된 결과 무시용)
        self._last_capture: dict[str, float] = {}

    def now(self) -> float:
        """모델이 쓰는 시계의 현재 시각 (초)."""
        return self._clock()

    def items(self) -> list[DisplayItem]:
        """현재 추적 중인 모든 항목 (번역 대기·유예 중인 항목 포함)."""
        return list(self._items.values())

    def items_for(self, source_id: str) -> list[DisplayItem]:
        """한 출처의 항목."""
        return [item for item in self._items.values() if item.source_id == source_id]

    def has_items(self, source_id: str) -> bool:
        """한 출처에 추적 중인 항목이 있는지 여부."""
        return any(item.source_id == source_id for item in self._items.values())

    def apply_ocr(self, result, force_stable: bool = False) -> ApplyOutcome:
        """인식 결과(`OcrResult`)를 같은 출처의 기존 항목과 맞춰 반영한다.

        `force_stable`이면 안정화를 기다리지 않는다 (사용자가 직접 요청한 번역).
        """
        if not self._accept_capture(result.source_id, result.captured_at):
            return ApplyOutcome(accepted=False)
        observed = [
            _Observed(unit.text, unit.bounds, fingerprint)
            for unit, fingerprint in zip(result.units, result.unit_fingerprints)
        ]
        return self._apply(result.source_id, observed, dict(result.probes), force_stable)

    def apply_translations(self, source_id: str, translations: dict[str, tuple[str, str]]) -> bool:
        """번역 결과를 반영한다. 원문이 그사이 바뀐 항목에는 적용하지 않는다 (오래된 번역 방지)."""
        changed = False
        for item in self.items_for(source_id):
            if item.stable_text is None:
                continue
            value = translations.get(item.stable_text)
            if value is None or item.translated_for == item.stable_text:
                continue
            item.translated_text = value[0]
            item.translated_for = item.stable_text
            changed = True
        return changed

    def apply_batch(self, batch: TranslationBatch, ttl_seconds: float | None = None) -> None:
        """이미 번역된 결과(`TranslationBatch`)를 바로 반영한다 (안정화를 거치지 않는 경로)."""
        if batch.captured_at is not None and not self._accept_capture(batch.source_id, batch.captured_at):
            return
        observed = [_Observed(unit.source_text, unit.bounds, None) for unit in batch.units]
        self._apply(batch.source_id, observed, {}, force_stable=True)
        self.apply_translations(
            batch.source_id,
            {normalize_text(u.source_text): (u.translated_text, u.engine) for u in batch.units},
        )
        if ttl_seconds is not None:
            expires_at = self._clock() + ttl_seconds
            for item in self.items_for(batch.source_id):
                if item.missing_since is None:
                    item.expires_at = expires_at

    def texts_to_translate(self, source_id: str) -> tuple[str, ...]:
        """출처에서 안정됐지만 아직 번역하지 않은 원문들 (중복 번역 판단)."""
        pending = (
            item.stable_text
            for item in self.items_for(source_id)
            if item.stable_text and item.translated_for != item.stable_text
        )
        return tuple(dict.fromkeys(pending))

    def remove_source(self, source_id: str) -> None:
        """한 출처의 항목을 모두 지운다 (박스 삭제 등)."""
        for item_id in [i.item_id for i in self.items_for(source_id)]:
            del self._items[item_id]
        self._last_capture.pop(source_id, None)

    def clear(self) -> None:
        """모든 항목을 지운다."""
        self._items.clear()
        self._last_capture.clear()

    def expire(self) -> bool:
        """유예 시간이 지난 항목과 만료된 항목을 지운다. 지운 항목이 있으면 True."""
        now = self._clock()
        expired = [
            item.item_id
            for item in self._items.values()
            if (item.missing_since is not None and now - item.missing_since >= self._grace)
            or (item.expires_at is not None and now >= item.expires_at)
        ]
        for item_id in expired:
            del self._items[item_id]
        return bool(expired)

    def next_deadline(self) -> float | None:
        """가장 먼저 지워질 항목의 시각 (없으면 None). UI가 타이머를 맞추는 데 쓴다."""
        deadlines = []
        for item in self._items.values():
            if item.missing_since is not None:
                deadlines.append(item.missing_since + self._grace)
            if item.expires_at is not None:
                deadlines.append(item.expires_at)
        return min(deadlines) if deadlines else None

    def _accept_capture(self, source_id: str, captured_at: float) -> bool:
        """이미 반영한 프레임보다 오래된 프레임이면 False (늦게 끝난 결과가 최신을 덮지 않게)."""
        last = self._last_capture.get(source_id)
        if last is not None and captured_at < last:
            return False
        self._last_capture[source_id] = captured_at
        return True

    def _apply(
        self,
        source_id: str,
        observed: list[_Observed],
        probes: dict[ScreenRegion, Fingerprint],
        force_stable: bool,
    ) -> ApplyOutcome:
        """새 인식 텍스트를 기존 항목과 짝지어 반영하고 판단 결과를 돌려준다."""
        now = self._clock()
        existing = self.items_for(source_id)
        matches = self._match(existing, observed)
        matched_ids = {id(item) for item in matches.values()}
        unstable = False

        for index, seen in enumerate(observed):
            item = matches.get(index)
            if item is None:
                # 짝이 없는 새 텍스트 → 새 항목
                item_id = f"{source_id}#{next(self._ids)}"
                item = DisplayItem(
                    item_id=item_id,
                    source_id=source_id,
                    source_text=seen.text,
                    bounds=seen.bounds,
                    last_seen=now,
                    stabilizer=self._stabilizer_factory(),
                )
                self._items[item_id] = item
            # 이동했거나 내용이 바뀐 기존 항목 → ID를 유지한 채 갱신
            item.source_text = seen.text
            item.bounds = seen.bounds
            item.area_fingerprint = seen.area_fingerprint
            item.last_seen = now
            item.missing_since = None
            item.expires_at = None
            item.ocr_misses = 0
            stable = item.stabilizer.observe(seen.text, now, force=force_stable)
            if stable is None:
                unstable = True
                if item.stable_text is not None and not normalize_text(seen.text).startswith(item.stable_text):
                    # 다른 텍스트로 바뀌는 중: 이전 안정 텍스트는 더 이상 유효하지 않다.
                    item.stable_text = None
            else:
                item.stable_text = stable

        ocr_misses = 0
        disappeared = 0
        for item in existing:
            if id(item) in matched_ids:
                continue
            if self._is_ocr_miss(item, probes):
                # 텍스트 자리 픽셀이 그대로다: OCR이 놓친 것으로 보고 번역문을 유지한다.
                item.ocr_misses += 1
                ocr_misses += 1
                continue
            if item.missing_since is None:
                # 텍스트가 실제로 바뀌거나 사라졌다: 유예 시간 뒤 지운다.
                item.missing_since = now
                disappeared += 1

        return ApplyOutcome(
            accepted=True,
            texts_to_translate=self.texts_to_translate(source_id),
            unstable=unstable,
            ocr_misses=ocr_misses,
            disappeared=disappeared,
            recognized=len(observed),
        )

    def _is_ocr_miss(self, item: DisplayItem, probes: dict[ScreenRegion, Fingerprint]) -> bool:
        """항목이 인식 결과에 없지만 그 자리 픽셀이 그대로인지 (OCR이 놓친 것인지)."""
        probe = probes.get(item.bounds)
        if probe is None or item.area_fingerprint is None:
            return False
        return not is_changed(item.area_fingerprint, probe, self._area_ratio)

    def _match(self, existing: list[DisplayItem], observed: list[_Observed]) -> dict[int, DisplayItem]:
        """새 텍스트(순번) → 짝지은 기존 항목. 점수가 높은 쌍부터 1:1로 정한다."""
        candidates: list[tuple[float, int, int]] = []
        for unit_index, seen in enumerate(observed):
            for item_index, item in enumerate(existing):
                score = self._score(item, seen.text, seen.bounds)
                if score > 0:
                    candidates.append((score, unit_index, item_index))
        candidates.sort(reverse=True)

        matches: dict[int, DisplayItem] = {}
        used_items: set[int] = set()
        for _score, unit_index, item_index in candidates:
            if unit_index in matches or item_index in used_items:
                continue
            matches[unit_index] = existing[item_index]
            used_items.add(item_index)
        return matches

    def _score(self, item: DisplayItem, source_text: str, bounds: ScreenRegion) -> float:
        """기존 항목과 새 텍스트가 같은 항목일 가능성. 0이면 짝이 아니다.

        - 같은 원문: 일정 거리(글자 높이 기준) 안에서만 이동으로 본다. 2점 + 가까운 정도.
          멀리 떨어진 같은 문자열은 우연히 같은 다른 문장으로 보고 짝짓지 않는다.
        - 같은 자리에서 글자가 늘어남(타자 효과): 1.5점 + 겹침 정도.
        - 원문이 다르면 같은 자리(IoU 기준 이상)일 때만 내용 변경으로 본다.
        """
        (x1, y1), (x2, y2) = item.bounds.center, bounds.center
        distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        # 글자 높이 기준으로 거리를 재야 화면 해상도와 관계없이 비교된다.
        scale = max(1, item.bounds.height, bounds.height)
        iou = item.bounds.intersection_over_union(bounds)
        old_text = normalize_text(item.source_text)
        new_text = normalize_text(source_text)
        if old_text == new_text:
            if distance / scale > self._move_max_ratio:
                return 0.0
            return 2.0 + 1.0 / (1.0 + distance / scale)
        growing = bool(old_text) and (new_text.startswith(old_text) or old_text.startswith(new_text))
        if growing and iou > 0 and abs(item.bounds.left - bounds.left) <= scale:
            return 1.5 + iou
        if iou >= self._same_place_iou:
            return iou
        return 0.0
