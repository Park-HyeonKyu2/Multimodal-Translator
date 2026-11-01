"""번역 텍스트 상태(`TextDisplayModel`)를 화면의 말풍선 창들과 맞춰 유지한다.

번역 결과가 오면 모델에 반영한 뒤, 모델의 항목마다 말풍선을 만들거나 옮기고, 사라진
항목의 말풍선은 닫는다. 모델의 유예/만료 시각에 맞춰 타이머로 다시 정리한다.
말풍선 위치의 기준은 항상 OCR이 인식한 원문 텍스트 좌표이며, 방향과 색은 출처
(바운딩 박스)별 설정을 `style_for`로 받아 쓴다.

말풍선 단위는 출처마다 고른다 (`groups_source`).
- 묶는 출처(바운딩 박스): 출처의 번역문을 모두 말풍선 하나에 읽는 순서로 넣는다. 원문이
  같은 줄이면 이어 쓰고 다른 줄이면 줄을 바꾼다. 말풍선은 원문 전체 범위를 기준으로 놓는다.
- 묶지 않는 출처(전체 화면): 원문 텍스트마다 말풍선을 따로 놓는다.
"""

import logging
from collections.abc import Callable

from PySide6.QtCore import QObject, QTimer
from PySide6.QtGui import QColor

from translator.application.bubble_text import BubbleEntry, compose_bubble
from translator.application.text_display import ApplyOutcome, DisplayItem, TextDisplayModel
from translator.core.models import (
    DEFAULT_TRANSLATION_DIRECTION,
    ScreenRegion,
    TranslationBatch,
    TranslationDirection,
)
from translator.tuning import RESULT_WINDOW_DEFAULT_OPACITY
from translator.ui.screen_geometry import physical_region_to_logical
from translator.ui.translation_bubble import TranslationBubble

logger = logging.getLogger(__name__)

# 출처 ID → (출력 방향, 박스 색)
StyleLookup = Callable[[str], tuple[TranslationDirection, QColor]]
# 출처 ID → 그 출처의 번역문을 말풍선 하나로 묶을지 여부
GroupLookup = Callable[[str], bool]
# 출처 하나로 묶은 말풍선의 키 접미사 (항목 ID와 겹치지 않는다: 항목 ID는 "출처#숫자")
_GROUP_SUFFIX = "#all"


def _default_style(_source_id: str) -> tuple[TranslationDirection, QColor]:
    """방향/색 조회 함수가 등록되기 전에 쓰는 기본값 (아래, 회색)."""
    return DEFAULT_TRANSLATION_DIRECTION, QColor(190, 190, 190)


def _never_group(_source_id: str) -> bool:
    """묶기 조회 함수가 등록되기 전의 기본값: 원문마다 말풍선을 따로 놓는다."""
    return False


def group_bubble_key(source_id: str) -> str:
    """출처 하나로 묶은 말풍선의 키."""
    return source_id + _GROUP_SUFFIX


class TranslationOverlay(QObject):
    """번역 말풍선 창들의 생성·배치·제거를 담당한다. UI 스레드에서만 사용한다."""

    def __init__(
        self,
        model: TextDisplayModel,
        style_for: StyleLookup = _default_style,
        bubble_factory: Callable[[], TranslationBubble] = TranslationBubble,
        groups_source: GroupLookup = _never_group,
        parent: QObject | None = None,
    ) -> None:
        """표시 상태 모델, 출처별 방향/색 조회 함수, 말풍선 생성 함수, 출처별 묶기 여부를 받는다."""
        super().__init__(parent)
        self._model = model
        self._style_for = style_for
        self._groups_source = groups_source
        self._bubble_factory = bubble_factory
        # 항목 ID(또는 묶은 출처의 키) → 말풍선 창
        self._bubbles: dict[str, TranslationBubble] = {}
        # 말풍선 배경 불투명도
        self._opacity = RESULT_WINDOW_DEFAULT_OPACITY
        # 유예 시간이 끝나거나 만료되는 항목을 정리하는 타이머
        self._expiry_timer = QTimer(self)
        self._expiry_timer.setSingleShot(True)
        self._expiry_timer.timeout.connect(self._on_expiry_timer)

    def set_style_lookup(self, style_for: StyleLookup) -> None:
        """출처별 방향/색 조회 함수를 바꾼다 (coordinator가 만들어진 뒤 등록)."""
        self._style_for = style_for

    def set_group_lookup(self, groups_source: GroupLookup) -> None:
        """출처별 묶기 여부 조회 함수를 바꾼다 (coordinator가 만들어진 뒤 등록)."""
        self._groups_source = groups_source
        self.refresh()

    @property
    def model(self) -> TextDisplayModel:
        """표시 상태 모델."""
        return self._model

    def bubble_for(self, item_id: str) -> TranslationBubble | None:
        """항목 ID의 말풍선 (없으면 None). 묶는 출처의 항목은 따로 말풍선이 없다."""
        return self._bubbles.get(item_id)

    def bubble_for_source(self, source_id: str) -> TranslationBubble | None:
        """출처 하나로 묶은 말풍선 (묶지 않는 출처이거나 보여 줄 번역이 없으면 None)."""
        return self._bubbles.get(group_bubble_key(source_id))

    def bubbles(self) -> dict[str, TranslationBubble]:
        """현재 말풍선 목록 (복사본)."""
        return dict(self._bubbles)

    def show_batch(self, batch: TranslationBatch, ttl_seconds: float | None = None) -> None:
        """새 번역 결과를 반영하고 말풍선을 갱신한다."""
        self._model.apply_batch(batch, ttl_seconds)
        self.refresh()

    def show_ocr(self, result, force_stable: bool = False) -> ApplyOutcome:
        """인식 결과를 반영하고(위치 갱신, 안정화) 말풍선을 갱신한다. 판단 결과를 돌려준다."""
        outcome = self._model.apply_ocr(result, force_stable)
        if outcome.accepted:
            self.refresh()
        return outcome

    def show_translations(self, source_id: str, translations: dict[str, tuple[str, str]]) -> None:
        """번역 결과를 반영하고 말풍선을 갱신한다."""
        if self._model.apply_translations(source_id, translations):
            self.refresh()

    def has_items(self, source_id: str) -> bool:
        """출처에 표시 중인 번역이 있는지 여부."""
        return self._model.has_items(source_id)

    def texts_to_translate(self, source_id: str) -> tuple[str, ...]:
        """출처에서 안정됐지만 아직 번역하지 않은 원문들."""
        return self._model.texts_to_translate(source_id)

    def item_bounds(self, source_id: str) -> tuple[ScreenRegion, ...]:
        """출처에서 표시 중인 텍스트들의 위치 (OCR 누락 판단에 쓰는 텍스트 자리)."""
        return tuple(item.bounds for item in self._model.items_for(source_id))

    def remove_source(self, source_id: str) -> None:
        """출처의 번역을 모두 지운다 (박스 삭제 등)."""
        self._model.remove_source(source_id)
        self.refresh()

    def clear(self) -> None:
        """모든 번역을 지운다."""
        self._model.clear()
        self.refresh()

    def set_background_opacity(self, opacity: float) -> None:
        """모든 말풍선의 배경 불투명도를 바꾼다."""
        self._opacity = opacity
        for bubble in self._bubbles.values():
            bubble.set_background_opacity(opacity)

    def refresh(self) -> None:
        """모델의 항목과 말풍선을 맞춘다: 새로 만들고, 옮기고, 남는 것은 닫는다."""
        alive: set[str] = set()
        # 묶는 출처 → 그 출처의 항목 (표시 순서 유지)
        grouped: dict[str, list[DisplayItem]] = {}
        for item in self._model.items():
            if self._groups_source(item.source_id):
                grouped.setdefault(item.source_id, []).append(item)
                continue
            text = item.display_text
            if text is None:
                # 아직 번역이 없거나(안정화·번역 대기) 원문이 바뀌어 새 번역을 기다리는 중
                continue
            if self._present(item.item_id, item.source_id, text, item.bounds):
                alive.add(item.item_id)

        for source_id, items in grouped.items():
            # 번역을 기다리는 항목은 빼고, 보여 줄 번역문만 읽는 순서로 합친다.
            composed = compose_bubble(
                [
                    BubbleEntry(item.display_text, item.bounds)
                    for item in items
                    if item.display_text is not None
                ]
            )
            if composed is None:
                continue
            key = group_bubble_key(source_id)
            if self._present(key, source_id, composed.text, composed.bounds):
                alive.add(key)

        for item_id in [i for i in self._bubbles if i not in alive]:
            bubble = self._bubbles.pop(item_id)
            bubble.close()
            bubble.deleteLater()
        self._schedule_expiry()

    def _present(self, key: str, source_id: str, text: str, bounds: ScreenRegion) -> bool:
        """키의 말풍선을 (없으면 만들어) 원문 위치 옆에 보여 준다. 표시했으면 True."""
        placement = physical_region_to_logical(bounds)
        if placement is None:
            # 원문 위치가 어느 모니터에도 없으면(모니터 분리 등) 표시하지 않는다.
            return False
        text_rect, area = placement
        direction, accent = self._style_for(source_id)
        bubble = self._bubbles.get(key)
        if bubble is None:
            bubble = self._bubble_factory()
            bubble.set_background_opacity(self._opacity)
            self._bubbles[key] = bubble
        bubble.present(text, text_rect, area, direction, accent)
        return True

    def _schedule_expiry(self) -> None:
        """가장 먼저 지워질 항목의 시각에 맞춰 타이머를 다시 건다."""
        deadline = self._model.next_deadline()
        if deadline is None:
            self._expiry_timer.stop()
            return
        remaining_ms = max(0, int((deadline - self._model_now()) * 1000)) + 1
        self._expiry_timer.start(remaining_ms)

    def _model_now(self) -> float:
        """모델과 같은 시계의 현재 시각."""
        return self._model.now()

    def _on_expiry_timer(self) -> None:
        """유예/만료 시각이 되면 모델을 정리하고 말풍선을 갱신한다."""
        self._model.expire()
        self.refresh()
