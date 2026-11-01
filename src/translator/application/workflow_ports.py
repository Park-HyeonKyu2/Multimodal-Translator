"""번역 흐름(`TranslationWorkflow`)이 화면 쪽과 대화하는 인터페이스.

흐름 로직은 application 계층에 있고 Qt 위젯을 모른다. 화면(UI) 쪽이 이 두 인터페이스를
구현해 주입하므로, 흐름은 박스 창이나 말풍선 창 없이 가짜 구현만으로 테스트할 수 있다.

- `BoxPort`: 바운딩 박스를 묻고(있는지, 자동 감지 중인지, 어디를 캡처할지) 알리는(처리 상태) 창구
- `DisplayPort`: 번역 표시 상태를 반영하고 묻는 창구
"""

from typing import Protocol

from translator.application.pipeline import OcrResult
from translator.application.text_display import ApplyOutcome
from translator.core.models import ScreenRegion


class BoxPort(Protocol):
    """바운딩 박스 목록과 박스별 처리 상태. 박스 ID로 다룬다."""

    def switch_content(self, content_id: str) -> None:
        """박스 목록을 콘텐츠의 것으로 바꾼다 (지금 박스는 저장하고 닫고, 새 콘텐츠의 박스를 복원). 같은 콘텐츠면 그대로."""

    def box_ids(self) -> tuple[str, ...]:
        """지금 있는 모든 박스의 ID."""

    def exists(self, box_id: str) -> bool:
        """박스가 아직 있는지 (지워졌으면 False)."""

    def is_auto(self, box_id: str) -> bool:
        """박스가 자동 감지 중인지 (후속 OCR·재검사 대상인지)."""

    def capture_region(self, box_id: str) -> ScreenRegion | None:
        """박스가 지금 캡처할 물리 픽셀 영역 (없으면 None)."""

    def all_capture_regions(self) -> tuple[ScreenRegion, ...]:
        """모든 박스의 캡처 영역 (전체 화면 인식에서 제외할 영역)."""

    def mark_pending(self, box_id: str, request_id: int) -> None:
        """박스의 OCR 요청이 큐에 들어갔음을 기록한다."""

    def on_ocr(self, box_id: str, result: OcrResult, manual: bool) -> None:
        """박스의 인식 결과를 반영한다 (처리 상태 갱신, 수동이면 그 화면을 자동 감지 기준으로 기록)."""

    def on_failed(self, box_id: str, message: str) -> None:
        """박스의 인식·번역 실패를 기록한다."""


class DisplayPort(Protocol):
    """번역 표시 상태 (위치 추적, 안정화, 번역 대기 판단)."""

    def show_ocr(self, result: OcrResult, force_stable: bool = False) -> ApplyOutcome:
        """인식 결과를 반영하고 판단 결과를 돌려준다."""

    def show_translations(self, source_id: str, translations: dict[str, tuple[str, str]]) -> None:
        """번역 결과를 반영한다."""

    def texts_to_translate(self, source_id: str) -> tuple[str, ...]:
        """출처에서 안정됐지만 아직 번역하지 않은 원문들."""

    def item_bounds(self, source_id: str) -> tuple[ScreenRegion, ...]:
        """출처에서 표시 중인 텍스트들의 위치 (OCR 누락 판단용 자리)."""

    def has_items(self, source_id: str) -> bool:
        """출처에 표시 중인 텍스트가 있는지."""

    def remove_source(self, source_id: str) -> None:
        """출처의 표시를 모두 지운다."""

    def clear(self) -> None:
        """모든 표시를 지운다."""
