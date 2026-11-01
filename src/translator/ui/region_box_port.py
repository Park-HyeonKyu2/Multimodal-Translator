"""`RegionManager`를 번역 흐름의 `BoxPort`로 보이게 하는 어댑터.

흐름 로직(`TranslationWorkflow`)은 박스 창이나 `RegionSession`을 모른다. 이 어댑터가 박스 ID로
묻고 알리는 요청을 `RegionManager`의 세션 호출로 바꿔 준다.
"""

from translator.application.pipeline import OcrResult
from translator.core.models import ScreenRegion
from translator.ui.region_manager import RegionManager


class RegionBoxPort:
    """`BoxPort` 구현: 바운딩 박스 관리자(`RegionManager`)를 박스 ID 기준으로 다룬다."""

    def __init__(self, regions: RegionManager) -> None:
        """박스 관리자를 받는다."""
        # 박스들을 관리하는 객체
        self._regions = regions

    def switch_content(self, content_id: str) -> None:
        """박스 목록을 콘텐츠의 것으로 바꾼다 (저장된 박스 복원은 박스 관리자가 한다)."""
        self._regions.switch_content(content_id)

    def box_ids(self) -> tuple[str, ...]:
        """지금 있는 모든 박스의 ID."""
        return tuple(session.region_id for session in self._regions.sessions())

    def exists(self, box_id: str) -> bool:
        """박스가 아직 있는지 (지워졌으면 False)."""
        return self._regions.get(box_id) is not None

    def is_auto(self, box_id: str) -> bool:
        """박스가 자동 감지 중인지."""
        session = self._regions.get(box_id)
        return session is not None and session.config.auto_enabled

    def capture_region(self, box_id: str) -> ScreenRegion | None:
        """박스가 지금 캡처할 물리 픽셀 영역 (박스가 없으면 None)."""
        session = self._regions.get(box_id)
        return session.capture_region() if session is not None else None

    def all_capture_regions(self) -> tuple[ScreenRegion, ...]:
        """모든 박스의 현재 캡처 영역 (전체 화면 인식에서 제외할 영역)."""
        return tuple(
            region
            for region in (session.capture_region() for session in self._regions.sessions())
            if region is not None
        )

    def mark_pending(self, box_id: str, request_id: int) -> None:
        """박스의 OCR 요청이 큐에 들어갔음을 기록한다."""
        session = self._regions.get(box_id)
        if session is not None:
            session.mark_pending(request_id)

    def on_ocr(self, box_id: str, result: OcrResult, manual: bool) -> None:
        """박스의 인식 결과를 처리 상태에 반영하고, 수동이면 그 화면을 자동 감지 기준으로 기록한다."""
        session = self._regions.get(box_id)
        if session is None:
            return
        session.on_ocr(tuple(unit.text for unit in result.units))
        if manual:
            # 자동 감지가 같은 화면을 다시 OCR하지 않도록 이 화면의 지문을 기록한다 (추가 캡처 없음).
            session.mark_ocr_fingerprint(result.fingerprint, result.capture_region)

    def on_failed(self, box_id: str, message: str) -> None:
        """박스의 인식·번역 실패를 기록한다."""
        session = self._regions.get(box_id)
        if session is not None:
            session.on_failed(message)
