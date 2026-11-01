"""전체 흐름 테스트: 여러 박스, 편집/잠금 모드, 원문 기준 말풍선, 전체 화면 추적, 자동 감지,
텍스트 안정화, OCR 재검사, 종료.

가짜 캡처와 가짜 OCR(`ScreenTextRecognizer`), Mock 번역을 쓴다. offscreen 화면은 800×800,
배율 1이다. 가짜 OCR은 실제 글자를 읽지 않고 `FakeScreen`에 정해 둔 텍스트를 돌려주므로,
OCR 오인식이나 글자 단위 흔들림은 이 테스트로 검증할 수 없다 (Mock OCR의 한계).
"""

import pytest
from PySide6.QtCore import QRect, Qt

from translator.adapters.ocr.mock import MockTextRecognizer
from translator.adapters.translation.mock import MockTranslator
from translator.application.controller import TranslationOutcome
from translator.application.pipeline import OcrResult
from translator.application.recheck import ObservationScheduler
from translator.core.cancellation import CancellationToken
from translator.core.interfaces import ScreenCapturer, TextRecognizer
from translator.core.models import (
    CapturedImage,
    RecognizedText,
    ScreenRegion,
    TextGranularity,
    TextSegment,
    TranslationDirection,
)
from translator.main import build_coordinator
from translator.tuning import OCR_RECHECK_MAX_ATTEMPTS
from translator.application.workflow import FULLSCREEN_SOURCE_ID
from translator.ui.coordinator import FULLSCREEN_ACCENT

# offscreen 가상 데스크톱 크기 (물리 픽셀)
DESKTOP = ScreenRegion(0, 0, 800, 800)


class FakeScreen(ScreenCapturer):
    """가짜 화면: 캡처 요청을 기록하고 지정한 밝기의 단색 이미지를 돌려준다.

    가짜 OCR이 읽을 "화면 속 텍스트"도 여기에 둔다.
    - `region_text`: 박스 캡처에서 인식될 텍스트. None이면 Mock OCR 예시 문장, ""이면 텍스트 없음.
    - `full_texts`: 전체 화면에서 인식될 (텍스트, 위치) 목록. None이면 Mock OCR 예시 문장.
    - `region_texts`: 박스 캡처에서 인식될 (텍스트, 위치) 목록. 있으면 `region_text`보다 우선한다.
    """

    def __init__(self) -> None:
        """기록 목록과 화면 밝기, 화면 속 텍스트를 준비한다."""
        # 캡처 요청된 영역 (전체 화면은 None). 번역과 자동 감지 표본이 함께 기록된다.
        self.regions: list[ScreenRegion | None] = []
        # 화면 밝기를 바꿔 "화면이 바뀜"을 흉내 낸다.
        self.gray = 0
        self.region_text: str | None = None
        self.full_texts: list[tuple[str, ScreenRegion]] | None = None
        # 박스 캡처에서 인식될 (텍스트, 위치) 목록. 있으면 region_text보다 우선한다.
        self.region_texts: list[tuple[str, ScreenRegion]] | None = None

    def capture_region(self, region: ScreenRegion) -> CapturedImage:
        """영역을 기록하고 현재 밝기의 단색 이미지를 반환한다."""
        self.regions.append(region)
        pixel = bytes((self.gray, self.gray, self.gray, 255))
        return CapturedImage(region, region.width, region.height, pixel * (region.width * region.height))

    def capture_full_screen(self) -> CapturedImage:
        """전체 화면 요청을 기록하고 현재 밝기의 화면 이미지를 반환한다."""
        image = self.capture_region(DESKTOP)
        self.regions[-1] = None
        return image


class ScreenTextRecognizer(TextRecognizer):
    """`FakeScreen`에 정해 둔 텍스트를 돌려주는 가짜 OCR (지연 없음)."""

    def __init__(self, screen: FakeScreen) -> None:
        """읽을 가짜 화면을 받는다."""
        self._screen = screen
        self._mock = MockTextRecognizer(delay_seconds=0)
        # 엔진 호출 횟수 (OCR 캐시 재사용 확인용)
        self.calls = 0

    def recognize(self, image: CapturedImage, cancellation: CancellationToken) -> RecognizedText:
        """전체 화면이면 `full_texts`, 박스면 `region_text`를 인식 결과로 돌려준다."""
        self.calls += 1
        if image.region == DESKTOP:
            if self._screen.full_texts is None:
                return self._mock.recognize(image, cancellation)
            segments = tuple(
                TextSegment(text, bounds, TextGranularity.LINE)
                for text, bounds in self._screen.full_texts
            )
            return RecognizedText(" ".join(t for t, _ in self._screen.full_texts), segments=segments)
        if self._screen.region_texts is not None:
            segments = tuple(
                TextSegment(text, bounds, TextGranularity.LINE)
                for text, bounds in self._screen.region_texts
            )
            return RecognizedText(" ".join(t for t, _ in self._screen.region_texts), segments=segments)
        text = self._screen.region_text
        if text is None:
            return self._mock.recognize(image, cancellation)
        if not text:
            return RecognizedText("")
        # Mock OCR과 같은 자리(가운데 띠)에 텍스트가 있다고 본다.
        band = self._mock.recognize(image, cancellation).segments[0].bounds
        return RecognizedText(text, segments=(TextSegment(text, band, TextGranularity.LINE),))


@pytest.fixture
def app_parts(qapp):
    """가짜 화면·가짜 OCR·지연 없는 Mock 번역으로 조립한 coordinator (설정 저장 없음)."""
    screen = FakeScreen()
    recognizer = ScreenTextRecognizer(screen)
    coordinator = build_coordinator(
        capturer=screen,
        recognizer=recognizer,
        translator=MockTranslator(delay_seconds=0),
    )
    coordinator._capture_delay_ms = 0
    coordinator.start()
    yield coordinator, screen
    coordinator.shutdown()


def fast_observation(coordinator) -> None:
    """후속 OCR·재검사 간격을 테스트용으로 짧게 바꾼다."""
    coordinator.workflow.use_observation(
        ObservationScheduler(
            followup_seconds=0.03,
            recheck_base_seconds=0.03,
            recheck_max_seconds=0.12,
            max_attempts=OCR_RECHECK_MAX_ATTEMPTS,
        ),
        tick_ms=10,
    )


def ocr_results_of(coordinator) -> list[OcrResult]:
    """이후 도착하는 인식 결과를 모으는 목록을 반환한다."""
    received: list[OcrResult] = []
    coordinator.workflow.controller.ocr_ready.connect(received.append)
    return received


def translations_of(coordinator) -> list[TranslationOutcome]:
    """이후 도착하는 번역 결과를 모으는 목록을 반환한다."""
    received: list[TranslationOutcome] = []
    coordinator.workflow.controller.translations_ready.connect(received.append)
    return received


def translated_texts(outcomes: list[TranslationOutcome], source_id: str) -> list[str]:
    """출처의 번역 요청 원문들을 도착 순서대로 펼친다."""
    return [text for o in outcomes if o.source_id == source_id for text in o.translations]


def add_region(coordinator, rect: QRect):
    """영역 추가 버튼을 누르고 드래그 결과로 `rect`가 선택된 것처럼 처리한다."""
    coordinator._overlay.region_button.click()
    coordinator._selector._on_selection_made(rect)
    return coordinator._regions.sessions()[-1]


def add_auto_region(coordinator, rect: QRect):
    """자동 감지가 켜진 박스를 수동 번역 없이 추가한다 (자동 감지 경로만 시험할 때)."""
    coordinator._set_edit_mode(True)
    session = add_region(coordinator, rect)  # 편집 중 추가: 바로 번역하지 않는다
    session.detector._timer.setInterval(20)
    session.set_auto_enabled(True)
    # 편집을 끌 때 자동 감지 박스는 수동 번역하지 않는다 (감지기가 맡는다).
    coordinator._set_edit_mode(False)
    return session


def bubbles_for(coordinator, source_id: str):
    """출처의 말풍선 목록.

    박스는 박스마다 말풍선 하나(번역이 없으면 빈 목록), 전체 화면은 원문마다 말풍선을 놓는다.
    """
    overlay = coordinator._translation_overlay
    grouped = overlay.bubble_for_source(source_id)
    if grouped is not None:
        return [grouped]
    bubbles = (overlay.bubble_for(item.item_id) for item in overlay.model.items_for(source_id))
    return [bubble for bubble in bubbles if bubble is not None]


def items_for(coordinator, source_id: str):
    """출처의 표시 항목."""
    return coordinator._translation_overlay.model.items_for(source_id)


def text_bounds(coordinator, source_id: str) -> ScreenRegion:
    """출처의 첫 번째 원문 위치."""
    return items_for(coordinator, source_id)[0].bounds


def spy_ocr_requests(coordinator) -> list[tuple[tuple, dict]]:
    """컨트롤러의 OCR 요청 인자를 기록한다 (요청은 그대로 실행)."""
    calls: list[tuple[tuple, dict]] = []
    original = coordinator.workflow.controller.request_ocr

    def spy(*args, **kwargs):
        """인자를 기록하고 원래 메서드를 호출한다."""
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    coordinator.workflow.controller.request_ocr = spy
    return calls


# --- 기본 UI 흐름 -------------------------------------------------------------------


def test_starts_locked_without_regions(app_parts) -> None:
    """시작은 잠금 모드이고 박스가 없으면 영역 번역/자동 버튼을 쓸 수 없다."""
    coordinator, _ = app_parts

    assert not coordinator.edit_mode
    assert not coordinator._overlay.translate_region_button.isEnabled()
    assert not coordinator._overlay.full_screen_button.isChecked()
    assert coordinator._overlay.isVisible()


def test_added_region_is_faint_and_translated_below_source_text(qtbot, app_parts) -> None:
    """영역을 추가하면 박스는 흐린(클릭 통과) 상태가 되고, 번역문은 원문 텍스트 아래에 놓인다."""
    coordinator, _ = app_parts
    rect = QRect(100, 100, 300, 100)

    session = add_region(coordinator, rect)

    assert session.box.isVisible() and not session.box.is_interactive
    assert coordinator._overlay.translate_region_button.isEnabled()
    qtbot.waitUntil(lambda: bool(bubbles_for(coordinator, "box-1")), timeout=3000)
    text = text_bounds(coordinator, "box-1")
    [bubble] = bubbles_for(coordinator, "box-1")
    # 말풍선 기준은 박스가 아니라 OCR 원문 위치 (Mock OCR: 박스 가운데 띠)
    assert text != ScreenRegion(100, 100, 300, 100)
    assert bubble.geometry().top() > text.bottom
    assert bubble.geometry().top() < rect.bottom()  # 박스 아래가 아니라 원문 바로 아래


def test_two_regions_results_go_to_their_own_bubbles(qtbot, app_parts) -> None:
    """두 박스의 번역 결과가 각자의 원문 옆, 각자의 색으로 표시된다."""
    coordinator, _ = app_parts
    first = add_region(coordinator, QRect(50, 50, 300, 100))
    second = add_region(coordinator, QRect(50, 500, 200, 80))

    qtbot.waitUntil(
        lambda: bool(bubbles_for(coordinator, "box-1") and bubbles_for(coordinator, "box-2")),
        timeout=3000,
    )
    [bubble_1] = bubbles_for(coordinator, "box-1")
    [bubble_2] = bubbles_for(coordinator, "box-2")
    assert "300x100" in bubble_1.display_text and "200x80" in bubble_2.display_text
    assert bubble_1.geometry().top() < 200 < bubble_2.geometry().top()
    assert bubble_1._accent == first.accent and bubble_2._accent == second.accent


def test_direction_is_per_region_and_relayouts_immediately(qtbot, app_parts) -> None:
    """박스별 방향을 바꾸면 다시 번역하지 않고 그 박스의 말풍선만 원문 위로 옮겨진다."""
    coordinator, capturer = app_parts
    first = add_region(coordinator, QRect(100, 300, 300, 100))
    add_region(coordinator, QRect(100, 600, 300, 100))
    qtbot.waitUntil(lambda: len(coordinator._translation_overlay.bubbles()) == 2, timeout=3000)
    captures = len(capturer.regions)

    first.set_direction(TranslationDirection.TOP)

    [bubble_1] = bubbles_for(coordinator, "box-1")
    [bubble_2] = bubbles_for(coordinator, "box-2")
    assert bubble_1.geometry().bottom() < text_bounds(coordinator, "box-1").top
    assert bubble_2.geometry().top() > text_bounds(coordinator, "box-2").bottom
    assert len(capturer.regions) == captures  # 다시 캡처·번역하지 않음


def test_edit_mode_shows_boxes_and_leaving_it_retranslates_moved_box(qtbot, app_parts) -> None:
    """편집 모드에서 박스를 옮기면, 편집을 끌 때 새 위치를 인식하고 말풍선이 따라간다."""
    coordinator, _ = app_parts
    session = add_region(coordinator, QRect(100, 100, 300, 100))
    qtbot.waitUntil(lambda: bool(bubbles_for(coordinator, "box-1")), timeout=3000)
    [bubble] = bubbles_for(coordinator, "box-1")
    item_id = items_for(coordinator, "box-1")[0].item_id
    translations = translations_of(coordinator)

    coordinator._overlay.edit_button.click()
    assert coordinator.edit_mode and session.box.is_interactive
    assert coordinator._status_window.isVisible()  # 편집 중에는 설정 창이 보인다
    session.box.move(session.box.x(), session.box.y() + 300)
    coordinator._overlay.edit_button.click()

    assert not coordinator.edit_mode and not session.box.is_interactive
    qtbot.waitUntil(lambda: text_bounds(coordinator, "box-1").top > 300, timeout=3000)
    # 같은 원문이 이동한 것이므로 같은 항목·같은 말풍선이 새 위치로 따라가고, 다시 번역하지 않는다.
    assert items_for(coordinator, "box-1")[0].item_id == item_id
    assert bubbles_for(coordinator, "box-1") == [bubble]
    assert bubble.geometry().top() > 400
    qtbot.wait(50)
    assert translations == []


def test_translate_button_leaves_edit_mode(qtbot, app_parts) -> None:
    """편집 중에 영역 번역을 누르면 편집이 꺼지고(박스 흐림) 인식·번역된다."""
    coordinator, _ = app_parts
    add_region(coordinator, QRect(100, 100, 300, 100))
    coordinator._overlay.edit_button.click()
    received = ocr_results_of(coordinator)

    coordinator._overlay.translate_region_button.click()

    assert not coordinator.edit_mode
    qtbot.waitUntil(lambda: len(received) == 1, timeout=3000)


def test_lock_makes_boxes_bubbles_and_status_click_through(qtbot, app_parts) -> None:
    """잠금(편집 꺼짐)에서는 박스·말풍선·상태 창이 클릭을 통과시키고, 오버레이는 클릭을 받는다."""
    coordinator, _ = app_parts
    session = add_region(coordinator, QRect(50, 50, 300, 100))
    qtbot.waitUntil(lambda: bool(bubbles_for(coordinator, "box-1")), timeout=3000)
    transparent = Qt.WindowType.WindowTransparentForInput

    assert session.box.windowHandle().flags() & transparent
    assert bubbles_for(coordinator, "box-1")[0].windowFlags() & transparent
    assert coordinator._status_window.windowFlags() & transparent
    assert not coordinator._overlay.windowHandle().flags() & transparent

    coordinator._overlay.edit_button.click()
    assert not session.box.windowHandle().flags() & transparent
    assert not coordinator._status_window.windowHandle().flags() & transparent


def test_selection_cancel_restores_overlay_and_boxes(app_parts) -> None:
    """선택을 취소하면 오버레이와 기존 박스가 다시 나타난다."""
    coordinator, _ = app_parts
    session = add_region(coordinator, QRect(50, 50, 300, 100))

    coordinator._overlay.region_button.click()
    assert not session.box.isVisible()
    coordinator._selector.cancel()

    assert coordinator._overlay.isVisible()
    assert session.box.isVisible()


def test_box_menu_translation_leaves_edit_mode(qtbot, app_parts) -> None:
    """박스 우클릭 메뉴의 "지금 번역"도 오버레이 버튼처럼 편집 모드를 끄고 인식한다."""
    coordinator, _ = app_parts
    session = add_region(coordinator, QRect(50, 50, 300, 100))
    coordinator._overlay.edit_button.click()
    received = ocr_results_of(coordinator)

    session.box.translate_requested.emit()

    assert not coordinator.edit_mode and not session.box.is_interactive
    qtbot.waitUntil(lambda: any(r.source_id == "box-1" for r in received), timeout=3000)


def test_box_menu_auto_on_leaves_edit_mode_but_auto_off_does_not(app_parts) -> None:
    """박스 메뉴에서 자동 감지를 켜면 편집 모드가 꺼지고, 끄는 것은 편집 모드에 영향이 없다."""
    coordinator, _ = app_parts
    session = add_region(coordinator, QRect(50, 50, 300, 100))
    coordinator._overlay.edit_button.click()

    session.box.auto_toggled.emit(True)
    assert not coordinator.edit_mode and session.config.auto_enabled

    coordinator._overlay.edit_button.click()
    session.box.auto_toggled.emit(False)
    assert coordinator.edit_mode and not session.config.auto_enabled


# --- 자동 감지, 안정화, 재검사 --------------------------------------------------------


def test_auto_detection_works_while_box_is_faint(qtbot, app_parts) -> None:
    """박스가 흐린 잠금 상태여도 자동 감지와 번역은 계속 동작한다."""
    coordinator, screen = app_parts
    fast_observation(coordinator)
    session = add_region(coordinator, QRect(100, 100, 300, 100))
    session.detector._timer.setInterval(20)
    translations = translations_of(coordinator)
    qtbot.waitUntil(lambda: bool(bubbles_for(coordinator, "box-1")), timeout=3000)

    session.set_auto_enabled(True)
    screen.region_text = "새 자막"
    screen.gray = 255

    assert not session.box.is_interactive
    qtbot.waitUntil(lambda: "새 자막" in translated_texts(translations, "box-1"), timeout=3000)
    assert coordinator._overlay.auto_button.isChecked()  # 모든 박스가 자동 감지 중


def test_auto_detection_reuses_detector_frame_without_recapture(qtbot, app_parts) -> None:
    """자동 감지의 OCR은 감지기가 찍은 프레임을 그대로 쓴다 (같은 화면을 다시 캡처하지 않음)."""
    coordinator, screen = app_parts
    session = add_region(coordinator, QRect(100, 100, 300, 100))
    qtbot.waitUntil(lambda: bool(bubbles_for(coordinator, "box-1")), timeout=3000)
    session.detector._timer.setInterval(20)
    requests = spy_ocr_requests(coordinator)
    results = ocr_results_of(coordinator)

    session.set_auto_enabled(True)
    screen.gray = 200
    qtbot.waitUntil(lambda: bool(results), timeout=3000)

    _args, kwargs = requests[0]
    frame = kwargs["frame"]
    assert frame is not None and frame.image.bgra[0] == 200
    assert results[0].frame_id == frame.frame_id
    assert results[0].captured_at == frame.captured_at


def test_enabling_auto_after_manual_translation_does_not_repeat_same_screen(
    qtbot, app_parts
) -> None:
    """회귀 테스트: 수동 번역 직후 자동 감지를 켜도 같은 화면을 다시 OCR하지 않는다."""
    coordinator, capturer = app_parts
    received = ocr_results_of(coordinator)
    session = add_region(coordinator, QRect(100, 100, 300, 100))
    session.detector._timer.setInterval(20)
    qtbot.waitUntil(lambda: len(received) == 1, timeout=3000)

    coordinator._overlay.auto_button.click()
    samples_before = len(capturer.regions)
    qtbot.waitUntil(lambda: len(capturer.regions) >= samples_before + 4, timeout=3000)
    assert len(received) == 1

    capturer.gray = 255  # 새 자막이 나타남
    qtbot.waitUntil(lambda: len(received) == 2, timeout=3000)


def test_manual_translation_captures_once(qtbot, app_parts) -> None:
    """수동 번역은 한 번만 캡처한다 (자동 감지 기준 화면은 그 결과의 지문으로 기록)."""
    coordinator, capturer = app_parts

    add_region(coordinator, QRect(100, 100, 300, 100))
    qtbot.waitUntil(lambda: bool(bubbles_for(coordinator, "box-1")), timeout=3000)
    qtbot.wait(50)

    assert capturer.regions == [ScreenRegion(100, 100, 300, 100)]


def test_auto_text_waits_until_stable_and_is_translated_once(qtbot, app_parts) -> None:
    """자동 감지 텍스트는 같은 텍스트가 연속으로 인식된 뒤 번역하고, 같은 원문은 다시 번역하지 않는다."""
    coordinator, screen = app_parts
    fast_observation(coordinator)
    results = ocr_results_of(coordinator)
    translations = translations_of(coordinator)
    screen.region_text = "안녕하세요"

    add_auto_region(coordinator, QRect(100, 100, 300, 100))

    qtbot.waitUntil(lambda: bool(translations), timeout=3000)
    # 첫 인식만으로는 번역하지 않는다: 후속 OCR로 한 번 더 같은 텍스트를 확인한 뒤 번역.
    assert len(results) >= 2
    assert results[0].units[0].text == "안녕하세요"
    qtbot.wait(150)
    assert translated_texts(translations, "box-1") == ["안녕하세요"]


def test_typing_text_keeps_previous_translation_until_new_text_is_stable(qtbot, app_parts) -> None:
    """타자 효과: 글자가 늘어나는 동안 이전 번역을 유지하고, 늘어난 텍스트가 안정되면 새로 번역한다."""
    coordinator, screen = app_parts
    fast_observation(coordinator)
    translations = translations_of(coordinator)
    screen.region_text = "Hello"
    add_auto_region(coordinator, QRect(100, 100, 300, 100))
    qtbot.waitUntil(lambda: "Hello" in translated_texts(translations, "box-1"), timeout=3000)
    [item] = items_for(coordinator, "box-1")

    screen.region_text = "Hello world"
    screen.gray = 255

    qtbot.waitUntil(lambda: "Hello world" in translated_texts(translations, "box-1"), timeout=3000)
    [same_item] = items_for(coordinator, "box-1")
    assert same_item.item_id == item.item_id  # 같은 자리에서 글자가 늘어난 같은 항목
    assert "Hello world" in same_item.display_text


def test_missed_text_is_found_by_recheck_without_screen_change(qtbot, app_parts) -> None:
    """OCR이 처음에 텍스트를 놓쳐도, 감지 기준 아래의 작은 화면 차이에서 재검사로 다시 인식한다."""
    coordinator, screen = app_parts
    fast_observation(coordinator)
    results = ocr_results_of(coordinator)
    screen.region_text = ""  # 첫 OCR은 텍스트를 놓친다
    add_auto_region(coordinator, QRect(100, 100, 300, 100))
    qtbot.waitUntil(lambda: bool(results), timeout=3000)
    assert results[0].units == ()

    # 화면 변화 감지 기준(밝기 차 40) 아래의 작은 차이: 자동 감지는 다시 요청하지 않는다.
    screen.gray = 1
    screen.region_text = "놓친 대사"

    qtbot.waitUntil(lambda: bool(bubbles_for(coordinator, "box-1")), timeout=3000)
    assert "놓친 대사" in items_for(coordinator, "box-1")[0].source_text


def test_recheck_is_limited_when_screen_has_no_text(qtbot, app_parts) -> None:
    """텍스트가 없는 화면은 정해진 횟수까지만 다시 확인한다 (무한 반복 방지)."""
    coordinator, screen = app_parts
    fast_observation(coordinator)
    results = ocr_results_of(coordinator)
    screen.region_text = ""
    add_auto_region(coordinator, QRect(100, 100, 300, 100))

    qtbot.waitUntil(lambda: len(results) == 1 + OCR_RECHECK_MAX_ATTEMPTS, timeout=3000)
    qtbot.wait(400)  # 최대 재검사 간격(0.12초)의 몇 배를 기다려도 더 요청하지 않는다
    assert len(results) == 1 + OCR_RECHECK_MAX_ATTEMPTS
    # 같은 화면의 재검사는 OCR 캐시를 써서 엔진을 다시 부르지 않는다.
    assert all(r.ocr_reused for r in results[1:])


def test_text_kept_when_ocr_misses_but_pixels_are_unchanged(qtbot, app_parts) -> None:
    """텍스트 자리의 픽셀이 그대로인데 OCR 결과에서 빠지면 번역문을 유지한다 (OCR 누락)."""
    coordinator, screen = app_parts
    fast_observation(coordinator)
    screen.region_text = "대사"
    add_auto_region(coordinator, QRect(100, 100, 300, 100))
    qtbot.waitUntil(lambda: bool(bubbles_for(coordinator, "box-1")), timeout=3000)
    results = ocr_results_of(coordinator)

    screen.region_text = ""  # OCR이 놓침
    screen.gray = 1  # 감지 기준 아래의 작은 차이 → 재검사 경로로 새 OCR 실행
    coordinator.workflow.observation.after_result("box-1", unstable=False, needs_recheck=True)

    qtbot.waitUntil(lambda: any(not r.units for r in results), timeout=3000)
    qtbot.wait(100)
    assert bubbles_for(coordinator, "box-1")  # 사라진 것으로 보지 않는다
    assert items_for(coordinator, "box-1")[0].ocr_misses >= 1


def test_text_removed_when_it_really_disappears(qtbot, app_parts) -> None:
    """텍스트 자리의 픽셀이 바뀌고 OCR 결과에서도 빠지면 유예 시간 뒤 번역문을 지운다."""
    coordinator, screen = app_parts
    fast_observation(coordinator)
    coordinator._translation_overlay.model._grace = 0.05
    screen.region_text = "대사"
    add_auto_region(coordinator, QRect(100, 100, 300, 100))
    qtbot.waitUntil(lambda: bool(bubbles_for(coordinator, "box-1")), timeout=3000)

    screen.region_text = ""
    screen.gray = 255  # 화면이 바뀌고 텍스트가 사라짐

    qtbot.waitUntil(lambda: not items_for(coordinator, "box-1"), timeout=3000)
    assert bubbles_for(coordinator, "box-1") == []


# --- 전체 화면 -----------------------------------------------------------------------


def test_full_screen_uses_text_positions_and_excludes_regions(qtbot, app_parts) -> None:
    """전체 화면은 원문 좌표 기준으로 말풍선을 놓고, 박스 영역은 제외해 중복 번역하지 않는다."""
    coordinator, _ = app_parts
    add_region(coordinator, QRect(50, 50, 300, 100))
    qtbot.waitUntil(lambda: bool(bubbles_for(coordinator, "box-1")), timeout=3000)
    requests = spy_ocr_requests(coordinator)

    coordinator._overlay.full_screen_button.click()

    qtbot.waitUntil(lambda: bool(bubbles_for(coordinator, FULLSCREEN_SOURCE_ID)), timeout=3000)
    args, kwargs = requests[0]
    assert args == (FULLSCREEN_SOURCE_ID, None)
    assert kwargs["exclude"] == (coordinator._regions.get("box-1").capture_region(),)
    [bubble] = bubbles_for(coordinator, FULLSCREEN_SOURCE_ID)
    assert bubble._accent == FULLSCREEN_ACCENT
    assert "800x800" in bubble.display_text
    assert coordinator._overlay.full_screen_button.isChecked()


def test_full_screen_also_translates_regions_without_results(qtbot, app_parts) -> None:
    """아직 결과가 없는 박스는 전체 화면 번역 때 함께 인식된다 (박스 안 텍스트 누락 방지)."""
    coordinator, _ = app_parts
    coordinator._set_edit_mode(True)
    add_region(coordinator, QRect(50, 50, 300, 100))  # 편집 중에 추가해 아직 번역되지 않음
    received = ocr_results_of(coordinator)

    coordinator._overlay.full_screen_button.click()

    qtbot.waitUntil(
        lambda: {r.source_id for r in received} == {FULLSCREEN_SOURCE_ID, "box-1"}, timeout=3000
    )


def test_empty_region_during_full_screen_does_not_show_no_text_notice(qtbot, app_parts) -> None:
    """전체 화면 번역에 딸려 인식한 박스가 비어 있어도 "텍스트 없음" 알림을 띄우지 않는다."""
    coordinator, screen = app_parts
    coordinator._set_edit_mode(True)
    add_region(coordinator, QRect(50, 50, 300, 100))  # 편집 중 추가: 아직 결과 없음
    received = ocr_results_of(coordinator)
    screen.region_text = ""

    coordinator._overlay.full_screen_button.click()

    qtbot.waitUntil(lambda: {r.source_id for r in received} == {FULLSCREEN_SOURCE_ID, "box-1"}, timeout=3000)
    assert coordinator._status_window.status_label.text() == ""


def test_manual_empty_region_shows_no_text_notice(qtbot, app_parts) -> None:
    """사용자가 직접 번역한 박스에 텍스트가 없으면 "텍스트 없음"을 알린다."""
    coordinator, screen = app_parts
    screen.region_text = ""
    received = ocr_results_of(coordinator)

    add_region(coordinator, QRect(50, 50, 300, 100))

    qtbot.waitUntil(lambda: bool(received), timeout=3000)
    assert coordinator._status_window.status_label.text() != ""


def test_full_screen_tracks_moving_text_without_retranslation(qtbot, app_parts) -> None:
    """전체 화면을 켜 두면 원문이 움직일 때 위치만 갱신하고 다시 번역하지 않는다."""
    coordinator, screen = app_parts
    fast_observation(coordinator)
    screen.full_texts = [("움직이는 대사", ScreenRegion(100, 100, 200, 20))]
    translations = translations_of(coordinator)
    coordinator._overlay.full_screen_button.click()
    qtbot.waitUntil(lambda: bool(bubbles_for(coordinator, FULLSCREEN_SOURCE_ID)), timeout=3000)
    detector = coordinator.workflow.fullscreen_detector
    qtbot.waitUntil(lambda: detector.is_running, timeout=3000)
    detector._timer.setInterval(20)
    [item] = items_for(coordinator, FULLSCREEN_SOURCE_ID)

    screen.full_texts = [("움직이는 대사", ScreenRegion(160, 140, 200, 20))]
    screen.gray = 255  # 화면이 바뀜

    qtbot.waitUntil(
        lambda: text_bounds(coordinator, FULLSCREEN_SOURCE_ID) == ScreenRegion(160, 140, 200, 20),
        timeout=3000,
    )
    [moved] = items_for(coordinator, FULLSCREEN_SOURCE_ID)
    assert moved.item_id == item.item_id
    qtbot.wait(100)
    assert translated_texts(translations, FULLSCREEN_SOURCE_ID) == ["움직이는 대사"]


def test_full_screen_same_text_in_two_places_are_separate_items(qtbot, app_parts) -> None:
    """같은 문자열이 멀리 떨어진 두 곳에 있으면 서로 다른 항목(말풍선)으로 표시한다."""
    coordinator, screen = app_parts
    screen.full_texts = [
        ("예", ScreenRegion(50, 50, 40, 20)),
        ("예", ScreenRegion(600, 700, 40, 20)),
    ]

    coordinator._overlay.full_screen_button.click()

    qtbot.waitUntil(lambda: len(bubbles_for(coordinator, FULLSCREEN_SOURCE_ID)) == 2, timeout=3000)
    bounds = {item.bounds for item in items_for(coordinator, FULLSCREEN_SOURCE_ID)}
    assert bounds == {ScreenRegion(50, 50, 40, 20), ScreenRegion(600, 700, 40, 20)}


def test_full_screen_removes_text_that_disappears(qtbot, app_parts) -> None:
    """전체 화면 추적 중 사라진 텍스트의 번역문은 정리된다."""
    coordinator, screen = app_parts
    coordinator._translation_overlay.model._grace = 0.05
    screen.full_texts = [("사라질 대사", ScreenRegion(100, 100, 200, 20))]
    coordinator._overlay.full_screen_button.click()
    qtbot.waitUntil(lambda: bool(bubbles_for(coordinator, FULLSCREEN_SOURCE_ID)), timeout=3000)
    detector = coordinator.workflow.fullscreen_detector
    qtbot.waitUntil(lambda: detector.is_running, timeout=3000)
    detector._timer.setInterval(20)

    screen.full_texts = []
    screen.gray = 255

    qtbot.waitUntil(lambda: not items_for(coordinator, FULLSCREEN_SOURCE_ID), timeout=3000)


def test_full_screen_does_not_ocr_unchanged_screen_again(qtbot, app_parts) -> None:
    """전체 화면을 켠 뒤 화면이 그대로면 추적 감지기가 같은 화면을 다시 OCR하지 않는다."""
    coordinator, screen = app_parts
    received = ocr_results_of(coordinator)
    coordinator._overlay.full_screen_button.click()
    qtbot.waitUntil(lambda: bool(received), timeout=3000)
    detector = coordinator.workflow.fullscreen_detector
    qtbot.waitUntil(lambda: detector.is_running, timeout=3000)
    detector._timer.setInterval(20)
    captures = len(screen.regions)

    qtbot.waitUntil(lambda: len(screen.regions) >= captures + 4, timeout=3000)

    assert len(received) == 1


def test_full_screen_toggle_off_clears_and_stops_tracking(qtbot, app_parts) -> None:
    """전체 화면을 끄면 그 번역문을 지우고 추적과 예약을 멈추며, 늦게 온 결과도 버린다."""
    coordinator, screen = app_parts
    coordinator._overlay.full_screen_button.click()
    qtbot.waitUntil(lambda: bool(bubbles_for(coordinator, FULLSCREEN_SOURCE_ID)), timeout=3000)
    qtbot.waitUntil(lambda: coordinator.workflow.fullscreen_detector.is_running, timeout=3000)
    late = coordinator.workflow.controller  # 같은 컨트롤러로 늦게 오는 결과를 흉내 낸다

    coordinator._overlay.full_screen_button.click()

    assert not coordinator._overlay.full_screen_button.isChecked()
    assert not coordinator.workflow.fullscreen_detector.is_running
    assert items_for(coordinator, FULLSCREEN_SOURCE_ID) == []
    assert coordinator.workflow.observation.due_at(FULLSCREEN_SOURCE_ID) is None
    stale = OcrResult(
        FULLSCREEN_SOURCE_ID, 99, 99.0, DESKTOP,
        units=(), unit_fingerprints=(), probes=(), fingerprint=None, ocr_reused=False,
    )
    coordinator.workflow.handle_ocr_ready(stale)
    assert items_for(coordinator, FULLSCREEN_SOURCE_ID) == []
    assert late.is_pending(FULLSCREEN_SOURCE_ID) is False


def test_region_selection_pauses_full_screen_tracking(qtbot, app_parts) -> None:
    """영역 선택 화면이 떠 있는 동안 전체 화면 추적을 멈추고, 선택이 끝나면 다시 켠다."""
    coordinator, _ = app_parts
    coordinator._overlay.full_screen_button.click()
    detector = coordinator.workflow.fullscreen_detector
    qtbot.waitUntil(lambda: detector.is_running, timeout=3000)

    coordinator._overlay.region_button.click()
    assert not detector.is_running
    coordinator._selector.cancel()

    assert detector.is_running


# --- 삭제, 동시 요청, 종료 ----------------------------------------------------------


def test_removing_region_keeps_other_region_working(qtbot, app_parts) -> None:
    """박스 하나를 지우면 그 말풍선만 사라지고, 다른 박스는 계속 인식된다."""
    coordinator, _ = app_parts
    first = add_region(coordinator, QRect(50, 50, 300, 100))
    add_region(coordinator, QRect(50, 500, 300, 100))
    qtbot.waitUntil(lambda: len(coordinator._translation_overlay.bubbles()) == 2, timeout=3000)
    received = ocr_results_of(coordinator)

    first.box.remove_requested.emit()

    assert coordinator._regions.get("box-1") is None
    assert bubbles_for(coordinator, "box-1") == []
    assert len(bubbles_for(coordinator, "box-2")) == 1
    coordinator._overlay.translate_region_button.click()
    qtbot.waitUntil(lambda: [r.source_id for r in received] == ["box-2"], timeout=3000)


def test_result_for_deleted_region_is_ignored(qtbot, app_parts) -> None:
    """삭제된 박스의 인식·번역 결과가 늦게 도착해도 말풍선을 만들지 않는다."""
    coordinator, _ = app_parts
    add_region(coordinator, QRect(50, 50, 300, 100))
    qtbot.waitUntil(lambda: bool(bubbles_for(coordinator, "box-1")), timeout=3000)
    coordinator._remove_region("box-1")

    region = ScreenRegion(0, 0, 10, 10)
    stale = OcrResult("box-1", 1, 999.0, region, (), (), (), None, False)
    coordinator.workflow.handle_ocr_ready(stale)
    coordinator.workflow.handle_translations_ready(TranslationOutcome("box-1", 1, {"x": ("y", "mock")}))

    assert coordinator._translation_overlay.bubbles() == {}
    assert items_for(coordinator, "box-1") == []


def test_region_removed_before_delayed_translation_is_ignored(qtbot, app_parts) -> None:
    """영역을 추가한 직후 지워도, 예약된 번역이 지워진 박스를 건드리지 않는다."""
    coordinator, capturer = app_parts
    coordinator._capture_delay_ms = 50
    session = add_region(coordinator, QRect(50, 50, 300, 100))

    coordinator._remove_region(session.region_id)
    qtbot.wait(150)

    assert capturer.regions == []
    assert coordinator._translation_overlay.bubbles() == {}


def test_many_regions_requested_at_once_all_get_results(qtbot, app_parts) -> None:
    """여러 박스를 한꺼번에 번역해도 각 박스가 자기 결과를 받는다 (서로 취소하지 않음)."""
    coordinator, _ = app_parts
    coordinator._set_edit_mode(True)
    for index in range(4):
        add_region(coordinator, QRect(50, 50 + index * 150, 200 + index * 10, 60))
    coordinator._set_edit_mode(False)  # 편집 중 추가된 박스를 한꺼번에 번역

    qtbot.waitUntil(
        lambda: all(bubbles_for(coordinator, f"box-{i}") for i in range(1, 5)), timeout=5000
    )
    for index in range(4):
        [bubble] = bubbles_for(coordinator, f"box-{index + 1}")
        assert f"{200 + index * 10}x60" in bubble.display_text


def test_close_button_requests_quit(qtbot, app_parts) -> None:
    """오버레이의 닫기 버튼은 프로그램 종료를 요청한다."""
    coordinator, _ = app_parts

    with qtbot.waitSignal(coordinator.quit_requested, timeout=1000):
        coordinator._overlay.close_button.click()


def test_shutdown_while_work_is_running_stops_everything(qtbot, app_parts) -> None:
    """작업 중에 종료해도 멈추지 않고, 감지기·예약이 멈추며 늦은 결과가 창을 만들지 않는다."""
    coordinator, screen = app_parts
    fast_observation(coordinator)
    coordinator._overlay.full_screen_button.click()
    session = add_region(coordinator, QRect(50, 50, 300, 100))
    session.set_auto_enabled(True)
    quits: list[bool] = []
    coordinator.quit_requested.connect(lambda: quits.append(True))

    coordinator._overlay.close_button.click()
    coordinator.shutdown()  # 두 번 불려도 안전하다
    screen.gray = 255
    qtbot.wait(100)

    assert quits == [True]
    assert not coordinator.workflow.fullscreen_detector.is_running
    assert not coordinator.workflow.is_observing
    assert coordinator._translation_overlay.bubbles() == {}
    assert not coordinator.workflow.controller.is_busy


# --- 박스 하나 = 말풍선 하나 ---------------------------------------------------------------


def test_box_shows_all_translations_in_one_bubble_in_reading_order(qtbot, app_parts) -> None:
    """박스 안 텍스트가 여러 개여도 말풍선은 하나이며, 다른 줄은 줄을 바꾸고 같은 줄은 이어 쓴다."""
    coordinator, screen = app_parts
    # 박스 (100, 100, 400, 200) 안: 첫 줄에 두 텍스트(멀리 떨어져 서로 다른 대사), 아래에 한 줄
    screen.region_texts = [
        ("아래 줄", ScreenRegion(120, 220, 120, 20)),
        ("오른쪽", ScreenRegion(360, 120, 100, 20)),
        ("왼쪽", ScreenRegion(120, 120, 100, 20)),
    ]

    add_region(coordinator, QRect(100, 100, 400, 200))

    qtbot.waitUntil(lambda: len(items_for(coordinator, "box-1")) == 3 and bool(bubbles_for(coordinator, "box-1")), timeout=3000)
    [bubble] = bubbles_for(coordinator, "box-1")
    assert len(coordinator._translation_overlay.bubbles()) == 1
    lines = bubble.display_text.split("\n")
    assert len(lines) == 2
    assert lines[0].index("왼쪽") < lines[0].index("오른쪽")
    assert "아래 줄" in lines[1]
    # 말풍선은 원문 전체의 아래(가장 아래 원문 밑)에 놓인다.
    assert bubble.geometry().top() > 240


def test_full_screen_keeps_one_bubble_per_text(qtbot, app_parts) -> None:
    """전체 화면은 텍스트가 흩어져 있으므로 박스와 달리 원문마다 말풍선을 놓는다."""
    coordinator, screen = app_parts
    screen.full_texts = [
        ("위", ScreenRegion(50, 50, 100, 20)),
        ("아래", ScreenRegion(50, 600, 100, 20)),
    ]

    coordinator._overlay.full_screen_button.click()

    qtbot.waitUntil(lambda: len(bubbles_for(coordinator, FULLSCREEN_SOURCE_ID)) == 2, timeout=3000)
    assert coordinator._translation_overlay.bubble_for_source(FULLSCREEN_SOURCE_ID) is None


def test_settings_opacity_slider_also_changes_bubble_background(qtbot, app_parts) -> None:
    """편집 모드 설정 창의 불투명도를 바꾸면 설정 창과 번역 말풍선 배경이 함께 바뀐다."""
    coordinator, _ = app_parts
    add_region(coordinator, QRect(100, 100, 300, 100))
    qtbot.waitUntil(lambda: bool(bubbles_for(coordinator, "box-1")), timeout=3000)
    [bubble] = bubbles_for(coordinator, "box-1")
    status = coordinator._status_window

    coordinator._overlay.edit_button.click()
    status.opacity_slider.setValue(30)

    assert status.background_opacity == 0.3
    assert bubble._opacity == 0.3
    # 그 뒤에 새로 생기는 말풍선도 같은 불투명도로 시작한다.
    add_region(coordinator, QRect(100, 500, 300, 100))
    coordinator._overlay.edit_button.click()  # 편집을 끄면 새 박스를 번역한다
    qtbot.waitUntil(lambda: bool(bubbles_for(coordinator, "box-2")), timeout=3000)
    assert bubbles_for(coordinator, "box-2")[0]._opacity == 0.3


def test_background_work_flag_follows_controller_shutdown(app_parts, monkeypatch) -> None:
    """종료할 때 컨트롤러가 끝나지 않은 작업을 알리면 coordinator가 그 상태를 기억한다."""
    coordinator, _ = app_parts
    assert coordinator.background_work_finished

    monkeypatch.setattr(coordinator.workflow.controller, "shutdown", lambda: False)
    coordinator.shutdown()

    assert coordinator.background_work_finished is False
    monkeypatch.undo()
    coordinator.workflow.controller.shutdown()  # 실제 작업 큐도 정리한다


def test_finish_process_exits_without_waiting_only_when_work_remains(app_parts, monkeypatch) -> None:
    """추론이 남아 있을 때만 정리 없이 종료하고, 아니면 종료 코드를 그대로 돌려준다."""
    from translator.main import finish_process

    coordinator, _ = app_parts
    exits: list[int] = []

    assert finish_process(coordinator, 0, hard_exit=exits.append) == 0
    assert exits == []

    monkeypatch.setattr(coordinator.workflow.controller, "shutdown", lambda: False)
    coordinator.shutdown()
    finish_process(coordinator, 3, hard_exit=exits.append)

    assert exits == [3]
    monkeypatch.undo()
    coordinator.workflow.controller.shutdown()  # 실제 작업 큐도 정리한다


def test_slow_full_screen_uses_longer_interval_and_shows_notice(qapp) -> None:
    """OCR이 CPU로 실행되면 전체 화면 확인 간격을 늘리고, 켤 때 느리다는 안내를 보여 준다."""
    from translator.main import SLOW_FULLSCREEN_NOTICE
    from translator.tuning import FULLSCREEN_DETECT_INTERVAL_CPU_MS, FULLSCREEN_DETECT_INTERVAL_MS

    screen = FakeScreen()
    slow = build_coordinator(
        capturer=screen,
        recognizer=ScreenTextRecognizer(screen),
        translator=MockTranslator(delay_seconds=0),
        slow_full_screen=True,
    )
    fast = build_coordinator(
        capturer=screen,
        recognizer=ScreenTextRecognizer(screen),
        translator=MockTranslator(delay_seconds=0),
    )
    notices: list[str] = []
    for coordinator in (slow, fast):
        coordinator._status_window.show_notice = lambda text, is_error: notices.append(text)
        coordinator.start()
    try:
        assert slow.workflow.fullscreen_detector._timer.interval() == FULLSCREEN_DETECT_INTERVAL_CPU_MS
        assert fast.workflow.fullscreen_detector._timer.interval() == FULLSCREEN_DETECT_INTERVAL_MS

        slow._set_full_screen(True)
        fast._set_full_screen(True)

        assert notices == [SLOW_FULLSCREEN_NOTICE]
    finally:
        slow.shutdown()
        fast.shutdown()


# --- 세션 전환 ----------------------------------------------------------------------------


def test_content_change_clears_bubbles_and_reads_auto_box_again(qtbot, app_parts) -> None:
    """콘텐츠가 바뀌면 이전 말풍선을 지우고, 자동 감지 박스는 현재 화면을 새 세션에서 다시 인식해 번역한다."""
    coordinator, screen = app_parts
    fast_observation(coordinator)
    results = ocr_results_of(coordinator)
    translations = translations_of(coordinator)
    screen.region_text = "First scene"
    add_auto_region(coordinator, QRect(100, 100, 300, 100))
    qtbot.waitUntil(lambda: "First scene" in translated_texts(translations, "box-1"), timeout=3000)
    assert bubbles_for(coordinator, "box-1")
    old_session = coordinator.workflow.session
    results.clear()
    translations.clear()
    # 장면이 바뀌면 픽셀도 바뀐다 (같은 픽셀이면 OCR 캐시가 이전 결과를 재사용한다).
    screen.region_text = "Second scene"
    screen.gray = 255

    new_context = coordinator.workflow.change_content("other-game")

    assert new_context != old_session and new_context.content_id == "other-game"
    assert not items_for(coordinator, "box-1")  # 이전 세션의 표시는 바로 사라진다.
    qtbot.waitUntil(lambda: "Second scene" in translated_texts(translations, "box-1"), timeout=3000)
    assert results and all(result.session == new_context for result in results)
    assert all(outcome.session == new_context for outcome in translations)
    assert "Second scene" in bubbles_for(coordinator, "box-1")[0].display_text


def test_manual_box_is_not_translated_again_after_content_change(qtbot, app_parts) -> None:
    """수동으로 번역한 박스는 콘텐츠가 바뀌면 말풍선만 지워지고, 사용자가 다시 요청하기 전에는 번역하지 않는다."""
    coordinator, screen = app_parts
    translations = translations_of(coordinator)
    screen.region_text = "Manual scene"
    add_region(coordinator, QRect(100, 100, 300, 100))
    qtbot.waitUntil(lambda: bool(translated_texts(translations, "box-1")), timeout=3000)
    translations.clear()

    coordinator.workflow.change_content("other-game")
    qtbot.wait(300)

    assert not items_for(coordinator, "box-1")
    assert translations == []


def test_content_detector_switches_session_and_shuts_down_with_the_app(qapp, qtbot) -> None:
    """식별기를 주입하면 활성 콘텐츠가 확정될 때 번역 세션의 콘텐츠가 바뀌고, 종료하면 감시도 끝난다."""
    from translator.adapters.content.mock import ScriptedContentDetector
    from translator.core.content import ContentIdentity, IdentificationMethod

    screen = FakeScreen()
    coordinator = build_coordinator(
        capturer=screen,
        recognizer=ScreenTextRecognizer(screen),
        translator=MockTranslator(delay_seconds=0),
        content_detector=ScriptedContentDetector(
            [ContentIdentity("app:game.exe", "game", IdentificationMethod.PROCESS_NAME)]
        ),
    )
    monitor = coordinator.workflow._content_monitor
    monitor._timer.setInterval(5)  # 테스트에서는 확인 간격을 짧게 한다.
    coordinator.start()
    try:
        qtbot.waitUntil(lambda: coordinator.workflow.session.content_id == "app:game.exe", timeout=3000)
    finally:
        coordinator.shutdown()

    assert not monitor.is_running


def test_without_content_detector_session_stays_on_default_content(qapp) -> None:
    """식별기를 주지 않으면 콘텐츠를 식별하지 않고 기본 콘텐츠 세션을 유지한다 (기존 동작)."""
    from translator.core.session import DEFAULT_CONTENT_ID

    screen = FakeScreen()
    coordinator = build_coordinator(
        capturer=screen,
        recognizer=ScreenTextRecognizer(screen),
        translator=MockTranslator(delay_seconds=0),
    )
    coordinator.start()
    try:
        assert coordinator.workflow._content_monitor is None
        assert coordinator.workflow.session.content_id == DEFAULT_CONTENT_ID
    finally:
        coordinator.shutdown()
