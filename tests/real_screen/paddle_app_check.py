"""실제 화면에서 PaddleOCR를 연결한 앱 흐름을 확인하는 스크립트 (Phase 2 2단계).

`tests/test_paddle_real_screen.py`가 별도 프로세스로 실행한다. 실제 앱과 같은 구성
(MSS 캡처 + PaddleOCR + 공용 추론 큐 + 텍스트 안정화 + 말풍선 + 번역 엔진)을 조립하고,
화면에 "가짜 게임 대사 창"(그라데이션 배경, 반투명 대사 패널, 외곽선 글자, 화자 이름)을 띄워
다음을 확인한다.

1. 박스 수동 번역: 일본어 대사를 읽고, 화자 이름과 대사를 따로 묶고, 말풍선을 원문 아래에 놓는다.
2. 타자 효과: 자동 감지 중 글자가 하나씩 나타날 때 부분 번역 없이 완성된 문장만 번역한다.
3. 대사 교체: 다음 대사로 바뀌면 새로 번역하고, 이전 번역을 다시 보여 주지 않는다.
4. 원문 이동: 글자가 박스 안에서 움직이면 다시 번역하지 않고 말풍선만 따라간다.
5. 전체 화면: 다른 모니터의 텍스트를 읽고, 박스 안 텍스트는 제외하며, 가만히 있는 동안의 OCR 횟수를 센다.

`--shutdown-during-load`를 주면 모델을 불러오는 도중 종료해도 프로세스가 멈추지 않는지만 확인한다.

개인정보: 전체 화면 인식은 사용자 바탕 화면 전체를 읽으므로, 결과에는 검사용 창의 텍스트와
일치 여부, 개수, 시간만 남긴다. 그 밖에 인식한 텍스트는 출력하지 않는다. 캡처 이미지는 저장하지 않는다.

이 검사는 직접 그린 가짜 창으로 하는 시험이다. 실제 게임·영상 화면(글꼴 효과, 움직이는 배경,
전체 화면 독점 모드)을 검증하지는 않는다.
"""

import difflib
import json
import logging
import os
import re
import sys
import time
from collections.abc import Callable

from PySide6.QtCore import QEventLoop, QPoint, QPointF, QRect, Qt
from PySide6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QApplication, QWidget

from translator.adapters.ocr.factory import select_text_recognizer
from translator.adapters.translation.factory import select_translator
from translator.core.cancellation import CancellationToken
from translator.core.interfaces import TextRecognizer
from translator.core.models import CapturedImage, RecognizedText
from translator.main import build_coordinator, finish_process
from translator.application.workflow import FULLSCREEN_SOURCE_ID

# 한글 글자 (말풍선에 한국어 번역이 있는지 확인)
HANGUL = re.compile(r"[가-힣]")
# 같은 텍스트로 볼 최소 유사도 (검사 판정용. 앱 동작에는 쓰지 않는다)
MATCH_RATIO = 0.85
# 대사 창 크기 (논리 픽셀)
DIALOGUE_SIZE = (860, 170)
# 화자 이름과 대사 글자 크기 (픽셀). 크기 비가 0.7보다 작아 다른 텍스트로 묶여야 한다.
SPEAKER_FONT_PX = 18
DIALOGUE_FONT_PX = 28
# 타자 효과에서 한 글자가 나타나는 간격 (초). 초당 약 20글자.
TYPING_INTERVAL_SECONDS = 0.05

# 시나리오별 원문
SCENE1_SPEAKER = "アリス"
SCENE1_LINES = ("おはようございます、勇者さま。", "今日は一緒にお城へ向かいましょう。")
SCENE2_SPEAKER = "Guard"
SCENE2_TEXT = "The ancient gate opens only when the three moons align."
SCENE3_TEXT = "Then we must wait here until midnight."
SIGN_LINES = ("Welcome to Riverside Village",)


def normalize(text: str) -> str:
    """비교용으로 공백을 지운다 (OCR 띄어쓰기 차이를 무시)."""
    return "".join(text.split())


def similarity(left: str, right: str) -> float:
    """두 텍스트의 유사도 (0~1, 공백 무시)."""
    return difflib.SequenceMatcher(None, normalize(left), normalize(right)).ratio()


def pump(app: QApplication, seconds: float) -> None:
    """정해진 시간 동안 Qt 이벤트를 처리한다 (워커 결과 시그널을 받기 위함)."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 20)
        time.sleep(0.005)


def pump_until(app: QApplication, predicate: Callable[[], bool], timeout: float) -> bool:
    """조건이 참이 되거나 시간이 다 될 때까지 이벤트를 처리한다. 조건 결과를 돌려준다."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 20)
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


class DialogueWindow(QWidget):
    """게임 대사 창을 흉내 낸 창: 그라데이션 배경, 반투명 패널, 화자 이름, 외곽선이 있는 대사."""

    def __init__(self, title_font: str = "Yu Gothic UI") -> None:
        """테두리 없이 항상 위에 뜨고, 나타날 때 포커스를 가져가지 않는 창을 만든다."""
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        # 글꼴 이름
        self._font_family = title_font
        # 화자 이름과 대사 줄
        self.speaker = ""
        self.lines: tuple[str, ...] = ()
        # 글자 전체를 옮기는 오프셋 (원문 이동 시나리오)
        self.offset = QPoint(0, 0)

    def set_content(self, speaker: str, lines: tuple[str, ...]) -> None:
        """화자 이름과 대사를 바꾸고 다시 그린다."""
        self.speaker, self.lines = speaker, lines
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt 이름 규칙)
        """배경, 패널, 화자 이름, 외곽선 대사를 그린다."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        width, height = self.width(), self.height()
        # 장면 배경: 대각선 그라데이션과 흐린 무늬 (글자가 아닌 잡음)
        gradient = QLinearGradient(0, 0, width, height)
        gradient.setColorAt(0, QColor(30, 45, 80))
        gradient.setColorAt(1, QColor(90, 40, 90))
        painter.fillRect(self.rect(), gradient)
        painter.setPen(QPen(QColor(255, 255, 255, 25), 2))
        for index in range(0, width, 37):
            painter.drawLine(index, 0, index + 60, height)
        # 반투명 대사 패널
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 150))
        painter.drawRoundedRect(QRect(10, 10, width - 20, height - 20), 12, 12)

        origin = self.offset
        if self.speaker:
            speaker_font = QFont(self._font_family)
            speaker_font.setPixelSize(SPEAKER_FONT_PX)
            speaker_font.setBold(True)
            painter.setFont(speaker_font)
            painter.setPen(QColor(255, 215, 120))
            painter.drawText(QPoint(32, 40) + origin, self.speaker)
        dialogue_font = QFont(self._font_family)
        dialogue_font.setPixelSize(DIALOGUE_FONT_PX)
        for index, line in enumerate(self.lines):
            # 외곽선(검정) 위에 흰 글자를 채운다 (게임 자막에서 흔한 효과)
            path = QPainterPath()
            path.addText(QPointF(32 + origin.x(), 88 + index * 40 + origin.y()), dialogue_font, line)
            painter.strokePath(path, QPen(QColor(0, 0, 0), 3))
            painter.fillPath(path, QColor(250, 250, 250))
        painter.end()


class TimedRecognizer(TextRecognizer):
    """실제 OCR을 감싸 호출마다 이미지 크기와 걸린 시간을 기록한다."""

    def __init__(self, inner: TextRecognizer) -> None:
        """감쌀 OCR 엔진을 받는다."""
        self._inner = inner
        # (끝난 시각, 걸린 시간(ms), 너비, 높이) 목록
        self.calls: list[tuple[float, float, int, int]] = []
        # 첫 호출이 시작되었는지 (모델 로딩 중 종료 시나리오)
        self.started = False
        # (시작 시각, 너비, 높이) 목록. 끝나지 않은 호출도 센다.
        self.starts: list[tuple[float, int, int]] = []

    @property
    def device(self) -> str | None:
        """감싼 엔진의 추론 장치 (PaddleOCR가 아니면 None)."""
        settings = getattr(self._inner, "settings", None)
        return getattr(settings, "device", None)

    def recognize(self, image: CapturedImage, cancellation: CancellationToken) -> RecognizedText:
        """감싼 엔진으로 인식하고 시간을 기록한다."""
        self.started = True
        self.starts.append((time.monotonic(), image.width, image.height))
        started = time.perf_counter()
        try:
            return self._inner.recognize(image, cancellation)
        finally:
            self.calls.append(
                (time.monotonic(), (time.perf_counter() - started) * 1000, image.width, image.height)
            )


class Recorder:
    """컨트롤러의 인식·번역 결과를 시각과 함께 기록한다."""

    def __init__(self, coordinator) -> None:
        """coordinator의 컨트롤러 시그널에 연결한다."""
        # (시각, 출처, 원문 목록)
        self.ocr: list[tuple[float, str, list[str]]] = []
        # (시각, 출처, 번역 요청 원문 목록)
        self.translations: list[tuple[float, str, list[str]]] = []
        # (시각, 출처, 오류 메시지)
        self.failures: list[tuple[float, str, str]] = []
        controller = coordinator.workflow.controller  # noqa: SLF001 (검사 스크립트)
        controller.ocr_ready.connect(self._on_ocr)
        controller.translations_ready.connect(self._on_translations)
        controller.failed.connect(lambda source, message: self.failures.append((time.monotonic(), source, message)))

    def _on_ocr(self, result) -> None:
        """인식 결과의 원문 목록을 기록한다."""
        self.ocr.append((time.monotonic(), result.source_id, [unit.text for unit in result.units]))

    def _on_translations(self, outcome) -> None:
        """번역된 원문 목록을 기록한다."""
        self.translations.append((time.monotonic(), outcome.source_id, list(outcome.translations)))

    def translated_since(self, source_id: str, since: float) -> list[str]:
        """시각 이후 출처에서 번역된 원문을 순서대로 펼친다."""
        return [t for at, source, texts in self.translations if source == source_id and at >= since for t in texts]


def bubble_text(coordinator, source_id: str) -> str:
    """박스 말풍선의 표시 텍스트 (없으면 빈 문자열)."""
    bubble = coordinator._translation_overlay.bubble_for_source(source_id)  # noqa: SLF001
    return bubble.display_text if bubble is not None and bubble.isVisible() else ""


def bubble_geometry(coordinator, source_id: str) -> QRect | None:
    """박스 말풍선의 위치 (없으면 None)."""
    bubble = coordinator._translation_overlay.bubble_for_source(source_id)  # noqa: SLF001
    return bubble.geometry() if bubble is not None and bubble.isVisible() else None


def items(coordinator, source_id: str):
    """출처의 표시 항목 목록."""
    return coordinator._translation_overlay.model.items_for(source_id)  # noqa: SLF001


def best_match(texts: list[str], expected: str) -> tuple[str | None, float]:
    """목록에서 기대 텍스트와 가장 비슷한 것과 그 유사도."""
    scored = [(text, similarity(text, expected)) for text in texts]
    return max(scored, key=lambda pair: pair[1], default=(None, 0.0))


def scene_manual(app, coordinator, recorder, timed, window, results) -> None:
    """1. 박스 수동 번역 (모델 첫 로딩 포함)."""
    window.set_content(SCENE1_SPEAKER, SCENE1_LINES)
    pump(app, 0.5)
    started = time.monotonic()
    coordinator._on_region_selected(window.geometry())  # noqa: SLF001 (드래그 선택을 흉내 낸다)
    shown = pump_until(app, lambda: bool(bubble_text(coordinator, "box-1")), 90)
    texts = [item.source_text for item in items(coordinator, "box-1")]
    dialogue = "".join(SCENE1_LINES)
    dialogue_text, dialogue_ratio = best_match(texts, dialogue)
    speaker_text, speaker_ratio = best_match(texts, SCENE1_SPEAKER)
    dialogue_item = next((i for i in items(coordinator, "box-1") if i.source_text == dialogue_text), None)
    geometry = bubble_geometry(coordinator, "box-1")
    results["scene1_manual"] = {
        "bubble_shown": shown,
        "bubble_text": bubble_text(coordinator, "box-1"),
        "bubble_has_hangul": bool(HANGUL.search(bubble_text(coordinator, "box-1"))),
        "seconds_to_bubble": round(time.monotonic() - started, 2),
        "first_ocr_ms": round(timed.calls[0][1]) if timed.calls else None,
        "units": texts,
        "dialogue_ratio": round(dialogue_ratio, 3),
        "speaker_ratio": round(speaker_ratio, 3),
        "speaker_separate": speaker_ratio >= MATCH_RATIO and speaker_text != dialogue_text,
        "bubble_below_text": bool(
            geometry is not None and dialogue_item is not None and geometry.top() >= dialogue_item.bounds.bottom
        ),
        "failures": [message for _, _, message in recorder.failures],
    }


def scene_typing(app, coordinator, recorder, timed, window, results) -> None:
    """2. 타자 효과: 자동 감지 중 글자가 하나씩 나타난다."""
    session = coordinator._regions.get("box-1")  # noqa: SLF001
    session.set_auto_enabled(True)
    window.set_content(SCENE2_SPEAKER, ())
    pump(app, 2.5)  # 이전 대사가 사라진 화면을 처리하게 둔다.
    started = time.monotonic()
    calls_before = len(timed.calls)
    for count in range(1, len(SCENE2_TEXT) + 1):
        window.set_content(SCENE2_SPEAKER, (SCENE2_TEXT[:count],))
        pump(app, TYPING_INTERVAL_SECONDS)
    typed_at = time.monotonic()
    pump_until(
        app,
        lambda: best_match(recorder.translated_since("box-1", started), SCENE2_TEXT)[1] >= MATCH_RATIO,
        20,
    )
    final_at = next(
        (at for at, source, texts in recorder.translations
         if source == "box-1" and at >= started and best_match(texts, SCENE2_TEXT)[1] >= MATCH_RATIO),
        None,
    )
    pump(app, 2.0)  # 완성 뒤 추가 번역이 생기는지 본다.
    results.setdefault("bubbles", {})["scene2"] = bubble_text(coordinator, "box-1")
    translated = recorder.translated_since("box-1", started)
    dialogue_translations = [t for t in translated if similarity(t, SCENE2_SPEAKER) < MATCH_RATIO]
    final_matches = [t for t in dialogue_translations if similarity(t, SCENE2_TEXT) >= MATCH_RATIO]
    results["scene2_typing"] = {
        "typing_seconds": round(typed_at - started, 2),
        "translated": translated,
        # 완성 문장이 아닌 대사 번역 (부분 번역 또는 오인식)
        "partial_translations": [t for t in dialogue_translations if t not in final_matches],
        "final_translations": len(final_matches),
        "final_ratio": round(best_match(dialogue_translations, SCENE2_TEXT)[1], 3),
        "seconds_after_typing_to_translation": round(final_at - typed_at, 2) if final_at else None,
        "ocr_calls_during_scene": len(timed.calls) - calls_before,
        "ocr_ms": [round(ms) for _, ms, _, _ in timed.calls[calls_before:]],
    }


def scene_replace(app, coordinator, recorder, window, results) -> None:
    """3. 대사 교체: 다음 대사로 한 번에 바뀐다."""
    # 바꾸기 전 말풍선 (이전 대사의 번역)
    old_bubble = bubble_text(coordinator, "box-1")
    started = time.monotonic()
    window.set_content(SCENE2_SPEAKER, (SCENE3_TEXT,))
    # 바뀐 대사를 처음 인식한 뒤에도 이전 대사 번역이 보이는지 감시한다.
    stale_seen = False
    new_seen_at: float | None = None
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        pump(app, 0.02)
        if new_seen_at is None:
            for at, source, texts in recorder.ocr:
                if source == "box-1" and at >= started and best_match(texts, SCENE3_TEXT)[1] >= MATCH_RATIO:
                    new_seen_at = at
        current = bubble_text(coordinator, "box-1")
        if new_seen_at is not None and old_bubble and current == old_bubble:
            stale_seen = True
        new_translated = any(
            source == "box-1" and at >= started and best_match(texts, SCENE3_TEXT)[1] >= MATCH_RATIO
            for at, source, texts in recorder.translations
        )
        if new_translated and current and current != old_bubble:
            break
    translated_at = next(
        (at for at, source, texts in recorder.translations
         if source == "box-1" and at >= started and best_match(texts, SCENE3_TEXT)[1] >= MATCH_RATIO),
        None,
    )
    results["scene3_replace"] = {
        "new_translated": translated_at is not None,
        "seconds_to_translation": round(translated_at - started, 2) if translated_at else None,
        "stale_translation_shown": stale_seen,
        "bubble_has_old": bool(old_bubble) and bubble_text(coordinator, "box-1") == old_bubble,
    }
    results.setdefault("bubbles", {})["scene3"] = bubble_text(coordinator, "box-1")


def scene_move(app, coordinator, recorder, window, results) -> None:
    """4. 원문 이동: 글자가 박스 안에서 오른쪽 아래로 움직인다."""
    pump(app, 1.0)
    before = bubble_geometry(coordinator, "box-1")
    started = time.monotonic()
    window.offset = QPoint(150, 12)
    window.update()
    moved = pump_until(
        app,
        lambda: (geometry := bubble_geometry(coordinator, "box-1")) is not None
        and before is not None
        and geometry.left() - before.left() >= 100,
        10,
    )
    pump(app, 2.0)
    after = bubble_geometry(coordinator, "box-1")
    results["scene4_move"] = {
        "bubble_followed": moved,
        "bubble_dx": (after.left() - before.left()) if before and after else None,
        "retranslated": recorder.translated_since("box-1", started),
    }


def scene_fullscreen(app, coordinator, recorder, timed, sign, results) -> None:
    """5. 전체 화면: 다른 모니터의 안내판을 읽고, 박스 안 대사는 제외한다."""
    started = time.monotonic()
    calls_before = len(timed.calls)
    coordinator._set_full_screen(True)  # noqa: SLF001
    found = pump_until(
        app,
        lambda: any(
            similarity(item.source_text, SIGN_LINES[0]) >= MATCH_RATIO and item.translated_text
            for item in items(coordinator, FULLSCREEN_SOURCE_ID)
        ),
        60,
    )
    found_at = time.monotonic()
    full_items = items(coordinator, FULLSCREEN_SOURCE_ID)
    full_calls = [c for c in timed.calls[calls_before:] if c[2] * c[3] > 2_000_000]
    # 가만히 있는 동안 (검사 창은 바뀌지 않음) OCR 횟수와 CPU 사용 시간
    idle_seconds = 10.0
    idle_start_calls = len(timed.calls)
    idle_start_starts = len(timed.starts)
    # 전체 화면 감지기가 "화면이 바뀌었다"며 OCR을 요청한 횟수
    detector_frames: list[float] = []
    coordinator.workflow.fullscreen_detector.frame_ready.connect(lambda _frame: detector_frames.append(time.monotonic()))  # noqa: SLF001
    # 대기 시작 시점에 이미 실행 중인 전체 화면 OCR이 있었는지
    running_at_idle_start = len(timed.starts) > len(timed.calls)
    cpu_before = time.process_time()
    pump(app, idle_seconds)
    cpu_used = time.process_time() - cpu_before
    idle_calls = timed.calls[idle_start_calls:]
    results["scene5_fullscreen"] = {
        "sign_found": found,
        "seconds_to_sign_translation": round(found_at - started, 2),
        "fullscreen_ocr_ms": [round(ms) for _, ms, _, _ in full_calls],
        "fullscreen_image": [full_calls[0][2], full_calls[0][3]] if full_calls else None,
        "fullscreen_text_count": len(full_items),
        "box_dialogue_in_fullscreen": any(
            similarity(item.source_text, SCENE3_TEXT) >= MATCH_RATIO for item in full_items
        ),
        "idle_seconds": idle_seconds,
        "idle_fullscreen_ocr": sum(1 for c in idle_calls if c[2] * c[3] > 2_000_000),
        "idle_box_ocr": sum(1 for c in idle_calls if c[2] * c[3] <= 2_000_000),
        "idle_fullscreen_ocr_started": sum(1 for c in timed.starts[idle_start_starts:] if c[1] * c[2] > 2_000_000),
        "idle_fullscreen_change_frames": len(detector_frames),
        "ocr_running_at_idle_start": running_at_idle_start,
        "idle_cpu_percent_of_one_core": round(cpu_used / idle_seconds * 100),
    }
    coordinator._set_full_screen(False)  # noqa: SLF001
    pump(app, 0.5)


def build(app: QApplication):
    """실제 앱과 같은 구성으로 coordinator를 만든다 (설정 저장 없음). (coordinator, OCR 기록, 번역 서버)를 돌려준다.

    OCR 엔진·장치와 번역 엔진은 앱과 같이 환경 변수로 고른다 (`TRANSLATOR_OCR_DEVICE=cpu`로 CPU 검사,
    `TRANSLATOR_TRANSLATION_ENGINE=mock`으로 Mock 번역 검사). 앱이 띄울 번역 서버가 있으면 시작한다.
    """
    selection = select_text_recognizer({**os.environ, "TRANSLATOR_OCR_ENGINE": "paddle"})
    translation = select_translator()
    if translation.server is not None:
        translation.server.start()
    timed = TimedRecognizer(selection.recognizer)
    coordinator = build_coordinator(
        recognizer=timed, translator=translation.translator, slow_full_screen=selection.slow_full_screen
    )
    coordinator.start()
    return coordinator, timed, translation


def run_scenarios():
    """시나리오 1~5를 차례로 실행하고 (결과, 종료한 coordinator)를 돌려준다. 예외가 나도 결과에 남긴다."""
    app = QApplication(sys.argv)
    results: dict = {"screens": [[s.name(), s.geometry().getRect(), s.devicePixelRatio()] for s in app.screens()]}
    coordinator, timed, translation = build(app)
    results["device"] = timed.device
    settings = getattr(translation.translator, "settings", None)
    results["translation_engine"] = getattr(settings, "engine_name", type(translation.translator).__name__)
    recorder = Recorder(coordinator)
    primary = app.primaryScreen().geometry()
    window = DialogueWindow()
    window.setGeometry(QRect(primary.x() + 480, primary.y() + 220, *DIALOGUE_SIZE))
    window.show()
    # 안내판: 두 번째 모니터가 있으면 그쪽, 없으면 주 모니터 아래쪽
    screens = app.screens()
    target = screens[1].geometry() if len(screens) > 1 else primary
    sign = DialogueWindow(title_font="Segoe UI")
    sign.setGeometry(QRect(target.x() + 300, target.y() + (300 if len(screens) > 1 else 600), 600, 120))
    sign.set_content("", SIGN_LINES)
    sign.show()
    try:
        scene_manual(app, coordinator, recorder, timed, window, results)
        scene_typing(app, coordinator, recorder, timed, window, results)
        scene_replace(app, coordinator, recorder, window, results)
        scene_move(app, coordinator, recorder, window, results)
        scene_fullscreen(app, coordinator, recorder, timed, sign, results)
        results["failures"] = [[source, message] for _, source, message in recorder.failures]
    except Exception as exc:  # 검사 실패 원인을 결과로 남긴다
        logging.getLogger("translator").exception("검사 중 예외")
        results["error"] = repr(exc)
    finally:
        started = time.monotonic()
        coordinator.shutdown()
        pump(app, 0.2)
        results["shutdown_seconds"] = round(time.monotonic() - started, 2)
        results["background_work_finished"] = coordinator.background_work_finished
        if translation.server is not None:
            translation.server.stop()
        window.close()
        sign.close()
        app.quit()
    return results, coordinator


def run_shutdown_during_load():
    """모델을 불러오는 도중 종료를 요청한다. (결과, 종료한 coordinator)를 돌려준다."""
    app = QApplication(sys.argv)
    coordinator, timed, translation = build(app)
    window = DialogueWindow()
    primary = app.primaryScreen().geometry()
    window.setGeometry(QRect(primary.x() + 480, primary.y() + 220, *DIALOGUE_SIZE))
    window.set_content(SCENE1_SPEAKER, SCENE1_LINES)
    window.show()
    results: dict = {}
    try:
        pump(app, 0.3)
        coordinator._on_region_selected(window.geometry())  # noqa: SLF001
        results["ocr_started"] = pump_until(app, lambda: timed.started, 10)
        pump(app, 0.2)  # 모델 로딩이 진행되는 중
    finally:
        started = time.monotonic()
        coordinator.shutdown()
        pump(app, 0.2)
        results["shutdown_seconds"] = round(time.monotonic() - started, 2)
        results["ocr_finished_before_exit"] = bool(timed.calls)
        results["background_work_finished"] = coordinator.background_work_finished
        if translation.server is not None:
            translation.server.stop()
        window.close()
        app.quit()
    return results, coordinator


def main() -> int:
    """모드에 맞는 검사를 실행하고 결과 JSON을 마지막 줄에 출력한다."""
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    if "--shutdown-during-load" in sys.argv:
        results, coordinator = run_shutdown_during_load()
    else:
        results, coordinator = run_scenarios()
    # 결과를 출력한 벽시계 시각. 실행한 쪽이 프로세스가 실제로 끝난 시각과 비교한다 (종료 지연 확인).
    results["printed_at"] = time.time()
    print(json.dumps(results, ensure_ascii=False), flush=True)
    exit_code = 1 if "error" in results else 0
    # 실제 앱 진입점과 같은 종료 처리 (끝나지 않은 추론이 있으면 기다리지 않고 끝낸다)
    return finish_process(coordinator, exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
