"""오버레이와 상태·설정 창(예전 결과 창) UI 테스트."""

from PySide6.QtCore import Qt

from translator.core.models import TranslationDirection
from translator.ui import result_window
from translator.ui.overlay import MiniOverlay
from translator.ui.result_window import ResultWindow


def test_overlay_is_frameless_always_on_top_tool_window(qtbot) -> None:
    """오버레이는 테두리 없는 항상 위 도구 창이다."""
    overlay = MiniOverlay()
    qtbot.addWidget(overlay)

    flags = overlay.windowFlags()
    assert flags & Qt.WindowType.WindowStaysOnTopHint
    assert flags & Qt.WindowType.FramelessWindowHint
    assert flags & Qt.WindowType.Tool


def test_overlay_buttons_emit_requests(qtbot) -> None:
    """오버레이 버튼이 각 요청 시그널을 보낸다."""
    overlay = MiniOverlay()
    qtbot.addWidget(overlay)
    overlay.show()

    assert overlay.region_button.text() == "영역 추가"
    with qtbot.waitSignal(overlay.region_requested, timeout=1000):
        qtbot.mouseClick(overlay.region_button, Qt.MouseButton.LeftButton)
    # 전체 화면 버튼은 켜고 끄는 버튼이다 (켜져 있는 동안 화면 전체를 추적).
    with qtbot.waitSignal(overlay.full_screen_toggled, timeout=1000) as blocker:
        qtbot.mouseClick(overlay.full_screen_button, Qt.MouseButton.LeftButton)
    assert blocker.args == [True]
    with qtbot.waitSignal(overlay.full_screen_toggled, timeout=1000) as blocker:
        qtbot.mouseClick(overlay.full_screen_button, Qt.MouseButton.LeftButton)
    assert blocker.args == [False]
    with qtbot.waitSignal(overlay.close_requested, timeout=1000):
        qtbot.mouseClick(overlay.close_button, Qt.MouseButton.LeftButton)
    assert not overlay.isVisible()


def test_region_actions_are_disabled_until_region_exists(qtbot) -> None:
    """박스가 없으면 영역 번역/자동 버튼을 쓸 수 없다."""
    overlay = MiniOverlay()
    qtbot.addWidget(overlay)

    assert not overlay.translate_region_button.isEnabled()
    assert not overlay.auto_button.isEnabled()

    overlay.set_region_available(True)
    with qtbot.waitSignal(overlay.region_translate_requested, timeout=1000):
        overlay.translate_region_button.click()
    with qtbot.waitSignal(overlay.auto_toggled, timeout=1000) as blocker:
        overlay.auto_button.click()
    assert blocker.args == [True]

    overlay.set_region_available(False)
    assert not overlay.auto_button.isChecked()


def test_set_checked_helpers_do_not_emit(qtbot) -> None:
    """자동/편집/전체 화면 버튼 상태를 코드로 맞출 때는 시그널을 보내지 않는다."""
    overlay = MiniOverlay()
    qtbot.addWidget(overlay)
    overlay.set_region_available(True)
    emitted: list[bool] = []
    overlay.auto_toggled.connect(emitted.append)
    overlay.edit_toggled.connect(emitted.append)

    overlay.full_screen_toggled.connect(emitted.append)

    overlay.set_auto_checked(True)
    overlay.set_edit_checked(True)
    overlay.set_full_screen_checked(True)

    assert overlay.auto_button.isChecked() and overlay.edit_button.isChecked()
    assert overlay.full_screen_button.isChecked()
    assert emitted == []


def test_edit_button_toggles(qtbot) -> None:
    """편집 버튼은 켜고 끄는 상태를 시그널로 보낸다 (꺼짐 = 잠금)."""
    overlay = MiniOverlay()
    qtbot.addWidget(overlay)

    with qtbot.waitSignal(overlay.edit_toggled, timeout=1000) as blocker:
        overlay.edit_button.click()
    assert blocker.args == [True]


def make_status(qtbot) -> ResultWindow:
    """상태 창을 만든다."""
    window = ResultWindow()
    qtbot.addWidget(window)
    return window


def test_status_window_starts_locked_and_hidden(qtbot) -> None:
    """상태 창은 숨겨진 잠금 상태(조작 요소 숨김, 클릭 통과)로 시작한다."""
    window = make_status(qtbot)

    assert not window.isVisible()
    assert not window.is_interactive
    assert window.opacity_slider.isHidden()
    assert window.windowFlags() & Qt.WindowType.WindowTransparentForInput


def test_notices_show_and_hide_automatically_when_locked(qtbot, monkeypatch) -> None:
    """잠금 상태의 알림은 잠깐 보였다가 스스로 사라진다 (닫기 버튼을 누를 수 없으므로)."""
    monkeypatch.setattr(result_window, "STATUS_NOTICE_HIDE_MS", 30)
    monkeypatch.setattr(result_window, "STATUS_ERROR_HIDE_MS", 60)
    window = make_status(qtbot)

    window.show_no_text()
    assert window.isVisible()
    assert window.status_label.text() == "인식된 텍스트가 없습니다."
    assert window.status_label.property("error") is False
    qtbot.waitUntil(lambda: not window.isVisible(), timeout=1000)

    window.show_error("캡처 실패")
    assert window.status_label.text() == "오류: 캡처 실패"
    assert window.status_label.property("error") is True
    qtbot.waitUntil(lambda: not window.isVisible(), timeout=1000)


def test_notice_is_plain_text(qtbot) -> None:
    """알림에 섞인 태그는 해석하지 않고 글자 그대로 보여 준다."""
    window = make_status(qtbot)

    window.show_error("<img src=y>")

    assert window.status_label.textFormat() == Qt.TextFormat.PlainText
    assert window.status_label.text() == "오류: <img src=y>"


def test_edit_mode_shows_settings_panel(qtbot) -> None:
    """편집 모드에서는 설정 창으로 보이고 클릭을 받는다. 끄면 숨고 클릭을 통과시킨다."""
    window = make_status(qtbot)
    transparent = Qt.WindowType.WindowTransparentForInput

    window.set_interactive(True)
    assert window.isVisible()
    assert window.opacity_slider.isVisible() and window.direction_combo.isVisible()
    assert not window.windowHandle().flags() & transparent

    window.set_interactive(False)
    assert not window.isVisible()
    assert window.windowHandle().flags() & transparent


def test_opacity_is_adjustable_clamped_and_announced(qtbot) -> None:
    """불투명도는 슬라이더로 바꿀 수 있고, 최솟값 아래로 내려가지 않으며, 변경을 알린다."""
    window = make_status(qtbot)
    changes: list[float] = []
    window.opacity_changed.connect(changes.append)

    window.opacity_slider.setValue(40)
    assert window.background_opacity == 0.4
    assert window.opacity_label.text() == "말풍선 배경 불투명도 40%"

    window.set_background_opacity(0.0)
    assert window.background_opacity == 0.2  # 클릭을 받을 수 있는 최솟값
    assert window.opacity_slider.value() == 20
    assert changes == [0.4, 0.2]


def test_fullscreen_direction_defaults_to_bottom_and_can_change(qtbot) -> None:
    """전체 화면 번역 위치는 기본 아래이고, 콤보 상자로 바꾸면 알린다."""
    window = make_status(qtbot)
    assert window.fullscreen_direction is TranslationDirection.BOTTOM

    with qtbot.waitSignal(window.fullscreen_direction_changed, timeout=1000) as blocker:
        window.direction_combo.setCurrentIndex(
            window.direction_combo.findData(TranslationDirection.LEFT)
        )
    assert blocker.args == [TranslationDirection.LEFT]
    assert window.fullscreen_direction is TranslationDirection.LEFT


def test_status_window_close_emits_closed(qtbot) -> None:
    """편집 모드에서 닫기 버튼은 closed 시그널을 보낸다."""
    window = make_status(qtbot)
    window.set_interactive(True)

    with qtbot.waitSignal(window.closed, timeout=1000):
        qtbot.mouseClick(window.close_button, Qt.MouseButton.LeftButton)
    assert not window.isVisible()
