"""번역 말풍선 관리 테스트 (offscreen 화면: 800×800, 배율 1)."""

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor

from translator.application.text_display import TextDisplayModel
from translator.core.models import (
    ScreenRegion,
    TranslatedUnit,
    TranslationBatch,
    TranslationDirection,
)
from translator.tuning import BUBBLE_GAP
from translator.ui.translation_overlay import TranslationOverlay

CAPTURE = ScreenRegion(0, 0, 800, 800)
ORANGE = QColor(255, 140, 0)


def batch(source: str, *units: tuple[str, ScreenRegion]) -> TranslationBatch:
    """(원문, 위치) 목록으로 번역 결과를 만든다."""
    return TranslationBatch(
        source,
        CAPTURE,
        tuple(TranslatedUnit(text, f"번역:{text}", bounds, "fake") for text, bounds in units),
    )


def make_overlay(qtbot, directions: dict[str, TranslationDirection] | None = None, grace=1.0):
    """출처별 방향을 지정한 말풍선 관리자를 만든다."""
    directions = directions or {}
    overlay = TranslationOverlay(
        TextDisplayModel(hide_grace_seconds=grace),
        style_for=lambda source: (directions.get(source, TranslationDirection.BOTTOM), ORANGE),
    )
    qtbot.addWidget  # noqa: B018 (qtbot이 QApplication을 준비하게 한다)
    return overlay


def only_bubble(overlay):
    """말풍선이 하나뿐일 때 그 말풍선."""
    [bubble] = overlay.bubbles().values()
    return bubble


def test_bubble_is_placed_below_source_text(qtbot) -> None:
    """기본 방향(아래)으로 원문 바로 아래에 간격을 두고 놓인다."""
    overlay = make_overlay(qtbot)

    overlay.show_batch(batch("box-1", ("hello", ScreenRegion(300, 200, 200, 20))))

    bubble = only_bubble(overlay)
    assert bubble.isVisible()
    assert bubble.geometry().top() == 220 + BUBBLE_GAP
    assert bubble.geometry().center().x() in range(398, 403)  # 원문 가운데
    assert bubble.display_text == "번역:hello"
    overlay.clear()


def test_each_source_uses_its_own_direction(qtbot) -> None:
    """박스마다 설정한 방향으로 놓인다 (box-1 위, box-2 오른쪽)."""
    overlay = make_overlay(
        qtbot, {"box-1": TranslationDirection.TOP, "box-2": TranslationDirection.RIGHT}
    )
    text_1 = ScreenRegion(300, 300, 100, 20)
    text_2 = ScreenRegion(100, 600, 100, 20)

    overlay.show_batch(batch("box-1", ("a", text_1)))
    overlay.show_batch(batch("box-2", ("b", text_2)))

    bubbles = {item.source_id: overlay.bubble_for(item.item_id) for item in overlay.model.items()}
    assert bubbles["box-1"].geometry().bottom() < 300  # 원문 위
    assert bubbles["box-2"].geometry().left() == 200 + BUBBLE_GAP  # 원문 오른쪽
    assert bubbles["box-1"].placed_direction is TranslationDirection.TOP
    overlay.clear()


def test_bubble_follows_moved_text(qtbot) -> None:
    """원문이 이동하면 같은 말풍선이 새 위치로 따라간다."""
    overlay = make_overlay(qtbot)
    overlay.show_batch(batch("box-1", ("hello", ScreenRegion(100, 100, 200, 20))))
    bubble = only_bubble(overlay)

    overlay.show_batch(batch("box-1", ("hello", ScreenRegion(100, 400, 200, 20))))

    assert only_bubble(overlay) is bubble
    assert bubble.geometry().top() == 420 + BUBBLE_GAP
    overlay.clear()


def test_direction_change_relayouts_without_new_translation(qtbot) -> None:
    """방향 설정이 바뀌면 refresh만으로 말풍선이 다시 놓인다."""
    directions = {"box-1": TranslationDirection.BOTTOM}
    overlay = make_overlay(qtbot, directions)
    overlay.show_batch(batch("box-1", ("hello", ScreenRegion(300, 300, 200, 20))))

    directions["box-1"] = TranslationDirection.TOP
    overlay.refresh()

    assert only_bubble(overlay).geometry().bottom() < 300
    overlay.clear()


def test_bubble_near_screen_edge_stays_on_screen(qtbot) -> None:
    """화면 아래 끝의 원문은 말풍선이 위로 옮겨져 화면 안에 남는다."""
    overlay = make_overlay(qtbot)

    overlay.show_batch(batch("box-1", ("edge", ScreenRegion(300, 780, 200, 20))))

    bubble = only_bubble(overlay)
    assert bubble.placed_direction is TranslationDirection.TOP
    assert QRect(0, 0, 800, 800).contains(bubble.geometry())
    overlay.clear()


def test_disappeared_text_bubble_closes_after_grace(qtbot) -> None:
    """원문이 사라지면 유예 시간 뒤 말풍선이 닫힌다."""
    overlay = make_overlay(qtbot, grace=0.05)
    overlay.show_batch(batch("box-1", ("bye", ScreenRegion(100, 100, 200, 20))))
    bubble = only_bubble(overlay)

    overlay.show_batch(batch("box-1"))
    assert bubble.isVisible()  # 바로 지우지 않는다 (깜빡임 방지)

    # 유예 시간이 지나면 말풍선이 닫히고 정리된다.
    qtbot.waitUntil(lambda: overlay.bubbles() == {}, timeout=2000)
    assert overlay.model.items() == []


def test_remove_source_closes_only_its_bubbles(qtbot) -> None:
    """박스를 지우면 그 박스의 말풍선만 닫힌다."""
    overlay = make_overlay(qtbot)
    overlay.show_batch(batch("box-1", ("a", ScreenRegion(100, 100, 100, 20))))
    overlay.show_batch(batch("box-2", ("b", ScreenRegion(100, 500, 100, 20))))

    overlay.remove_source("box-1")

    assert [i.source_id for i in overlay.model.items()] == ["box-2"]
    assert len(overlay.bubbles()) == 1
    overlay.clear()


def test_long_text_wraps_and_is_height_limited(qtbot) -> None:
    """긴 번역문은 줄을 바꾸고, 그래도 넘치면 말줄임표로 잘려 화면 안에 들어간다."""
    overlay = make_overlay(qtbot)
    long_text = "아주 긴 번역문 " * 400

    overlay.show_batch(batch("box-1", (long_text, ScreenRegion(100, 100, 600, 20))))

    bubble = only_bubble(overlay)
    assert bubble.display_text.endswith("…")
    assert bubble.geometry().height() <= 800 * 0.4 + 1
    assert QRect(0, 0, 800, 800).contains(bubble.geometry())
    overlay.clear()


def test_bubbles_are_click_through_and_do_not_take_focus(qtbot) -> None:
    """말풍선은 표시 전용: 클릭을 통과시키고 포커스를 받지 않는다."""
    overlay = make_overlay(qtbot)
    overlay.show_batch(batch("box-1", ("a", ScreenRegion(100, 100, 100, 20))))

    flags = only_bubble(overlay).windowFlags()
    assert flags & Qt.WindowType.WindowTransparentForInput
    assert flags & Qt.WindowType.WindowDoesNotAcceptFocus
    overlay.clear()


def test_opacity_applies_to_existing_bubbles(qtbot) -> None:
    """불투명도를 바꾸면 이미 떠 있는 말풍선에도 적용된다."""
    overlay = make_overlay(qtbot)
    overlay.show_batch(batch("box-1", ("a", ScreenRegion(100, 100, 100, 20))))

    overlay.set_background_opacity(0.3)

    assert only_bubble(overlay)._opacity == 0.3
    overlay.clear()


# --- 박스 하나 = 말풍선 하나 ---------------------------------------------------------------


def make_grouping_overlay(qtbot, directions: dict[str, TranslationDirection] | None = None, grace=1.0):
    """박스(box-*)는 말풍선 하나로 묶고, 다른 출처(전체 화면)는 원문마다 따로 놓는 관리자."""
    directions = directions or {}
    overlay = TranslationOverlay(
        TextDisplayModel(hide_grace_seconds=grace),
        style_for=lambda source: (directions.get(source, TranslationDirection.BOTTOM), ORANGE),
        groups_source=lambda source: source.startswith("box-"),
    )
    qtbot.addWidget  # noqa: B018 (qtbot이 QApplication을 준비하게 한다)
    return overlay


def test_box_translations_share_one_bubble_in_reading_order(qtbot) -> None:
    """박스 하나의 번역문은 말풍선 하나에 들어가며, 다른 줄은 줄을 바꾸고 같은 줄은 이어 쓴다."""
    overlay = make_grouping_overlay(qtbot)

    overlay.show_batch(
        batch(
            "box-1",
            ("둘째", ScreenRegion(100, 300, 100, 20)),
            ("첫째-오른쪽", ScreenRegion(300, 200, 100, 20)),
            ("첫째-왼쪽", ScreenRegion(100, 200, 100, 20)),
        )
    )

    bubble = only_bubble(overlay)
    assert overlay.bubble_for_source("box-1") is bubble
    assert bubble.display_text == "번역:첫째-왼쪽 번역:첫째-오른쪽\n번역:둘째"
    overlay.clear()


def test_group_bubble_is_placed_below_all_source_texts(qtbot) -> None:
    """묶은 말풍선은 원문 전체 범위를 기준으로, 맨 아래 원문의 바로 아래에 놓인다."""
    overlay = make_grouping_overlay(qtbot)

    overlay.show_batch(
        batch(
            "box-1",
            ("a", ScreenRegion(100, 200, 100, 20)),
            ("b", ScreenRegion(100, 260, 300, 20)),
        )
    )

    bubble = only_bubble(overlay)
    assert bubble.geometry().top() == 280 + BUBBLE_GAP
    assert bubble.geometry().center().x() in range(247, 254)  # 원문 전체(100~400)의 가운데
    overlay.clear()


def test_line_break_bubble_is_taller_than_single_line(qtbot) -> None:
    """줄을 바꾼 번역문은 한 줄 번역문보다 말풍선이 높다 (줄바꿈이 실제로 그려짐)."""
    overlay = make_grouping_overlay(qtbot)
    overlay.show_batch(batch("box-1", ("a", ScreenRegion(100, 100, 100, 20))))
    single_height = only_bubble(overlay).height()

    overlay.show_batch(
        batch("box-1", ("a", ScreenRegion(100, 100, 100, 20)), ("b", ScreenRegion(100, 160, 100, 20)))
    )

    assert only_bubble(overlay).height() > single_height
    overlay.clear()


def test_each_box_has_its_own_bubble_and_full_screen_stays_per_text(qtbot) -> None:
    """박스마다 말풍선이 하나씩이고, 묶지 않는 출처(전체 화면)는 원문마다 말풍선을 놓는다."""
    overlay = make_grouping_overlay(qtbot)

    overlay.show_batch(batch("box-1", ("a", ScreenRegion(100, 100, 100, 20)), ("b", ScreenRegion(100, 160, 100, 20))))
    overlay.show_batch(batch("box-2", ("c", ScreenRegion(100, 400, 100, 20))))
    overlay.show_batch(
        batch("fullscreen", ("x", ScreenRegion(500, 100, 100, 20)), ("y", ScreenRegion(500, 600, 100, 20)))
    )

    assert overlay.bubble_for_source("box-1").display_text == "번역:a\n번역:b"
    assert overlay.bubble_for_source("box-2").display_text == "번역:c"
    assert overlay.bubble_for_source("fullscreen") is None
    full_screen = [overlay.bubble_for(i.item_id) for i in overlay.model.items_for("fullscreen")]
    assert [b.display_text for b in full_screen] == ["번역:x", "번역:y"]
    assert len(overlay.bubbles()) == 4
    overlay.clear()


def test_group_bubble_is_reused_and_updated_when_texts_change(qtbot) -> None:
    """번역문이 늘거나 바뀌어도 같은 말풍선을 고쳐 쓴다 (새 창을 만들지 않음)."""
    overlay = make_grouping_overlay(qtbot)
    overlay.show_batch(batch("box-1", ("a", ScreenRegion(100, 100, 100, 20))))
    bubble = only_bubble(overlay)

    overlay.show_batch(
        batch("box-1", ("a", ScreenRegion(100, 100, 100, 20)), ("b", ScreenRegion(100, 160, 100, 20)))
    )

    assert only_bubble(overlay) is bubble
    assert bubble.display_text == "번역:a\n번역:b"
    overlay.clear()


def test_group_bubble_skips_texts_waiting_for_translation(qtbot) -> None:
    """번역을 기다리는 원문은 빼고 보여 주며, 보여 줄 번역이 없으면 말풍선을 닫는다."""
    overlay = make_grouping_overlay(qtbot)
    overlay.show_batch(batch("box-1", ("a", ScreenRegion(100, 100, 100, 20))))
    [item] = overlay.model.items_for("box-1")

    # 원문이 다른 문장으로 바뀌어 새 번역을 기다리는 상태 (이전 번역은 보여 주지 않는다)
    item.stable_text = "다른 문장"
    item.source_text = "다른 문장"
    overlay.refresh()

    assert overlay.bubbles() == {}
    overlay.clear()


def test_removing_box_closes_its_group_bubble(qtbot) -> None:
    """박스를 지우면 그 박스의 묶은 말풍선이 닫힌다."""
    overlay = make_grouping_overlay(qtbot)
    overlay.show_batch(batch("box-1", ("a", ScreenRegion(100, 100, 100, 20))))
    overlay.show_batch(batch("box-2", ("b", ScreenRegion(100, 500, 100, 20))))

    overlay.remove_source("box-1")

    assert overlay.bubble_for_source("box-1") is None
    assert list(overlay.bubbles()) == ["box-2#all"]
    overlay.clear()
