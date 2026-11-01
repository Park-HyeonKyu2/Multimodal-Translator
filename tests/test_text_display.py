"""번역 텍스트 추적(고유 ID, 이동, 내용 변경, 깜빡임 방지, 안정화, OCR 누락) 테스트. 가짜 시계를 쓴다."""

from translator.application.change_detection import Fingerprint
from translator.application.pipeline import OcrResult
from translator.application.text_display import TextDisplayModel
from translator.application.text_stabilizer import TextStabilizer
from translator.core.models import ScreenRegion, TextUnit, TranslatedUnit, TranslationBatch

CAPTURE = ScreenRegion(0, 0, 1000, 1000)


class FakeClock:
    """테스트에서 직접 넘기는 시계."""

    def __init__(self) -> None:
        """0초에서 시작한다."""
        self.now = 0.0

    def __call__(self) -> float:
        """현재 시각."""
        return self.now


def unit(text: str, left: int, top: int, width: int = 200, height: int = 20) -> TranslatedUnit:
    """번역 결과 하나를 만든다."""
    return TranslatedUnit(text, f"번역:{text}", ScreenRegion(left, top, width, height), "fake")


def batch(source: str, *units: TranslatedUnit) -> TranslationBatch:
    """출처의 번역 결과 묶음을 만든다."""
    return TranslationBatch(source, CAPTURE, units)


def make_model(grace: float = 1.0) -> tuple[TextDisplayModel, FakeClock]:
    """가짜 시계를 쓰는 모델을 만든다."""
    clock = FakeClock()
    return TextDisplayModel(clock=clock, hide_grace_seconds=grace), clock


def test_new_texts_get_unique_ids_per_source() -> None:
    """새 텍스트마다 출처가 붙은 고유 ID가 생긴다."""
    model, _ = make_model()

    model.apply_batch(batch("box-1", unit("a", 0, 0), unit("b", 0, 100)))
    model.apply_batch(batch("box-2", unit("a", 0, 0)))

    ids = [item.item_id for item in model.items()]
    assert len(set(ids)) == 3
    assert [i.source_id for i in model.items()] == ["box-1", "box-1", "box-2"]
    assert all(i.item_id.startswith(i.source_id + "#") for i in model.items())


def test_moved_text_keeps_id_and_updates_position() -> None:
    """같은 원문이 다른 위치에 보이면 같은 항목이 이동한 것으로 본다."""
    model, _ = make_model()
    model.apply_batch(batch("box-1", unit("hello", 0, 0)))
    [before] = model.items()

    model.apply_batch(batch("box-1", unit("hello", 300, 50)))

    [after] = model.items()
    assert after.item_id == before.item_id
    assert after.bounds == ScreenRegion(300, 50, 200, 20)


def test_changed_text_in_same_place_keeps_id_and_updates_translation() -> None:
    """같은 자리에 다른 원문이 보이면 같은 항목의 내용이 바뀐 것으로 본다 (말풍선이 깜빡이지 않음)."""
    model, _ = make_model()
    model.apply_batch(batch("box-1", unit("line one", 0, 0)))
    [before] = model.items()

    model.apply_batch(batch("box-1", unit("line two", 5, 0)))

    [after] = model.items()
    assert after.item_id == before.item_id
    assert after.translated_text == "번역:line two"


def test_same_text_in_two_places_is_tracked_separately() -> None:
    """같은 원문이 두 곳에 있으면 각자 가까운 쪽과 짝지어 따로 추적한다."""
    model, _ = make_model()
    model.apply_batch(batch("fullscreen", unit("OK", 0, 0), unit("OK", 0, 800)))
    top_id, bottom_id = (item.item_id for item in model.items())

    # 둘 다 조금씩 움직임
    model.apply_batch(batch("fullscreen", unit("OK", 0, 810), unit("OK", 0, 10)))

    by_id = {item.item_id: item.bounds.top for item in model.items()}
    assert by_id == {top_id: 10, bottom_id: 810}


def test_missing_text_is_kept_during_grace_then_removed() -> None:
    """이번 결과에서 빠진 텍스트는 유예 시간 동안 남았다가 지워진다."""
    model, clock = make_model(grace=1.0)
    model.apply_batch(batch("box-1", unit("bye", 0, 0)))

    clock.now = 0.5
    model.apply_batch(batch("box-1"))  # 텍스트가 사라짐
    assert len(model.items()) == 1
    assert model.next_deadline() == 1.5

    clock.now = 1.4
    assert model.expire() is False
    clock.now = 1.5
    assert model.expire() is True
    assert model.items() == []


def test_brief_recognition_miss_does_not_flicker() -> None:
    """한 번 인식되지 않았다가 유예 시간 안에 다시 보이면 같은 항목이 그대로 유지된다."""
    model, clock = make_model(grace=1.0)
    model.apply_batch(batch("box-1", unit("stay", 0, 0)))
    [item] = model.items()

    clock.now = 0.3
    model.apply_batch(batch("box-1"))  # 인식 실패
    clock.now = 0.6
    model.apply_batch(batch("box-1", unit("stay", 0, 0)))  # 다시 인식
    clock.now = 5.0
    model.expire()

    assert [i.item_id for i in model.items()] == [item.item_id]
    assert model.items()[0].missing_since is None


def test_sources_do_not_affect_each_other() -> None:
    """다른 출처의 결과는 서로의 항목을 지우거나 바꾸지 않는다."""
    model, clock = make_model(grace=0.1)
    model.apply_batch(batch("box-1", unit("one", 0, 0)))
    model.apply_batch(batch("box-2", unit("two", 0, 0)))

    model.apply_batch(batch("box-2"))
    clock.now = 1.0
    model.expire()

    assert [i.source_text for i in model.items()] == ["one"]


def test_ttl_expires_items_even_if_never_missing() -> None:
    """TTL이 있는 결과(전체 화면)는 시간이 지나면 지워진다."""
    model, clock = make_model()
    model.apply_batch(batch("fullscreen", unit("x", 0, 0)), ttl_seconds=10)

    clock.now = 9.9
    assert model.expire() is False
    clock.now = 10.0
    assert model.expire() is True


def test_remove_source_clears_only_that_source() -> None:
    """박스를 지우면 그 박스의 항목만 사라진다."""
    model, _ = make_model()
    model.apply_batch(batch("box-1", unit("one", 0, 0)))
    model.apply_batch(batch("box-2", unit("two", 0, 0)))

    model.remove_source("box-1")

    assert [i.source_id for i in model.items()] == ["box-2"]
    assert model.has_items("box-2") and not model.has_items("box-1")


# --- 인식 결과(OcrResult) 경로: 안정화, 중복 번역 판단, OCR 누락/실제 사라짐 -----------------

# 텍스트 자리 픽셀 지문 (같으면 그 자리 화면이 그대로라는 뜻)
SAME_AREA = Fingerprint(4, 1, bytes((10, 10, 10, 10)))
CHANGED_AREA = Fingerprint(4, 1, bytes((250, 250, 250, 250)))


def region(left: int, top: int, width: int = 200, height: int = 20) -> ScreenRegion:
    """텍스트 위치를 만든다."""
    return ScreenRegion(left, top, width, height)


def ocr(source: str, *texts: tuple[str, ScreenRegion], at: float = 0.0, probes=()) -> OcrResult:
    """(원문, 위치) 목록으로 인식 결과를 만든다. 텍스트 자리 지문은 모두 SAME_AREA다."""
    units = tuple(TextUnit(text, bounds) for text, bounds in texts)
    return OcrResult(
        source_id=source,
        frame_id=int(at * 1000),
        captured_at=at,
        capture_region=CAPTURE,
        units=units,
        unit_fingerprints=(SAME_AREA,) * len(units),
        probes=tuple(probes),
        fingerprint=SAME_AREA,
        ocr_reused=False,
    )


def stable_model(grace: float = 1.0) -> tuple[TextDisplayModel, FakeClock]:
    """같은 텍스트 2번이면 안정으로 보는 모델."""
    clock = FakeClock()
    model = TextDisplayModel(
        clock=clock,
        hide_grace_seconds=grace,
        stabilizer_factory=lambda: TextStabilizer(settle_observations=2, max_wait_seconds=10),
    )
    return model, clock


def test_auto_text_is_translated_only_after_it_is_stable() -> None:
    """자동 인식 텍스트는 처음에는 번역하지 않고, 같은 텍스트가 다시 인식되면 번역 대상이 된다."""
    model, clock = stable_model()

    first = model.apply_ocr(ocr("box-1", ("Hello", region(0, 0)), at=0.0))
    clock.now = 0.4
    second = model.apply_ocr(ocr("box-1", ("Hello", region(0, 0)), at=0.4))

    assert first.unstable and first.texts_to_translate == ()
    assert not second.unstable and second.texts_to_translate == ("Hello",)
    assert model.items_for("box-1")[0].display_text is None  # 번역 전에는 보여 줄 것이 없다


def test_stable_text_is_not_translated_twice() -> None:
    """이미 번역한 원문은 다시 인식되거나 위치가 바뀌어도 다시 번역하지 않는다."""
    model, clock = stable_model()
    model.apply_ocr(ocr("box-1", ("Hello", region(0, 0))), force_stable=True)
    model.apply_translations("box-1", {"Hello": ("안녕", "fake")})

    clock.now = 1.0
    again = model.apply_ocr(ocr("box-1", ("Hello", region(0, 0)), at=1.0))
    clock.now = 2.0
    moved = model.apply_ocr(ocr("box-1", ("Hello", region(30, 40)), at=2.0))

    assert again.texts_to_translate == () and moved.texts_to_translate == ()
    [item] = model.items_for("box-1")
    assert item.bounds == region(30, 40) and item.display_text == "안녕"


def test_replaced_text_hides_old_translation_until_new_one_arrives() -> None:
    """같은 자리의 텍스트가 다른 문장으로 바뀌면 이전 번역을 보여 주지 않는다 (오래된 번역 방지)."""
    model, clock = stable_model()
    model.apply_ocr(ocr("box-1", ("첫 대사", region(0, 0))), force_stable=True)
    model.apply_translations("box-1", {"첫 대사": ("first", "fake")})

    clock.now = 1.0
    outcome = model.apply_ocr(ocr("box-1", ("다른 대사", region(0, 0)), at=1.0))

    [item] = model.items_for("box-1")
    assert outcome.unstable and item.display_text is None
    clock.now = 1.4
    outcome = model.apply_ocr(ocr("box-1", ("다른 대사", region(0, 0)), at=1.4))
    assert outcome.texts_to_translate == ("다른 대사",)
    assert model.apply_translations("box-1", {"다른 대사": ("second", "fake")})
    assert model.items_for("box-1")[0].display_text == "second"


def test_typing_text_keeps_previous_translation_while_growing() -> None:
    """글자가 늘어나는 동안에는 이전 번역을 유지한다 (깜빡임 없이 타자 효과 대응)."""
    model, clock = stable_model()
    model.apply_ocr(ocr("box-1", ("Hello", region(0, 0, 100))), force_stable=True)
    model.apply_translations("box-1", {"Hello": ("안녕", "fake")})

    clock.now = 0.4
    outcome = model.apply_ocr(ocr("box-1", ("Hello wor", region(0, 0, 180)), at=0.4))

    [item] = model.items_for("box-1")
    assert outcome.unstable and item.display_text == "안녕"


def test_translation_for_outdated_text_is_not_applied() -> None:
    """번역이 도착하기 전에 원문이 바뀌었으면 그 번역은 적용하지 않는다."""
    model, clock = stable_model()
    model.apply_ocr(ocr("box-1", ("A", region(0, 0))), force_stable=True)
    clock.now = 1.0
    model.apply_ocr(ocr("box-1", ("B", region(0, 0)), at=1.0), force_stable=True)

    assert not model.apply_translations("box-1", {"A": ("번역 A", "fake")})
    assert model.items_for("box-1")[0].display_text is None


def test_ocr_miss_with_unchanged_pixels_keeps_text() -> None:
    """결과에서 빠졌지만 그 자리 픽셀이 그대로면 OCR이 놓친 것으로 보고 항목을 유지한다."""
    model, clock = stable_model(grace=0.5)
    model.apply_ocr(ocr("box-1", ("대사", region(0, 0))), force_stable=True)

    clock.now = 1.0
    outcome = model.apply_ocr(ocr("box-1", at=1.0, probes=[(region(0, 0), SAME_AREA)]))
    clock.now = 5.0
    model.expire()

    assert (outcome.ocr_misses, outcome.disappeared) == (1, 0)
    [item] = model.items_for("box-1")
    assert item.ocr_misses == 1 and item.missing_since is None


def test_text_with_changed_pixels_is_treated_as_disappeared() -> None:
    """결과에서 빠지고 그 자리 픽셀도 바뀌었으면 실제로 사라진 것으로 보고 유예 뒤 지운다."""
    model, clock = stable_model(grace=0.5)
    model.apply_ocr(ocr("box-1", ("대사", region(0, 0))), force_stable=True)

    clock.now = 1.0
    outcome = model.apply_ocr(ocr("box-1", at=1.0, probes=[(region(0, 0), CHANGED_AREA)]))
    assert (outcome.ocr_misses, outcome.disappeared) == (0, 1)
    clock.now = 1.6
    model.expire()

    assert model.items_for("box-1") == []


def test_text_without_probe_is_treated_as_disappeared() -> None:
    """그 자리를 확인할 수 없으면(캡처 밖) 사라진 것으로 본다 (영구히 남지 않게)."""
    model, clock = stable_model(grace=0.5)
    model.apply_ocr(ocr("box-1", ("대사", region(0, 0))), force_stable=True)

    clock.now = 1.0
    outcome = model.apply_ocr(ocr("box-1", at=1.0))

    assert outcome.disappeared == 1


def test_older_frame_result_does_not_overwrite_newer_one() -> None:
    """늦게 끝난 오래된 프레임의 결과는 무시한다 (최신 결과를 덮어쓰지 않음)."""
    model, _ = stable_model()
    model.apply_ocr(ocr("box-1", ("최신", region(0, 0)), at=2.0), force_stable=True)

    outcome = model.apply_ocr(ocr("box-1", ("오래된", region(0, 0)), at=1.0), force_stable=True)

    assert not outcome.accepted
    assert model.items_for("box-1")[0].source_text == "최신"
    # 다른 출처는 각자의 시각으로 판단한다.
    assert model.apply_ocr(ocr("box-2", ("다른 박스", region(0, 0)), at=1.0)).accepted


def test_same_text_far_away_is_a_different_sentence() -> None:
    """같은 문자열이라도 멀리(글자 높이의 정해진 배수 밖) 나타나면 다른 문장으로 본다."""
    model, clock = stable_model()
    model.apply_ocr(ocr("fullscreen", ("예", region(0, 0, 40, 20))), force_stable=True)
    first_id = model.items_for("fullscreen")[0].item_id

    clock.now = 1.0
    model.apply_ocr(
        ocr("fullscreen", ("예", region(0, 0, 40, 20)), ("예", region(900, 900, 40, 20)), at=1.0),
        force_stable=True,
    )

    items = model.items_for("fullscreen")
    assert len(items) == 2
    assert items[0].item_id == first_id and items[0].bounds == region(0, 0, 40, 20)


def test_same_text_moved_a_little_is_the_same_sentence() -> None:
    """같은 문자열이 가까이 움직이면 같은 문장의 이동으로 본다 (재번역 없이 위치만 갱신)."""
    model, clock = stable_model()
    model.apply_ocr(ocr("fullscreen", ("예", region(0, 0, 40, 20))), force_stable=True)
    model.apply_translations("fullscreen", {"예": ("yes", "fake")})
    first_id = model.items_for("fullscreen")[0].item_id

    clock.now = 1.0
    outcome = model.apply_ocr(ocr("fullscreen", ("예", region(100, 50, 40, 20)), at=1.0))

    [item] = model.items_for("fullscreen")
    assert item.item_id == first_id and item.bounds == region(100, 50, 40, 20)
    assert outcome.texts_to_translate == ()
