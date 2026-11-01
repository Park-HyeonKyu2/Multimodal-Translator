"""UI 설정 저장/복원 테스트. 실제 레지스트리 대신 임시 INI 파일을 쓴다."""

import pytest
from PySide6.QtCore import QPoint, QRect, QSettings

from translator.adapters.ocr.mock import MockTextRecognizer
from translator.adapters.storage.sqlite_profiles import SqliteContentProfileStore
from translator.adapters.translation.mock import MockTranslator
from translator.core.interfaces import ScreenCapturer
from translator.core.models import CapturedImage, ScreenRegion
from translator.application.regions import RegionConfig
from translator.core.models import TranslationDirection
from translator.main import build_coordinator
from translator.ui import ui_settings
from translator.ui.ui_settings import UiSettingsStore


class BlankCapturer(ScreenCapturer):
    """검은 화면만 돌려주는 가짜 캡처 (설정 테스트에서는 캡처 내용이 중요하지 않다)."""

    def capture_region(self, region: ScreenRegion) -> CapturedImage:
        """영역 크기의 검은 이미지를 반환한다."""
        return CapturedImage(region, region.width, region.height, bytes(region.width * region.height * 4))

    def capture_full_screen(self) -> CapturedImage:
        """작은 검은 이미지를 반환한다."""
        return CapturedImage(ScreenRegion(0, 0, 4, 4), 4, 4, bytes(64))


@pytest.fixture
def ini_path(tmp_path):
    """테스트마다 새 INI 파일 경로."""
    return str(tmp_path / "ui.ini")


@pytest.fixture
def db_path(tmp_path):
    """테스트마다 새 SQLite 파일 경로 (같은 경로로 다시 열면 재실행을 흉내 낸다)."""
    return str(tmp_path / "translator.db")


def make_store(path: str) -> UiSettingsStore:
    """INI 파일을 쓰는 설정 저장소를 만든다 (재실행을 흉내 내려면 같은 경로로 다시 만든다)."""
    return UiSettingsStore(QSettings(path, QSettings.Format.IniFormat))


def test_empty_settings_return_none(qapp, ini_path) -> None:
    """처음 실행하면 저장된 값이 없다."""
    store = make_store(ini_path)

    assert store.load_overlay_position() is None
    assert store.load_result_window_geometry() is None
    assert store.load_result_opacity() is None


def test_values_survive_reopen(qapp, ini_path) -> None:
    """저장한 값은 저장소를 다시 열어도(재실행) 그대로 읽힌다. 음수 좌표 포함."""
    store = make_store(ini_path)
    store.save_overlay_position(QPoint(-1500, 20))
    store.save_result_window_geometry(QRect(100, 200, 500, 220))
    store.save_result_opacity(0.4)
    store.sync()

    reopened = make_store(ini_path)

    assert reopened.load_overlay_position() == QPoint(-1500, 20)
    assert reopened.load_result_window_geometry() == QRect(100, 200, 500, 220)
    assert reopened.load_result_opacity() == pytest.approx(0.4)


@pytest.mark.parametrize(
    "key, value",
    [
        ("overlay/x", "not-a-number"),
        ("overlay/y", "1.5"),
    ],
)
def test_corrupted_overlay_values_are_ignored(qapp, ini_path, key, value) -> None:
    """손상된 값은 무시하고 None을 돌려준다 (기본 위치 사용)."""
    store = make_store(ini_path)
    store.save_overlay_position(QPoint(10, 10))
    QSettings(ini_path, QSettings.Format.IniFormat).setValue(key, value)

    assert make_store(ini_path).load_overlay_position() is None


@pytest.mark.parametrize("value", ["abc", "-0.1", "1.5", "nan"])
def test_invalid_opacity_is_ignored(qapp, ini_path, value) -> None:
    """범위를 벗어나거나 숫자가 아닌 불투명도는 무시한다."""
    store = make_store(ini_path)
    store.save_result_opacity(0.5)
    store.sync()
    raw = QSettings(ini_path, QSettings.Format.IniFormat)
    raw.setValue("result_window/background_opacity", value)
    raw.sync()

    assert make_store(ini_path).load_result_opacity() is None


def test_non_positive_window_size_is_ignored(qapp, ini_path) -> None:
    """크기가 0 이하인 결과 창 값은 무시한다."""
    store = make_store(ini_path)
    store.save_result_window_geometry(QRect(10, 10, 0, 100))

    assert store.load_result_window_geometry() is None


def test_other_settings_version_is_ignored(qapp, ini_path, monkeypatch) -> None:
    """저장 형식 버전이 다르면 값을 쓰지 않는다."""
    store = make_store(ini_path)
    store.save_overlay_position(QPoint(10, 10))
    store.sync()
    monkeypatch.setattr(ui_settings, "SETTINGS_VERSION", 2)

    assert make_store(ini_path).load_overlay_position() is None


# --- coordinator 연동 ------------------------------------------------------------------


def start_app(path: str, database: str | None = None):
    """설정 저장소를 연결한 coordinator를 만들어 시작한다. `database`가 있으면 박스를 SQLite에 저장·복원한다."""
    coordinator = build_coordinator(
        capturer=BlankCapturer(),
        recognizer=MockTextRecognizer(delay_seconds=0),
        translator=MockTranslator(delay_seconds=0),
        settings=make_store(path),
        profile_store=SqliteContentProfileStore(database) if database else None,
    )
    coordinator.start()
    return coordinator


def test_ui_settings_are_restored_after_restart(qtbot, ini_path) -> None:
    """종료할 때 저장한 오버레이 위치, 상태 창 위치/크기, 불투명도, 전체 화면 번역 위치가 복원된다."""
    first = start_app(ini_path)
    first._overlay.move(30, 500)
    first._status_window.set_interactive(True)  # 편집 모드에서 설정 창으로 띄운다
    first._status_window.setGeometry(QRect(50, 100, 400, 200))
    first._status_window.set_background_opacity(0.35)
    first._status_window.set_fullscreen_direction(TranslationDirection.RIGHT)
    first.shutdown()

    second = start_app(ini_path)
    try:
        assert second._overlay.pos() == QPoint(30, 500)
        assert second._status_window.geometry() == QRect(50, 100, 400, 200)
        assert second._status_window.background_opacity == pytest.approx(0.35)
        assert second._status_window.opacity_slider.value() == 35
        assert second._status_window.fullscreen_direction is TranslationDirection.RIGHT
    finally:
        second.shutdown()


def test_regions_are_restored_after_restart(qtbot, ini_path, db_path) -> None:
    """박스의 ID, 위치, 크기, 출력 방향, 자동 감지 상태가 다음 실행에 복원된다."""
    first = start_app(ini_path, db_path)
    first._overlay.region_button.click()
    first._selector._on_selection_made(QRect(100, 100, 300, 80))
    first._overlay.region_button.click()
    first._selector._on_selection_made(QRect(100, 400, 200, 60))
    box_1, box_2 = first._regions.sessions()
    box_1.set_direction(TranslationDirection.LEFT)
    box_2.set_auto_enabled(True)
    box_2.box.move(box_2.box.x() + 40, box_2.box.y())
    expected = first._regions.configs()
    first.shutdown()

    second = start_app(ini_path, db_path)
    try:
        assert second._regions.configs() == expected
        restored_1, restored_2 = second._regions.sessions()
        assert restored_1.config.direction is TranslationDirection.LEFT
        assert restored_2.config.auto_enabled and restored_2.detector.is_running
        assert restored_2.capture_region().left == 140
        assert not restored_1.box.is_interactive  # 잠금 모드로 시작
        assert second._overlay.translate_region_button.isEnabled()
    finally:
        second.shutdown()


def test_removed_region_stays_removed_after_restart(qtbot, ini_path, db_path) -> None:
    """삭제한 박스는 다음 실행에 다시 나타나지 않는다 (종료할 때 마지막 상태를 저장)."""
    first = start_app(ini_path, db_path)
    first._overlay.region_button.click()
    first._selector._on_selection_made(QRect(100, 100, 300, 80))
    first._remove_region("box-1")
    first.shutdown()

    second = start_app(ini_path, db_path)
    try:
        assert second._regions.sessions() == []
    finally:
        second.shutdown()


def test_region_store_roundtrip_and_corrupted_entries(qapp, ini_path) -> None:
    """박스 목록 저장/읽기와, 손상되거나 ID가 겹친 항목 건너뛰기."""
    store = make_store(ini_path)
    configs = [
        RegionConfig("box-1", -1500, 10, 300, 80, TranslationDirection.TOP, True),
        RegionConfig("box-2", 0, 0, 100, 40),
    ]
    store.save_regions(configs)
    store.sync()
    assert make_store(ini_path).load_regions() == configs

    raw = QSettings(ini_path, QSettings.Format.IniFormat)
    raw.setValue("profiles/default/regions/2/direction", "sideways")  # 알 수 없는 방향
    raw.sync()
    assert make_store(ini_path).load_regions() == [configs[0]]


def test_regions_are_stored_per_profile(qapp, ini_path) -> None:
    """박스 목록은 프로필별로 따로 저장된다 (콘텐츠별 설정 확장 대비)."""
    store = make_store(ini_path)
    store.save_regions([RegionConfig("box-1", 0, 0, 100, 40)], profile="game-a")

    assert store.load_regions() == []
    assert [c.region_id for c in store.load_regions(profile="game-a")] == ["box-1"]


def test_offscreen_saved_positions_fall_back_to_default(qtbot, ini_path, db_path) -> None:
    """저장된 위치가 현재 모니터 밖이면(모니터 분리 등) 기본 위치를 쓰고, 화면 밖 박스는 복원하지 않는다."""
    store = make_store(ini_path)
    store.save_overlay_position(QPoint(5000, 5000))
    store.save_result_window_geometry(QRect(-5000, 0, 400, 200))
    SqliteContentProfileStore(db_path).save_regions("default", [RegionConfig("box-1", 5000, 5000, 100, 40)])
    store.sync()

    coordinator = start_app(ini_path, db_path)
    try:
        overlay = coordinator._overlay
        expected = QPoint(overlay.pos())
        overlay.move_to_default_position()
        assert expected == overlay.pos()
        assert not coordinator._status_window.has_position
        assert coordinator._regions.sessions() == []
    finally:
        coordinator.shutdown()


def test_never_shown_status_window_does_not_overwrite_saved_geometry(qtbot, ini_path) -> None:
    """상태 창을 한 번도 띄우지 않고 종료해도 저장된 상태 창 위치를 덮어쓰지 않는다."""
    store = make_store(ini_path)
    store.save_result_window_geometry(QRect(-5000, 0, 400, 200))  # 이번 실행에서는 복원되지 않는 값
    store.sync()

    start_app(ini_path).shutdown()

    assert make_store(ini_path).load_result_window_geometry() == QRect(-5000, 0, 400, 200)
