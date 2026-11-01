"""모니터별 DPI 배율과 좌표 변환 테스트.

여기 있는 테스트는 가짜 모니터 정보로 **계산식**을 검증한다. 실제 Windows 디스플레이
배율(125%, 150% 등)을 바꾼 환경에서 동작을 확인한 것은 아니다. 실제 환경 검증
현황은 docs/dpi-verification.md에 따로 기록한다.
"""

from dataclasses import dataclass

import pytest
from PySide6.QtCore import QPoint, QRect

from translator.core.models import ScreenRegion
from translator.ui import screen_geometry
from translator.ui.screen_geometry import (
    global_logical_rect_to_physical,
    is_rect_on_any_screen,
    logical_rect_to_physical,
    physical_region_to_logical,
)


@dataclass
class FakeScreen:
    """QScreen 대신 쓰는 가짜 모니터. 논리 좌표 영역과 배율만 가진다."""

    logical_geometry: QRect
    ratio: float

    def geometry(self) -> QRect:
        """논리 좌표 기준 모니터 영역."""
        return self.logical_geometry

    def devicePixelRatio(self) -> float:  # noqa: N802 (QScreen과 같은 이름)
        """모니터 배율."""
        return self.ratio

    def availableGeometry(self) -> QRect:  # noqa: N802 (QScreen과 같은 이름)
        """작업 영역: 아래 40 논리 픽셀을 작업표시줄로 뺀다."""
        geometry = self.logical_geometry
        return geometry.adjusted(0, 0, 0, -40)


def make_screen(native_x: int, native_y: int, native_w: int, native_h: int, ratio: float) -> FakeScreen:
    """물리 해상도와 배율로 Qt 6 방식의 논리 영역을 만든다 (원점은 물리 좌표 그대로)."""
    return FakeScreen(
        QRect(native_x, native_y, round(native_w / ratio), round(native_h / ratio)), ratio
    )


@pytest.fixture
def fake_screens(monkeypatch):
    """`QGuiApplication`의 모니터 조회를 가짜 모니터 목록으로 바꾼다."""

    def install(*screens: FakeScreen, primary_index: int = 0) -> None:
        """가짜 모니터를 등록한다."""

        def screen_at(point: QPoint):
            """점이 속한 모니터 (없으면 None)."""
            return next((s for s in screens if s.geometry().contains(point)), None)

        monkeypatch.setattr(screen_geometry.QGuiApplication, "screens", lambda: list(screens))
        monkeypatch.setattr(screen_geometry.QGuiApplication, "screenAt", screen_at)
        monkeypatch.setattr(
            screen_geometry.QGuiApplication, "primaryScreen", lambda: screens[primary_index]
        )

    return install


# --- 순수 계산식 -----------------------------------------------------------------


@pytest.mark.parametrize("ratio", [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0])
def test_full_screen_rect_maps_to_native_resolution(ratio: float) -> None:
    """모니터 전체 논리 영역은 배율과 관계없이 물리 해상도 전체로 변환된다."""
    screen = make_screen(0, 0, 3840, 2160, ratio)
    local = QRect(0, 0, screen.geometry().width(), screen.geometry().height())

    region = logical_rect_to_physical(local, screen.geometry(), ratio)

    assert region.left == 0 and region.top == 0
    # 논리 크기를 반올림해서 만들었으므로 물리 해상도와 최대 1픽셀(× 배율 반올림) 차이를 허용한다.
    assert abs(region.width - 3840) <= ratio
    assert abs(region.height - 2160) <= ratio


@pytest.mark.parametrize(
    "local, origin, ratio, expected",
    [
        # 125%: 오프셋과 크기 모두 1.25배, 원점은 그대로
        (QRect(100, 40, 400, 80), QRect(0, 0, 1536, 864), 1.25, ScreenRegion(125, 50, 500, 100)),
        # 175% 보조 모니터 (오른쪽)
        (QRect(10, 10, 100, 20), QRect(2560, 0, 1463, 823), 1.75, ScreenRegion(2578, 18, 175, 35)),
        # 주 모니터 위쪽에 있는 모니터 (음수 y)
        (QRect(0, 0, 50, 50), QRect(0, -1440, 1280, 720), 2.0, ScreenRegion(0, -1440, 100, 100)),
        # 주 모니터 왼쪽 아래 대각선 위치 (음수 x, 양수 y)
        (QRect(20, 30, 40, 10), QRect(-1920, 1080, 1280, 720), 1.5, ScreenRegion(-1890, 1125, 60, 15)),
    ],
)
def test_logical_rect_to_physical_for_various_layouts(local, origin, ratio, expected) -> None:
    """모니터 배치(오른쪽, 위, 왼쪽 아래)와 배율별 변환 결과."""
    assert logical_rect_to_physical(local, origin, ratio) == expected


@pytest.mark.parametrize("ratio", [1.25, 1.5, 1.75, 2.25])
def test_adjacent_logical_rects_have_no_physical_gap_or_overlap(ratio: float) -> None:
    """나란히 붙은 논리 사각형은 물리 좌표에서도 틈이나 겹침 없이 이어진다."""
    origin = QRect(0, 0, 2000, 2000)
    left = logical_rect_to_physical(QRect(0, 0, 33, 10), origin, ratio)
    right = logical_rect_to_physical(QRect(33, 0, 17, 10), origin, ratio)

    assert left.left + left.width == right.left


def test_fractional_ratio_never_produces_empty_region() -> None:
    """작은 사각형도 반올림 때문에 크기가 0이 되지 않는다 (1 논리 픽셀 × 1.25)."""
    region = logical_rect_to_physical(QRect(1, 1, 1, 1), QRect(0, 0, 100, 100), 1.25)

    assert region.width >= 1 and region.height >= 1


# --- 모니터 선택 (가짜 모니터) --------------------------------------------------------


def test_rect_on_secondary_high_dpi_screen_uses_that_screen_ratio(fake_screens) -> None:
    """100% 주 모니터 + 150% 보조 모니터에서 보조 모니터 위 사각형은 1.5배로 변환된다."""
    primary = make_screen(0, 0, 1920, 1080, 1.0)
    secondary = make_screen(1920, 0, 2560, 1440, 1.5)  # 논리 (1920, 0, 1707, 960)
    fake_screens(primary, secondary)

    region = global_logical_rect_to_physical(QRect(1920 + 100, 200, 300, 60))

    assert region == ScreenRegion(1920 + 150, 300, 450, 90)


def test_rect_on_primary_screen_ignores_other_screen_ratio(fake_screens) -> None:
    """같은 구성에서 주 모니터 위 사각형은 1.0배 그대로다."""
    fake_screens(make_screen(0, 0, 1920, 1080, 1.0), make_screen(1920, 0, 2560, 1440, 1.5))

    assert global_logical_rect_to_physical(QRect(100, 200, 300, 60)) == ScreenRegion(100, 200, 300, 60)


def test_rect_spanning_two_screens_uses_center_screen(fake_screens) -> None:
    """두 모니터에 걸친 사각형은 중심이 있는 모니터의 배율을 쓴다 (알려진 한계)."""
    primary = make_screen(0, 0, 1920, 1080, 1.0)
    secondary = make_screen(1920, 0, 3840, 2160, 2.0)  # 논리 (1920, 0, 1920, 1080)
    fake_screens(primary, secondary)

    # 중심 x = 1900 + 200/2 = 2000 → 보조 모니터
    region = global_logical_rect_to_physical(QRect(1900, 100, 200, 50))

    # 보조 모니터 원점 기준 오프셋 -20 → 물리 -40. 주 모니터 부분은 실제와 어긋날 수 있다.
    assert region == ScreenRegion(1920 - 40, 200, 400, 100)


def test_rect_in_gap_between_scaled_screens_falls_back_to_primary(fake_screens) -> None:
    """배율 때문에 생긴 논리 좌표의 빈 공간에 중심이 있으면 주 모니터 기준으로 변환한다."""
    # 150% 주 모니터는 논리 너비가 1280이라 1280~1919 구간은 어느 모니터에도 속하지 않는다.
    primary = make_screen(0, 0, 1920, 1080, 1.5)
    secondary = make_screen(1920, 0, 1920, 1080, 1.0)
    fake_screens(primary, secondary)

    region = global_logical_rect_to_physical(QRect(1500, 10, 100, 10))

    assert region == ScreenRegion(2250, 15, 150, 15)


def test_empty_rect_is_not_converted(fake_screens) -> None:
    """크기가 0인 사각형은 변환하지 않는다."""
    fake_screens(make_screen(0, 0, 1920, 1080, 1.0))

    assert global_logical_rect_to_physical(QRect(10, 10, 0, 5)) is None


# --- 화면 안 판정 ---------------------------------------------------------------------


def test_is_rect_on_any_screen_checks_center(fake_screens) -> None:
    """중심이 어느 모니터 위에 있는지로 화면 안/밖을 판단한다."""
    fake_screens(make_screen(0, 0, 1920, 1080, 1.0), make_screen(-1920, 0, 1920, 1080, 1.0))

    assert is_rect_on_any_screen(QRect(-500, 100, 200, 100))  # 왼쪽 모니터
    assert is_rect_on_any_screen(QRect(1850, 100, 100, 100))  # 중심이 주 모니터 안 (일부 걸침)
    assert not is_rect_on_any_screen(QRect(3000, 100, 200, 100))  # 분리된 모니터 위치
    assert not is_rect_on_any_screen(QRect(100, 1200, 200, 100))  # 화면 아래


@pytest.mark.parametrize(
    "value, expected",
    [(192.5, 193), (17.5, 18), (0.5, 1), (0.49, 0), (-0.5, -1), (-2.5, -3), (-40.0, -40)],
)
def test_round_like_qt_rounds_half_away_from_zero(value: float, expected: int) -> None:
    """Qt의 qRound처럼 0.5는 0에서 먼 쪽으로 반올림한다 (파이썬 round와 다름)."""
    assert screen_geometry.round_like_qt(value) == expected


# --- 물리 → 논리 (OCR 원문 위치에 말풍선을 놓을 때) ------------------------------------


def test_physical_text_on_high_dpi_secondary_maps_back_to_logical(fake_screens) -> None:
    """150% 보조 모니터의 OCR 물리 좌표가 그 모니터의 논리 좌표와 작업 영역으로 돌아온다."""
    primary = make_screen(0, 0, 1920, 1080, 1.0)
    secondary = make_screen(1920, 0, 2560, 1440, 1.5)  # 논리 (1920, 0, 1707, 960)
    fake_screens(primary, secondary)

    logical, area = physical_region_to_logical(ScreenRegion(1920 + 150, 300, 450, 90))

    assert logical == QRect(1920 + 100, 200, 300, 60)
    assert area == secondary.availableGeometry()


def test_physical_text_on_primary_uses_primary_area(fake_screens) -> None:
    """주 모니터(100%)의 텍스트는 좌표가 그대로이고 주 모니터 작업 영역을 쓴다."""
    primary = make_screen(0, 0, 1920, 1080, 1.0)
    fake_screens(primary, make_screen(1920, 0, 2560, 1440, 1.5))

    logical, area = physical_region_to_logical(ScreenRegion(100, 200, 300, 60))

    assert logical == QRect(100, 200, 300, 60)
    assert area == primary.availableGeometry()


@pytest.mark.parametrize("ratio", [1.25, 1.5, 1.75, 2.0])
def test_physical_and_logical_round_trip(fake_screens, ratio: float) -> None:
    """논리 → 물리 → 논리로 바꿔도 같은 사각형으로 돌아온다 (반올림 1픽셀 이내)."""
    screen = make_screen(-2560, 0, 2560, 1440, ratio)
    fake_screens(make_screen(0, 0, 1920, 1080, 1.0), screen)
    original = QRect(screen.geometry().x() + 101, 57, 333, 41)

    physical = global_logical_rect_to_physical(original)
    logical, _ = physical_region_to_logical(physical)

    for a, b in zip(logical.getRect(), original.getRect()):
        assert abs(a - b) <= 1


def test_physical_text_outside_all_screens_is_not_placed(fake_screens) -> None:
    """어느 모니터에도 없는 위치(분리된 모니터)의 텍스트는 말풍선을 놓지 않는다."""
    fake_screens(make_screen(0, 0, 1920, 1080, 1.0))

    assert physical_region_to_logical(ScreenRegion(5000, 100, 50, 20)) is None
