"""Qt 논리 좌표와 MSS 물리 픽셀 좌표 사이의 변환."""

from PySide6.QtCore import QRect
from PySide6.QtGui import QGuiApplication

from translator.core.models import ScreenRegion


def round_like_qt(value: float) -> int:
    """Qt의 qRound와 같은 방식(0.5는 0에서 먼 쪽)으로 반올림한다.

    파이썬 `round()`는 0.5를 짝수 쪽으로 보내므로(192.5 → 192), Qt가 창을 놓은
    물리 위치와 1픽셀 어긋날 수 있다. Qt와 같은 규칙을 써서 박스 위치와 캡처
    영역을 일치시킨다.
    """
    return int(value + 0.5) if value >= 0 else int(value - 0.5)


def logical_rect_to_physical(
    local_rect: QRect, screen_geometry: QRect, device_pixel_ratio: float
) -> ScreenRegion:
    """모니터 내부 논리 좌표 사각형을 가상 데스크톱 물리 픽셀 좌표로 변환한다.

    Qt 6(Windows)는 각 모니터의 좌상단을 물리 좌표 그대로 두고 모니터 내부만
    배율로 나누어 논리 좌표를 만든다. 따라서 모니터 원점 + 내부 오프셋 × 배율로
    물리 좌표를 얻는다. 가장자리를 각각 반올림해 인접 영역 사이에 틈이 생기지 않게 한다.
    """
    # 모니터 좌상단의 물리 좌표 (Qt 6에서는 논리 좌표와 같다)
    origin_x = screen_geometry.x()
    origin_y = screen_geometry.y()
    ratio = device_pixel_ratio
    # 네 변을 따로 변환한 뒤 너비/높이를 계산해야 인접 영역이 정확히 맞닿는다.
    left = origin_x + round_like_qt(local_rect.x() * ratio)
    top = origin_y + round_like_qt(local_rect.y() * ratio)
    right = origin_x + round_like_qt((local_rect.x() + local_rect.width()) * ratio)
    bottom = origin_y + round_like_qt((local_rect.y() + local_rect.height()) * ratio)
    return ScreenRegion(left, top, right - left, bottom - top)


def physical_region_to_logical(region: ScreenRegion) -> tuple[QRect, QRect] | None:
    """물리 픽셀 영역을 Qt 논리 좌표로 바꾼다. (논리 사각형, 그 모니터의 작업 영역)을 반환한다.

    OCR이 돌려준 원문 위치에 번역 말풍선을 놓을 때 쓴다. 영역 중심이 있는 모니터의
    배율로 변환하며, 중심이 어느 모니터에도 없으면 None을 반환한다.
    `logical_rect_to_physical`의 역변환이다 (Qt 6은 모니터 원점을 물리 좌표 그대로 둔다).
    """
    center_x, center_y = region.center
    for screen in QGuiApplication.screens():
        geometry = screen.geometry()
        ratio = screen.devicePixelRatio()
        # 모니터의 물리 픽셀 범위: 원점은 같고 크기만 배율만큼 크다.
        physical_width = round_like_qt(geometry.width() * ratio)
        physical_height = round_like_qt(geometry.height() * ratio)
        if not (
            geometry.x() <= center_x < geometry.x() + physical_width
            and geometry.y() <= center_y < geometry.y() + physical_height
        ):
            continue
        left = geometry.x() + round_like_qt((region.left - geometry.x()) / ratio)
        top = geometry.y() + round_like_qt((region.top - geometry.y()) / ratio)
        right = geometry.x() + round_like_qt((region.right - geometry.x()) / ratio)
        bottom = geometry.y() + round_like_qt((region.bottom - geometry.y()) / ratio)
        logical = QRect(left, top, max(1, right - left), max(1, bottom - top))
        return logical, screen.availableGeometry()
    return None


def virtual_desktop_region() -> ScreenRegion | None:
    """연결된 모든 모니터를 감싸는 물리 픽셀 영역 (전체 화면 추적용). 모니터가 없으면 None."""
    lefts, tops, rights, bottoms = [], [], [], []
    for screen in QGuiApplication.screens():
        geometry = screen.geometry()
        ratio = screen.devicePixelRatio()
        # 모니터의 물리 범위: 원점은 논리 좌표와 같고 크기만 배율만큼 크다.
        lefts.append(geometry.x())
        tops.append(geometry.y())
        rights.append(geometry.x() + round_like_qt(geometry.width() * ratio))
        bottoms.append(geometry.y() + round_like_qt(geometry.height() * ratio))
    if not lefts:
        return None
    left, top = min(lefts), min(tops)
    return ScreenRegion(left, top, max(rights) - left, max(bottoms) - top)


def is_rect_on_any_screen(rect: QRect) -> bool:
    """사각형의 중심이 연결된 모니터 중 하나 위에 있으면 True.

    저장된 창 위치를 복원하거나 모니터가 분리된 뒤 창이 화면 밖에 남았는지 판단할 때 쓴다.
    """
    center = rect.center()
    return any(screen.geometry().contains(center) for screen in QGuiApplication.screens())


def global_logical_rect_to_physical(rect: QRect) -> ScreenRegion | None:
    """가상 데스크톱 논리 좌표 사각형을 물리 픽셀 좌표로 변환한다.

    사각형 중심이 있는 모니터의 배율을 사용한다. 배율이 다른 두 모니터에 걸친
    사각형은 다른 쪽 모니터 부분이 조금 어긋날 수 있다.
    """
    if rect.width() <= 0 or rect.height() <= 0:
        return None
    # 중심이 어느 모니터에도 없으면(배율 때문에 생긴 논리 좌표의 빈 공간 등) 주 모니터 기준으로 변환한다.
    screen = QGuiApplication.screenAt(rect.center()) or QGuiApplication.primaryScreen()
    if screen is None:
        return None
    geometry = screen.geometry()
    # 모니터 기준 내부 좌표로 바꾼다.
    local = rect.translated(-geometry.x(), -geometry.y())
    return logical_rect_to_physical(local, geometry, screen.devicePixelRatio())
