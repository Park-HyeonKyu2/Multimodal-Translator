"""계층 간에 주고받는 불변 데이터 모델.

좌표 규칙: `ScreenRegion`은 항상 **가상 데스크톱 기준 물리 픽셀** 좌표다. OCR 어댑터는
이미지 안 좌표를 `CapturedImage.to_screen_region`으로 바꿔서 돌려줘야 하며, UI는
화면에 그릴 때만 Qt 논리 좌표로 변환한다.
"""

from dataclasses import dataclass, field
from enum import Enum


@dataclass(frozen=True)
class ScreenRegion:
    """가상 데스크톱 기준 물리 픽셀 좌표의 사각형 영역.

    다중 모니터 환경에서는 주 모니터 왼쪽/위쪽에 있는 모니터 때문에
    `left`, `top`이 음수일 수 있다.
    """

    left: int
    top: int
    width: int
    height: int

    def __post_init__(self) -> None:
        """크기가 0 이하인 영역을 거부한다."""
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"영역 크기는 양수여야 합니다: {self.width}x{self.height}")

    @property
    def right(self) -> int:
        """오른쪽 경계 (영역 밖 첫 픽셀의 x)."""
        return self.left + self.width

    @property
    def bottom(self) -> int:
        """아래쪽 경계 (영역 밖 첫 픽셀의 y)."""
        return self.top + self.height

    @property
    def center(self) -> tuple[float, float]:
        """영역 중심 좌표."""
        return (self.left + self.width / 2, self.top + self.height / 2)

    def contains_point(self, x: float, y: float) -> bool:
        """점 (x, y)가 영역 안에 있는지 여부."""
        return self.left <= x < self.right and self.top <= y < self.bottom

    def intersection_over_union(self, other: "ScreenRegion") -> float:
        """두 영역이 겹치는 정도 (0: 안 겹침, 1: 같은 영역)."""
        overlap_w = min(self.right, other.right) - max(self.left, other.left)
        overlap_h = min(self.bottom, other.bottom) - max(self.top, other.top)
        if overlap_w <= 0 or overlap_h <= 0:
            return 0.0
        overlap = overlap_w * overlap_h
        union = self.width * self.height + other.width * other.height - overlap
        return overlap / union

    def union(self, other: "ScreenRegion") -> "ScreenRegion":
        """두 영역을 모두 감싸는 가장 작은 영역."""
        left = min(self.left, other.left)
        top = min(self.top, other.top)
        return ScreenRegion(
            left, top, max(self.right, other.right) - left, max(self.bottom, other.bottom) - top
        )


@dataclass(frozen=True)
class CapturedImage:
    """메모리에만 보관되는 캡처 이미지 (BGRA 픽셀, 행 우선)."""

    region: ScreenRegion
    width: int
    height: int
    bgra: bytes = field(repr=False)

    def to_screen_region(self, x: int, y: int, width: int, height: int) -> ScreenRegion:
        """이미지 안 사각형(픽셀)을 화면 물리 좌표 영역으로 바꾼다. 이미지 밖은 잘라낸다."""
        # 캡처 이미지는 보통 영역과 같은 크기지만, 다를 경우를 대비해 비율로 환산한다.
        scale_x = self.region.width / self.width
        scale_y = self.region.height / self.height
        left = max(0, x)
        top = max(0, y)
        right = min(self.width, x + width)
        bottom = min(self.height, y + height)
        screen_left = self.region.left + round(left * scale_x)
        screen_top = self.region.top + round(top * scale_y)
        screen_right = self.region.left + round(right * scale_x)
        screen_bottom = self.region.top + round(bottom * scale_y)
        return ScreenRegion(
            screen_left,
            screen_top,
            max(1, screen_right - screen_left),
            max(1, screen_bottom - screen_top),
        )


class TextGranularity(Enum):
    """OCR 조각의 단위. 엔진마다 돌려주는 단위가 다르므로 함께 기록한다."""

    CHARACTER = "character"
    WORD = "word"
    LINE = "line"
    # 엔진이 이미 문단/대사 단위로 묶어 준 경우
    BLOCK = "block"


@dataclass(frozen=True)
class TextSegment:
    """OCR이 인식한 텍스트 조각 하나와 그 화면 위치 (물리 픽셀)."""

    text: str
    bounds: ScreenRegion
    granularity: TextGranularity
    # 엔진이 제공할 때만 채운다. 보정된 확률이 아니다.
    confidence: float | None = None


@dataclass(frozen=True)
class RecognizedText:
    """OCR 결과.

    `segments`는 위치가 있는 조각 목록이고, `text`는 조각을 읽는 순서로 이은 전체
    텍스트다. `confidence`는 엔진이 제공할 때만 채우며 보정된 확률이 아니다.
    """

    text: str
    confidence: float | None = None
    segments: tuple[TextSegment, ...] = ()


@dataclass(frozen=True)
class TextUnit:
    """번역 단위로 묶은 텍스트 (대사 한 덩어리). 원본 조각과 전체 범위를 함께 가진다."""

    text: str
    bounds: ScreenRegion
    segments: tuple[TextSegment, ...] = ()


class TranslationDirection(Enum):
    """번역문을 원문 텍스트의 어느 쪽에 표시할지."""

    TOP = "top"
    BOTTOM = "bottom"
    LEFT = "left"
    RIGHT = "right"


# 요구사항으로 정해진 기본 출력 방향 (원문 아래)
DEFAULT_TRANSLATION_DIRECTION = TranslationDirection.BOTTOM


@dataclass(frozen=True)
class TranslationRequest:
    """번역 엔진 입력.

    `context`는 향후 Context Memory가 채울 관련 문맥(최근 대사, 장면 요약 등)이다.
    """

    source_text: str
    source_language: str = "auto"
    target_language: str = "ko"
    context: tuple[str, ...] = ()


@dataclass(frozen=True)
class TranslationResult:
    """번역 엔진 출력."""

    source_text: str
    translated_text: str
    engine: str


@dataclass(frozen=True)
class TranslatedUnit:
    """화면 위치가 있는 번역 결과 하나 (원문 텍스트 단위)."""

    source_text: str
    translated_text: str
    bounds: ScreenRegion
    engine: str


@dataclass(frozen=True)
class TranslationBatch:
    """한 번의 캡처에서 나온 번역 결과 전체.

    `source_id`는 결과를 낸 출처(바운딩 박스 ID 또는 전체 화면)다. 결과를 올바른
    박스에 연결하고, 같은 출처의 이전 결과와 비교(추적)하는 데 쓴다.
    """

    source_id: str
    capture_region: ScreenRegion
    units: tuple[TranslatedUnit, ...] = ()
    # 결과를 낸 캡처(프레임) 번호, 작업 요청 번호, 캡처 시각(단조 시계, 초). 모르면 None.
    frame_id: int | None = None
    request_id: int | None = None
    captured_at: float | None = None
