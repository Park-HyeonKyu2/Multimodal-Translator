"""캡처 이미지의 변화를 감지해 "새 텍스트가 나타났을 수 있는 시점"을 판단한다.

구성은 두 부분으로 나뉜다.

- 관찰(observer): 이미지 → 비교 가능한 값. 현재는 밝기 지문(`compute_fingerprint`)이고,
  향후 OCR 텍스트 기반 감지에서는 "정규화된 인식 텍스트"가 된다.
- 판단 정책(`ChangePolicy`): 연속된 관찰값을 보고 번역할 시점을 정한다. 현재는
  "바뀐 뒤 멈추면 한 번" 규칙의 `ChangeTracker`이고, 향후 타자 효과를 처리하는
  텍스트 안정화 정책으로 바꿔 끼울 수 있다.

`AutoDetector`는 이 두 가지를 주입받으므로, 감지 방식을 바꿔도 자동 감지기와
UI 코드는 고치지 않아도 된다. 이 모듈은 Qt에 의존하지 않는다.
"""

from dataclasses import dataclass
from typing import Protocol, TypeVar

from translator.core.models import CapturedImage, ScreenRegion
from translator.tuning import (
    AUTO_DETECT_CHANGED_RATIO,
    AUTO_OCR_MIN_SAMPLE_GAP,
    AUTO_DETECT_GRID_HEIGHT,
    AUTO_DETECT_GRID_WIDTH,
    AUTO_DETECT_PIXEL_DELTA,
    AUTO_DETECT_SETTLE_SAMPLES,
)

# 관찰값 타입 (밝기 지문, 인식 텍스트 등). 정책은 관찰값을 받기만 하므로 반공변으로 둔다.
ObservationT_contra = TypeVar("ObservationT_contra", contravariant=True)


class ChangePolicy(Protocol[ObservationT_contra]):
    """연속된 관찰값을 보고 번역 시점을 정하는 정책의 공통 형태.

    모든 메서드는 UI 스레드에서 호출되며, 구현은 스레드 안전할 필요가 없다.
    """

    def observe(self, observation: ObservationT_contra) -> bool:
        """새 관찰값을 기록하고, 지금 번역해야 하면 True를 반환한다."""
        ...

    def mark_translated(self, observation: ObservationT_contra) -> None:
        """자동 감지 밖에서(수동 번역 등) 이미 번역한 화면을 기록한다."""
        ...

    def restart(self) -> None:
        """관찰 진행 상태만 비우고, 마지막으로 번역한 화면 기록은 유지한다."""
        ...

    def reset(self) -> None:
        """번역한 화면 기록까지 모두 비운다 (감시 영역이 바뀐 경우)."""
        ...


@dataclass(frozen=True)
class Fingerprint:
    """격자 표본의 밝기(0~255) 값. 원본 이미지는 보관하지 않는다."""

    grid_width: int
    grid_height: int
    samples: bytes


def compute_fingerprint(
    image: CapturedImage,
    grid_width: int = AUTO_DETECT_GRID_WIDTH,
    grid_height: int = AUTO_DETECT_GRID_HEIGHT,
    area: ScreenRegion | None = None,
) -> Fingerprint:
    """이미지를 격자 지점에서 표본 추출해 밝기 지문을 만든다.

    `area`(화면 물리 좌표)를 주면 이미지 중 그 부분만 표본으로 쓴다. 텍스트 자리의 픽셀이
    그대로인지(OCR이 놓친 것인지, 텍스트가 사라진 것인지) 판단할 때 쓴다.
    """
    left, top, width, height = _crop_box(image, area)
    # 영역이 격자보다 작으면 픽셀 수만큼만 표본을 뽑는다.
    columns = max(1, min(grid_width, width))
    rows = max(1, min(grid_height, height))
    stride = image.width * 4  # BGRA 한 행의 바이트 수
    data = image.bgra
    samples = bytearray(columns * rows)
    if not data:
        return Fingerprint(columns, rows, bytes(samples))
    index = 0
    for row in range(rows):
        # 각 격자 칸의 가운데 지점을 표본으로 쓴다.
        y = top + (2 * row + 1) * height // (2 * rows)
        row_offset = y * stride
        for column in range(columns):
            x = left + (2 * column + 1) * width // (2 * columns)
            offset = row_offset + x * 4
            blue, green, red = data[offset], data[offset + 1], data[offset + 2]
            # ITU-R BT.601 근사 휘도 (정수 연산)
            samples[index] = (299 * red + 587 * green + 114 * blue) // 1000
            index += 1
    return Fingerprint(columns, rows, bytes(samples))


def _crop_box(image: CapturedImage, area: ScreenRegion | None) -> tuple[int, int, int, int]:
    """화면 물리 좌표 영역을 이미지 안 픽셀 사각형 (왼쪽, 위, 너비, 높이)으로 바꾼다."""
    if area is None:
        return 0, 0, image.width, image.height
    # 캡처 이미지와 영역 크기가 다를 수 있어 비율로 환산한다.
    scale_x = image.width / image.region.width
    scale_y = image.height / image.region.height
    left = max(0, min(image.width - 1, int((area.left - image.region.left) * scale_x)))
    top = max(0, min(image.height - 1, int((area.top - image.region.top) * scale_y)))
    right = max(left + 1, min(image.width, int((area.right - image.region.left) * scale_x)))
    bottom = max(top + 1, min(image.height, int((area.bottom - image.region.top) * scale_y)))
    return left, top, right - left, bottom - top


def difference(first: Fingerprint, second: Fingerprint) -> float:
    """두 지문의 평균 밝기 차이(0~255). 진단용이며 변화 판단에는 `changed_ratio`를 쓴다.

    격자 크기가 다르면 완전히 다른 화면으로 본다.
    """
    if (first.grid_width, first.grid_height) != (second.grid_width, second.grid_height):
        return 255.0
    total = sum(abs(a - b) for a, b in zip(first.samples, second.samples))
    return total / len(first.samples)


def changed_ratio(
    first: Fingerprint, second: Fingerprint, pixel_delta: int = AUTO_DETECT_PIXEL_DELTA
) -> float:
    """밝기가 `pixel_delta` 이상 바뀐 표본의 비율(0~1). 격자 크기가 다르면 1.

    큰 박스 안의 작은 글자가 바뀌어도 바뀐 표본 수로 잡아낼 수 있다. 평균 차이는
    박스 대부분이 그대로인 배경에 묻혀 버린다.
    """
    if (first.grid_width, first.grid_height) != (second.grid_width, second.grid_height):
        return 1.0
    changed = sum(1 for a, b in zip(first.samples, second.samples) if abs(a - b) >= pixel_delta)
    return changed / len(first.samples)


class ChangeTracker:
    """연속으로 관찰한 지문을 보고 번역이 필요한 시점을 판단한다 (`ChangePolicy` 구현).

    1. 직전 지문과 비교해 바뀐 표본 비율이 기준 이상이면 "바뀌는 중"으로 보고 기다린다.
    2. `settle_samples`번 연속으로 거의 같으면 "멈춤"으로 본다.
    3. 멈춘 화면이 마지막으로 번역한 화면과 다를 때만 번역을 요청한다.
    """

    def __init__(
        self,
        pixel_delta: int = AUTO_DETECT_PIXEL_DELTA,
        changed_ratio_threshold: float = AUTO_DETECT_CHANGED_RATIO,
        settle_samples: int = AUTO_DETECT_SETTLE_SAMPLES,
    ) -> None:
        """표본 하나의 변화 기준, 화면 변화로 볼 비율, 멈춤 판단 횟수를 정한다."""
        self._pixel_delta = pixel_delta
        self._ratio_threshold = changed_ratio_threshold
        self._settle_samples = settle_samples
        # 마지막으로 "바뀌었다"고 판단한 시점의 지문
        self._last_seen: Fingerprint | None = None
        # 그 뒤로 연속해서 같은 화면이 관찰된 횟수
        self._stable_count = 0
        # 마지막으로 번역한(또는 수동 번역으로 기록된) 화면의 지문
        self._last_translated: Fingerprint | None = None

    def reset(self) -> None:
        """번역한 화면 기록까지 모두 비운다 (박스 위치/크기가 바뀐 경우)."""
        self.restart()
        self._last_translated = None

    def restart(self) -> None:
        """관찰 진행 상태만 비운다. 자동 감지를 다시 켤 때 사용한다."""
        self._last_seen = None
        self._stable_count = 0

    def mark_translated(self, observation: Fingerprint) -> None:
        """수동 번역한 화면을 기록해, 자동 감지가 같은 화면을 다시 번역하지 않게 한다."""
        self._last_translated = observation

    def observe(self, observation: Fingerprint) -> bool:
        """새 지문을 기록하고, 지금 번역해야 하면 True를 반환한다."""
        if self._last_seen is None or self._is_changed(observation, self._last_seen):
            # 화면이 바뀌는 중: 기준을 새 화면으로 옮기고 멈출 때까지 기다린다.
            self._last_seen = observation
            self._stable_count = 0
            return False

        self._stable_count += 1
        if self._stable_count < self._settle_samples:
            return False
        if (
            self._last_translated is not None
            and not self._is_changed(observation, self._last_translated)
        ):
            # 이미 번역한 화면과 같으면 다시 번역하지 않는다.
            return False
        self._last_translated = observation
        return True

    def _is_changed(self, first: Fingerprint, second: Fingerprint) -> bool:
        """두 지문이 다른 화면인지 (바뀐 표본 비율이 기준 이상)."""
        return changed_ratio(first, second, self._pixel_delta) >= self._ratio_threshold


def is_changed(
    first: Fingerprint,
    second: Fingerprint,
    ratio_threshold: float = AUTO_DETECT_CHANGED_RATIO,
    pixel_delta: int = AUTO_DETECT_PIXEL_DELTA,
) -> bool:
    """두 지문이 다른 화면인지 (바뀐 표본 비율이 기준 이상)."""
    return changed_ratio(first, second, pixel_delta) >= ratio_threshold


class ChangeTriggerPolicy:
    """화면이 바뀌면 멈추기를 기다리지 않고 OCR을 요청하는 판단 정책 (`ChangePolicy` 구현).

    - 마지막으로 OCR한 화면과 다르면 OCR을 요청한다.
    - 화면이 계속 바뀌는 동안에는 `min_sample_gap` 표본마다 한 번만 요청한다.
    - 화면이 멈추면 마지막 화면이 OCR한 화면과 다를 때 한 번 더 요청한다.

    화면이 멈추기를 기다리지 않으므로, 배경이 계속 움직여 화면이 "멈추지 않아도" OCR이
    영구히 미뤄지지 않는다. 글자가 다 나왔는지(타자 효과)는 OCR 텍스트로 따로 판단한다
    (`text_stabilizer`). 중복 번역 여부도 텍스트로 따로 판단한다.
    """

    def __init__(
        self,
        pixel_delta: int = AUTO_DETECT_PIXEL_DELTA,
        changed_ratio_threshold: float = AUTO_DETECT_CHANGED_RATIO,
        min_sample_gap: int = AUTO_OCR_MIN_SAMPLE_GAP,
    ) -> None:
        """표본 하나의 변화 기준, 화면 변화로 볼 비율, OCR 요청 사이 최소 표본 수를 정한다."""
        self._pixel_delta = pixel_delta
        self._ratio_threshold = changed_ratio_threshold
        self._min_gap = max(1, min_sample_gap)
        # 마지막으로 OCR을 요청한(또는 수동 OCR로 기록된) 화면의 지문
        self._last_requested: Fingerprint | None = None
        # 마지막 요청 뒤 관찰한 표본 수 (처음에는 바로 요청할 수 있게 최소 간격으로 둔다)
        self._samples_since_request = self._min_gap

    def observe(self, observation: Fingerprint) -> bool:
        """새 지문을 기록하고, 지금 OCR해야 하면 True를 반환한다."""
        self._samples_since_request += 1
        if self._last_requested is not None and not is_changed(
            observation, self._last_requested, self._ratio_threshold, self._pixel_delta
        ):
            # 이미 OCR한 화면과 같다.
            return False
        if self._samples_since_request < self._min_gap:
            # 화면이 계속 바뀌는 중: 너무 자주 OCR하지 않는다.
            return False
        self._last_requested = observation
        self._samples_since_request = 0
        return True

    def mark_translated(self, observation: Fingerprint) -> None:
        """다른 경로(수동 번역)로 이미 OCR한 화면을 기록한다."""
        self._last_requested = observation

    def restart(self) -> None:
        """다시 켤 때 바로 요청할 수 있게 한다. OCR한 화면 기록은 유지한다."""
        self._samples_since_request = self._min_gap

    def reset(self) -> None:
        """감시 영역이 바뀌면 OCR한 화면 기록까지 비운다."""
        self._last_requested = None
        self._samples_since_request = self._min_gap
