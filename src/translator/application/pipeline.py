"""캡처 → 텍스트 인식 → 대사 단위 묶기 → 번역으로 이어지는 동기식 작업 흐름.

두 단계로 나뉜다.

1. 인식(`recognize`): 캡처(또는 이미 찍은 프레임 사용) → OCR(공유 캐시) → 대사 묶기.
   결과(`OcrResult`)에는 텍스트별 위치와 그 자리의 픽셀 지문이 담긴다.
2. 번역(`translate_texts`): 안정된 원문만 받아 번역한다 (번역 캐시 사용).

두 단계 사이(텍스트 안정화, 중복 번역 판단)는 UI 스레드의 `TextDisplayModel`이 맡는다.
`run`은 두 단계를 바로 이어 실행하는 호환용 진입점이다. Qt에 의존하지 않으므로
백그라운드 스레드에서 실행하고 단독으로 테스트할 수 있다.
"""

import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass

from translator.application.change_detection import Fingerprint, compute_fingerprint
from translator.application.frames import Frame, next_frame_id, now
from translator.application.ocr_service import OcrService
from translator.application.text_grouping import group_segments
from translator.core.cancellation import CancellationToken
from translator.core.errors import TranslationError
from translator.core.interfaces import ScreenCapturer, TextRecognizer, Translator
from translator.core.session import SessionContext
from translator.core.models import (
    CapturedImage,
    RecognizedText,
    ScreenRegion,
    TextUnit,
    TranslatedUnit,
    TranslationBatch,
    TranslationRequest,
)
from translator.tuning import (
    AUTO_DETECT_GRID_HEIGHT,
    AUTO_DETECT_GRID_WIDTH,
    TEXT_AREA_GRID_HEIGHT,
    TEXT_AREA_GRID_WIDTH,
    TRANSLATION_CACHE_SIZE,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineRequest:
    """인식 작업 하나의 입력.

    - `source_id`: 결과를 연결할 출처 (바운딩 박스 ID 또는 전체 화면)
    - `region`: 캡처할 물리 픽셀 영역. None이면 전체 화면
    - `exclude`: 이 영역들 안에 중심이 있는 텍스트는 처리하지 않는다 (다른 출처가 담당)
    - `frame`: 이미 찍은 캡처. 있으면 다시 캡처하지 않는다 (자동 감지 표본 재사용)
    - `probe_areas`: 지금 표시 중인 텍스트 자리. 그 자리의 픽셀 지문을 함께 돌려줘
      OCR이 놓친 것인지 텍스트가 사라진 것인지 판단하게 한다.
    - `fingerprint_grid`: 화면 전체 지문의 격자 크기 (출처의 자동 감지 격자와 맞춘다)
    """

    source_id: str
    region: ScreenRegion | None
    exclude: tuple[ScreenRegion, ...] = ()
    frame: Frame | None = None
    probe_areas: tuple[ScreenRegion, ...] = ()
    fingerprint_grid: tuple[int, int] = (AUTO_DETECT_GRID_WIDTH, AUTO_DETECT_GRID_HEIGHT)


@dataclass(frozen=True)
class OcrResult:
    """인식 단계의 결과. 이미지는 담지 않는다."""

    source_id: str
    frame_id: int
    captured_at: float
    capture_region: ScreenRegion
    units: tuple[TextUnit, ...]
    # units와 같은 순서의, 각 텍스트 자리의 픽셀 지문
    unit_fingerprints: tuple[Fingerprint, ...]
    # (표시 중인 텍스트 자리, 지금 그 자리의 픽셀 지문)
    probes: tuple[tuple[ScreenRegion, Fingerprint], ...]
    # 화면 전체 지문 (수동 OCR 뒤 자동 감지의 기준 화면으로 기록할 때 쓴다)
    fingerprint: Fingerprint
    # True면 OCR 엔진을 새로 호출하지 않고 이전 결과를 재사용했다.
    ocr_reused: bool
    # 작업 요청 번호 (컨트롤러가 채운다)
    request_id: int | None = None
    # 요청한 시점의 콘텐츠·세션 (컨트롤러가 채운다). 이 값이 현재 세션과 다르면 결과를 버린다.
    session: SessionContext | None = None


class TranslationPipeline:
    """인식(캡처·OCR·묶기)과 번역 단계를 실행한다. 여러 워커 스레드에서 호출해도 안전하다."""

    def __init__(
        self,
        capturer: ScreenCapturer,
        recognizer: TextRecognizer | OcrService,
        translator: Translator,
        target_language: str = "ko",
        cache_size: int = TRANSLATION_CACHE_SIZE,
    ) -> None:
        """캡처, OCR(엔진 또는 공유 서비스), 번역 구현과 목표 언어, 번역 캐시 크기를 주입받는다."""
        self._capturer = capturer
        # OCR 엔진을 직접 받으면 공유 서비스로 감싸 캐시와 중복 합치기를 적용한다.
        self._ocr = recognizer if isinstance(recognizer, OcrService) else OcrService(recognizer)
        self._translator = translator
        # 번역 목표 언어 코드 (예: "ko")
        self._target_language = target_language
        # (목표 언어, 원문) → (번역문, 엔진). 같은 원문을 반복 번역하지 않기 위한 LRU 캐시.
        # 향후 문맥(context)을 번역에 쓰게 되면 키에 문맥도 포함해야 한다.
        self._cache: OrderedDict[tuple[str, str], tuple[str, str]] = OrderedDict()
        self._cache_size = cache_size
        # 워커가 여러 개가 되어도 캐시가 깨지지 않도록 보호한다.
        self._cache_lock = threading.Lock()

    @property
    def ocr_service(self) -> OcrService:
        """OCR 공유 서비스 (측정·테스트용)."""
        return self._ocr

    def recognize(self, request: PipelineRequest, cancellation: CancellationToken) -> OcrResult:
        """캡처(또는 받은 프레임)를 OCR해 대사 단위 텍스트와 위치를 반환한다.

        텍스트가 없으면 오류가 아니라 `units`가 빈 결과를 반환한다. 캡처 이미지는 이 함수
        안에서만 사용하고 보관하지 않는다.
        """
        cancellation.raise_if_cancelled()
        image, frame_id, captured_at, frame_fingerprint = self._acquire(request)

        cancellation.raise_if_cancelled()
        outcome = self._ocr.recognize(image, cancellation)
        units = tuple(
            unit
            for unit in _units_from(outcome.recognized, image.region)
            if not _is_excluded(unit, request.exclude)
        )
        # 텍스트 자리의 픽셀 지문: 다음 결과에서 이 텍스트가 빠졌을 때 화면이 그대로였는지 판단한다.
        unit_fingerprints = tuple(_area_fingerprint(image, unit.bounds) for unit in units)
        probes = tuple(
            (area, _area_fingerprint(image, area))
            for area in request.probe_areas
            if _overlaps(area, image.region)
        )
        if frame_fingerprint is None:
            grid_width, grid_height = request.fingerprint_grid
            frame_fingerprint = compute_fingerprint(image, grid_width, grid_height)
        capture_region = image.region
        del image

        logger.info(
            "%s: 프레임 %d에서 텍스트 %d개 인식%s",
            request.source_id, frame_id, len(units), " (OCR 재사용)" if outcome.reused else "",
        )
        return OcrResult(
            source_id=request.source_id,
            frame_id=frame_id,
            captured_at=captured_at,
            capture_region=capture_region,
            units=units,
            unit_fingerprints=unit_fingerprints,
            probes=probes,
            fingerprint=frame_fingerprint,
            ocr_reused=outcome.reused,
        )

    def translate_texts(
        self, source_texts: list[str] | tuple[str, ...], cancellation: CancellationToken
    ) -> dict[str, tuple[str, str]]:
        """원문들을 번역해 원문 → (번역문, 엔진)으로 돌려준다.

        최근에 번역한 원문은 캐시를 쓰고, 나머지 서로 다른 원문만 한 번에 번역 엔진에 넘긴다.
        """
        results: dict[str, tuple[str, str]] = {}
        missing: list[str] = []
        with self._cache_lock:
            for text in dict.fromkeys(source_texts):  # 순서를 유지하며 중복 제거
                cached = self._cache.get((self._target_language, text))
                if cached is None:
                    missing.append(text)
                else:
                    self._cache.move_to_end((self._target_language, text))
                    results[text] = cached
        if not missing:
            return results
        cancellation.raise_if_cancelled()
        # TODO(Context Memory): 콘텐츠별 관련 문맥을 조회해 context에 채운다.
        requests = [
            TranslationRequest(source_text=text, target_language=self._target_language)
            for text in missing
        ]
        translated = self._translator.translate_batch(requests, cancellation)
        if len(translated) != len(requests):
            # 엔진 출력을 그대로 믿지 않는다: 개수가 다르면 원문과 번역을 짝지을 수 없다.
            raise TranslationError("번역 엔진이 요청과 다른 개수의 결과를 돌려주었습니다.")
        with self._cache_lock:
            for text, result in zip(missing, translated):
                value = (result.translated_text, result.engine)
                results[text] = value
                self._cache[(self._target_language, text)] = value
            # 오래된 항목부터 지워 크기를 제한한다.
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return results

    def run(self, request: PipelineRequest, cancellation: CancellationToken) -> TranslationBatch:
        """인식과 번역을 바로 이어 실행한다 (안정화 없이 한 번에 번역하는 호환용 진입점)."""
        result = self.recognize(request, cancellation)
        cancellation.raise_if_cancelled()
        translations = self.translate_texts([unit.text for unit in result.units], cancellation)
        translated = tuple(
            TranslatedUnit(unit.text, translations[unit.text][0], unit.bounds, translations[unit.text][1])
            for unit in result.units
        )
        return TranslationBatch(
            request.source_id,
            result.capture_region,
            translated,
            frame_id=result.frame_id,
            captured_at=result.captured_at,
        )

    def _acquire(
        self, request: PipelineRequest
    ) -> tuple[CapturedImage, int, float, Fingerprint | None]:
        """처리할 이미지를 얻는다. 받은 프레임이 있으면 다시 캡처하지 않는다."""
        frame = request.frame
        if frame is not None:
            fingerprint = frame.fingerprint if isinstance(frame.fingerprint, Fingerprint) else None
            return frame.image, frame.frame_id, frame.captured_at, fingerprint
        captured_at = now()
        if request.region is None:
            image = self._capturer.capture_full_screen()
        else:
            image = self._capturer.capture_region(request.region)
        return image, next_frame_id(), captured_at, None


def _units_from(recognized: RecognizedText, capture_region: ScreenRegion) -> list[TextUnit]:
    """OCR 결과를 대사 단위로 묶는다. 위치 정보가 없는 엔진이면 캡처 영역 전체를 위치로 쓴다."""
    if recognized.segments:
        return group_segments(recognized.segments)
    if recognized.text.strip():
        return [TextUnit(recognized.text.strip(), capture_region)]
    return []


def _is_excluded(unit: TextUnit, exclude: tuple[ScreenRegion, ...]) -> bool:
    """텍스트 중심이 제외 영역 안에 있으면 True (다른 출처가 이미 맡은 텍스트)."""
    x, y = unit.bounds.center
    return any(area.contains_point(x, y) for area in exclude)


def _area_fingerprint(image: CapturedImage, area: ScreenRegion) -> Fingerprint:
    """화면 영역 중 텍스트 자리만의 작은 픽셀 지문."""
    return compute_fingerprint(image, TEXT_AREA_GRID_WIDTH, TEXT_AREA_GRID_HEIGHT, area)


def _overlaps(area: ScreenRegion, region: ScreenRegion) -> bool:
    """두 영역이 겹치는지 여부."""
    return area.intersection_over_union(region) > 0
