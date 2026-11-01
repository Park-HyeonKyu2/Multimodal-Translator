"""PaddleOCR(3.x) 기반 텍스트 인식 구현.

- 모델은 처음 인식할 때(백그라운드 워커 스레드) 불러온다. UI 스레드에서는 엔진을 만들지 않는다.
- PaddleOCR가 돌려주는 줄 단위 텍스트와 위치를 `TextSegment`(LINE)로 바꾼다. 줄을 대사로 묶는
  일은 기존 `text_grouping`이 그대로 맡는다.
- 엔진 출력은 그대로 믿지 않는다. 필드가 없거나 개수가 맞지 않으면 오류로 처리하고, 항목별로
  잘못된 값(빈 텍스트, 범위 밖 점수, 이미지 밖 위치)은 버린다.
- 캡처 이미지는 메모리에서만 쓰고 저장하지 않는다. 로그에는 인식한 텍스트를 남기지 않는다.

PaddleOCR 추론은 중간에 멈출 수 없으므로, 취소는 추론 전후에만 확인한다. 추론 중에 취소된
요청은 추론이 끝난 뒤 결과를 버린다.
"""

import logging
import math
import os
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol

import numpy as np

from translator.adapters.ocr.unavailable import PADDLE_NOT_INSTALLED_MESSAGE
from translator.core.cancellation import CancellationToken
from translator.core.errors import TextRecognitionError
from translator.core.interfaces import TextRecognizer
from translator.core.models import (
    CapturedImage,
    RecognizedText,
    ScreenRegion,
    TextGranularity,
    TextSegment,
)
from translator.tuning import PADDLE_OCR_CPU_THREADS, PADDLE_OCR_MIN_SCORE

logger = logging.getLogger(__name__)

# 엔진 잠금을 기다리는 동안 취소 여부를 확인하는 간격 (초). 반응성만 정하는 구현 세부값이다.
_LOCK_POLL_SECONDS = 0.05
# CPU 가속(oneDNN)을 켜면 검출 모델 실행 중 `ConvertPirAttribute2RuntimeAttribute not support` 오류가 나는
# PaddlePaddle 버전 앞부분 (3.3.1에서 확인, 2026-10-12). 이 버전이면 oneDNN을 켜지 않는다.
_ONEDNN_BROKEN_VERSION_PREFIXES = ("3.3.",)

# 사용자용 오류 메시지는 번역 대상 텍스트를 담지 않는다.
# 모델을 불러오지 못했을 때의 사용자용 오류 메시지
LOAD_FAILED_MESSAGE = (
    "OCR 모델을 불러오지 못했습니다. 처음 실행할 때는 모델을 내려받으므로 네트워크 연결을 확인하고, "
    "로그 파일을 확인한 뒤 프로그램을 다시 시작해 주세요."
)
# 추론 중 메모리가 부족할 때의 사용자용 오류 메시지
OUT_OF_MEMORY_MESSAGE = "OCR 실행 중 메모리가 부족합니다. 박스를 작게 하거나 다른 프로그램을 닫은 뒤 다시 시도해 주세요."
# 추론 중 그 밖의 오류가 났을 때의 사용자용 오류 메시지
INFERENCE_FAILED_MESSAGE = "텍스트 인식(OCR) 중 오류가 발생했습니다. 로그 파일을 확인해 주세요."
# 엔진 출력 형식이 예상과 다를 때의 사용자용 오류 메시지
INVALID_OUTPUT_MESSAGE = "OCR 엔진 출력 형식이 예상과 다릅니다. PaddleOCR 버전을 확인해 주세요."


class OcrEngine(Protocol):
    """PaddleOCR 엔진에서 이 어댑터가 쓰는 부분 (테스트에서는 가짜 엔진으로 바꾼다)."""

    def predict(self, input: Any) -> Iterable[Mapping[str, Any]]:  # noqa: A002 (PaddleOCR 인자 이름)
        """BGR 이미지 배열을 받아 페이지별 인식 결과(`rec_texts`, `rec_scores`, `rec_boxes` 등)를 돌려준다."""


@dataclass(frozen=True)
class PaddleOcrSettings:
    """PaddleOCR 엔진 설정 (숫자 조정값이 아닌 선택값).

    기본 모델 PP-OCRv5 mobile은 중국어(간체·번체), 영어, 일본어를 한 모델로 인식한다.
    한국어 원문을 인식하려면 인식 모델을 `korean_PP-OCRv5_mobile_rec`로 바꾼다.
    서버 모델(`PP-OCRv5_server_det`/`_rec`)은 느리고 VRAM을 많이 쓰며, 3단계 측정에서 작은 글자를
    오히려 덜 읽었다 (docs/phase2-performance-evaluation.md).
    """

    # 추론 장치: "cpu" 또는 "gpu:0" 같은 PaddlePaddle 장치 이름
    device: str = "cpu"
    # `device`로 모델을 불러오지 못했을 때 한 번 더 시도할 장치 (예: GPU 실패 시 "cpu"). None이면 다시 시도하지 않는다.
    fallback_device: str | None = None
    # 텍스트 위치 검출 모델 이름
    detection_model: str = "PP-OCRv5_mobile_det"
    # 텍스트 인식 모델 이름
    recognition_model: str = "PP-OCRv5_mobile_rec"
    # CPU 장치에서 CPU 가속(oneDNN)을 쓸지. 3단계 측정에서 약 5배 빨랐다 (PaddlePaddle 3.2.2).
    # 오류가 나는 버전(3.3.x)이 설치되어 있으면 켜지 않는다 (`effective_onednn`).
    enable_onednn: bool = True
    # CPU 장치에서 쓸 추론 스레드 수 (tuning.PADDLE_OCR_CPU_THREADS)
    cpu_threads: int = PADDLE_OCR_CPU_THREADS


def effective_onednn(requested: bool, paddle_version: str) -> bool:
    """설치된 PaddlePaddle 버전에서 oneDNN을 실제로 켤지 정한다.

    매개변수:
        requested: 설정에서 oneDNN을 켜기로 했는지.
        paddle_version: 설치된 PaddlePaddle 버전 문자열 (예: "3.2.2").
    반환값:
        켜기로 했고, 오류가 알려진 버전이 아니면 True.
    """
    return requested and not paddle_version.startswith(_ONEDNN_BROKEN_VERSION_PREFIXES)


def create_paddle_engine(settings: PaddleOcrSettings) -> OcrEngine:
    """PaddleOCR 엔진을 만든다. 처음이면 모델을 내려받으므로 수 초 이상 걸릴 수 있다.

    매개변수:
        settings: 장치, 모델 이름, CPU 설정.
    반환값:
        `predict`를 가진 PaddleOCR 엔진.
    예외:
        ImportError: PaddleOCR나 PaddlePaddle이 설치되어 있지 않을 때.
        Exception: 모델 다운로드·로딩 실패 등 PaddleOCR가 던지는 오류.
    """
    # 가져올 때마다 모델 저장소 연결을 확인하는 단계를 건너뛴다 (시작 지연 방지). 사용자가 정했으면 그대로 둔다.
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    # oneDNN은 `cpu_threads`가 아니라 OpenMP 스레드 수를 따른다(3단계 측정). paddle을 처음 가져오기 전에만
    # 효과가 있으므로 장치와 관계없이 먼저 정한다. 사용자가 정한 값이 있으면 그대로 둔다.
    os.environ.setdefault("OMP_NUM_THREADS", str(settings.cpu_threads))
    import paddle  # 무거운 의존성이므로 실제로 쓸 때만 가져온다.
    from paddleocr import PaddleOCR

    # 설치된 버전에서 실제로 쓸 oneDNN 여부
    onednn = effective_onednn(settings.enable_onednn, paddle.__version__)
    if settings.enable_onednn and not onednn:
        logger.warning("PaddlePaddle %s에서는 oneDNN 오류가 있어 끕니다 (CPU OCR이 느려집니다).", paddle.__version__)
    return PaddleOCR(
        text_detection_model_name=settings.detection_model,
        text_recognition_model_name=settings.recognition_model,
        # 화면 텍스트는 똑바로 놓여 있으므로 문서용 방향 보정·펴기 단계를 끈다 (지연 감소).
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device=settings.device,
        enable_mkldnn=onednn,
        cpu_threads=settings.cpu_threads,
    )


class PaddleTextRecognizer(TextRecognizer):
    """PaddleOCR로 캡처 이미지의 텍스트 줄과 위치를 인식한다. 여러 스레드에서 호출해도 안전하다."""

    def __init__(
        self,
        settings: PaddleOcrSettings | None = None,
        engine_factory: Callable[[PaddleOcrSettings], OcrEngine] = create_paddle_engine,
        min_score: float = PADDLE_OCR_MIN_SCORE,
    ) -> None:
        """설정, 엔진을 만드는 함수(테스트에서는 가짜 엔진), 최소 인식 점수를 받는다.

        매개변수:
            settings: 엔진 설정. None이면 기본값(CPU, PP-OCRv5 mobile).
            engine_factory: 설정을 받아 엔진을 만드는 함수. 처음 인식할 때 한 번만 호출한다.
            min_score: 이 점수보다 낮은 줄은 버린다 (보정된 확률이 아닌 엔진 점수).
        """
        # 엔진 설정 (장치, 모델 이름)
        self._settings = settings or PaddleOcrSettings()
        # 엔진 생성 함수. 생성자에서는 호출하지 않는다 (UI 스레드에서 모델을 불러오지 않기 위함).
        self._engine_factory = engine_factory
        # 남길 최소 인식 점수
        self._min_score = min_score
        # 불러온 엔진. 처음 인식할 때 만든다.
        self._engine: OcrEngine | None = None
        # 엔진 생성 실패 시 사용자용 메시지. 한 번 실패하면 다시 시도하지 않는다 (무한 재시도 방지).
        # 예외 객체 대신 메시지를 기억해, 다시 던질 때마다 traceback이 쌓이지 않게 한다.
        self._load_error_message: str | None = None
        # 엔진 생성과 추론을 한 번에 하나씩만 실행한다 (PaddleOCR 엔진은 동시 호출을 보장하지 않음).
        self._lock = threading.Lock()

    @property
    def settings(self) -> PaddleOcrSettings:
        """엔진 설정."""
        return self._settings

    def recognize(
        self, image: CapturedImage, cancellation: CancellationToken
    ) -> RecognizedText:
        """캡처 이미지의 텍스트 줄을 화면 물리 좌표와 함께 돌려준다.

        매개변수:
            image: 인식할 캡처 이미지 (BGRA).
            cancellation: 취소 토큰. 추론 전후에 확인한다.
        반환값:
            줄 단위 조각(`TextGranularity.LINE`)과 이어 붙인 전체 텍스트. 텍스트가 없으면 빈 결과.
        예외:
            OperationCancelledError: 추론 전후에 취소되었을 때.
            TextRecognitionError: 설치·모델 로딩·추론 실패, 엔진 출력 형식 오류일 때.
        """
        cancellation.raise_if_cancelled()
        if image.width <= 0 or image.height <= 0 or not image.bgra:
            return RecognizedText(text="")
        # 엔진 입력용 BGR 배열 (이 함수 안에서만 쓰고 버린다)
        pixels = bgra_to_bgr_array(image)

        self._acquire_lock(cancellation)
        try:
            # 처음 호출이면 여기서 모델을 불러온다 (이 워커 스레드에서).
            engine = self._ensure_engine()
            cancellation.raise_if_cancelled()
            # 추론 시작 시각 (지연 진단용)
            started = time.perf_counter()
            # 엔진이 돌려준 페이지 결과 목록
            pages = self._predict(engine, pixels)
            # 추론에 걸린 시간 (밀리초)
            elapsed_ms = (time.perf_counter() - started) * 1000
        finally:
            self._lock.release()
        del pixels

        # 추론 중에 취소되었으면 결과를 버린다.
        cancellation.raise_if_cancelled()
        # 검증을 통과한 화면 좌표의 줄 조각
        segments = parse_paddle_output(pages, image, self._min_score)
        logger.debug(
            "PaddleOCR: %dx%d에서 줄 %d개, %.0fms", image.width, image.height, len(segments), elapsed_ms
        )
        return RecognizedText(
            text="\n".join(segment.text for segment in segments), segments=tuple(segments)
        )

    def _acquire_lock(self, cancellation: CancellationToken) -> None:
        """엔진 잠금을 얻는다. 기다리는 동안 취소되면 `OperationCancelledError`를 던진다."""
        while not self._lock.acquire(timeout=_LOCK_POLL_SECONDS):
            cancellation.raise_if_cancelled()

    def _ensure_engine(self) -> OcrEngine:
        """엔진을 돌려준다. 아직 없으면 만든다 (잠금을 잡은 상태에서 호출).

        예외:
            TextRecognitionError: 설치되어 있지 않거나 모델을 불러오지 못했을 때. 이후 호출도 같은 오류.
        """
        if self._engine is not None:
            return self._engine
        if self._load_error_message is not None:
            raise TextRecognitionError(self._load_error_message)
        logger.info(
            "PaddleOCR 모델을 불러옵니다 (장치 %s, 검출 %s, 인식 %s).",
            self._settings.device, self._settings.detection_model, self._settings.recognition_model,
        )
        # 모델 로딩 시작 시각 (로딩 시간 기록용)
        started = time.perf_counter()
        try:
            self._engine = self._create_engine()
        except ImportError as exc:
            logger.error("PaddleOCR를 가져오지 못했습니다: %s", exc)
            self._load_error_message = PADDLE_NOT_INSTALLED_MESSAGE
            raise TextRecognitionError(PADDLE_NOT_INSTALLED_MESSAGE) from exc
        except Exception as exc:
            logger.exception("PaddleOCR 모델을 불러오지 못했습니다.")
            self._load_error_message = LOAD_FAILED_MESSAGE
            raise TextRecognitionError(LOAD_FAILED_MESSAGE) from exc
        logger.info("PaddleOCR 모델 준비 완료 (%.1f초).", time.perf_counter() - started)
        return self._engine

    def _create_engine(self) -> OcrEngine:
        """설정한 장치로 엔진을 만든다. 실패하면 대체 장치가 있을 때 한 번만 다시 시도한다.

        반환값:
            만든 엔진. 대체 장치로 만들었으면 `settings.device`도 그 장치로 바뀐다.
        예외:
            ImportError, Exception: 대체 장치까지 실패했거나 대체 장치가 없을 때 마지막 오류.
        """
        try:
            return self._engine_factory(self._settings)
        except ImportError:
            raise  # 설치 문제는 장치를 바꿔도 해결되지 않는다.
        except Exception:
            fallback = self._settings.fallback_device
            if fallback is None or fallback == self._settings.device:
                raise
            logger.warning(
                "장치 %s에서 PaddleOCR 모델을 불러오지 못해 %s로 다시 시도합니다.",
                self._settings.device, fallback, exc_info=True,
            )
            self._settings = replace(self._settings, device=fallback, fallback_device=None)
            return self._engine_factory(self._settings)

    @staticmethod
    def _predict(engine: OcrEngine, pixels: np.ndarray) -> list[Mapping[str, Any]]:
        """엔진을 실행해 페이지 결과 목록을 돌려준다. 엔진 예외는 `TextRecognitionError`로 바꾼다."""
        try:
            return list(engine.predict(pixels))
        except MemoryError as exc:
            logger.exception("PaddleOCR 추론 중 메모리 부족 (%dx%d)", pixels.shape[1], pixels.shape[0])
            raise TextRecognitionError(OUT_OF_MEMORY_MESSAGE) from exc
        except Exception as exc:
            logger.exception("PaddleOCR 추론 실패 (%dx%d)", pixels.shape[1], pixels.shape[0])
            raise TextRecognitionError(INFERENCE_FAILED_MESSAGE) from exc


def bgra_to_bgr_array(image: CapturedImage) -> np.ndarray:
    """캡처 이미지(BGRA 바이트)를 PaddleOCR 입력용 BGR 배열(높이 × 너비 × 3, uint8)로 바꾼다.

    매개변수:
        image: 캡처 이미지.
    반환값:
        새로 복사한 연속 메모리 배열 (원본 바이트를 수정하지 않는다).
    예외:
        TextRecognitionError: 픽셀 바이트 수가 이미지 크기와 맞지 않을 때.
    """
    # BGRA 한 픽셀이 차지하는 바이트 수 (형식이 정한 값)
    bytes_per_pixel = 4
    # 이미지 크기로 계산한 바이트 수
    expected = image.width * image.height * bytes_per_pixel
    if len(image.bgra) != expected:
        logger.error("캡처 바이트 수가 맞지 않습니다: %d (기대 %d)", len(image.bgra), expected)
        raise TextRecognitionError("캡처 이미지 형식이 올바르지 않습니다.")
    # 원본 바이트를 복사 없이 본 읽기 전용 배열 (높이 × 너비 × 4)
    pixels = np.frombuffer(image.bgra, dtype=np.uint8).reshape(image.height, image.width, bytes_per_pixel)
    # 알파 채널을 빼고 복사한다. PaddleOCR는 OpenCV와 같은 BGR 순서를 쓴다.
    return np.ascontiguousarray(pixels[:, :, :3])


def parse_paddle_output(
    pages: Sequence[Mapping[str, Any]], image: CapturedImage, min_score: float
) -> list[TextSegment]:
    """PaddleOCR 페이지 결과를 화면 좌표의 줄 조각 목록으로 바꾼다.

    매개변수:
        pages: `predict`가 돌려준 페이지 결과. 이미지 하나를 넣었으므로 보통 하나다.
        image: 인식한 캡처 이미지 (이미지 좌표 → 화면 좌표 변환에 쓴다).
        min_score: 이 점수보다 낮은 줄은 버린다.
    반환값:
        읽는 순서(위 → 아래, 왼쪽 → 오른쪽)로 정렬한 줄 조각 목록.
    예외:
        TextRecognitionError: 필요한 필드가 없거나 필드끼리 개수가 다를 때 (엔진 버전 불일치 등).
    """
    segments: list[TextSegment] = []
    # 값이 잘못되어 버린 항목 수, 점수가 낮아 버린 항목 수 (진단용)
    invalid = low_score = 0
    for page in pages:
        texts, scores, boxes = _page_fields(page)
        for text, score, box in zip(texts, scores, boxes):
            if not isinstance(text, str):
                invalid += 1
                continue
            text = text.strip()
            if not text:
                continue
            # 검증한 인식 점수와 화면 좌표 위치 (잘못된 값이면 None)
            confidence = _valid_score(score)
            bounds = _box_to_screen(box, image)
            if confidence is None or bounds is None:
                invalid += 1
                continue
            if confidence < min_score:
                low_score += 1
                continue
            segments.append(
                TextSegment(
                    text=text,
                    bounds=bounds,
                    granularity=TextGranularity.LINE,
                    # PaddleOCR 인식 점수. 보정된 확률이 아니다.
                    confidence=confidence,
                )
            )
    if invalid or low_score:
        logger.debug("PaddleOCR 결과 중 잘못된 항목 %d개, 점수 미달 %d개를 버렸습니다.", invalid, low_score)
    return sorted(segments, key=lambda segment: (segment.bounds.top, segment.bounds.left))


def _page_fields(page: Mapping[str, Any]) -> tuple[list[Any], list[Any], list[Any]]:
    """페이지 결과에서 텍스트, 점수, 위치 목록을 꺼낸다. 위치는 `rec_boxes`가 없으면 `rec_polys`를 쓴다."""
    try:
        # 줄마다의 인식 텍스트, 인식 점수, 위치 (같은 순서)
        texts = list(page["rec_texts"])
        scores = list(page["rec_scores"])
        boxes = page.get("rec_boxes")
        if boxes is None or len(boxes) != len(texts):
            boxes = page["rec_polys"]
        boxes = list(boxes)
    except (KeyError, TypeError, AttributeError) as exc:
        logger.error("PaddleOCR 결과에 필요한 필드가 없습니다: %s", type(exc).__name__)
        raise TextRecognitionError(INVALID_OUTPUT_MESSAGE) from exc
    if not len(texts) == len(scores) == len(boxes):
        logger.error(
            "PaddleOCR 결과 개수가 맞지 않습니다: 텍스트 %d, 점수 %d, 위치 %d",
            len(texts), len(scores), len(boxes),
        )
        raise TextRecognitionError(INVALID_OUTPUT_MESSAGE)
    return texts, scores, boxes


def _valid_score(score: Any) -> float | None:
    """점수를 0~1 실수로 돌려준다. 숫자가 아니거나 범위를 벗어나면 None."""
    try:
        # numpy 실수도 파이썬 실수로 바꾼다.
        value = float(score)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        return None
    return value


def _box_to_screen(box: Any, image: CapturedImage) -> ScreenRegion | None:
    """이미지 좌표의 위치(사각형 [x1, y1, x2, y2] 또는 꼭짓점 목록)를 화면 물리 좌표로 바꾼다.

    이미지와 겹치지 않거나 넓이가 없으면 None.
    """
    try:
        # 위치 값을 실수 배열로 읽는다 (rec_boxes는 int16, rec_polys는 꼭짓점 배열).
        values = np.asarray(box, dtype=float)
    except (TypeError, ValueError):
        return None
    if values.shape == (4,):
        # 사각형 [왼쪽, 위, 오른쪽, 아래] (이미지 픽셀)
        left, top, right, bottom = values
    elif values.ndim == 2 and values.shape[1] == 2 and values.shape[0] >= 3:
        # 꼭짓점 목록은 모두 감싸는 사각형으로 바꾼다.
        left, top = values.min(axis=0)
        right, bottom = values.max(axis=0)
    else:
        return None
    if not np.isfinite([left, top, right, bottom]).all():
        return None
    # 이미지 안으로 잘라낸 뒤 넓이가 없으면 버린다.
    left, right = max(0.0, left), min(float(image.width), right)
    top, bottom = max(0.0, top), min(float(image.height), bottom)
    if right - left < 1 or bottom - top < 1:
        return None
    # 글자를 잘라 먹지 않도록 바깥쪽 정수 픽셀로 맞춘 왼쪽 위 좌표
    x, y = math.floor(left), math.floor(top)
    return image.to_screen_region(x, y, math.ceil(right) - x, math.ceil(bottom) - y)
