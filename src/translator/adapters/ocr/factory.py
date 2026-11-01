"""OCR 엔진 선택: 설정(환경 변수)에 따라 PaddleOCR 또는 Mock OCR을 만든다.

- `TRANSLATOR_OCR_ENGINE`: `auto`(기본), `paddle`, `mock`
  - `auto`: PaddleOCR가 설치되어 있으면 PaddleOCR, 없으면 경고 로그를 남기고 Mock OCR
  - `paddle`: 항상 PaddleOCR. 설치되어 있지 않으면 인식할 때마다 설치 안내 오류를 보여 준다.
  - `mock`: 화면에서 글자처럼 보이는 영역을 찾는 Mock OCR (모델 없이 UI 확인용)
- `TRANSLATOR_OCR_DEVICE`: PaddleOCR 추론 장치
  - `auto`(기본): PaddlePaddle GPU 빌드가 설치되어 있고 CUDA GPU가 있으면 `gpu:0`, 아니면 `cpu`.
    자동으로 GPU를 골랐는데 모델을 불러오지 못하면 CPU로 한 번 다시 시도한다.
  - `cpu`, `gpu:0` 등: 그 장치만 쓴다.

PaddleOCR 모듈은 무거우므로 선택되었을 때만 가져온다. 이 모듈은 모델을 불러오지 않고 paddle도
가져오지 않는다 (모델은 처음 인식할 때 워커 스레드에서 불러온다). GPU 확인은 NVIDIA 드라이버에
장치 수만 묻는다.
"""

import ctypes
import importlib.metadata
import importlib.util
import logging
import os
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum

from translator.adapters.ocr.mock_layout import MockLayoutTextRecognizer
from translator.adapters.ocr.unavailable import (
    PADDLE_NOT_INSTALLED_MESSAGE,
    UnavailableTextRecognizer,
)
from translator.core.interfaces import TextRecognizer

logger = logging.getLogger(__name__)

# OCR 엔진 종류를 고르는 환경 변수 이름
ENGINE_ENV = "TRANSLATOR_OCR_ENGINE"
# PaddleOCR 추론 장치를 고르는 환경 변수 이름
DEVICE_ENV = "TRANSLATOR_OCR_DEVICE"
# 장치를 자동으로 고르는 값
DEVICE_AUTO = "auto"
# GPU 장치 이름 (첫 번째 GPU)
GPU_DEVICE = "gpu:0"
# CPU 장치 이름
CPU_DEVICE = "cpu"
# PaddleOCR가 동작하는 데 필요한 모듈 (가져오지 않고 설치 여부만 확인한다)
_PADDLE_MODULES = ("paddleocr", "paddle", "numpy")
# PaddlePaddle GPU 빌드의 배포 패키지 이름
_PADDLE_GPU_DISTRIBUTION = "paddlepaddle-gpu"


class OcrEngineKind(Enum):
    """선택할 수 있는 OCR 엔진 종류."""

    AUTO = "auto"
    PADDLE = "paddle"
    MOCK = "mock"


@dataclass(frozen=True)
class OcrSelection:
    """고른 OCR 엔진과, 앱 동작을 맞추는 데 필요한 정보."""

    # 고른 OCR 엔진
    recognizer: TextRecognizer
    # True면 OCR이 CPU로 실행되어 전체 화면 OCR 한 번에 수 초가 걸린다 (전체 화면 확인 간격을 늘린다).
    slow_full_screen: bool = False


def paddle_available() -> bool:
    """PaddleOCR 실행에 필요한 모듈이 설치되어 있는지 여부 (모듈을 가져오지 않고 확인).

    반환값:
        모두 설치되어 있으면 True.
    """
    return all(importlib.util.find_spec(name) is not None for name in _PADDLE_MODULES)


def paddle_gpu_build_installed() -> bool:
    """PaddlePaddle GPU 빌드(`paddlepaddle-gpu`)가 설치되어 있는지 여부.

    반환값:
        설치되어 있으면 True.
    """
    try:
        importlib.metadata.version(_PADDLE_GPU_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


def cuda_gpu_available() -> bool:
    """NVIDIA 드라이버로 CUDA GPU가 하나 이상 있는지 확인한다 (paddle을 가져오지 않는다).

    반환값:
        Windows에서 NVIDIA 드라이버가 CUDA 장치를 하나 이상 알려 주면 True. 그 밖에는 False.
    """
    if sys.platform != "win32":
        return False
    try:
        # NVIDIA 드라이버가 설치되어 있으면 시스템에 있는 CUDA 드라이버 API 라이브러리
        driver = ctypes.WinDLL("nvcuda.dll")
    except OSError:
        return False
    try:
        if driver.cuInit(0) != 0:
            return False
        # 드라이버가 알려 준 CUDA 장치 수
        count = ctypes.c_int(0)
        return driver.cuDeviceGetCount(ctypes.byref(count)) == 0 and count.value > 0
    except (AttributeError, OSError):
        logger.warning("CUDA 장치 수를 확인하지 못했습니다.", exc_info=True)
        return False


def parse_engine_kind(value: str | None) -> OcrEngineKind:
    """환경 변수 값을 엔진 종류로 바꾼다. 비었거나 모르는 값이면 `AUTO` (경고 로그).

    매개변수:
        value: 환경 변수 값 (대소문자·앞뒤 공백 무시).
    반환값:
        엔진 종류.
    """
    if value is None or not value.strip():
        return OcrEngineKind.AUTO
    try:
        return OcrEngineKind(value.strip().lower())
    except ValueError:
        logger.warning(
            "%s 값 '%s'을(를) 알 수 없어 auto로 처리합니다 (가능한 값: auto, paddle, mock).",
            ENGINE_ENV, value,
        )
        return OcrEngineKind.AUTO


def resolve_device(value: str | None, gpu_usable: Callable[[], bool]) -> tuple[str, str | None]:
    """장치 설정값을 실제 장치와 대체 장치로 바꾼다.

    매개변수:
        value: 환경 변수 값. 비었거나 `auto`면 자동으로 고른다.
        gpu_usable: GPU 빌드와 CUDA GPU가 모두 있는지 확인하는 함수.
    반환값:
        (쓸 장치, 모델 로딩이 실패하면 다시 시도할 장치 또는 None).
    """
    # 앞뒤 공백을 뺀 소문자 설정값
    requested = (value or "").strip().lower()
    if requested and requested != DEVICE_AUTO:
        return requested, None
    if gpu_usable():
        return GPU_DEVICE, CPU_DEVICE
    return CPU_DEVICE, None


def _gpu_usable() -> bool:
    """PaddlePaddle GPU 빌드가 설치되어 있고 CUDA GPU가 있는지 여부."""
    return paddle_gpu_build_installed() and cuda_gpu_available()


def select_text_recognizer(
    environ: Mapping[str, str] | None = None,
    is_paddle_available: Callable[[], bool] = paddle_available,
    is_gpu_usable: Callable[[], bool] = _gpu_usable,
) -> OcrSelection:
    """설정에 맞는 OCR 엔진을 고른다. 모델을 불러오지 않으므로 UI 스레드에서 호출해도 된다.

    매개변수:
        environ: 설정을 읽을 환경 변수 (None이면 `os.environ`).
        is_paddle_available: PaddleOCR 설치 여부를 확인하는 함수 (테스트에서 바꾼다).
        is_gpu_usable: GPU를 쓸 수 있는지 확인하는 함수 (테스트에서 바꾼다).
    반환값:
        고른 엔진과 전체 화면 속도 정보.
    """
    # 설정을 읽을 환경 변수 모음
    environ = os.environ if environ is None else environ
    # 사용자가 고른 엔진 종류
    kind = parse_engine_kind(environ.get(ENGINE_ENV))
    if kind is OcrEngineKind.MOCK:
        logger.info("OCR 엔진: Mock (%s=mock)", ENGINE_ENV)
        return OcrSelection(MockLayoutTextRecognizer())

    if not is_paddle_available():
        if kind is OcrEngineKind.AUTO:
            logger.warning(
                "PaddleOCR가 설치되어 있지 않아 Mock OCR을 사용합니다. 설치: pip install -e .[ocr]"
            )
            return OcrSelection(MockLayoutTextRecognizer())
        logger.error("%s=paddle이지만 PaddleOCR가 설치되어 있지 않습니다.", ENGINE_ENV)
        return OcrSelection(UnavailableTextRecognizer(PADDLE_NOT_INSTALLED_MESSAGE))

    # numpy 등 무거운 모듈을 쓰므로 PaddleOCR를 고른 경우에만 가져온다.
    from translator.adapters.ocr.paddle import PaddleOcrSettings, PaddleTextRecognizer

    # 쓸 장치와, 그 장치로 모델을 불러오지 못했을 때 다시 시도할 장치
    device, fallback = resolve_device(environ.get(DEVICE_ENV), is_gpu_usable)
    settings = PaddleOcrSettings(device=device, fallback_device=fallback)
    logger.info(
        "OCR 엔진: PaddleOCR (장치 %s%s, 모델은 처음 인식할 때 불러옵니다)",
        settings.device, f", 실패 시 {fallback}" if fallback else "",
    )
    return OcrSelection(
        PaddleTextRecognizer(settings),
        slow_full_screen=not settings.device.startswith("gpu"),
    )


def create_text_recognizer(
    environ: Mapping[str, str] | None = None,
    is_paddle_available: Callable[[], bool] = paddle_available,
    is_gpu_usable: Callable[[], bool] = _gpu_usable,
) -> TextRecognizer:
    """설정에 맞는 OCR 엔진만 돌려준다 (`select_text_recognizer`의 간단한 형태).

    매개변수:
        environ: 설정을 읽을 환경 변수 (None이면 `os.environ`).
        is_paddle_available: PaddleOCR 설치 여부를 확인하는 함수.
        is_gpu_usable: GPU를 쓸 수 있는지 확인하는 함수.
    반환값:
        `TextRecognizer` 구현.
    """
    return select_text_recognizer(environ, is_paddle_available, is_gpu_usable).recognizer
