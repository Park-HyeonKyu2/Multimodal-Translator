"""OCR 엔진 선택 테스트: 환경 변수와 설치 여부에 따라 PaddleOCR·Mock OCR을 고른다 (모델은 불러오지 않음)."""

import pytest

from translator.adapters.ocr.factory import (
    DEVICE_ENV,
    ENGINE_ENV,
    OcrEngineKind,
    create_text_recognizer,
    cuda_gpu_available,
    paddle_gpu_build_installed,
    parse_engine_kind,
    resolve_device,
    select_text_recognizer,
)
from translator.adapters.ocr.mock_layout import MockLayoutTextRecognizer
from translator.adapters.ocr.unavailable import UnavailableTextRecognizer
from translator.core.cancellation import CancellationToken
from translator.core.errors import TextRecognitionError
from translator.core.models import CapturedImage, ScreenRegion


def installed() -> bool:
    """PaddleOCR가 설치된 상황."""
    return True


def not_installed() -> bool:
    """PaddleOCR가 설치되지 않은 상황."""
    return False


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, OcrEngineKind.AUTO),
        ("", OcrEngineKind.AUTO),
        (" Paddle ", OcrEngineKind.PADDLE),
        ("MOCK", OcrEngineKind.MOCK),
        ("tesseract", OcrEngineKind.AUTO),  # 모르는 값은 auto
    ],
)
def test_engine_kind_is_parsed(value: str | None, expected: OcrEngineKind) -> None:
    """환경 변수 값을 대소문자·공백과 관계없이 읽고, 모르는 값은 auto로 본다."""
    assert parse_engine_kind(value) is expected


def test_mock_is_used_when_requested() -> None:
    """mock을 고르면 PaddleOCR가 설치되어 있어도 Mock OCR을 쓴다 (기존 동작 유지)."""
    recognizer = create_text_recognizer({ENGINE_ENV: "mock"}, is_paddle_available=installed)

    assert isinstance(recognizer, MockLayoutTextRecognizer)


def test_auto_falls_back_to_mock_when_paddle_is_missing(caplog: pytest.LogCaptureFixture) -> None:
    """기본(auto)은 PaddleOCR가 없으면 경고를 남기고 Mock OCR로 실행한다."""
    recognizer = create_text_recognizer({}, is_paddle_available=not_installed)

    assert isinstance(recognizer, MockLayoutTextRecognizer)
    assert "Mock OCR을 사용합니다" in caplog.text


def test_explicit_paddle_without_install_reports_on_use() -> None:
    """paddle을 직접 골랐는데 설치되어 있지 않으면, 인식할 때마다 설치 안내 오류를 보여 준다."""
    recognizer = create_text_recognizer({ENGINE_ENV: "paddle"}, is_paddle_available=not_installed)
    image = CapturedImage(ScreenRegion(0, 0, 1, 1), 1, 1, b"\x00" * 4)

    assert isinstance(recognizer, UnavailableTextRecognizer)
    with pytest.raises(TextRecognitionError, match="pip install"):
        recognizer.recognize(image, CancellationToken())


def no_gpu() -> bool:
    """GPU를 쓸 수 없는 상황 (GPU 빌드 없음 또는 CUDA GPU 없음)."""
    return False


def has_gpu() -> bool:
    """GPU 빌드와 CUDA GPU가 모두 있는 상황."""
    return True


def test_paddle_is_selected_with_device_without_loading_model() -> None:
    """장치를 직접 정하면 그 장치만 쓰고(대체 장치 없음), 모델은 아직 불러오지 않는다."""
    pytest.importorskip("numpy")
    from translator.adapters.ocr.paddle import PaddleTextRecognizer

    recognizer = create_text_recognizer(
        {DEVICE_ENV: "gpu:0"}, is_paddle_available=installed, is_gpu_usable=no_gpu
    )

    assert isinstance(recognizer, PaddleTextRecognizer)
    assert recognizer.settings.device == "gpu:0"
    assert recognizer.settings.fallback_device is None
    assert recognizer._engine is None  # noqa: SLF001 (지연 로딩 확인)


def test_auto_device_uses_gpu_with_cpu_fallback_when_available() -> None:
    """장치 기본값(auto)은 GPU를 쓸 수 있으면 GPU를 고르고, 실패 시 CPU로 다시 시도하게 한다."""
    pytest.importorskip("numpy")

    selection = select_text_recognizer({}, is_paddle_available=installed, is_gpu_usable=has_gpu)

    assert selection.recognizer.settings.device == "gpu:0"
    assert selection.recognizer.settings.fallback_device == "cpu"
    assert selection.slow_full_screen is False


def test_auto_device_uses_cpu_without_gpu_and_marks_full_screen_slow() -> None:
    """GPU를 쓸 수 없으면 CPU를 고르고, 전체 화면이 느리다고 표시한다."""
    pytest.importorskip("numpy")

    selection = select_text_recognizer(
        {DEVICE_ENV: " AUTO "}, is_paddle_available=installed, is_gpu_usable=no_gpu
    )

    assert selection.recognizer.settings.device == "cpu"
    assert selection.recognizer.settings.fallback_device is None
    assert selection.slow_full_screen is True


def test_explicit_cpu_is_slow_even_with_gpu() -> None:
    """GPU가 있어도 CPU를 직접 고르면 CPU를 쓰고 전체 화면이 느리다고 표시한다."""
    pytest.importorskip("numpy")

    selection = select_text_recognizer(
        {DEVICE_ENV: "cpu"}, is_paddle_available=installed, is_gpu_usable=has_gpu
    )

    assert selection.recognizer.settings.device == "cpu"
    assert selection.slow_full_screen is True


def test_mock_and_unavailable_are_not_marked_slow() -> None:
    """Mock OCR과 설치되지 않은 PaddleOCR는 전체 화면을 느리다고 표시하지 않는다."""
    mock = select_text_recognizer({ENGINE_ENV: "mock"}, is_paddle_available=installed)
    missing = select_text_recognizer({ENGINE_ENV: "paddle"}, is_paddle_available=not_installed)

    assert mock.slow_full_screen is False and missing.slow_full_screen is False


@pytest.mark.parametrize(
    ("value", "gpu", "expected"),
    [
        (None, True, ("gpu:0", "cpu")),
        ("", False, ("cpu", None)),
        ("auto", True, ("gpu:0", "cpu")),
        ("GPU:1", True, ("gpu:1", None)),
        ("cpu", True, ("cpu", None)),
    ],
)
def test_resolve_device(value, gpu, expected) -> None:
    """장치 설정값을 (쓸 장치, 대체 장치)로 바꾼다."""
    assert resolve_device(value, lambda: gpu) == expected


def test_gpu_check_does_not_import_paddle() -> None:
    """GPU 확인은 paddle을 가져오지 않는다 (시작 지연 방지). 결과는 환경마다 다를 수 있다."""
    import sys

    already_loaded = "paddle" in sys.modules
    assert isinstance(cuda_gpu_available(), bool)
    assert isinstance(paddle_gpu_build_installed(), bool)
    if not already_loaded:
        assert "paddle" not in sys.modules
