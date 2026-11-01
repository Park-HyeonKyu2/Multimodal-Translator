"""애플리케이션 공통 예외.

`TranslatorError` 계열 예외의 메시지는 사용자에게 그대로 보여 줄 수 있도록
번역 대상 텍스트 같은 민감한 내용을 포함하지 않는다.
"""

# 예상하지 못한 예외가 났을 때 사용자에게 보여 줄 일반 메시지. 예외 내용은 로그에만 남긴다.
UNEXPECTED_ERROR_MESSAGE = "예기치 못한 오류가 발생했습니다. 로그 파일을 확인해 주세요."


class TranslatorError(Exception):
    """처리 가능한(예상된) 실패의 기반 예외."""


class CaptureError(TranslatorError):
    """화면 캡처 실패."""


class TextRecognitionError(TranslatorError):
    """텍스트 인식(OCR) 실패."""


class TranslationError(TranslatorError):
    """번역 엔진 실패."""


class StorageError(TranslatorError):
    """로컬 저장소(SQLite 등) 읽기·쓰기 실패."""


class OperationCancelledError(TranslatorError):
    """작업이 취소되었거나 더 새로운 요청으로 대체됨."""
