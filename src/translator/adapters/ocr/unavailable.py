"""사용할 수 없는 OCR 엔진 자리를 채우는 구현.

선택한 엔진을 쓸 수 없을 때(설치되지 않음 등) 프로그램을 멈추지 않고, 인식을 요청할 때마다
해결 방법이 담긴 오류를 상태 창에 보여 주기 위해 쓴다.
"""

from translator.core.cancellation import CancellationToken
from translator.core.errors import TextRecognitionError
from translator.core.interfaces import TextRecognizer
from translator.core.models import CapturedImage, RecognizedText

# PaddleOCR가 설치되어 있지 않을 때의 사용자용 오류 메시지 (설치 방법 안내)
PADDLE_NOT_INSTALLED_MESSAGE = (
    "PaddleOCR가 설치되어 있지 않습니다. `pip install -e .[ocr]`로 설치한 뒤 프로그램을 다시 시작해 주세요."
)


class UnavailableTextRecognizer(TextRecognizer):
    """인식할 때마다 정해진 오류를 던지는 OCR (엔진을 쓸 수 없음을 알리는 용도)."""

    def __init__(self, message: str) -> None:
        """사용자에게 보여 줄 오류 메시지를 받는다.

        매개변수:
            message: 해결 방법이 담긴 메시지. 번역 대상 텍스트를 담지 않는다.
        """
        # 인식 요청마다 던질 오류 메시지
        self._message = message

    def recognize(
        self, image: CapturedImage, cancellation: CancellationToken
    ) -> RecognizedText:
        """항상 `TextRecognitionError`를 던진다.

        매개변수:
            image: 사용하지 않는다.
            cancellation: 취소되었으면 취소 예외를 먼저 던진다.
        예외:
            OperationCancelledError: 이미 취소되었을 때.
            TextRecognitionError: 그 밖의 모든 경우.
        """
        cancellation.raise_if_cancelled()
        raise TextRecognitionError(self._message)
