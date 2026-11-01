"""캡처 한 번의 결과(프레임)와 그 식별 정보.

자동 감지가 찍은 캡처를 번역 단계까지 그대로 넘겨, 같은 화면을 다시 캡처하지 않게 한다.
프레임마다 고유 번호(frame_id)와 캡처 시각(captured_at)을 붙여, 결과가 어느 화면에서
나왔는지와 결과 사이의 선후 관계를 확인할 수 있게 한다.
"""

import hashlib
import itertools
import threading
import time
from dataclasses import dataclass, field

from translator.core.models import CapturedImage, ScreenRegion

# 프레임 번호 발급기. 여러 워커 스레드가 함께 쓰므로 잠금으로 보호한다.
_frame_ids = itertools.count(1)
_frame_id_lock = threading.Lock()


def next_frame_id() -> int:
    """새 프레임 번호를 발급한다 (프로그램 실행 중 유일, 증가)."""
    with _frame_id_lock:
        return next(_frame_ids)


def now() -> float:
    """프레임 시각에 쓰는 시계 (초, 단조 증가). 결과 사이 선후 비교에만 쓴다."""
    return time.monotonic()


def content_key(image: CapturedImage) -> tuple[ScreenRegion, int, int, bytes]:
    """캡처 내용의 식별값: 영역, 크기, 픽셀 해시.

    픽셀이 하나라도 다르면 값이 달라지므로, 바뀐 화면이 이전 OCR 결과를 재사용하는 일이 없다.
    해시만 보관하고 픽셀은 보관하지 않는다.
    """
    digest = hashlib.blake2b(image.bgra, digest_size=16).digest()
    return image.region, image.width, image.height, digest


@dataclass(frozen=True)
class Frame:
    """캡처 한 번의 결과. 이미지는 처리할 때까지만 메모리에 두고 저장하지 않는다."""

    frame_id: int
    image: CapturedImage = field(repr=False)
    captured_at: float
    # 자동 감지가 계산한 화면 지문 (없으면 None)
    fingerprint: object | None = None

    @property
    def region(self) -> ScreenRegion:
        """캡처한 물리 픽셀 영역."""
        return self.image.region
