"""프레임 식별 정보 테스트: 프레임 번호, 캡처 내용 식별값 (Qt 없이 실행)."""

import threading

from translator.application.frames import Frame, content_key, next_frame_id
from translator.core.models import CapturedImage, ScreenRegion

REGION = ScreenRegion(0, 0, 2, 1)


def image(data: bytes, region: ScreenRegion = REGION) -> CapturedImage:
    """2×1 이미지를 만든다."""
    return CapturedImage(region, 2, 1, data)


def test_frame_ids_are_unique_across_threads() -> None:
    """여러 워커 스레드가 동시에 발급해도 프레임 번호가 겹치지 않는다."""
    ids: list[int] = []
    lock = threading.Lock()

    def worker() -> None:
        """번호 200개를 발급한다."""
        local = [next_frame_id() for _ in range(200)]
        with lock:
            ids.extend(local)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(set(ids)) == 800


def test_content_key_changes_with_any_pixel_or_region() -> None:
    """픽셀 하나나 영역이 달라도 식별값이 달라진다 (바뀐 화면은 OCR 캐시를 쓰지 않음)."""
    base = bytes(8)

    assert content_key(image(base)) == content_key(image(bytes(8)))
    assert content_key(image(base)) != content_key(image(bytes(7) + b"\x01"))
    assert content_key(image(base)) != content_key(image(base, ScreenRegion(5, 0, 2, 1)))


def test_frame_exposes_capture_region() -> None:
    """프레임의 영역은 캡처 이미지의 영역이다."""
    frame = Frame(1, image(bytes(8)), captured_at=0.5)

    assert frame.region == REGION
    assert "image" not in repr(frame)  # 로그에 픽셀이 찍히지 않게 한다
