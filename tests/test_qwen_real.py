"""실제 llama.cpp 서버와 Qwen 모델로 번역되는지 확인하는 선택 실행 테스트.

GPU와 모델 파일(기본 폴더 또는 환경 변수)이 필요하므로 기본 실행에서는 건너뛴다. 실행 방법:

    TRANSLATOR_QWEN_TESTS=1 pytest tests/test_qwen_real.py

번역 품질은 판정하지 않는다 (docs/phase2-qwen-translation.md의 비교 실험 참고). 형식·언어·종료만 확인한다.
"""

import os
import re

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("TRANSLATOR_QWEN_TESTS") != "1",
    reason="실제 Qwen 모델이 필요한 테스트 (TRANSLATOR_QWEN_TESTS=1로 실행)",
)

# 한글 글자
HANGUL = re.compile(r"[가-힣]")


@pytest.fixture(scope="module")
def qwen():
    """기본 설정으로 고른 Qwen 번역 엔진과 서버를 띄우고, 끝나면 끈다."""
    from translator.adapters.translation.factory import select_translator

    selection = select_translator({**os.environ, "TRANSLATOR_TRANSLATION_ENGINE": "qwen"})
    if selection.server is None:
        pytest.skip("앱이 띄울 llama.cpp 서버 설정이 없습니다 (외부 서버 주소를 쓰는 경우 포함).")
    selection.server.start()
    yield selection
    selection.server.stop()
    assert not selection.server.is_running()


def test_lines_are_translated_into_korean(qwen) -> None:
    """여러 줄을 한 번에 보내면 같은 개수의 한국어 번역이 같은 순서로 온다."""
    from translator.core.cancellation import CancellationToken
    from translator.core.models import TranslationRequest

    sources = ["Save", "Shut up and fight!", "Meet me at the harbor before sunrise."]

    results = qwen.translator.translate_batch(
        [TranslationRequest(text) for text in sources], CancellationToken()
    )

    assert [r.source_text for r in results] == sources
    assert all(HANGUL.search(r.translated_text) for r in results)


def test_single_line_after_warm_up_is_fast_enough(qwen) -> None:
    """준비된 뒤 한 줄 번역이 요청 시간 제한 안에 끝난다 (지연 값은 기록용)."""
    import time

    from translator.core.cancellation import CancellationToken
    from translator.core.models import TranslationRequest
    from translator.tuning import LLM_REQUEST_TIMEOUT_SECONDS

    started = time.perf_counter()
    [result] = qwen.translator.translate_batch([TranslationRequest("Watch your back.")], CancellationToken())
    elapsed = time.perf_counter() - started

    print(f"한 줄 번역 {elapsed * 1000:.0f}ms")
    assert HANGUL.search(result.translated_text)
    assert elapsed < LLM_REQUEST_TIMEOUT_SECONDS
