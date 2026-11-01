"""콘텐츠·세션 식별 정보 테스트: 빈 ID 거부, 새 세션 생성, 값 비교."""

import pytest

from translator.core.session import DEFAULT_CONTENT_ID, SessionContext, new_session


def test_new_sessions_are_unique_and_use_default_content() -> None:
    """콘텐츠를 정하지 않으면 기본 콘텐츠이고, 세션 ID는 매번 다르다."""
    first, second = new_session(), new_session()

    assert first.content_id == second.content_id == DEFAULT_CONTENT_ID
    assert first.session_id != second.session_id
    assert first != second


def test_same_content_gets_a_new_session_each_time() -> None:
    """같은 콘텐츠라도 다시 시작하면 다른 세션이다 (늦게 온 이전 결과를 알아보기 위함)."""
    first, second = new_session("game-a"), new_session("game-a")

    assert first.content_id == second.content_id == "game-a"
    assert first != second


def test_explicit_session_id_is_kept_and_compared_by_value() -> None:
    """세션 ID를 정해 주면 그대로 쓰고, 같은 값이면 같은 세션이다."""
    assert new_session("game-a", "s1") == SessionContext("game-a", "s1")
    assert new_session("game-a", "s1") != new_session("game-b", "s1")


@pytest.mark.parametrize("content_id, session_id", [("", "s1"), ("game", "")])
def test_empty_ids_are_rejected(content_id: str, session_id: str) -> None:
    """빈 ID는 식별되지 않는 세션이 섞이는 원인이 되므로 만들 수 없다."""
    with pytest.raises(ValueError):
        SessionContext(content_id, session_id)
