"""콘텐츠 바뀜 판단 정책(`ContentSwitchPolicy`) 테스트: 연속 관찰 확인, 일시 전환 무시, 식별 실패 처리."""

import pytest

from translator.application.content_switch import ContentSwitchPolicy
from translator.core.content import ContentIdentity, IdentificationMethod
from translator.core.session import DEFAULT_CONTENT_ID


def app(name: str) -> ContentIdentity:
    """테스트용 프로세스 기반 콘텐츠 (`app:<name>.exe`)."""
    return ContentIdentity(f"app:{name}.exe", name, IdentificationMethod.PROCESS_NAME)


def feed(policy: ContentSwitchPolicy, observations: list[ContentIdentity | None]) -> list[str | None]:
    """관찰을 차례로 넣고, 매번 확정된 콘텐츠 ID(없으면 None)를 모아 돌려준다."""
    results = []
    for observation in observations:
        changed = policy.observe(observation)
        results.append(changed.content_id if changed else None)
    return results


def test_new_content_is_confirmed_after_consecutive_observations() -> None:
    """현재와 다른 콘텐츠가 정해진 횟수만큼 연속 관찰되면 그 횟수째에 바뀐 것으로 확정한다."""
    policy = ContentSwitchPolicy(confirm_samples=3)

    assert feed(policy, [app("a")] * 3) == [None, None, "app:a.exe"]
    assert policy.current_content_id == "app:a.exe"


def test_first_identified_content_replaces_default_content() -> None:
    """식별 전의 기본 콘텐츠에서 처음 식별된 콘텐츠로 바뀐다."""
    policy = ContentSwitchPolicy(confirm_samples=1)

    assert policy.current_content_id == DEFAULT_CONTENT_ID
    assert feed(policy, [app("a")]) == ["app:a.exe"]


def test_brief_switch_and_return_does_not_change_content() -> None:
    """다른 창을 잠깐 보고 돌아오면(Alt-Tab 중간 화면 등) 콘텐츠가 바뀌지 않는다."""
    policy = ContentSwitchPolicy(confirm_samples=3, initial_content_id="app:a.exe")

    assert feed(policy, [app("b"), app("b"), app("a"), app("b"), app("b"), app("a")]) == [None] * 6
    assert policy.current_content_id == "app:a.exe"


def test_alternating_candidates_never_confirm() -> None:
    """후보가 계속 바뀌면 연속 횟수가 처음부터 다시 세어져 확정되지 않는다."""
    policy = ContentSwitchPolicy(confirm_samples=2, initial_content_id="app:a.exe")

    assert feed(policy, [app("b"), app("c"), app("b"), app("c")]) == [None] * 4


def test_unidentified_observation_is_ignored_without_resetting_candidate() -> None:
    """식별하지 못한 관찰(None)은 현재 콘텐츠도 후보의 연속 횟수도 바꾸지 않는다."""
    policy = ContentSwitchPolicy(confirm_samples=3, initial_content_id="app:a.exe")

    assert feed(policy, [app("b"), None, app("b"), None, app("b")]) == [None, None, None, None, "app:b.exe"]


def test_unidentified_observations_alone_never_change_content() -> None:
    """식별 실패만 계속되어도 현재 콘텐츠는 그대로다."""
    policy = ContentSwitchPolicy(confirm_samples=1, initial_content_id="app:a.exe")

    assert feed(policy, [None] * 5) == [None] * 5
    assert policy.current_content_id == "app:a.exe"


def test_same_content_as_current_never_reports_change() -> None:
    """현재 콘텐츠가 계속 관찰되면 바뀜을 알리지 않는다."""
    policy = ContentSwitchPolicy(confirm_samples=1, initial_content_id="app:a.exe")

    assert feed(policy, [app("a")] * 4) == [None] * 4


def test_returning_to_previous_content_is_reported_as_change_again() -> None:
    """A → B → A로 확정되면 매번 바뀜으로 알린다 (돌아와도 새 세션이다)."""
    policy = ContentSwitchPolicy(confirm_samples=2)

    assert feed(policy, [app("a")] * 2 + [app("b")] * 2 + [app("a")] * 2) == [
        None, "app:a.exe", None, "app:b.exe", None, "app:a.exe",
    ]


def test_reset_sets_current_content_and_discards_candidate() -> None:
    """`reset`은 현재 콘텐츠를 다시 정하고 진행 중이던 후보를 버린다."""
    policy = ContentSwitchPolicy(confirm_samples=2, initial_content_id="app:a.exe")
    feed(policy, [app("b")])

    policy.reset("app:b.exe")

    assert policy.current_content_id == "app:b.exe"
    assert feed(policy, [app("b")] * 3) == [None] * 3
    # 후보를 버렸으므로 이전에 센 b 1회는 이어지지 않는다.
    policy.reset("app:a.exe")
    assert feed(policy, [app("b")]) == [None]


@pytest.mark.parametrize("samples", [0, -1])
def test_confirm_samples_must_be_positive(samples: int) -> None:
    """확인에 필요한 연속 관찰 횟수가 1보다 작으면 만들 수 없다."""
    with pytest.raises(ValueError):
        ContentSwitchPolicy(confirm_samples=samples)
