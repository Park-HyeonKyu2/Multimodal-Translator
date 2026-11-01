"""OCR 텍스트의 안정화 판단 (타자 효과 대응).

글자가 하나씩 나타나는 동안에는 번역하지 않고, 같은 텍스트가 연속으로 인식되면
"안정됐다"고 본다. 이 모듈은 **안정 여부만** 판단한다. 안정된 텍스트를 이미 번역했는지
(중복 번역 판단)는 `text_display`가 따로 판단한다.

Mock OCR 단계의 한계: 기본 Mock OCR은 글자를 읽지 못해 "단어1 단어2"처럼 단어 개수만
나타낸다. 그래서 단어가 늘어나는 타자 효과는 흉내 낼 수 있지만, 한 단어 안에서 글자가
늘어나는 경우나 OCR 오인식으로 글자가 흔들리는 경우는 검증할 수 없다.
"""

from translator.tuning import (
    TEXT_MAX_STABILIZE_SECONDS,
    TEXT_SETTLE_OBSERVATIONS,
    TEXT_TRAILING_MARKERS,
)


def normalize_text(text: str, trailing_markers: str = TEXT_TRAILING_MARKERS) -> str:
    """비교용 정규화: 공백을 하나로 줄이고 끝의 진행 표시·커서 문자를 지운다."""
    collapsed = " ".join(text.split())
    return collapsed.rstrip(trailing_markers + " ")


class TextStabilizer:
    """텍스트 항목 하나의 연속 인식 결과로 안정 여부를 판단한다."""

    def __init__(
        self,
        settle_observations: int = TEXT_SETTLE_OBSERVATIONS,
        max_wait_seconds: float = TEXT_MAX_STABILIZE_SECONDS,
    ) -> None:
        """같은 텍스트가 몇 번 연속되면 안정으로 볼지, 최대 몇 초까지 기다릴지 정한다."""
        self._settle = max(1, settle_observations)
        self._max_wait = max_wait_seconds
        # 직전에 인식된 텍스트 (정규화)
        self._last: str | None = None
        # 직전 텍스트가 연속으로 인식된 횟수
        self._count = 0
        # 지금 이어지는 텍스트(늘어나는 중 포함)가 처음 보인 시각
        self._since: float | None = None

    def observe(self, text: str, now: float, force: bool = False) -> str | None:
        """새 인식 결과를 기록하고, 안정됐으면 그 텍스트(정규화)를, 아니면 None을 반환한다.

        - 같은 텍스트가 `settle_observations`번 연속이면 안정.
        - 이전 텍스트 뒤에 글자가 늘어난 경우(타자 효과)는 기다리되, 처음 보인 뒤
          `max_wait_seconds`가 지나면 현재 텍스트로 안정 처리한다.
        - `force`면 바로 안정으로 본다 (사용자가 직접 요청한 번역).
        """
        normalized = normalize_text(text)
        if normalized == self._last:
            self._count += 1
        else:
            growing = bool(self._last) and normalized.startswith(self._last)
            if not growing or self._since is None:
                # 완전히 다른 텍스트로 바뀌었다: 기다림을 처음부터 다시 잰다.
                self._since = now
            self._last = normalized
            self._count = 1
        if force or self._count >= self._settle:
            return normalized
        if self._since is not None and now - self._since >= self._max_wait:
            return normalized
        return None

    @property
    def last_text(self) -> str | None:
        """직전에 인식된 텍스트 (정규화)."""
        return self._last
