"""재실행 시 복원할 UI 설정(창 위치, 상태 창 크기, 불투명도, 바운딩 박스 목록)의 저장소.

QSettings를 사용하므로 Windows에서는 사용자 레지스트리
(HKCU\\Software\\MultimodalTranslator)에 로컬로만 저장된다. 번역 내용이나 화면
정보는 저장하지 않는다.

저장된 값은 손상되었거나 다른 버전에서 왔을 수 있으므로, 읽을 때 형식이 맞지
않으면 None을 반환하고 호출하는 쪽이 기본값을 쓰게 한다.

바운딩 박스는 Phase 3부터 콘텐츠별로 SQLite에 저장한다 (`ContentProfileStore`). 이 저장소의
`load_regions`·`save_regions`·`clear_regions`는 그 이전 버전이 레지스트리(`profiles/<프로필>/regions`)에
저장한 박스를 SQLite로 옮기는 데(`region_migration.py`) 쓰는 이전 형식 지원이다.
"""

import logging
from collections.abc import Sequence

from PySide6.QtCore import QPoint, QRect, QSettings

from translator.application.profile_store import ContentProfileStore
from translator.application.regions import RegionConfig
from translator.core.models import TranslationDirection

logger = logging.getLogger(__name__)

ORGANIZATION_NAME = "MultimodalTranslator"
APPLICATION_NAME = "MultimodalTranslator"

# 저장 형식 버전. 키 구조를 바꾸면 올리고, 다른 버전의 값은 무시한다.
SETTINGS_VERSION = 1

_KEY_VERSION = "ui/version"
_KEY_OVERLAY_X = "overlay/x"
_KEY_OVERLAY_Y = "overlay/y"
_KEY_RESULT_X = "result_window/x"
_KEY_RESULT_Y = "result_window/y"
_KEY_RESULT_WIDTH = "result_window/width"
_KEY_RESULT_HEIGHT = "result_window/height"
_KEY_RESULT_OPACITY = "result_window/background_opacity"
_KEY_FULLSCREEN_DIRECTION = "translation/fullscreen_direction"

# 콘텐츠별 설정을 도입하기 전까지 쓰는 기본 프로필 이름
DEFAULT_PROFILE = "default"


class UiSettingsStore:
    """UI 설정을 읽고 쓰는 얇은 래퍼. UI 스레드에서만 사용한다."""

    def __init__(self, settings: QSettings | None = None) -> None:
        """테스트에서는 임시 INI 파일을 쓰는 QSettings를 주입한다."""
        self._settings = settings or QSettings(ORGANIZATION_NAME, APPLICATION_NAME)

    def load_overlay_position(self) -> QPoint | None:
        """저장된 오버레이 위치. 없거나 손상되었으면 None."""
        if not self._is_current_version():
            return None
        x = self._read_int(_KEY_OVERLAY_X)
        y = self._read_int(_KEY_OVERLAY_Y)
        if x is None or y is None:
            return None
        return QPoint(x, y)

    def save_overlay_position(self, position: QPoint) -> None:
        """오버레이 위치를 저장한다."""
        self._mark_version()
        self._settings.setValue(_KEY_OVERLAY_X, position.x())
        self._settings.setValue(_KEY_OVERLAY_Y, position.y())

    def load_result_window_geometry(self) -> QRect | None:
        """저장된 결과 창 위치와 크기. 없거나 손상되었으면 None."""
        if not self._is_current_version():
            return None
        values = [
            self._read_int(key)
            for key in (_KEY_RESULT_X, _KEY_RESULT_Y, _KEY_RESULT_WIDTH, _KEY_RESULT_HEIGHT)
        ]
        if any(value is None for value in values):
            return None
        x, y, width, height = values
        if width <= 0 or height <= 0:
            logger.warning("저장된 결과 창 크기가 올바르지 않아 무시합니다.")
            return None
        return QRect(x, y, width, height)

    def save_result_window_geometry(self, geometry: QRect) -> None:
        """결과 창 위치와 크기를 저장한다."""
        self._mark_version()
        self._settings.setValue(_KEY_RESULT_X, geometry.x())
        self._settings.setValue(_KEY_RESULT_Y, geometry.y())
        self._settings.setValue(_KEY_RESULT_WIDTH, geometry.width())
        self._settings.setValue(_KEY_RESULT_HEIGHT, geometry.height())

    def load_result_opacity(self) -> float | None:
        """저장된 결과 창 배경 불투명도(0~1). 범위 조정은 결과 창이 한다."""
        if not self._is_current_version():
            return None
        raw = self._settings.value(_KEY_RESULT_OPACITY)
        if raw is None:
            return None
        try:
            value = float(raw)
        except (TypeError, ValueError):
            logger.warning("저장된 불투명도 값을 읽을 수 없어 무시합니다.")
            return None
        # NaN은 자기 자신과 같지 않으므로 이 비교로 걸러낸다.
        if value != value or not 0.0 <= value <= 1.0:
            logger.warning("저장된 불투명도 값이 범위를 벗어나 무시합니다.")
            return None
        return value

    def save_result_opacity(self, opacity: float) -> None:
        """결과 창 배경 불투명도를 저장한다."""
        self._mark_version()
        self._settings.setValue(_KEY_RESULT_OPACITY, float(opacity))

    def load_fullscreen_direction(self) -> TranslationDirection | None:
        """저장된 전체 화면 번역 출력 방향. 없거나 알 수 없는 값이면 None."""
        if not self._is_current_version():
            return None
        return _parse_direction(self._settings.value(_KEY_FULLSCREEN_DIRECTION))

    def save_fullscreen_direction(self, direction: TranslationDirection) -> None:
        """전체 화면 번역 출력 방향을 저장한다."""
        self._mark_version()
        self._settings.setValue(_KEY_FULLSCREEN_DIRECTION, direction.value)

    def load_regions(self, profile: str = DEFAULT_PROFILE) -> list[RegionConfig]:
        """프로필에 저장된 바운딩 박스 설정 목록. 손상된 항목은 건너뛴다."""
        if not self._is_current_version():
            return []
        configs: list[RegionConfig] = []
        seen_ids: set[str] = set()
        count = self._settings.beginReadArray(_regions_key(profile))
        try:
            for index in range(count):
                self._settings.setArrayIndex(index)
                config = self._read_region()
                if config is None or config.region_id in seen_ids:
                    logger.warning("저장된 박스 설정 %d번이 손상되었거나 중복되어 건너뜁니다.", index)
                    continue
                seen_ids.add(config.region_id)
                configs.append(config)
        finally:
            self._settings.endArray()
        return configs

    def save_regions(self, configs: list[RegionConfig], profile: str = DEFAULT_PROFILE) -> None:
        """프로필의 바운딩 박스 설정 목록을 통째로 바꿔 저장한다."""
        self._mark_version()
        key = _regions_key(profile)
        # 줄어든 목록의 이전 항목이 남지 않도록 먼저 지운다.
        self._settings.remove(key)
        self._settings.beginWriteArray(key, len(configs))
        try:
            for index, config in enumerate(configs):
                self._settings.setArrayIndex(index)
                self._settings.setValue("id", config.region_id)
                self._settings.setValue("left", config.left)
                self._settings.setValue("top", config.top)
                self._settings.setValue("width", config.width)
                self._settings.setValue("height", config.height)
                self._settings.setValue("direction", config.direction.value)
                # bool은 저장 형식마다 표현이 달라 정수로 저장한다.
                self._settings.setValue("auto", int(config.auto_enabled))
        finally:
            self._settings.endArray()

    def profiles(self) -> list[str]:
        """박스가 저장된 적 있는 프로필 이름 목록 (이전 형식 값을 옮길 때 쓴다)."""
        self._settings.beginGroup("profiles")
        try:
            return list(self._settings.childGroups())
        finally:
            self._settings.endGroup()

    def clear_regions(self, profile: str = DEFAULT_PROFILE) -> None:
        """프로필에 저장된 바운딩 박스 설정을 지운다 (SQLite로 옮긴 뒤 이전 형식 값을 정리)."""
        self._settings.remove(_regions_key(profile))

    def sync(self) -> None:
        """메모리에 있는 변경 사항을 저장소에 즉시 기록한다."""
        self._settings.sync()
        if self._settings.status() != QSettings.Status.NoError:
            logger.warning("UI 설정을 저장하지 못했습니다 (상태: %s).", self._settings.status())

    def _is_current_version(self) -> bool:
        """저장된 설정이 현재 형식인지 확인한다. 아무것도 저장되지 않았으면 False."""
        return self._read_int(_KEY_VERSION) == SETTINGS_VERSION

    def _mark_version(self) -> None:
        """현재 저장 형식 버전을 기록한다."""
        self._settings.setValue(_KEY_VERSION, SETTINGS_VERSION)

    def _read_region(self) -> RegionConfig | None:
        """현재 배열 위치의 박스 설정을 읽는다. 형식이 맞지 않으면 None."""
        region_id = self._settings.value("id")
        numbers = [self._read_int(key) for key in ("left", "top", "width", "height", "auto")]
        direction = _parse_direction(self._settings.value("direction"))
        if not isinstance(region_id, str) or not region_id or direction is None:
            return None
        if any(number is None for number in numbers):
            return None
        left, top, width, height, auto = numbers
        try:
            return RegionConfig(region_id, left, top, width, height, direction, bool(auto))
        except ValueError:
            return None

    def _read_int(self, key: str) -> int | None:
        """정수 값을 읽는다. 없거나 정수로 바꿀 수 없으면 None."""
        raw = self._settings.value(key)
        if raw is None:
            return None
        try:
            return int(raw)
        except (TypeError, ValueError):
            logger.warning("저장된 UI 설정 '%s' 값을 읽을 수 없어 무시합니다.", key)
            return None


class SettingsProfileStore(ContentProfileStore):
    """SQLite를 열지 못했을 때 쓰는 대체 저장소: 콘텐츠별 박스를 UI 설정 저장소(레지스트리)에 저장한다.

    콘텐츠 ID를 프로필 이름으로 쓴다 (`profiles/<콘텐츠 ID>/regions`). 이 저장소에 쌓인 박스는 다음에
    SQLite를 열 수 있을 때 `migrate_legacy_regions`가 SQLite로 옮긴다.
    """

    def __init__(self, settings: UiSettingsStore) -> None:
        """박스를 읽고 쓸 UI 설정 저장소를 받는다."""
        # 박스를 실제로 저장하는 UI 설정 저장소
        self._settings = settings

    def load_regions(self, content_id: str) -> list[RegionConfig]:
        """콘텐츠(프로필)에 저장된 박스 설정 목록."""
        return self._settings.load_regions(content_id)

    def save_regions(self, content_id: str, configs: Sequence[RegionConfig]) -> None:
        """콘텐츠(프로필)의 박스 설정 목록을 통째로 바꿔 저장하고 바로 기록한다."""
        self._settings.save_regions(list(configs), content_id)
        self._settings.sync()

    def adopt_regions(self, source_content_id: str, target_content_id: str) -> list[RegionConfig]:
        """`source`의 박스를 `target`으로 옮긴다. `target`에 박스가 이미 있으면 옮기지 않는다."""
        if source_content_id == target_content_id or self._settings.load_regions(target_content_id):
            return []
        adopted = self._settings.load_regions(source_content_id)
        if not adopted:
            return []
        self._settings.save_regions(adopted, target_content_id)
        self._settings.clear_regions(source_content_id)
        self._settings.sync()
        return adopted


def _regions_key(profile: str) -> str:
    """프로필의 박스 목록 키."""
    return f"profiles/{profile}/regions"


def _parse_direction(raw: object) -> TranslationDirection | None:
    """저장된 방향 문자열을 읽는다. 알 수 없는 값이면 None."""
    try:
        return TranslationDirection(raw)
    except ValueError:
        return None
