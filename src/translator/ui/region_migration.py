"""이전 방식(레지스트리, QSettings)에 저장된 바운딩 박스를 콘텐츠별 SQLite 저장소로 옮긴다.

이전 버전은 콘텐츠 구분 없이 박스를 하나의 프로필(`default`)에 저장했다. 옮긴 박스는 SQLite의
`default` 콘텐츠에 들어가고, 박스가 없는 콘텐츠가 처음 식별될 때 그 콘텐츠가 가져간다
(`RegionManager.switch_content`). SQLite를 열 수 없을 때 대체 저장소(`SettingsProfileStore`)가
콘텐츠 ID를 프로필 이름으로 레지스트리에 쌓아 둔 박스도 같은 방식으로 해당 콘텐츠로 옮긴다.
"""

import logging

from translator.application.profile_store import ContentProfileStore
from translator.core.errors import StorageError
from translator.ui.ui_settings import UiSettingsStore

logger = logging.getLogger(__name__)


def migrate_legacy_regions(settings: UiSettingsStore, store: ContentProfileStore) -> int:
    """레지스트리에 남은 이전 방식의 박스를 SQLite로 옮기고 레지스트리 값을 지운다.

    프로필 이름이 곧 콘텐츠 ID다 (`default`는 식별 전 콘텐츠). 콘텐츠마다 따로 처리하며, SQLite의 그
    콘텐츠에 이미 박스가 있으면 덮어쓰지 않고 레지스트리 값도 그대로 둔다 (어느 쪽이 최신인지 알 수
    없으므로 사용자 데이터를 잃지 않는 쪽을 택한다). 옮기는 데 실패하면 그 프로필의 레지스트리 값을
    지우지 않아 다음 실행에 다시 시도한다.

    매개변수:
        settings: 이전 방식을 읽을 UI 설정 저장소.
        store: 옮겨 넣을 콘텐츠별 저장소.
    반환값:
        옮긴 박스 수. 옮길 것이 없거나 옮기지 않았으면 0.
    """
    # 옮긴 박스 수의 합
    moved = 0
    for profile in settings.profiles():
        # 이 프로필에 이전 방식으로 저장된 박스 설정들
        legacy = settings.load_regions(profile)
        if not legacy:
            continue
        try:
            if store.load_regions(profile):
                logger.warning("SQLite에 이미 박스가 있어 이전 방식의 박스를 옮기지 않았습니다.")
                continue
            store.save_regions(profile, legacy)
        except StorageError:
            logger.exception("이전 방식의 박스를 SQLite로 옮기지 못했습니다. 다음 실행에 다시 시도합니다.")
            continue
        settings.clear_regions(profile)
        moved += len(legacy)
    if moved:
        settings.sync()
        logger.info("이전 방식의 박스 %d개를 SQLite로 옮겼습니다.", moved)
    return moved
