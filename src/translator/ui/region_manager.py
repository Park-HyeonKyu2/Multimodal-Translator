"""여러 바운딩 박스의 관리.

- `RegionSession`: 박스 하나. 박스 창(`RegionBox`), 자동 감지기(`AutoDetector`), 설정
  (`RegionConfig`), 처리 상태(`RegionRuntimeState`)를 묶고, 박스 창의 입력(이동, 우클릭
  메뉴)을 설정 변경과 번역 요청 시그널로 바꾼다. 각 세션은 서로 독립적으로 동작한다.
- `RegionManager`: 세션을 만들고 지우고 복원한다. 세션 사이의 공통 동작(편집 모드,
  전체 자동 감지)과 콘텐츠별 박스 저장·교체를 다루며, 번역 요청 처리나 화면 표시는 하지 않는다.

콘텐츠별 박스: 박스 목록은 콘텐츠(`content_id`)마다 따로 저장한다. 콘텐츠가 바뀌면
(`switch_content`) 지금 박스를 저장하고 닫은 뒤 새 콘텐츠의 박스를 복원한다. 식별 전
(`default`)에 있던 박스는, 박스가 없는 콘텐츠가 처음 식별될 때 그 콘텐츠가 가져간다.

OCR 요청은 세션이 `ocr_requested` 시그널로 알리고, coordinator가 번역
컨트롤러에 전달한다. 그래서 세션은 번역 엔진이나 작업 큐를 알지 못한다.
"""

import logging
from collections.abc import Callable
from dataclasses import replace

from PySide6.QtCore import QObject, QRect, QTimer, Signal
from PySide6.QtGui import QColor, QGuiApplication

from translator.application.auto_detector import AutoDetector
from translator.application.profile_store import ContentProfileStore
from translator.application.regions import (
    RegionConfig,
    RegionIdAllocator,
    RegionRuntimeState,
    RequestStatus,
)
from translator.core.errors import StorageError
from translator.core.models import (
    DEFAULT_TRANSLATION_DIRECTION,
    ScreenRegion,
    TranslationDirection,
)
from translator.core.session import DEFAULT_CONTENT_ID
from translator.tuning import REGION_SAVE_DELAY_MS
from translator.ui.region_box import RegionBox, color_for_index
from translator.ui.screen_geometry import is_rect_on_any_screen

logger = logging.getLogger(__name__)

# 박스 설정을 저장하지 못했을 때 사용자에게 보여 줄 메시지
SAVE_FAILED_MESSAGE = (
    "박스 설정을 저장하지 못했습니다. 이번 실행의 박스 변경은 다음 실행에 복원되지 않을 수 있습니다. "
    "로그 파일을 확인해 주세요."
)
# 저장된 박스 설정을 읽지 못했을 때 사용자에게 보여 줄 메시지
LOAD_FAILED_MESSAGE = "저장된 박스 설정을 읽지 못해 박스 없이 시작합니다. 로그 파일을 확인해 주세요."

# 박스 ID와 색으로 박스 창을 만드는 함수
BoxFactory = Callable[[str, QColor], RegionBox]
# 감시 영역 제공 함수로 자동 감지기를 만드는 함수
DetectorFactory = Callable[[Callable[[], ScreenRegion | None]], AutoDetector]


class RegionSession(QObject):
    """바운딩 박스 하나의 설정, 처리 상태, 창, 자동 감지기를 묶는다."""

    # 박스 ID, 캡처 영역(ScreenRegion), 이미 찍은 프레임(Frame 또는 None), 수동 여부
    ocr_requested = Signal(str, object, object, bool)
    config_changed = Signal(str)  # 박스 ID (저장할 설정이 바뀜)
    appearance_changed = Signal(str)  # 박스 ID (출력 방향이 바뀌어 말풍선을 다시 놓아야 함)
    remove_requested = Signal(str)  # 박스 ID
    auto_failed = Signal(str, str)  # 박스 ID, 오류 메시지
    run_requested = Signal(str)  # 박스 ID (사용자가 박스 메뉴에서 번역을 실행함 → 편집 모드를 끈다)

    def __init__(
        self,
        config: RegionConfig,
        box: RegionBox,
        detector: AutoDetector,
        interactive: bool,
        parent: QObject | None = None,
    ) -> None:
        """설정에 맞춰 박스를 배치하고 박스 창·자동 감지기 시그널을 연결한다."""
        super().__init__(parent)
        self._config = config
        self._state = RegionRuntimeState()
        self._box = box
        self._detector = detector

        # 편집 모드 여부를 먼저 정한 뒤 배치해야 이동 이벤트를 "편집 중 변경"으로 오해하지 않는다.
        box.set_interactive(interactive)
        box.set_direction(config.direction)
        box.set_auto_enabled(config.auto_enabled)
        box.set_capture_rect(QRect(config.left, config.top, config.width, config.height))

        box.geometry_changed.connect(self._on_geometry_changed)
        box.direction_selected.connect(self.set_direction)
        box.auto_toggled.connect(self._on_menu_auto_toggled)
        box.translate_requested.connect(self._on_menu_translate)
        box.remove_requested.connect(lambda: self.remove_requested.emit(self.region_id))
        detector.frame_ready.connect(self._on_frame_ready)
        detector.stopped_by_error.connect(self._on_detector_stopped)

        if config.auto_enabled:
            detector.start()

    @property
    def region_id(self) -> str:
        """박스 ID."""
        return self._config.region_id

    @property
    def config(self) -> RegionConfig:
        """현재 설정 (저장 대상)."""
        return self._config

    @property
    def state(self) -> RegionRuntimeState:
        """처리 상태 (저장하지 않음)."""
        return self._state

    @property
    def box(self) -> RegionBox:
        """박스 창."""
        return self._box

    @property
    def detector(self) -> AutoDetector:
        """자동 감지기."""
        return self._detector

    @property
    def accent(self) -> QColor:
        """박스와 말풍선을 잇는 색."""
        return self._box.accent

    def show(self) -> None:
        """박스 창을 보여 준다."""
        self._box.show()

    def hide(self) -> None:
        """박스 창을 잠시 숨긴다 (영역 선택 중). 자동 감지는 계속된다."""
        self._box.hide()

    def capture_region(self) -> ScreenRegion | None:
        """지금 캡처할 물리 픽셀 영역."""
        return self._box.capture_region()

    def set_interactive(self, interactive: bool) -> None:
        """편집 모드 여부를 박스 창에 적용한다."""
        self._box.set_interactive(interactive)

    def set_direction(self, direction: TranslationDirection) -> None:
        """번역 출력 방향을 바꾼다. 번역은 다시 하지 않고 말풍선 위치만 바뀐다."""
        if direction is self._config.direction:
            return
        self._config = replace(self._config, direction=direction)
        self._box.set_direction(direction)
        self.config_changed.emit(self.region_id)
        self.appearance_changed.emit(self.region_id)

    def set_auto_enabled(self, enabled: bool) -> None:
        """이 박스의 자동 감지를 켜거나 끈다."""
        if enabled == self._config.auto_enabled:
            return
        self._config = replace(self._config, auto_enabled=enabled)
        self._box.set_auto_enabled(enabled)
        if enabled:
            self._detector.start()
        else:
            self._detector.stop()
        self.config_changed.emit(self.region_id)

    def translate_now(self) -> None:
        """이 박스를 지금 수동으로 번역하도록 요청한다."""
        region = self.capture_region()
        if region is None:
            logger.debug("%s: 캡처할 영역이 없어 번역을 건너뜁니다.", self.region_id)
            return
        self._state.needs_translation = False
        # 수동 번역은 자동 감지 표본이 없으므로 새로 캡처한다 (프레임 없음).
        self.ocr_requested.emit(self.region_id, region, None, True)

    def mark_pending(self, request_id: int) -> None:
        """번역 요청이 큐에 들어갔음을 기록한다."""
        self._state.status = RequestStatus.PENDING
        self._state.last_request_id = request_id

    def mark_ocr_fingerprint(self, fingerprint: object, region: ScreenRegion) -> None:
        """수동 OCR한 화면의 지문을 자동 감지기에 기록한다 (추가 캡처 없음).

        자동 감지를 나중에 켜거나 이미 켜져 있어도 같은 화면을 다시 OCR하지 않게 한다.
        """
        self._detector.mark_fingerprint(fingerprint, region)

    def on_ocr(self, texts: tuple[str, ...]) -> None:
        """이 박스의 인식 결과로 처리 상태(마지막 텍스트)를 갱신한다."""
        self._state.last_texts = texts
        self._state.status = RequestStatus.DONE if texts else RequestStatus.NO_TEXT
        self._state.last_error = None

    def on_failed(self, message: str) -> None:
        """이 박스의 번역 실패를 기록한다."""
        self._state.status = RequestStatus.FAILED
        self._state.last_error = message

    def ensure_on_screen(self) -> None:
        """모니터가 분리되어 박스가 화면 밖에 남았으면 주 모니터 안으로 옮긴다."""
        if is_rect_on_any_screen(self._box.frameGeometry()):
            return
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return
        area = screen.availableGeometry()
        self._box.move(area.left() + 40, area.top() + 80)
        logger.info("%s: 화면 밖에 남은 박스를 주 모니터로 옮겼습니다.", self.region_id)

    def shutdown(self) -> None:
        """자동 감지를 멈추고 박스 창을 닫는다."""
        self._detector.shutdown()
        self._box.close()
        self._box.deleteLater()

    def _on_geometry_changed(self) -> None:
        """박스가 옮겨지거나 크기가 바뀌면 설정을 갱신하고 자동 감지 기록을 버린다."""
        rect = self._box.capture_rect()
        if rect.width() <= 0 or rect.height() <= 0:
            return
        self._config = self._config.with_rect(rect.x(), rect.y(), rect.width(), rect.height())
        self._detector.reset()
        if self._box.is_interactive:
            # 편집 중에 바뀐 박스는 편집을 마칠 때 다시 번역한다.
            self._state.needs_translation = True
        self.config_changed.emit(self.region_id)

    def _on_menu_translate(self) -> None:
        """박스 메뉴의 "지금 번역": 번역 실행을 알린 뒤 번역한다."""
        self.run_requested.emit(self.region_id)
        self.translate_now()

    def _on_menu_auto_toggled(self, enabled: bool) -> None:
        """박스 메뉴의 "자동 감지": 켜는 것은 번역 실행이므로 알린 뒤 켠다."""
        if enabled:
            self.run_requested.emit(self.region_id)
        self.set_auto_enabled(enabled)

    def _on_frame_ready(self, frame: object) -> None:
        """자동 감지기가 화면 변화를 알리면 그 캡처(프레임)로 OCR을 요청한다."""
        self.ocr_requested.emit(self.region_id, frame.region, frame, False)

    def _on_detector_stopped(self, message: str) -> None:
        """연속 실패로 자동 감지가 멈추면 설정을 끄고 알린다."""
        self.set_auto_enabled(False)
        self.auto_failed.emit(self.region_id, message)


class RegionManager(QObject):
    """여러 `RegionSession`을 만들고 지우고 복원한다. UI 스레드에서만 사용한다."""

    regions_changed = Signal()  # 박스가 추가·삭제되거나 설정이 바뀜 (저장, 버튼 상태 갱신)
    ocr_requested = Signal(str, object, object, bool)  # 박스 ID, ScreenRegion, Frame 또는 None, 수동 여부
    appearance_changed = Signal(str)  # 박스 ID
    remove_requested = Signal(str)  # 박스 ID
    auto_failed = Signal(str, str)  # 박스 ID, 오류 메시지
    run_requested = Signal(str)  # 박스 ID
    storage_failed = Signal(str)  # 박스 설정 저장·읽기 실패 (사용자에게 보여 줄 메시지, 연속 실패는 한 번만 알림)

    def __init__(
        self,
        box_factory: BoxFactory,
        detector_factory: DetectorFactory,
        profile_store: ContentProfileStore | None = None,
        content_id: str = DEFAULT_CONTENT_ID,
        save_delay_ms: int = REGION_SAVE_DELAY_MS,
        parent: QObject | None = None,
    ) -> None:
        """박스 창과 자동 감지기를 만드는 함수, 콘텐츠별 저장소를 주입받는다.

        - `profile_store`가 없으면 박스를 저장하거나 복원하지 않는다 (테스트 기본값).
        - `content_id`는 처음 상태의 콘텐츠다 (식별 전에는 기본 콘텐츠).
        - 설정이 바뀐 뒤 `save_delay_ms`가 지나면 마지막 상태를 한 번 저장한다.
        """
        super().__init__(parent)
        self._box_factory = box_factory
        self._detector_factory = detector_factory
        # 콘텐츠별 박스 저장소 (없으면 저장하지 않는다)
        self._profile_store = profile_store
        # 지금 박스 목록이 속한 콘텐츠 ID
        self._content_id = content_id
        # 편집 모드 여부 (복원하는 박스에도 같은 모드를 적용한다)
        self._interactive = False
        # 직전 저장·읽기가 실패했는지 여부 (같은 실패를 계속 알리지 않기 위함)
        self._storage_failed = False
        # 복원하지 못해(지금 모니터 구성에서 화면 밖) 창으로 만들지 않은 박스 설정. 저장할 때 함께 써서
        # 모니터를 다시 연결했을 때 쓸 수 있게 보존한다. 박스 ID → 설정
        self._unrestored: dict[str, RegionConfig] = {}
        # 설정 변경을 모아 한 번에 저장하는 타이머
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(save_delay_ms)
        self._save_timer.timeout.connect(self.save_profile)
        # 박스 ID → 세션. 추가 순서를 유지한다.
        self._sessions: dict[str, RegionSession] = {}
        self._ids = RegionIdAllocator()
        # 지금까지 만든 박스 수 (색을 차례로 고르는 데 쓴다)
        self._created = 0

    @property
    def content_id(self) -> str:
        """지금 박스 목록이 속한 콘텐츠 ID."""
        return self._content_id

    def sessions(self) -> list[RegionSession]:
        """모든 박스 세션 (추가 순서)."""
        return list(self._sessions.values())

    def get(self, region_id: str) -> RegionSession | None:
        """박스 ID의 세션 (삭제되었으면 None)."""
        return self._sessions.get(region_id)

    def __len__(self) -> int:
        """박스 수."""
        return len(self._sessions)

    def configs(self) -> list[RegionConfig]:
        """저장할 박스 설정 목록."""
        return [session.config for session in self._sessions.values()]

    def all_auto_enabled(self) -> bool:
        """박스가 있고 모든 박스의 자동 감지가 켜져 있는지 여부."""
        return bool(self._sessions) and all(
            session.config.auto_enabled for session in self._sessions.values()
        )

    def add(
        self,
        rect: QRect,
        interactive: bool,
        direction: TranslationDirection = DEFAULT_TRANSLATION_DIRECTION,
        auto_enabled: bool = False,
    ) -> RegionSession:
        """새 박스를 만든다. `rect`는 캡처할 내부 영역(논리 좌표)이다."""
        config = RegionConfig(
            self._ids.allocate(), rect.x(), rect.y(), rect.width(), rect.height(),
            direction, auto_enabled,
        )
        session = self._create(config, interactive)
        logger.info("%s 추가: %s", config.region_id, rect)
        self.regions_changed.emit()
        return session

    def restore(self, configs: list[RegionConfig], interactive: bool) -> list[RegionSession]:
        """저장된 박스를 복원한다. 지금 모니터 구성에서 화면 밖인 박스는 건너뛴다."""
        restored = self._restore_quietly(configs, interactive)
        if restored:
            self.regions_changed.emit()
        return restored

    def load_profile(self) -> list[RegionSession]:
        """지금 콘텐츠의 저장된 박스를 복원한다 (시작할 때). 저장소가 없으면 아무것도 하지 않는다."""
        if self._profile_store is None:
            return []
        return self.restore(self._load_configs(self._content_id), self._interactive)

    def save_profile(self) -> None:
        """지금 콘텐츠의 박스 설정을 저장하고, 모아 둔 저장 예약을 취소한다.

        저장에 실패해도 예외를 던지지 않는다 (화면은 계속 동작해야 한다). 실패하면 한 번만
        `storage_failed`로 알린다.
        """
        self._save_timer.stop()
        if self._profile_store is None:
            return
        try:
            # 화면 밖이라 복원하지 않은 박스도 함께 저장한다 (그렇지 않으면 방문만으로 저장소에서 사라진다).
            self._profile_store.save_regions(
                self._content_id, [*self.configs(), *self._unrestored.values()]
            )
        except StorageError:
            logger.exception("박스 설정 저장 실패")
            self._report_storage_failure(SAVE_FAILED_MESSAGE)
        else:
            self._storage_failed = False

    def schedule_save(self) -> None:
        """설정이 바뀌었음을 알린다. 짧은 시간 모은 뒤 마지막 상태를 한 번 저장한다."""
        if self._profile_store is not None:
            self._save_timer.start()

    def switch_content(self, content_id: str) -> None:
        """박스 목록을 다른 콘텐츠의 것으로 바꾼다.

        지금 박스를 저장하고 닫은 뒤 새 콘텐츠의 박스를 복원한다. 식별 전(`default`)에서 처음
        식별된 콘텐츠로 바뀌는 경우, 새 콘텐츠에 저장된 박스가 없으면 `default`의 박스를 가져온다.
        같은 콘텐츠로 바꾸면 아무것도 하지 않는다. 저장소가 없으면 박스를 그대로 둔다.

        매개변수:
            content_id: 새 콘텐츠 ID.
        """
        if content_id == self._content_id:
            return
        if self._profile_store is None:
            # 저장소가 없으면 콘텐츠별로 나눌 수 없으므로 박스는 콘텐츠와 무관하게 그대로 둔다.
            self._content_id = content_id
            return
        previous = self._content_id
        self.save_profile()
        for session in self._sessions.values():
            session.shutdown()
            session.deleteLater()
        self._sessions.clear()
        self._unrestored.clear()
        self._ids = RegionIdAllocator()
        self._created = 0
        self._content_id = content_id
        logger.info("박스 목록을 다른 콘텐츠의 것으로 바꿨습니다.")
        configs = self._load_configs(content_id)
        if not configs and previous == DEFAULT_CONTENT_ID:
            configs = self._adopt_configs(previous, content_id)
        self._restore_quietly(configs, self._interactive)
        # 박스가 하나도 복원되지 않았어도 버튼 상태를 맞춰야 하므로 항상 알린다.
        self.regions_changed.emit()

    def remove(self, region_id: str) -> None:
        """박스 하나를 지운다. 다른 박스에는 영향을 주지 않는다."""
        session = self._sessions.pop(region_id, None)
        if session is None:
            return
        session.shutdown()
        session.deleteLater()
        logger.info("%s 삭제", region_id)
        self.regions_changed.emit()

    def set_interactive(self, interactive: bool) -> None:
        """모든 박스의 편집 모드를 바꾼다."""
        self._interactive = interactive
        for session in self._sessions.values():
            session.set_interactive(interactive)

    def set_visible(self, visible: bool) -> None:
        """모든 박스 창을 잠시 숨기거나 다시 보여 준다 (영역 선택 중)."""
        for session in self._sessions.values():
            if visible:
                session.show()
            else:
                session.hide()

    def set_auto_all(self, enabled: bool) -> None:
        """모든 박스의 자동 감지를 켜거나 끈다."""
        for session in self._sessions.values():
            session.set_auto_enabled(enabled)

    def ensure_on_screen(self) -> None:
        """화면 밖에 남은 박스를 화면 안으로 옮긴다."""
        for session in self._sessions.values():
            session.ensure_on_screen()

    def shutdown(self) -> None:
        """모든 박스의 자동 감지를 멈추고 창을 닫는다 (설정은 지우지 않는다)."""
        self._save_timer.stop()
        for session in self._sessions.values():
            session.shutdown()
        self._sessions.clear()

    def _restore_quietly(self, configs: list[RegionConfig], interactive: bool) -> list[RegionSession]:
        """박스를 복원하되 `regions_changed`는 보내지 않는다. 화면 밖 박스는 창을 만들지 않고 설정만 보존한다."""
        restored = []
        for config in configs:
            if config.region_id in self._sessions:
                continue
            # 화면 밖 박스도 ID는 예약해 새 박스가 같은 ID를 받지 않게 한다.
            self._ids.reserve(config.region_id)
            rect = QRect(config.left, config.top, config.width, config.height)
            if not is_rect_on_any_screen(rect):
                logger.info("%s: 저장된 위치가 화면 밖이라 복원하지 않고 설정만 보존합니다.", config.region_id)
                self._unrestored[config.region_id] = config
                continue
            restored.append(self._create(config, interactive))
        return restored

    def _load_configs(self, content_id: str) -> list[RegionConfig]:
        """콘텐츠의 저장된 박스 설정을 읽는다. 읽지 못하면 알리고 빈 목록."""
        if self._profile_store is None:
            return []
        try:
            return self._profile_store.load_regions(content_id)
        except StorageError:
            logger.exception("박스 설정 읽기 실패")
            self._report_storage_failure(LOAD_FAILED_MESSAGE)
            return []

    def _adopt_configs(self, source_content_id: str, target_content_id: str) -> list[RegionConfig]:
        """`source`의 박스를 `target`이 가져가게 한다. 실패하면 알리고 빈 목록."""
        if self._profile_store is None:
            return []
        try:
            return self._profile_store.adopt_regions(source_content_id, target_content_id)
        except StorageError:
            logger.exception("박스 설정 옮기기 실패")
            self._report_storage_failure(LOAD_FAILED_MESSAGE)
            return []

    def _report_storage_failure(self, message: str) -> None:
        """저장소 실패를 알린다. 이미 실패 상태면 다시 알리지 않는다."""
        if not self._storage_failed:
            self._storage_failed = True
            self.storage_failed.emit(message)

    def _create(self, config: RegionConfig, interactive: bool) -> RegionSession:
        """설정으로 세션을 만들고 시그널을 이어 준 뒤 박스를 보여 준다."""
        box = self._box_factory(config.region_id, color_for_index(self._created))
        self._created += 1
        detector = self._detector_factory(box.capture_region)
        session = RegionSession(config, box, detector, interactive, parent=self)
        session.ocr_requested.connect(self.ocr_requested)
        session.appearance_changed.connect(self.appearance_changed)
        session.remove_requested.connect(self.remove_requested)
        session.auto_failed.connect(self.auto_failed)
        session.run_requested.connect(self.run_requested)
        session.config_changed.connect(lambda _region_id: self.regions_changed.emit())
        self._sessions[config.region_id] = session
        session.show()
        return session
