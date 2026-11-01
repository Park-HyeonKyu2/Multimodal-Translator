"""오버레이 버튼, 영역 선택, 바운딩 박스들, 번역 흐름, 번역 말풍선, 상태 창을 연결한다.

이 객체는 화면(UI)을 연결하는 일만 한다: 사용자 동작을 흐름(`TranslationWorkflow`)과 박스에
전달하고, 흐름이 보내는 알림(전체 화면 켜짐·꺼짐, 실패, 텍스트 없음)을 창에 반영한다. OCR 요청,
텍스트 안정화, 번역 요청, 재검사 예약, 전체 화면 추적, 세션 전환은 모두 흐름이 맡는다
(`application/workflow.py`).

모드:
- 편집 모드: 박스가 또렷하게 보이고 이동·크기 조절·우클릭 메뉴를 쓸 수 있다. 상태 창은
  설정 창(말풍선 배경 불투명도, 전체 화면 번역 위치)으로 보인다.
- 잠금 모드(편집이 꺼짐): 박스는 흐린 점선으로 남고, 박스·말풍선·상태 창이 클릭을
  통과시킨다. 번역을 실행하면(영역 번역, 자동, 전체 화면) 이 모드로 바뀐다.

번역 결과는 출처(박스 ID 또는 전체 화면)별로 말풍선에 표시하며, 말풍선 위치의 기준은
항상 OCR이 인식한 원문 텍스트 좌표다. 박스는 박스마다 말풍선 하나에 번역문을 읽는 순서로
모두 넣고, 전체 화면은 원문마다 말풍선을 따로 놓는다. UI 설정과 박스 목록은 시작할 때 복원하고
종료할 때와 박스 설정이 바뀔 때 저장한다.
"""

import logging

from PySide6.QtCore import QObject, QRect, QTimer, Signal
from PySide6.QtGui import QColor, QGuiApplication, QScreen

from translator.application.workflow import FULLSCREEN_SOURCE_ID, TranslationWorkflow
from translator.core.models import TranslationDirection
from translator.tuning import CAPTURE_DELAY_MS
from translator.ui.overlay import MiniOverlay
from translator.ui.region_manager import RegionManager
from translator.ui.region_selector import RegionSelector
from translator.ui.result_window import ResultWindow
from translator.ui.screen_geometry import is_rect_on_any_screen
from translator.ui.translation_overlay import TranslationOverlay
from translator.ui.ui_settings import UiSettingsStore

logger = logging.getLogger(__name__)

# 전체 화면 번역 말풍선의 색 띠 (특정 박스와 연결되지 않음을 회색으로 표시)
FULLSCREEN_ACCENT = QColor(190, 190, 190)

class AppCoordinator(QObject):
    """사용자 동작을 각 구성 요소에 전달하고 결과를 올바른 박스·말풍선에 연결한다."""

    quit_requested = Signal()

    def __init__(
        self,
        overlay: MiniOverlay,
        selector: RegionSelector,
        regions: RegionManager,
        status_window: ResultWindow,
        workflow: TranslationWorkflow,
        translation_overlay: TranslationOverlay,
        settings: UiSettingsStore | None = None,
        capture_delay_ms: int = CAPTURE_DELAY_MS,
        fullscreen_notice: str | None = None,
        parent: QObject | None = None,
    ) -> None:
        """각 구성 요소를 주입받아 시그널을 연결한다.

        - `workflow`는 OCR·번역 흐름이다 (전체 화면 감지기와 재검사 예약도 거기에 들어 있다).
        - `settings`가 없으면 설정을 저장하지 않는다.
        - `fullscreen_notice`가 있으면 전체 화면 번역을 켤 때 상태 창에 이 안내를 잠시 보여 준다
          (예: OCR이 CPU로 실행되어 느릴 때).
        """
        super().__init__(parent)
        self._overlay = overlay
        self._selector = selector
        self._regions = regions
        self._status_window = status_window
        self._workflow = workflow
        self._translation_overlay = translation_overlay
        self._settings = settings
        self._capture_delay_ms = capture_delay_ms
        # 전체 화면 번역을 켤 때 보여 줄 안내 (없으면 None)
        self._fullscreen_notice = fullscreen_notice
        # 종료가 시작되면 늦게 도착한 타이머/시그널이 창을 다시 띄우지 않게 막는다.
        self._is_shutting_down = False
        # 종료할 때 실행 중이던 추론 작업이 대기 시간 안에 끝났는지 (끝나지 않았으면 진입점이 바로 종료한다)
        self._background_work_finished = True
        # 편집 모드 여부 (False = 잠금: 박스 흐림, 클릭 통과)
        self._edit_mode = False
        # 말풍선은 출처별 방향과 박스 색, 말풍선 묶기 여부를 이 객체에서 조회한다.
        translation_overlay.set_style_lookup(self.style_for)
        translation_overlay.set_group_lookup(self.groups_source)

        overlay.region_requested.connect(self._start_region_selection)
        overlay.full_screen_toggled.connect(self._set_full_screen)
        overlay.region_translate_requested.connect(self._translate_all_regions)
        overlay.auto_toggled.connect(self._set_auto_all)
        overlay.edit_toggled.connect(self._set_edit_mode)
        overlay.close_requested.connect(self.shutdown)
        selector.region_selected.connect(self._on_region_selected)
        selector.cancelled.connect(self._on_selection_cancelled)
        regions.ocr_requested.connect(workflow.on_box_ocr_requested)
        regions.appearance_changed.connect(lambda _region_id: translation_overlay.refresh())
        regions.remove_requested.connect(self._remove_region)
        regions.regions_changed.connect(self._on_regions_changed)
        regions.auto_failed.connect(self._on_auto_failed)
        regions.storage_failed.connect(self.show_error)
        # 박스 메뉴에서 번역을 실행해도 오버레이 버튼과 같이 편집 모드를 끈다.
        regions.run_requested.connect(lambda _region_id: self._leave_edit_mode())
        workflow.full_screen_changed.connect(overlay.set_full_screen_checked)
        workflow.full_screen_error.connect(self._on_full_screen_error)
        workflow.source_failed.connect(self._on_source_failed)
        workflow.no_text_found.connect(status_window.show_no_text)
        workflow.content_detection_error.connect(status_window.show_error)
        status_window.opacity_changed.connect(translation_overlay.set_background_opacity)
        status_window.fullscreen_direction_changed.connect(
            lambda _direction: translation_overlay.refresh()
        )

        app = QGuiApplication.instance()
        if app is not None:
            app.screenRemoved.connect(self._on_screen_removed)

    @property
    def workflow(self) -> TranslationWorkflow:
        """OCR·번역 흐름 (세션 전환 등 화면과 무관한 동작은 여기로 요청한다)."""
        return self._workflow

    @property
    def edit_mode(self) -> bool:
        """편집 모드인지 여부."""
        return self._edit_mode

    def style_for(self, source_id: str) -> tuple[TranslationDirection, QColor]:
        """출처의 말풍선 방향과 색. `TranslationOverlay`가 말풍선을 놓을 때 쓴다."""
        if source_id == FULLSCREEN_SOURCE_ID:
            return self._status_window.fullscreen_direction, FULLSCREEN_ACCENT
        session = self._regions.get(source_id)
        if session is None:
            return self._status_window.fullscreen_direction, FULLSCREEN_ACCENT
        return session.config.direction, session.accent

    def groups_source(self, source_id: str) -> bool:
        """출처의 번역문을 말풍선 하나로 묶을지 여부.

        바운딩 박스는 박스마다 말풍선 하나에 번역문을 모두 넣는다. 전체 화면은 텍스트가 화면
        곳곳에 흩어져 있으므로 원문마다 말풍선을 따로 놓는다.
        """
        return source_id != FULLSCREEN_SOURCE_ID

    def start(self) -> None:
        """저장된 UI 설정과 박스를 복원하고 잠금 모드로 시작한다."""
        self._restore_settings()
        self._status_window.set_interactive(False)
        self._sync_region_buttons()
        self._overlay.set_edit_checked(False)
        self._overlay.set_full_screen_checked(False)
        self._overlay.show()
        self._workflow.start()

    @property
    def background_work_finished(self) -> bool:
        """종료할 때 실행 중이던 추론 작업이 대기 시간 안에 모두 끝났는지 (종료 전에는 True)."""
        return self._background_work_finished

    def shutdown(self) -> None:
        """UI 설정을 저장하고 백그라운드 작업을 정리한 뒤 종료를 요청한다."""
        if self._is_shutting_down:
            return
        self._is_shutting_down = True
        logger.info("프로그램 종료")
        self._save_settings()
        # 새 OCR 예약과 전체 화면 감지를 먼저 멈춰 종료 중에 새 작업이 들어오지 않게 한다.
        self._workflow.stop()
        self._selector.cancel()
        self._regions.shutdown()
        self._background_work_finished = self._workflow.controller.shutdown()
        self._translation_overlay.clear()
        self._status_window.close()
        self.quit_requested.emit()

    # --- UI 설정 ----------------------------------------------------------------

    def _restore_settings(self) -> None:
        """저장된 위치가 현재 모니터 구성에서 보이는 경우에만 복원한다."""
        overlay_position = self._settings.load_overlay_position() if self._settings else None
        # 저장된 위치로 옮겼을 때의 오버레이 영역이 화면 위에 있는지 미리 확인한다.
        if overlay_position is not None and is_rect_on_any_screen(
            QRect(overlay_position, self._overlay.frameGeometry().size())
        ):
            self._overlay.move(overlay_position)
        else:
            self._overlay.move_to_default_position()

        # 박스는 콘텐츠별 저장소(SQLite)에서 복원한다 (UI 설정 저장소와는 별개).
        self._regions.load_profile()
        if self._settings is None:
            return
        geometry = self._settings.load_result_window_geometry()
        if geometry is not None and not self._status_window.restore_geometry(geometry):
            logger.info("저장된 상태 창 위치가 화면 밖이라 기본 위치를 사용합니다.")
        opacity = self._settings.load_result_opacity()
        if opacity is not None:
            self._status_window.set_background_opacity(opacity)
        direction = self._settings.load_fullscreen_direction()
        if direction is not None:
            self._status_window.set_fullscreen_direction(direction)

    def _save_settings(self) -> None:
        """현재 창 위치, 불투명도, 박스 목록을 저장한다. 저장 실패는 종료를 막지 않는다."""
        # 박스 목록은 기다리지 않고 바로 저장한다 (저장 실패는 박스 관리자가 기록하고 알린다).
        self._regions.save_profile()
        if self._settings is None:
            return
        try:
            self._settings.save_overlay_position(self._overlay.pos())
            # 한 번도 표시되지 않은 상태 창의 기본 위치로 저장값을 덮어쓰지 않는다.
            if self._status_window.has_position:
                self._settings.save_result_window_geometry(self._status_window.geometry())
            self._settings.save_result_opacity(self._status_window.background_opacity)
            self._settings.save_fullscreen_direction(self._status_window.fullscreen_direction)
            self._settings.sync()
        except Exception:
            logger.exception("UI 설정 저장 중 오류")

    def _save_regions(self) -> None:
        """박스 목록 저장을 예약한다 (추가·삭제·설정 변경 때). 짧은 시간 모은 뒤 한 번 저장한다."""
        if self._is_shutting_down:
            return
        self._regions.schedule_save()

    def show_error(self, message: str) -> None:
        """상태 창에 오류를 표시한다 (시작할 때 저장소를 열지 못한 경우 등)."""
        if not self._is_shutting_down:
            self._status_window.show_error(message)

    # --- 모드 -------------------------------------------------------------------

    def _set_edit_mode(self, enabled: bool) -> None:
        """편집 모드를 켜거나 끈다. 끌 때 편집 중 바뀐 박스를 다시 번역한다."""
        if enabled == self._edit_mode:
            self._overlay.set_edit_checked(enabled)
            return
        self._edit_mode = enabled
        logger.info("조작 모드: %s", "편집" if enabled else "잠금(클릭 통과)")
        self._overlay.set_edit_checked(enabled)
        self._regions.set_interactive(enabled)
        self._status_window.set_interactive(enabled)
        if not enabled:
            for session in self._regions.sessions():
                # 자동 감지 중인 박스는 감지기가 새 위치를 알아서 번역한다.
                if session.state.needs_translation and not session.config.auto_enabled:
                    session.translate_now()
                session.state.needs_translation = False

    def _leave_edit_mode(self) -> None:
        """번역을 실행하기 전에 편집 모드를 끈다 (박스를 흐리게, 클릭 통과)."""
        if self._edit_mode:
            self._set_edit_mode(False)

    # --- 영역 추가 / 삭제 -------------------------------------------------------

    def _start_region_selection(self) -> None:
        """새 영역 선택을 시작한다. 선택을 가리지 않도록 오버레이와 박스를 잠시 숨긴다."""
        self._overlay.hide()
        self._regions.set_visible(False)
        # 선택 화면(어둡게 덮은 화면)을 전체 화면 감지기가 "화면 변화"로 OCR하지 않게 잠시 멈춘다.
        self._workflow.pause_tracking_for_selection()
        self._selector.start()

    def _on_region_selected(self, rect: QRect) -> None:
        """선택한 영역에 새 박스를 만든다.

        편집 중이면 박스를 그대로 수정할 수 있게 두고(편집을 끌 때 번역), 아니면 바로
        번역한다. 전체 자동 감지가 켜져 있으면 새 박스도 자동 감지로 시작한다.
        """
        self._overlay.show()
        self._regions.set_visible(True)
        self._workflow.resume_tracking_after_selection()
        auto = self._overlay.auto_button.isChecked()
        session = self._regions.add(rect, interactive=self._edit_mode, auto_enabled=auto)
        if self._edit_mode:
            session.state.needs_translation = True
        elif not auto:
            # 선택 화면이 완전히 사라진 뒤 캡처한다. 그 사이 박스가 지워질 수 있으므로 ID로 다시 찾는다.
            region_id = session.region_id
            QTimer.singleShot(self._capture_delay_ms, lambda: self._translate_region(region_id))

    def _on_selection_cancelled(self) -> None:
        """선택을 취소하면 오버레이와 박스를 되돌린다."""
        if self._is_shutting_down:
            return
        self._overlay.show()
        self._regions.set_visible(True)
        self._workflow.resume_tracking_after_selection()

    def _remove_region(self, region_id: str) -> None:
        """박스 하나와 그 요청·예약·말풍선을 지운다. 다른 박스에는 영향을 주지 않는다."""
        self._workflow.forget_source(region_id)
        self._translation_overlay.remove_source(region_id)
        self._regions.remove(region_id)

    def _on_regions_changed(self) -> None:
        """박스 구성이 바뀌면 버튼 상태를 맞추고 저장한다."""
        self._sync_region_buttons()
        self._save_regions()

    def _sync_region_buttons(self) -> None:
        """박스 유무와 전체 자동 감지 상태를 오버레이 버튼에 반영한다."""
        self._overlay.set_region_available(len(self._regions) > 0)
        self._overlay.set_auto_checked(self._regions.all_auto_enabled())

    # --- 번역 요청 ----------------------------------------------------------------

    def _translate_region(self, region_id: str) -> None:
        """박스 하나를 수동 번역한다. 이미 지워졌거나 종료 중이면 무시한다."""
        session = self._regions.get(region_id)
        if session is not None and not self._is_shutting_down:
            session.translate_now()

    def _translate_all_regions(self) -> None:
        """모든 박스를 지금 수동으로 번역한다."""
        self._leave_edit_mode()
        for session in self._regions.sessions():
            session.translate_now()

    def _set_auto_all(self, enabled: bool) -> None:
        """모든 박스의 자동 감지를 켜거나 끈다."""
        if enabled:
            self._leave_edit_mode()
        self._regions.set_auto_all(enabled)
        self._sync_region_buttons()

    # --- 전체 화면 번역 -----------------------------------------------------------

    def _set_full_screen(self, enabled: bool) -> None:
        """전체 화면 번역을 켜거나 끈다 (오버레이의 전체 화면 버튼)."""
        if self._is_shutting_down:
            return
        if enabled == self._workflow.full_screen_on:
            self._overlay.set_full_screen_checked(enabled)
            return
        if enabled:
            self._start_full_screen()
        else:
            self._workflow.stop_full_screen()

    def _start_full_screen(self) -> None:
        """전체 화면 번역을 켠다. 편집 모드를 끄고(박스 재번역 포함), 느리면 안내를 보여 준다.

        편집을 끌 때 일어나는 박스 재번역도 전체 화면에 딸린 요청으로 보고 "텍스트 없음"을 알리지 않는다.
        """
        with self._workflow.quiet_box_requests():
            self._leave_edit_mode()
            if self._fullscreen_notice:
                self._status_window.show_notice(self._fullscreen_notice, is_error=False)
            self._workflow.start_full_screen()

    # --- 흐름의 알림 --------------------------------------------------------------

    def _on_source_failed(self, source_id: str, message: str) -> None:
        """인식·번역 실패를 상태 창에 알린다."""
        if self._is_shutting_down:
            return
        label = "전체 화면" if source_id == FULLSCREEN_SOURCE_ID else source_id
        self._status_window.show_error(f"{label}: {message}")

    def _on_full_screen_error(self, message: str) -> None:
        """전체 화면 추적이 연속 캡처 실패로 멈췄음을 알린다 (전체 화면 번역은 이미 꺼졌다)."""
        if self._is_shutting_down:
            return
        self._status_window.show_error(f"전체 화면 추적을 중지했습니다. {message}")

    def _on_auto_failed(self, region_id: str, message: str) -> None:
        """박스의 자동 감지가 연속 실패로 멈추면 알리고 버튼 상태를 맞춘다."""
        self._sync_region_buttons()
        self._status_window.show_error(f"{region_id} 자동 감지를 중지했습니다. {message}")

    # --- 모니터 변경 --------------------------------------------------------------

    def _on_screen_removed(self, _screen: QScreen) -> None:
        """모니터가 분리되면 창 위치를 다시 확인한다."""
        # Qt가 창을 옮긴 뒤에 위치를 확인하도록 이벤트 루프 한 바퀴 뒤에 실행한다.
        QTimer.singleShot(0, self._ensure_windows_on_screen)

    def _ensure_windows_on_screen(self) -> None:
        """화면 밖에 남은 창과 박스를 화면 안으로 옮기고 말풍선을 다시 놓는다."""
        if self._is_shutting_down:
            return
        if not is_rect_on_any_screen(self._overlay.frameGeometry()):
            self._overlay.move_to_default_position()
        if self._status_window.isVisible():
            self._status_window.ensure_on_screen()
        self._regions.ensure_on_screen()
        self._translation_overlay.refresh()
