"""애플리케이션 진입점이자 의존성을 조립하는 곳.

번역 엔진, OCR, 캡처 구현을 바꾸려면 `build_coordinator`에 다른 구현을 넘기면 된다.
"""

import logging
import os
import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path

from PySide6.QtWidgets import QApplication

from translator.adapters.capture.windows import MssScreenCapturer
from translator.adapters.content.windows import WindowsContentDetector
from translator.adapters.ocr.factory import select_text_recognizer
from translator.adapters.storage.sqlite_profiles import SqliteContentProfileStore, default_database_path
from translator.adapters.ocr.mock_layout import MockLayoutTextRecognizer
from translator.adapters.translation.factory import select_translator
from translator.adapters.translation.mock import MockTranslator
from translator.application.auto_detector import AutoDetector
from translator.application.change_detection import compute_fingerprint
from translator.application.content_monitor import ContentMonitor
from translator.application.controller import TranslationController
from translator.application.ocr_service import OcrService
from translator.application.pipeline import TranslationPipeline
from translator.application.profile_store import ContentProfileStore
from translator.application.scheduler import InferenceScheduler
from translator.application.text_display import TextDisplayModel
from translator.application.workflow import TranslationWorkflow
from translator.core.errors import StorageError
from translator.core.interfaces import ContentDetector, ScreenCapturer, TextRecognizer, Translator
from translator.tuning import (
    FULLSCREEN_DETECT_GRID_HEIGHT,
    FULLSCREEN_DETECT_GRID_WIDTH,
    FULLSCREEN_DETECT_INTERVAL_CPU_MS,
    FULLSCREEN_DETECT_INTERVAL_MS,
)
from translator.ui.coordinator import AppCoordinator
from translator.ui.overlay import MiniOverlay
from translator.ui.region_box import RegionBox
from translator.ui.region_box_port import RegionBoxPort
from translator.ui.region_manager import RegionManager
from translator.ui.region_migration import migrate_legacy_regions
from translator.ui.region_selector import RegionSelector
from translator.ui.result_window import ResultWindow
from translator.ui.screen_geometry import virtual_desktop_region
from translator.ui.translation_overlay import TranslationOverlay
from translator.ui.ui_settings import SettingsProfileStore, UiSettingsStore
from translator.utils.logger import install_exception_hooks, setup_logging

logger = logging.getLogger("translator.main")

# OCR이 CPU로 실행될 때 전체 화면 번역을 켜면 보여 주는 안내 (상태 창)
SLOW_FULLSCREEN_NOTICE = (
    "OCR이 CPU로 실행 중이라 전체 화면 번역이 느립니다(한 번에 수 초). "
    "화면 변화 확인 간격을 늘렸습니다. 바운딩 박스 사용을 권장합니다."
)


def build_coordinator(
    capturer: ScreenCapturer | None = None,
    recognizer: TextRecognizer | None = None,
    translator: Translator | None = None,
    settings: UiSettingsStore | None = None,
    slow_full_screen: bool = False,
    content_detector: ContentDetector | None = None,
    profile_store: ContentProfileStore | None = None,
) -> AppCoordinator:
    """구성 요소를 조립해 coordinator를 만든다. QApplication이 만들어진 뒤에 호출해야 한다.

    `settings`가 None이면 UI 설정을 저장하거나 복원하지 않는다 (테스트 기본값).
    `recognizer`가 None이면 화면에서 글자처럼 보이는 영역을 찾는 Mock OCR을 쓴다.
    `slow_full_screen`이 True면(OCR이 CPU로 실행됨) 전체 화면 변화 확인 간격을 늘리고, 전체 화면을
    켤 때 느리다는 안내를 보여 준다.
    `content_detector`가 있으면 활성 창으로 콘텐츠를 식별해 바뀔 때 세션을 바꾼다 (None이면 식별하지
    않는다. 테스트 기본값).
    `profile_store`가 있으면 바운딩 박스를 콘텐츠별로 저장·복원한다 (None이면 저장하지 않는다). `settings`도
    함께 있으면 이전 버전이 레지스트리에 저장한 박스를 먼저 `profile_store`로 옮긴다.
    """
    if (
        settings is not None
        and profile_store is not None
        and not isinstance(profile_store, SettingsProfileStore)  # 대체 저장소는 이미 레지스트리다
    ):
        migrate_legacy_regions(settings, profile_store)
    # 번역 파이프라인과 자동 감지기가 같은 캡처 구현을 공유한다.
    capturer = capturer or MssScreenCapturer()
    # 모든 출처가 OCR 서비스 하나를 공유해, 같은 캡처 내용은 한 번만 OCR한다 (캐시·중복 합치기).
    ocr_service = OcrService(recognizer or MockLayoutTextRecognizer())
    pipeline = TranslationPipeline(
        capturer=capturer,
        recognizer=ocr_service,
        translator=translator or MockTranslator(),
    )
    # OCR·번역(향후 장면 분석)이 함께 쓰는 추론 큐. 동시 실행 수는 tuning.INFERENCE_MAX_CONCURRENCY.
    scheduler = InferenceScheduler()
    controller = TranslationController(pipeline, scheduler)
    # 전체 화면 추적: 모든 모니터를 감싸는 영역을 박스보다 촘촘한 격자로 관찰한다.
    fullscreen_detector = AutoDetector(
        capturer,
        region_provider=virtual_desktop_region,
        observer=partial(
            compute_fingerprint,
            grid_width=FULLSCREEN_DETECT_GRID_WIDTH,
            grid_height=FULLSCREEN_DETECT_GRID_HEIGHT,
        ),
        interval_ms=(
            FULLSCREEN_DETECT_INTERVAL_CPU_MS if slow_full_screen else FULLSCREEN_DETECT_INTERVAL_MS
        ),
    )
    regions = RegionManager(
        box_factory=lambda region_id, accent: RegionBox(region_id, accent),
        # 박스마다 자기 자동 감지기와 표본 스레드를 가진다 (박스를 지워도 다른 박스에 영향 없음).
        detector_factory=lambda provider: AutoDetector(capturer, region_provider=provider),
        profile_store=profile_store,
    )
    # 활성 창을 지켜보다 콘텐츠가 바뀌면 흐름에 알리는 감시기 (식별기가 없으면 만들지 않는다).
    content_monitor = ContentMonitor(content_detector) if content_detector is not None else None
    status_window = ResultWindow()
    # 출처별 방향/색 조회 함수는 coordinator가 만들어질 때 등록된다.
    translation_overlay = TranslationOverlay(TextDisplayModel())
    # OCR·번역 흐름: 화면과 무관한 처리(요청, 안정화, 재검사, 전체 화면 추적, 세션)를 맡는다.
    # 박스와 말풍선은 인터페이스(BoxPort, DisplayPort)로만 알려 준다.
    workflow = TranslationWorkflow(
        controller=controller,
        display=translation_overlay,
        boxes=RegionBoxPort(regions),
        fullscreen_detector=fullscreen_detector,
        content_monitor=content_monitor,
    )
    coordinator = AppCoordinator(
        overlay=MiniOverlay(),
        selector=RegionSelector(),
        regions=regions,
        status_window=status_window,
        workflow=workflow,
        translation_overlay=translation_overlay,
        settings=settings,
        fullscreen_notice=SLOW_FULLSCREEN_NOTICE if slow_full_screen else None,
    )
    for child in (scheduler, controller, workflow, regions, translation_overlay, fullscreen_detector):
        child.setParent(coordinator)
    if content_monitor is not None:
        content_monitor.setParent(coordinator)
    translation_overlay.set_background_opacity(status_window.background_opacity)
    return coordinator


def open_profile_store(
    settings: UiSettingsStore, database_path: Path | None = None
) -> tuple[ContentProfileStore, str | None]:
    """콘텐츠별 박스 저장소를 연다. SQLite를 열 수 없으면 레지스트리 기반 대체 저장소를 돌려준다.

    대체 저장소를 쓰면 박스가 사라지지 않고(기존 박스는 레지스트리에 그대로 있다) 콘텐츠별로도
    나뉜다. 다음에 SQLite를 열 수 있게 되면 `migrate_legacy_regions`가 SQLite로 옮긴다.

    매개변수:
        settings: 대체 저장소가 쓸 UI 설정 저장소.
        database_path: SQLite 파일 경로 (None이면 기본 경로).
    반환값:
        (저장소, 사용자에게 보여 줄 안내). 정상이면 안내는 None이다.
    """
    path = database_path or default_database_path()
    try:
        return SqliteContentProfileStore(path), None
    except StorageError as exc:
        logger.error("SQLite를 열지 못해 레지스트리에 박스를 저장합니다.")
        notice = (
            f"{exc} (파일: {path}) 박스 설정은 이전 방식(레지스트리)으로 저장하며 기존 박스는 지워지지 "
            "않았습니다. 파일 문제를 해결하고 다시 시작하면 SQLite로 옮겨집니다."
        )
        return SettingsProfileStore(settings), notice


def main() -> int:
    """Multimodal Translator 애플리케이션을 시작한다."""
    log_path = setup_logging()
    install_exception_hooks()
    logger.info("시작 (로그 파일: %s)", log_path or "없음")

    app = QApplication(sys.argv)
    app.setApplicationName("Multimodal Translator")
    # 상태 창만 닫혀도 종료되지 않도록 하고, 종료는 오버레이의 닫기 버튼으로만 한다.
    app.setQuitOnLastWindowClosed(False)

    # OCR 엔진과 장치는 환경 변수로 고른다 (기본: PaddleOCR가 설치되어 있으면 PaddleOCR, GPU가 있으면 GPU).
    selection = select_text_recognizer()
    # 번역 엔진도 환경 변수로 고른다 (기본: Qwen 설정이 있으면 Qwen, 없으면 Mock).
    translation = select_translator()
    # UI 설정 저장소와 콘텐츠별 박스 저장소. SQLite를 열지 못하면 레지스트리 기반 저장소로 계속 실행하고
    # 시작 후 상태 창에 안내한다.
    ui_settings = UiSettingsStore()
    profile_store, storage_error = open_profile_store(ui_settings)
    try:
        if translation.server is not None:
            # 모델 로딩은 기다리지 않는다. 첫 번역이 워커 스레드에서 준비될 때까지 기다린다.
            try:
                translation.server.start()
            except OSError:
                logger.exception("번역 서버를 시작하지 못했습니다. 번역할 때 상태 창에 오류가 표시됩니다.")
        coordinator = build_coordinator(
            settings=ui_settings,
            recognizer=selection.recognizer,
            translator=translation.translator,
            slow_full_screen=selection.slow_full_screen,
            content_detector=WindowsContentDetector(),
            profile_store=profile_store,
        )
        coordinator.quit_requested.connect(app.quit)
        coordinator.start()
        if storage_error is not None:
            coordinator.show_error(storage_error)
        # Qt 이벤트 루프가 끝날 때의 종료 코드
        exit_code = app.exec()
    finally:
        # 정리 없이 끝내는 경우(finish_process)에도 서버가 VRAM을 쥔 채 남지 않게 먼저 끈다.
        if translation.server is not None:
            translation.server.stop()
        # 박스 설정은 coordinator 종료 때 이미 저장했다. 저장소 연결만 닫는다.
        profile_store.close()
    return finish_process(coordinator, exit_code)


def finish_process(
    coordinator: AppCoordinator,
    exit_code: int,
    hard_exit: Callable[[int], None] = os._exit,
) -> int:
    """이벤트 루프가 끝난 뒤 프로세스 종료 방법을 정한다.

    PaddleOCR 추론은 중간에 멈출 수 없다. 긴 추론(예: CPU 전체 화면 OCR, 수십 초)이 종료 대기
    시간을 넘겨 남아 있으면, 스레드 풀 정리가 그 작업을 끝까지 기다려 창이 닫힌 뒤에도 프로세스가
    남는다. 설정은 `shutdown`에서 이미 저장(sync)했으므로, 이때는 로그를 비운 뒤 기다리지 않고 끝낸다.

    매개변수:
        coordinator: 종료를 마친 coordinator.
        exit_code: Qt 이벤트 루프의 종료 코드.
        hard_exit: 정리 없이 프로세스를 끝내는 함수 (테스트에서 바꾼다).
    반환값:
        추론 작업이 모두 끝났으면 `exit_code` (보통 종료). 아니면 `hard_exit`가 돌아오지 않는다.
    """
    if not coordinator.background_work_finished:
        logger.warning("끝나지 않은 추론 작업이 있어 기다리지 않고 프로세스를 종료합니다.")
        logging.shutdown()
        hard_exit(exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
