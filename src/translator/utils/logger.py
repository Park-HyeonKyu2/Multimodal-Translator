"""로깅 설정.

로그에는 진단 정보(크기, 길이, 오류 종류)만 남기고 번역 대상 텍스트나 화면
내용은 기록하지 않는다.
"""

import logging
import os
import sys
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path
from types import TracebackType

from translator.tuning import LOG_BACKUP_COUNT, LOG_MAX_BYTES

LOGGER_NAME = "translator"
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(threadName)s %(name)s: %(message)s"


def default_log_dir() -> Path:
    """LOCALAPPDATA 아래 MultimodalTranslator/logs 폴더 경로를 반환한다."""
    base = os.environ.get("LOCALAPPDATA")
    root = Path(base) if base else Path.home()
    return root / "MultimodalTranslator" / "logs"


def setup_logging(level: int = logging.INFO, log_dir: Path | None = None) -> Path | None:
    """콘솔과 회전 파일 로그를 설정하고 로그 파일 경로를 반환한다.

    파일을 만들 수 없으면 콘솔 로그만 사용하고 None을 반환한다. 여러 번 호출해도
    핸들러가 중복되지 않는다.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(LOG_FORMAT)
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    log_path = (log_dir or default_log_dir()) / "translator.log"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT, encoding="utf-8"
        )
    except OSError as exc:
        logger.warning("로그 파일을 만들 수 없어 콘솔 로그만 사용합니다: %s", exc)
        return None
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return log_path


def install_exception_hooks() -> None:
    """처리되지 않은 예외(메인/워커 스레드, Qt 슬롯 포함)를 로그에 남긴다."""
    logger = logging.getLogger(LOGGER_NAME)

    def handle_exception(
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_traceback: TracebackType | None,
    ) -> None:
        """처리되지 않은 예외를 로그에 남긴다 (Ctrl+C는 기본 처리)."""
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.critical(
            "처리되지 않은 예외", exc_info=(exc_type, exc_value, exc_traceback)
        )

    def handle_thread_exception(args: threading.ExceptHookArgs) -> None:
        """워커 스레드의 처리되지 않은 예외를 같은 방식으로 기록한다."""
        handle_exception(args.exc_type, args.exc_value, args.exc_traceback)

    sys.excepthook = handle_exception
    threading.excepthook = handle_thread_exception
