"""llama.cpp 서버(`llama-server.exe`) 프로세스를 띄우고 끄는 구현.

- 비어 있는 로컬 포트를 골라 127.0.0.1에만 연다 (다른 컴퓨터에서 접근할 수 없음).
- Windows Job Object에 넣어, 앱이 비정상 종료되어도 서버가 VRAM을 쥔 채 남지 않게 한다.
- 콘솔 창을 띄우지 않고, 서버 출력은 버린다 (요청 내용이 로그 파일에 남지 않게 하기 위함).

시작은 기다리지 않고 바로 돌아온다. 모델을 다 불러왔는지는 번역 엔진이 `/health`로 확인한다.
"""

import ctypes
import logging
import socket
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from translator.tuning import LLM_CONTEXT_TOKENS, LLM_GPU_LAYERS, LLM_SERVER_STOP_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)

# 서버를 열 주소 (이 컴퓨터 안에서만 접근)
LOCAL_HOST = "127.0.0.1"
# Windows: 새 콘솔 창을 띄우지 않는 프로세스 생성 플래그 (Win32 API가 정한 값)
_CREATE_NO_WINDOW = 0x08000000
# Windows Job Object 설정 값 (Win32 API가 정한 값)
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS = 9
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000


def find_free_port(host: str = LOCAL_HOST) -> int:
    """지금 비어 있는 TCP 포트 번호를 운영체제에서 받는다.

    반환값:
        포트 번호. 돌려준 뒤 다른 프로그램이 먼저 쓸 수도 있지만, 그 가능성은 매우 낮다.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((host, 0))
        return probe.getsockname()[1]


def server_command(
    executable: Path, model: Path, port: int, context_tokens: int, gpu_layers: int
) -> list[str]:
    """llama-server 실행 인자를 만든다.

    매개변수:
        executable: llama-server 실행 파일 경로.
        model: GGUF 모델 파일 경로.
        port: 열 포트.
        context_tokens: 문맥 길이 (토큰).
        gpu_layers: GPU에 올릴 층 수.
    반환값:
        `subprocess`에 넘길 인자 목록.
    """
    return [
        str(executable),
        "--model", str(model),
        "--host", LOCAL_HOST,
        "--port", str(port),
        "--ctx-size", str(context_tokens),
        "--n-gpu-layers", str(gpu_layers),
        # 공용 추론 큐가 번역을 한 번에 하나씩 보내므로 동시 처리 칸은 하나면 된다 (VRAM 절약).
        "--parallel", "1",
        # 채팅 템플릿(Jinja)을 써야 thinking 끄기 같은 템플릿 인자가 적용된다.
        "--jinja",
        # 브라우저용 웹 화면은 쓰지 않는다.
        "--no-webui",
    ]


class _KillOnCloseJob:
    """닫히거나 앱이 끝나면 안에 든 프로세스를 함께 끝내는 Windows Job Object."""

    def __init__(self) -> None:
        """Job Object를 만들고 '닫히면 모두 종료' 설정을 건다. 실패하면 경고만 남긴다."""
        # Job Object 핸들 (만들지 못했으면 None)
        self._handle: int | None = None
        if sys.platform != "win32":
            return
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            logger.warning("Job Object를 만들지 못했습니다 (오류 %d).", ctypes.get_last_error())
            return
        info = _ExtendedLimitInformation()
        info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(
            handle, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS, ctypes.byref(info), ctypes.sizeof(info)
        ):
            logger.warning("Job Object 설정 실패 (오류 %d).", ctypes.get_last_error())
            kernel32.CloseHandle(handle)
            return
        self._handle = handle
        self._kernel32 = kernel32

    def add(self, process: subprocess.Popen) -> bool:
        """프로세스를 Job Object에 넣는다. 성공하면 True."""
        if self._handle is None:
            return False
        if not self._kernel32.AssignProcessToJobObject(self._handle, int(process._handle)):  # noqa: SLF001
            logger.warning("서버 프로세스를 Job Object에 넣지 못했습니다 (오류 %d).", ctypes.get_last_error())
            return False
        return True

    def close(self) -> None:
        """Job Object를 닫는다 (안에 남은 프로세스가 있으면 함께 끝난다)."""
        if self._handle is not None:
            self._kernel32.CloseHandle(self._handle)
            self._handle = None


class _BasicLimitInformation(ctypes.Structure):
    """Win32 JOBOBJECT_BASIC_LIMIT_INFORMATION 구조체."""

    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
    ]


class _IoCounters(ctypes.Structure):
    """Win32 IO_COUNTERS 구조체."""

    _fields_ = [(name, ctypes.c_uint64) for name in (
        "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
    )]


class _ExtendedLimitInformation(ctypes.Structure):
    """Win32 JOBOBJECT_EXTENDED_LIMIT_INFORMATION 구조체."""

    _fields_ = [
        ("BasicLimitInformation", _BasicLimitInformation),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class LlamaServerProcess:
    """llama-server 프로세스 하나를 관리한다. 메서드는 UI 스레드(진입점)에서 호출한다."""

    def __init__(
        self,
        executable: Path,
        model: Path,
        context_tokens: int = LLM_CONTEXT_TOKENS,
        gpu_layers: int = LLM_GPU_LAYERS,
        popen: Callable[..., Any] = subprocess.Popen,
        port_finder: Callable[[], int] = find_free_port,
        extra_args: Sequence[str] = (),
    ) -> None:
        """실행 파일, 모델, 서버 설정과 프로세스 생성 함수(테스트에서 바꾼다)를 받는다.

        매개변수:
            executable: llama-server 실행 파일 경로.
            model: GGUF 모델 파일 경로.
            context_tokens: 문맥 길이 (토큰).
            gpu_layers: GPU에 올릴 층 수.
            popen: 프로세스를 만드는 함수.
            port_finder: 빈 포트를 고르는 함수.
            extra_args: 추가 실행 인자.
        """
        # 실행 파일과 모델 경로
        self._executable = executable
        self._model = model
        # 서버 설정
        self._context_tokens = context_tokens
        self._gpu_layers = gpu_layers
        self._extra_args = tuple(extra_args)
        # 프로세스 생성 함수
        self._popen = popen
        # 실행 중인 서버 프로세스 (시작 전이거나 끝났으면 None)
        self._process: Any | None = None
        # 서버 포트. 만들 때 정해 두어 번역 엔진이 시작 전에도 주소를 알 수 있게 한다.
        self._port = port_finder()
        # 앱 종료 시 서버를 함께 끝내는 Job Object
        self._job: _KillOnCloseJob | None = None

    @property
    def base_url(self) -> str:
        """서버 주소 (예: http://127.0.0.1:51234). 시작하기 전에도 정해져 있다."""
        return f"http://{LOCAL_HOST}:{self._port}"

    @property
    def model_name(self) -> str:
        """모델 파일 이름 (확장자 제외, 결과의 엔진 이름에 쓴다)."""
        return self._model.stem

    def start(self) -> str:
        """서버 프로세스를 시작하고 주소를 돌려준다. 모델 로딩은 기다리지 않는다.

        반환값:
            서버 주소.
        예외:
            FileNotFoundError: 실행 파일이나 모델 파일이 없을 때.
            OSError: 프로세스를 만들지 못했을 때.
        """
        if self.is_running():
            return self.base_url
        for path in (self._executable, self._model):
            if not path.is_file():
                raise FileNotFoundError(f"파일이 없습니다: {path}")
        command = server_command(
            self._executable, self._model, self._port, self._context_tokens, self._gpu_layers
        ) + list(self._extra_args)
        self._job = _KillOnCloseJob()
        self._process = self._popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=_CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        attached = self._job.add(self._process) if isinstance(self._process, subprocess.Popen) else False
        logger.info(
            "번역 서버 시작: 모델 %s, 포트 %d%s",
            self._model.name, self._port, "" if attached else " (앱 비정상 종료 시 자동 종료 미보장)",
        )
        return self.base_url

    def is_running(self) -> bool:
        """서버 프로세스가 실행 중인지 여부."""
        return self._process is not None and self._process.poll() is None

    def stop(self, timeout_seconds: float = LLM_SERVER_STOP_TIMEOUT_SECONDS) -> None:
        """서버를 끝낸다. 시간 안에 끝나지 않으면 강제로 끝낸다. 여러 번 호출해도 된다."""
        process, self._process = self._process, None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout_seconds)
            except subprocess.TimeoutExpired:
                logger.warning("번역 서버가 %ss 안에 끝나지 않아 강제로 끝냅니다.", timeout_seconds)
                process.kill()
                process.wait(timeout_seconds)
            logger.info("번역 서버 종료 (종료 코드 %s)", process.returncode)
        if self._job is not None:
            self._job.close()
            self._job = None
