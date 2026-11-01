"""llama.cpp 서버 프로세스 관리 테스트: 실행 인자, 시작·종료, 강제 종료, Job Object (실제 모델 없이)."""

import subprocess
import sys
import time
from pathlib import Path

import pytest

from translator.adapters.translation.llama_server import (
    LOCAL_HOST,
    LlamaServerProcess,
    _KillOnCloseJob,
    find_free_port,
    server_command,
)


class FakeProcess:
    """가짜 프로세스: 종료 요청을 기록하고, `stubborn`이면 terminate로 끝나지 않는다."""

    def __init__(self, stubborn: bool = False) -> None:
        """실행 중 상태로 시작한다."""
        # 종료 코드 (실행 중이면 None)
        self.returncode: int | None = None
        # terminate로 끝나지 않는지
        self.stubborn = stubborn
        # 호출된 종료 동작 기록
        self.calls: list[str] = []

    def poll(self) -> int | None:
        """종료 코드를 돌려준다."""
        return self.returncode

    def terminate(self) -> None:
        """종료를 요청한다."""
        self.calls.append("terminate")
        if not self.stubborn:
            self.returncode = 0

    def kill(self) -> None:
        """강제로 끝낸다."""
        self.calls.append("kill")
        self.returncode = 1

    def wait(self, timeout: float) -> int:
        """끝날 때까지 기다린다. 아직 실행 중이면 시간 초과."""
        if self.returncode is None:
            raise subprocess.TimeoutExpired("llama-server", timeout)
        return self.returncode


@pytest.fixture
def files(tmp_path: Path) -> tuple[Path, Path]:
    """가짜 실행 파일과 모델 파일."""
    executable = tmp_path / "llama-server.exe"
    model = tmp_path / "Qwen-test.gguf"
    executable.write_bytes(b"")
    model.write_bytes(b"")
    return executable, model


def test_command_binds_localhost_only_with_jinja() -> None:
    """서버는 127.0.0.1에만 열고, 채팅 템플릿(Jinja)을 쓰며, 동시 처리 칸은 하나다."""
    command = server_command(Path("s.exe"), Path("m.gguf"), 1234, 4096, 999)

    assert command[0] == "s.exe"
    assert command[command.index("--host") + 1] == LOCAL_HOST == "127.0.0.1"
    assert command[command.index("--port") + 1] == "1234"
    assert command[command.index("--parallel") + 1] == "1"
    assert "--jinja" in command and "--no-webui" in command


def test_free_port_is_usable() -> None:
    """빈 포트 번호를 받는다."""
    assert 0 < find_free_port() < 65536


def test_start_and_stop(files) -> None:
    """시작하면 정한 포트로 서버를 띄우고 주소를 알려 주며, 종료하면 끝낸다. 주소는 시작 전에도 정해져 있다."""
    executable, model = files
    launched: list[tuple] = []
    process = FakeProcess()

    def popen(command, **kwargs):
        """실행 인자를 기록하고 가짜 프로세스를 돌려준다."""
        launched.append((command, kwargs))
        return process

    server = LlamaServerProcess(executable, model, popen=popen, port_finder=lambda: 5555)
    assert server.base_url == "http://127.0.0.1:5555"

    assert server.start() == "http://127.0.0.1:5555"
    assert server.is_running()
    assert server.model_name == "Qwen-test"
    command, kwargs = launched[0]
    assert command[command.index("--model") + 1] == str(model)
    assert kwargs["stdout"] is subprocess.DEVNULL and kwargs["stderr"] is subprocess.DEVNULL

    server.stop()
    server.stop()  # 여러 번 호출해도 된다.
    assert not server.is_running()
    assert process.calls == ["terminate"]


def test_start_twice_does_not_launch_again(files) -> None:
    """이미 실행 중이면 다시 띄우지 않는다."""
    executable, model = files
    launched: list = []
    server = LlamaServerProcess(
        executable, model, popen=lambda *a, **k: launched.append(1) or FakeProcess(), port_finder=lambda: 1
    )

    server.start()
    server.start()

    assert len(launched) == 1


def test_stubborn_server_is_killed(files) -> None:
    """정한 시간 안에 끝나지 않으면 강제로 끝낸다."""
    executable, model = files
    process = FakeProcess(stubborn=True)
    server = LlamaServerProcess(executable, model, popen=lambda *a, **k: process, port_finder=lambda: 1)
    server.start()

    server.stop(timeout_seconds=0.01)

    assert process.calls == ["terminate", "kill"]


def test_missing_files_are_reported(tmp_path: Path) -> None:
    """실행 파일이나 모델 파일이 없으면 시작하지 않고 알린다."""
    server = LlamaServerProcess(tmp_path / "none.exe", tmp_path / "none.gguf", port_finder=lambda: 1)

    with pytest.raises(FileNotFoundError):
        server.start()
    assert not server.is_running()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows Job Object 테스트")
def test_job_object_kills_child_when_closed() -> None:
    """Job Object를 닫으면(앱이 끝나면) 안에 든 프로세스도 함께 끝난다 (서버가 VRAM을 쥔 채 남지 않음)."""
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    job = _KillOnCloseJob()
    try:
        assert job.add(child)
        job.close()
        deadline = time.monotonic() + 10
        while child.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert child.poll() is not None
    finally:
        if child.poll() is None:
            child.kill()
