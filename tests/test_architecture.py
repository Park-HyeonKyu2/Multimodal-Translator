"""계층 의존 규칙 테스트: core와 application은 화면(ui)과 Qt 위젯·그래픽에 의존하지 않는다.

흐름 로직(application)이 화면 없이 테스트되고 재사용될 수 있게 하는 규칙이다. 이 테스트가
깨지면 흐름 로직이 화면 쪽 구현을 직접 가져다 쓰기 시작했다는 뜻이다 (인터페이스로 받아야 한다).
"""

import ast
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "src" / "translator"
# 화면에 의존하는 모듈 접두사: 우리 ui 패키지와 Qt의 위젯·그래픽 모듈
FORBIDDEN_PREFIXES = ("translator.ui", "PySide6.QtWidgets", "PySide6.QtGui")


def imported_modules(path: Path) -> list[str]:
    """파일이 가져오는 모듈 이름 목록."""
    modules: list[str] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            modules += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return modules


def lower_layer_files() -> list[Path]:
    """core와 application 패키지의 모든 파이썬 파일."""
    return sorted(p for package in ("core", "application") for p in (SOURCE / package).glob("*.py"))


@pytest.mark.parametrize("path", lower_layer_files(), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_core_and_application_do_not_import_ui(path: Path) -> None:
    """core·application 모듈은 ui 패키지와 Qt 위젯·그래픽 모듈을 가져오지 않는다."""
    violations = [m for m in imported_modules(path) if m.startswith(FORBIDDEN_PREFIXES)]

    assert violations == []


def test_core_does_not_import_application_or_adapters() -> None:
    """core는 가장 아래 계층이라 application과 adapters를 가져오지 않는다."""
    violations = {
        path.name: [m for m in imported_modules(path) if m.startswith(("translator.application", "translator.adapters"))]
        for path in (SOURCE / "core").glob("*.py")
    }

    assert {name: found for name, found in violations.items() if found} == {}
