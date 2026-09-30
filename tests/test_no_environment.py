import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

import karotte.load_tasks
from karotte.cli import app
from karotte.load_tasks import (
    _environment_is_installed as real_environment_is_installed,  # pyright: ignore[reportPrivateUsage]
)
from karotte.load_tasks import (
    _get_task_loader as real_task_loader,  # pyright: ignore[reportPrivateUsage]
)
from karotte.load_tasks import require_environment

runner = CliRunner(mix_stderr=False)

NO_ENVIRONMENT = (
    "No karotte environment found. cd into one and run `uv run karotte ...` there."
)

COMMANDS = [
    ["tasks", "list"],
    ["create-run-config"],
    ["check"],
    ["run", "--config", "{}"],
    ["run", "--config", "{}", "--no-containerized"],
]


def _parse_config_unreachable():
    return patch(
        "karotte.cli.run.parse_config", side_effect=AssertionError("config parsed")
    )


@pytest.fixture(autouse=True)
def _real_task_loader(monkeypatch: pytest.MonkeyPatch):  # pyright: ignore[reportUnusedFunction]
    monkeypatch.setattr(karotte.load_tasks, "_get_task_loader", real_task_loader)
    monkeypatch.setattr(
        karotte.load_tasks, "_environment_is_installed", real_environment_is_installed
    )
    monkeypatch.delitem(sys.modules, "environment", raising=False)
    monkeypatch.delenv("KAROTTE_CONTAINERIZED", raising=False)
    monkeypatch.setattr("karotte.build.which", lambda _name: "/usr/bin/engine")  # pyright: ignore[reportUnknownLambdaType]
    monkeypatch.setattr("karotte.build._buildx_available", lambda: True)


def _in_karotte_image_for_no_containerized(
    monkeypatch: pytest.MonkeyPatch, args: list[str]
) -> None:
    if "--no-containerized" in args:
        monkeypatch.setenv("KAROTTE_CONTAINERIZED", "1")


@pytest.mark.parametrize("args", COMMANDS, ids=" ".join)
def test_outside_an_environment_prints_one_line(
    monkeypatch: pytest.MonkeyPatch, args: list[str]
):
    _in_karotte_image_for_no_containerized(monkeypatch, args)
    with _parse_config_unreachable():
        result = runner.invoke(app, args)

    assert result.exit_code == 1
    assert result.stderr.strip() == NO_ENVIRONMENT
    assert result.stdout == ""


def _install_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, source: str
) -> None:
    (tmp_path / "environment").mkdir()
    (tmp_path / "environment" / "__init__.py").write_text(source)
    monkeypatch.syspath_prepend(str(tmp_path))


@pytest.mark.parametrize(
    ("source", "error"),
    [
        ("import not_a_real_dependency_xyz\n", ModuleNotFoundError),
        ("def get_tasks(:\n", SyntaxError),
        ("", ImportError),
    ],
)
@pytest.mark.parametrize(
    "args", [["tasks", "list"], ["create-run-config"], ["check"]], ids=" ".join
)
def test_a_broken_environment_keeps_its_traceback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    args: list[str],
    source: str,
    error: type[Exception],
):
    _install_environment(monkeypatch, tmp_path, source)

    result = runner.invoke(app, args)

    assert isinstance(result.exception, error)
    assert NO_ENVIRONMENT not in result.stderr


@pytest.mark.parametrize("args", COMMANDS[3:], ids=" ".join)
def test_run_goes_on_with_a_broken_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, args: list[str]
):
    _install_environment(monkeypatch, tmp_path, "import not_a_real_dependency_xyz\n")
    _in_karotte_image_for_no_containerized(monkeypatch, args)

    with _parse_config_unreachable():
        result = runner.invoke(app, args)

    assert str(result.exception) == "config parsed"
    assert NO_ENVIRONMENT not in result.stderr


def test_the_check_does_not_run_environment_code(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    _install_environment(monkeypatch, tmp_path, "raise RuntimeError('imported')\n")

    require_environment()

    assert "environment" not in sys.modules
