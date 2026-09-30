import sys
from collections.abc import Mapping
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer

from karotte.cli.check import check
from karotte.task import Task


def _make_task_class(task_id: str) -> type[Task]:
    """Create a minimal Task subclass with the given id."""
    return type(
        "FakeTask",
        (Task,),
        {
            "id": task_id,
            "steps": property(lambda self: []),
            "tools": property(lambda self: []),
        },
    )


@patch("karotte.cli.check.load_all_task_classes")
def test_check_passes_with_short_task_name(
    mock_load: MagicMock, capsys: pytest.CaptureFixture[str]
) -> None:
    mock_load.return_value = [_make_task_class("my-task")]
    check()
    assert capsys.readouterr().out.splitlines()[-1] == "Check passed."


@patch("karotte.cli.check.load_all_task_classes")
def test_check_fails_when_task_name_exceeds_255_chars(mock_load: MagicMock) -> None:
    long_name = "a" * 256
    mock_load.return_value = [_make_task_class(long_name)]
    with pytest.raises(typer.Exit) as exc_info:
        check()
    assert exc_info.value.exit_code == 1


@patch("karotte.cli.check.load_all_task_classes")
def test_check_passes_with_task_name_exactly_255_chars(mock_load: MagicMock) -> None:
    name_255 = "a" * 255
    mock_load.return_value = [_make_task_class(name_255)]
    check()


@patch("karotte.cli.check.load_all_task_classes")
def test_check_fails_when_any_task_name_exceeds_255_chars(mock_load: MagicMock) -> None:
    mock_load.return_value = [
        _make_task_class("ok-task"),
        _make_task_class("b" * 256),
    ]
    with pytest.raises(typer.Exit) as exc_info:
        check()
    assert exc_info.value.exit_code == 1


class TestCheckRunsTheCredentialGate:
    """The gate only runs inside a container, so nothing else covers this wiring."""

    @pytest.fixture
    def _containerized(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # By path this resolves to `karotte.cli`'s re-exported `check` function,
        # not the module, so patch the module object out of sys.modules.
        monkeypatch.setattr(
            sys.modules["karotte.cli.check"], "is_containerized", lambda: True
        )

    @patch("karotte.cli.check.load_all_task_classes")
    def test_check_fails_when_the_image_ships_a_credential(
        self,
        mock_load: MagicMock,
        _containerized: None,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_load.return_value = [_make_task_class("my-task")]
        token = tmp_path / ".local/share/uv/credentials/3859a629/tokens.json"
        token.parent.mkdir(parents=True)
        token.write_text('{"access_token": "eyJ"}')

        def only_tmp_path(_env: Mapping[str, str] | None = None) -> list[Path]:
            return [tmp_path]

        monkeypatch.setattr("karotte.check_credentials.home_directories", only_tmp_path)

        with pytest.raises(typer.Exit) as exc_info:
            check()

        assert exc_info.value.exit_code == 1
        assert "tokens.json" in capsys.readouterr().out


class TestCheckWithEnvironmentTools:
    @pytest.fixture(autouse=True)
    def _clean_environment_modules(self):
        yield
        for name in [k for k in sys.modules if k.startswith("environment")]:
            del sys.modules[name]

    @patch("karotte.cli.check.load_all_task_classes")
    def test_lists_tool_whose_config_has_required_fields(
        self,
        mock_load: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A tool is only configured at task setup, so check must not need its config."""
        mock_load.return_value = [_make_task_class("my-task")]

        env_tools_dir = tmp_path / "environment" / "tools"
        env_tools_dir.mkdir(parents=True)
        (tmp_path / "environment" / "__init__.py").write_text("")
        (env_tools_dir / "__init__.py").write_text("")
        (env_tools_dir / "needy.py").write_text('''
from typing import final

from fastmcp.tools.tool import ToolResult
from karotte import ToolBase
from pydantic import BaseModel, Field


class NeedyConfig(BaseModel):
    submissions: list[str] = Field(min_length=1)


@final
class needy(ToolBase[NeedyConfig]):
    """A tool whose config a task must write before it can be built."""

    config_schema = NeedyConfig

    async def __call__(self) -> ToolResult:
        """Does nothing."""
        return ToolResult(structured_content={"ok": True})
''')
        monkeypatch.syspath_prepend(str(tmp_path))
        for name in [k for k in sys.modules if k.startswith("environment")]:
            del sys.modules[name]

        check()

        assert "needy" in capsys.readouterr().out
