from types import SimpleNamespace
from typing import Any

import pytest
import typer
from typer.testing import CliRunner

from karotte.cli import PLUGIN_ENTRY_POINT_GROUP, add_plugin_commands

runner = CliRunner()


def _plugin(message: str) -> typer.Typer:
    plugin = typer.Typer()

    @plugin.command()
    def hello() -> None:  # pyright: ignore[reportUnusedFunction]
        print(message)

    @plugin.command()
    def other() -> None:  # pyright: ignore[reportUnusedFunction]
        pass

    return plugin


def _app_with_builtin() -> typer.Typer:
    app = typer.Typer()

    @app.command()
    def run() -> None:  # pyright: ignore[reportUnusedFunction]
        print("builtin run")

    @app.command()
    def check() -> None:  # pyright: ignore[reportUnusedFunction]
        pass

    return app


def _register(monkeypatch: pytest.MonkeyPatch, **loaders: Any) -> None:
    def fake_entry_points(*, group: str) -> list[SimpleNamespace]:
        assert group == PLUGIN_ENTRY_POINT_GROUP
        return [SimpleNamespace(name=name, load=load) for name, load in loaders.items()]

    monkeypatch.setattr("karotte.cli.entry_points", fake_entry_points)


def test_installed_packages_add_command_groups(monkeypatch: pytest.MonkeyPatch):
    _register(monkeypatch, extra=lambda: _plugin("from the plugin"))
    app = _app_with_builtin()

    add_plugin_commands(app)

    result = runner.invoke(app, ["extra", "hello"])
    assert result.exit_code == 0, result.output
    assert "from the plugin" in result.output


def test_a_plugin_cannot_replace_a_builtin_command(monkeypatch: pytest.MonkeyPatch):
    _register(monkeypatch, run=lambda: _plugin("hijacked"))
    app = _app_with_builtin()

    add_plugin_commands(app)

    result = runner.invoke(app, ["run"])
    assert "builtin run" in result.output


def test_a_broken_plugin_leaves_the_others_working(monkeypatch: pytest.MonkeyPatch):
    def broken() -> typer.Typer:
        raise ImportError("missing dependency")

    _register(monkeypatch, broken=broken, extra=lambda: _plugin("still here"))
    app = _app_with_builtin()

    add_plugin_commands(app)

    assert "still here" in runner.invoke(app, ["extra", "hello"]).output
    assert runner.invoke(app, ["run"]).exit_code == 0
