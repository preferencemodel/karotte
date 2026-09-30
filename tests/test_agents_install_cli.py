"""Tests for the `karotte agents install` command."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer

from karotte.cli.agents import install


@pytest.fixture(autouse=True)
def in_karotte_image(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("KAROTTE_CONTAINERIZED", "1")


def test_install_refuses_outside_a_karotte_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("KAROTTE_CONTAINERIZED")
    with (
        patch("karotte.cli.agents.subprocess.run") as mock_run,
        pytest.raises(typer.Abort),
    ):
        install(manifest=_write_manifest(tmp_path, ["mistral-vibe"]))
    mock_run.assert_not_called()


def _install_and_capture_commands(manifest: Path) -> list[str]:
    with patch(
        "karotte.cli.agents.subprocess.run",
        return_value=MagicMock(returncode=0),
    ) as mock_run:
        install(manifest=manifest)
    return [call.args[0] for call in mock_run.call_args_list]


def _write_manifest(tmp_path: Path, agents: list[str]) -> Path:
    manifest = tmp_path / ".manifest.json"
    manifest.write_text(
        json.dumps(
            {"karotte_version": "0.0.0", "templates": ["default"], "agents": agents}
        )
    )
    return manifest


def test_install_runs_pinned_recipe_from_manifest(tmp_path: Path):
    from karotte.agents.mistral_vibe import MistralVibeAgent

    commands = _install_and_capture_commands(
        _write_manifest(tmp_path, ["mistral-vibe"])
    )
    assert any(f"mistral-vibe=={MistralVibeAgent.version}" in c for c in commands)


def test_empty_manifest_agents_is_a_noop(tmp_path: Path):
    assert _install_and_capture_commands(_write_manifest(tmp_path, [])) == []


def test_install_clears_group_other_write_on_the_agents_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mode-666 file in the agents dir loses group/other write; subprocess is real on purpose."""
    agents_dir = tmp_path / "agents"
    (agents_dir / "tools").mkdir(parents=True)
    lock = agents_dir / "tools" / ".lock"
    lock.write_text("")
    lock.chmod(0o666)

    class _StubAgent:
        version: str = "0.0.0"

        @classmethod
        def install(cls) -> list[str]:
            return ["true"]  # a real, harmless command

    monkeypatch.setattr("karotte.agents.cli_agent.AGENTS_DIR", str(agents_dir))
    monkeypatch.setattr(
        "karotte.agents.get_cli_agent_type",
        lambda _name: _StubAgent,  # pyright: ignore[reportUnknownLambdaType]
    )

    install(manifest=_write_manifest(tmp_path, ["stub"]))

    assert lock.stat().st_mode & 0o022 == 0, "group/other write still set"
    # readability must be preserved: the student has to be able to run the agent
    assert lock.stat().st_mode & 0o044 != 0
