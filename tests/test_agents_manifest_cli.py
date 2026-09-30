"""Tests for the `karotte agents add` / `remove` commands."""

import json
from pathlib import Path

import pytest
import typer

from karotte.cli.agents import add, remove
from karotte.create_env import EnvManifest


def _write_manifest(tmp_path: Path, agents: list[str]) -> Path:
    manifest = tmp_path / ".manifest.json"
    manifest.write_text(
        json.dumps(
            {"karotte_version": "0.0.0", "templates": ["default"], "agents": agents}
        )
    )
    return manifest


def _agents(tmp_path: Path) -> list[str]:
    manifest = tmp_path / ".manifest.json"
    return EnvManifest.model_validate_json(manifest.read_text()).agents


def test_add_appends_known_agent(tmp_path: Path):
    _write_manifest(tmp_path, [])
    add("mistral-vibe", project_dir=tmp_path)
    assert _agents(tmp_path) == ["mistral-vibe"]


def test_add_is_idempotent(tmp_path: Path):
    _write_manifest(tmp_path, ["mistral-vibe"])
    add("mistral-vibe", project_dir=tmp_path)
    assert _agents(tmp_path) == ["mistral-vibe"]


def test_add_rejects_unknown_agent(tmp_path: Path):
    _write_manifest(tmp_path, [])
    with pytest.raises(ValueError, match="Unknown agent"):
        add("nope", project_dir=tmp_path)
    assert _agents(tmp_path) == []


def test_remove_drops_agent(tmp_path: Path):
    _write_manifest(tmp_path, ["mistral-vibe"])
    remove("mistral-vibe", project_dir=tmp_path)
    assert _agents(tmp_path) == []


def test_remove_absent_agent_is_a_noop(tmp_path: Path):
    _write_manifest(tmp_path, [])
    remove("mistral-vibe", project_dir=tmp_path)
    assert _agents(tmp_path) == []


def test_missing_manifest_raises(tmp_path: Path):
    with pytest.raises(typer.BadParameter, match="No .manifest.json"):
        add("mistral-vibe", project_dir=tmp_path)
