"""The triage script of the harbor-to-karotte skill."""

import importlib.util
from pathlib import Path
from types import ModuleType

SCRIPT = (
    Path(__file__).parent.parent
    / "skills"
    / "harbor-to-karotte"
    / "scripts"
    / "inventory.py"
)


def _inventory() -> ModuleType:
    spec = importlib.util.spec_from_file_location("inventory", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _task(root: Path, name: str, toml: str, dockerfile: str, solve: str) -> Path:
    task = root / name
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "solution").mkdir()
    (task / "task.toml").write_text(f'[task]\nname = "org/{name}"\n{toml}')
    (task / "environment" / "Dockerfile").write_text(dockerfile)
    (task / "tests" / "test_outputs.py").write_text("def test_ok():\n    pass\n")
    (task / "solution" / "solve.sh").write_text(solve)
    return task


def test_plain_task_is_portable(tmp_path: Path):
    task = _task(
        tmp_path,
        "plain",
        "[environment]\ncpus = 2\nmemory_mb = 4096\n",
        "FROM ubuntu:24.04\nWORKDIR /app\n",
        "echo 42 > /app/answer.txt\n",
    )
    row = _inventory().inspect(task)
    assert row["verdict"] == "portable"
    assert row["flags"] == []
    assert (row["cpus"], row["memory_mb"], row["gpus"]) == (2, 4096, 0)


def test_gpu_task_is_deferred(tmp_path: Path):
    task = _task(
        tmp_path,
        "gpu",
        "[environment]\ngpus = 1\n",
        "FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04\n",
        "sudo apt-get install -y foo\n",
    )
    row = _inventory().inspect(task)
    assert row["verdict"] == "skip-or-defer"
    assert "gpus=1" in row["flags"]
    assert "root-in-solution" in row["flags"]
    assert any(f.startswith("custom-base:") for f in row["flags"])


def test_notes_do_not_change_the_verdict(tmp_path: Path):
    task = _task(tmp_path, "noted", "", "FROM python:3.12\n", "true\n")
    (task / "tests" / "Dockerfile").write_text("FROM python:3.12\n")
    (task / "cheat").mkdir()
    row = _inventory().inspect(task)
    assert row["verdict"] == "portable"
    assert row["flags"] == ["verifier-image", "extra-dirs:cheat"]
