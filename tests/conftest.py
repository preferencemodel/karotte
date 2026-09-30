import json
import os
import re
import shutil
import socket
import sys
from collections.abc import AsyncGenerator, Callable
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast, final, override

import pytest
import pytest_asyncio

import karotte.load_tasks
from karotte import Step, Task
from karotte.hardware import (
    CONTAINER_RUN_ARGS_ENTRY_POINT_GROUP,
    DEFAULT_HARDWARE_ENTRY_POINT_GROUP,
    HARDWARE_LIMITS_ENTRY_POINT_GROUP,
)
from karotte.judges.regex_judge import RegexJudge
from karotte.mcp_servers.http_mcp_server import HttpMcpServer, run_server
from karotte.providers import SERVICE_TIER_ENV
from karotte.schemas.evaluation_run_config import EvaluationRunConfig
from karotte.schemas.http_mcp_server_config import HttpMcpServerConfig
from karotte.schemas.transcript import Transcript
from karotte.subprocess import (
    HARNESS_SECRET_ENTRY_POINT_GROUP,
    PLATFORM_TOOLING_ENTRY_POINT_GROUP,
)
from karotte.tools.bash import _BashSession, bash  # pyright: ignore[reportPrivateUsage]
from karotte.update_env import INDEX_OVERRIDE_VARS

# The characterization env is an `environment` package that only the MCP server
# subprocess may import (via PYTHONPATH). Importing it in the test process (e.g.
# through --doctest-modules collection) would make `environment.tools` visible
# to in-process `discover_tools` calls.
collect_ignore = ["resources/characterization_env"]


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    if sys.platform == "linux" and os.geteuid() == 0:
        return
    skip = pytest.mark.skip(reason="needs Linux + root")
    for item in items:
        if "requires_root" in item.keywords:
            item.add_marker(skip)


class Step42(Step):
    @property
    @override
    def instructions(self) -> str:
        return "Step 42."

    @property
    @override
    def judge(self) -> RegexJudge:
        return RegexJudge([re.compile(r"answer.*\b42\b")])


class Step43(Step):
    @property
    @override
    def instructions(self) -> str:
        return "Step 43."

    @property
    @override
    def judge(self) -> RegexJudge:
        return RegexJudge([re.compile(r"answer.*\b43\b")])


@final
class TestTask(Task):
    __test__ = False
    id = "test-task"

    @property
    @override
    def system_prompt(self) -> str | None:
        return None

    @property
    @override
    def tools(self):
        return ["bash"]

    @property
    @override
    def steps(self):
        return [
            Step42(config=self.config),
            Step43(config=self.config),
        ]


def find_free_port() -> int:
    """Find a free port on localhost."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.listen(1)
        port = cast(int, s.getsockname()[1])
    return port


_CHARACTERIZATION_ENV_DIR = Path(__file__).parent / "resources" / "characterization_env"


@pytest_asyncio.fixture
async def characterization_mcp_server(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncGenerator[HttpMcpServer]:
    """A real MCP server subprocess that can discover the deterministic
    characterization tools via `environment.tools` on its PYTHONPATH."""
    existing = os.environ.get("PYTHONPATH")
    pythonpath = str(_CHARACTERIZATION_ENV_DIR)
    if existing:
        pythonpath = f"{pythonpath}{os.pathsep}{existing}"
    monkeypatch.setenv("PYTHONPATH", pythonpath)

    config = HttpMcpServerConfig(host="0.0.0.0", port=find_free_port())
    with run_server(config) as server:
        yield server


@pytest_asyncio.fixture(scope="function")
async def mcp_server() -> AsyncGenerator[HttpMcpServer]:
    free_port = find_free_port()
    config = HttpMcpServerConfig(host="0.0.0.0", port=free_port)

    with run_server(config) as server:
        yield server


@pytest_asyncio.fixture
async def bash_tool(monkeypatch: pytest.MonkeyPatch):
    # Production hardcodes the hardened absolute path /usr/bin/bash. That path
    # does not exist on every dev platform (e.g. macOS ships bash elsewhere and
    # its system /bin/bash is too old to import BASHOPTS from the environment),
    # so point the session at a real, modern bash when it is missing.
    if not Path(*_BashSession._command).exists():  # pyright: ignore[reportPrivateUsage]
        bash_path = shutil.which("bash")
        if bash_path is None:
            pytest.skip("no bash binary found on PATH")
        monkeypatch.setattr(_BashSession, "_command", [bash_path])

    tool = bash()
    try:
        yield tool
    finally:
        await tool.dispose()


@pytest.fixture
def transcript() -> Transcript:
    return Transcript(run_id="run_id")


@pytest.fixture
def sample_config():
    return EvaluationRunConfig(
        run_id="test_run",
        task_id="test-task",
        model="test_model",
        model_api_key="test_key",
        mcp_server_config=HttpMcpServerConfig(
            host="0.0.0.0",
            port=8080,
        ),
        transcript_file="out/transcript.json",
    )


@pytest.fixture
def test_task(sample_config: EvaluationRunConfig) -> TestTask:
    return TestTask(sample_config)


@pytest.fixture(autouse=True)
def fresh_confinement():
    """Sandbox detection and the confinement are cached per process, but tests
    flip the env vars they are derived from. The vars themselves are restored
    too: ``_set_sandbox_env`` writes them to ``os.environ`` directly, past
    monkeypatch's bookkeeping."""
    from karotte.cgroups import (
        _student_groups,  # pyright: ignore[reportPrivateUsage]
    )
    from karotte.confinement import (
        _confinement_for,  # pyright: ignore[reportPrivateUsage]
        current_sandbox,
    )

    saved = {var: os.environ.get(var) for var in ("KAROTTE_SANDBOX", "KAROTTE_GVISOR")}
    current_sandbox.cache_clear()
    _confinement_for.cache_clear()
    _student_groups.clear()
    yield
    for var, value in saved.items():
        if value is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = value
    current_sandbox.cache_clear()
    _confinement_for.cache_clear()
    _student_groups.clear()


@pytest.fixture(autouse=True)
def test_env(monkeypatch: pytest.MonkeyPatch):
    """Fixture to set up test environment variables."""

    def get_task_loader_mock():
        def loader() -> list[type[Task]]:
            return [TestTask]

        return loader

    monkeypatch.setattr(karotte.load_tasks, "_get_task_loader", get_task_loader_mock)
    monkeypatch.setattr(karotte.load_tasks, "_environment_is_installed", lambda: True)
    for var in INDEX_OVERRIDE_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv(SERVICE_TIER_ENV, raising=False)
    yield


@pytest.fixture
def assert_matches_golden() -> Callable[[str, Any], None]:
    """Compare data against tests/resources/golden/<name>, or rewrite it when
    UPDATE_GOLDEN is set."""

    def compare(name: str, data: Any) -> None:
        path = Path(__file__).parent / "resources" / "golden" / name
        if os.environ.get("UPDATE_GOLDEN"):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2) + "\n")
        assert path.exists(), (
            f"Golden file {path} is missing. Generate it with UPDATE_GOLDEN=1."
        )
        assert data == json.loads(path.read_text()), (
            f"Behavior differs from golden file {name}. If the change is "
            "intentional, regenerate with UPDATE_GOLDEN=1 and review the diff."
        )

    return compare


@pytest.fixture(scope="session", name="resource_dir")
def resource_dir_fixture() -> Path:
    """Returns the path to the test resource directory.

    Returns:
        LocalPath: The resource directory path.
    """
    return Path(__file__).parent.joinpath("resources")


def register_harness_secrets(monkeypatch: pytest.MonkeyPatch, **names: object) -> None:
    _register_subprocess_entry_points(
        monkeypatch, HARNESS_SECRET_ENTRY_POINT_GROUP, **names
    )


def register_platform_tooling_dirs(
    monkeypatch: pytest.MonkeyPatch, **names: object
) -> None:
    _register_subprocess_entry_points(
        monkeypatch, PLATFORM_TOOLING_ENTRY_POINT_GROUP, **names
    )


def _register_subprocess_entry_points(
    monkeypatch: pytest.MonkeyPatch, expected_group: str, **names: object
) -> None:
    def fake_entry_points(*, group: str) -> list[SimpleNamespace]:
        assert group == expected_group

        def loader(value: object) -> object:
            if isinstance(value, Exception):
                raise value
            return value

        return [
            SimpleNamespace(name=name, load=lambda v=value: loader(v))
            for name, value in names.items()
        ]

    monkeypatch.setattr("karotte.subprocess.entry_points", fake_entry_points)


def register_hardware_plugins(
    monkeypatch: pytest.MonkeyPatch,
    *,
    default: dict[str, object] | None = None,
    limits: dict[str, object] | None = None,
    container_run_args: dict[str, object] | None = None,
) -> None:
    """Replace the installed hardware plugins; each dict maps entry point name to what it loads."""
    groups = {
        DEFAULT_HARDWARE_ENTRY_POINT_GROUP: default or {},
        HARDWARE_LIMITS_ENTRY_POINT_GROUP: limits or {},
        CONTAINER_RUN_ARGS_ENTRY_POINT_GROUP: container_run_args or {},
    }

    def loader(value: object) -> object:
        if isinstance(value, Exception):
            raise value
        return value

    def fake_entry_points(*, group: str) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(name=name, load=lambda v=value: loader(v))
            for name, value in groups[group].items()
        ]

    monkeypatch.setattr("karotte.hardware.entry_points", fake_entry_points)
