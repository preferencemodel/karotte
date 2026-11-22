import ast
import os
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import pytest

from karotte.plugin_path import PLUGIN_MOUNT_ROOT, PLUGIN_PATH_ENV, plugin_path_env


def _plugin(
    root: Path,
    package: str,
    agent: str,
    requires: str | None = None,
    *,
    decorated: bool = False,
) -> Path:
    (root / package).mkdir(parents=True)
    (root / package / "__init__.py").write_text(
        "from typing import ClassVar\n"
        + "from karotte.agents import register_cli_agent\n"
        + "from karotte.agents.claude_code import ClaudeCodeAgent\n\n\n"
        + ("@register_cli_agent\n" if decorated else "")
        + "class Agent(ClaudeCodeAgent):\n"
        + f"    name: ClassVar[str] = {agent!r}\n"
    )
    dist_info = root / f"{package}-0.1.dist-info"
    dist_info.mkdir()
    metadata = f"Metadata-Version: 2.1\nName: {package}\nVersion: 0.1\n"
    if requires is not None:
        metadata += f"Requires-Dist: {requires}\n"
    (dist_info / "METADATA").write_text(metadata)
    (dist_info / "entry_points.txt").write_text(
        f"[karotte.cli_agents]\n{package} = {package}:Agent\n"
    )
    return root


def _python(
    code: str,
    plugin_path: str,
    python_path: str | None = None,
    student: int | None = None,
) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, PLUGIN_PATH_ENV: plugin_path}
    if python_path is not None:
        env["PYTHONPATH"] = python_path
    if student is not None:
        env["KAROTTE_DEMOTE_ID"] = str(student)
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env
    )


def _agent_module(
    agent: str,
    plugin_path: str,
    python_path: str | None = None,
    student: int | None = None,
) -> subprocess.CompletedProcess[str]:
    return _python(
        "from karotte.agents.cli_agent import get_cli_agent_type; "
        + f"print(get_cli_agent_type({agent!r}).__module__)",
        plugin_path,
        python_path,
        student,
    )


def test_an_agent_on_the_plugin_path_can_be_used(tmp_path: Path):
    plugin = _plugin(tmp_path / "plugin", "probe_plugin", "probe")

    result = _agent_module("probe", str(plugin))

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "probe_plugin"


def test_the_plugin_path_takes_several_folders(tmp_path: Path):
    a = _plugin(tmp_path / "a", "plugin_a", "agent-a")
    b = _plugin(tmp_path / "b", "plugin_b", "agent-b")

    for agent, module in (("agent-a", "plugin_a"), ("agent-b", "plugin_b")):
        result = _agent_module(agent, f"{a}:{b}")
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == module


def test_the_plugin_path_wins_over_installed_packages(tmp_path: Path):
    installed = _plugin(tmp_path / "installed", "installed_plugin", "probe")
    mounted = _plugin(tmp_path / "mounted", "mounted_plugin", "probe")

    result = _agent_module("probe", str(mounted), python_path=str(installed))

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "mounted_plugin"


def test_the_plugin_path_wins_over_a_package_that_registers_itself(tmp_path: Path):
    installed = _plugin(tmp_path / "installed", "a_installed", "probe", decorated=True)
    mounted = _plugin(tmp_path / "mounted", "z_mounted", "probe")

    result = _agent_module("probe", str(mounted), python_path=str(installed))

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "z_mounted"


def test_a_plugin_that_registers_itself_loads(tmp_path: Path):
    plugin = _plugin(tmp_path / "plugin", "probe_plugin", "probe", decorated=True)

    result = _agent_module("probe", str(plugin))

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "probe_plugin"
    assert "is taken" not in result.stderr


def test_plugin_folders_come_first_in_their_order(tmp_path: Path):
    a, b, other = tmp_path / "a", tmp_path / "b", tmp_path / "other"
    for folder in (a, b, other):
        folder.mkdir()

    result = _python(
        "import sys, karotte; print(sys.path)",
        f"{a}:{b}",
        python_path=f"{other}:{a}",
    )

    assert result.returncode == 0, result.stderr
    sys_path: list[str] = ast.literal_eval(result.stdout)
    assert sys_path[:2] == [str(a), str(b)]
    assert sys_path.count(str(a)) == 1
    assert sys_path.index(str(other)) > 1


def test_a_relative_folder_stops_karotte():
    result = _agent_module("probe", "plugins")

    assert result.returncode != 0
    assert f"{PLUGIN_PATH_ENV} entry plugins isn't an absolute path" in result.stderr


def test_a_malformed_requirement_names_the_plugin(tmp_path: Path):
    plugin = _plugin(tmp_path / "plugin", "probe_plugin", "probe", "karotte >>> 1")

    result = _agent_module("probe", str(plugin))

    assert result.returncode != 0
    assert f"probe_plugin in {plugin} has an invalid Requires-Dist" in result.stderr


def test_a_plugin_for_another_karotte_stops_karotte(tmp_path: Path):
    plugin = _plugin(tmp_path / "plugin", "probe_plugin", "probe", "karotte>=999")

    result = _agent_module("probe", str(plugin))

    assert result.returncode != 0
    assert (
        f"probe_plugin in {plugin} needs karotte>=999, but karotte {version('karotte')} is installed"
        in result.stderr
    )


@pytest.mark.parametrize(
    "requires",
    [
        f"karotte>={version('karotte')}",
        f"Karotte[modal]>={version('karotte')}",
        'karotte>=999; extra == "dev"',
        'karotte>=999; python_version < "3"',
    ],
)
def test_a_plugin_for_this_karotte_loads(tmp_path: Path, requires: str):
    plugin = _plugin(tmp_path / "plugin", "probe_plugin", "probe", requires)

    result = _agent_module("probe", str(plugin))

    assert result.returncode == 0, result.stderr


def test_a_missing_plugin_folder_stops_karotte(tmp_path: Path):
    result = _agent_module("probe", str(tmp_path / "missing"))

    assert result.returncode != 0
    assert f"{PLUGIN_PATH_ENV} names {tmp_path / 'missing'}, which isn't a folder" in (
        result.stderr
    )


def test_a_folder_the_student_can_reach_stops_karotte(tmp_path: Path):
    plugin = _plugin(tmp_path / "plugin", "probe_plugin", "probe")

    result = _agent_module("probe", str(plugin), student=os.getuid())

    assert result.returncode != 0
    assert (
        f"{PLUGIN_PATH_ENV} names {plugin}, which the student (uid {os.getuid()}) can reach"
        in result.stderr
    )


def test_a_folder_hidden_from_the_student_loads(tmp_path: Path):
    locked = tmp_path / "root"
    locked.mkdir()
    locked.chmod(0o700)
    plugin = _plugin(locked / "plugin", "probe_plugin", "probe")

    result = _agent_module("probe", str(plugin), student=os.getuid() + 54321)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "probe_plugin"


def test_the_sandbox_gets_the_plugin_mounts():
    assert plugin_path_env(
        ["/data", f"{PLUGIN_MOUNT_ROOT}/0", f"{PLUGIN_MOUNT_ROOT}/1"]
    ) == {PLUGIN_PATH_ENV: f"{PLUGIN_MOUNT_ROOT}/0:{PLUGIN_MOUNT_ROOT}/1"}


def test_no_plugin_mounts_set_nothing():
    assert plugin_path_env(["/data"]) == {}
