"""Runtime layer: a process born with a poisoned loader env must re-exec so its work runs with a clean loader.

The real-loader tests need glibc, so they run on Linux only.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from karotte.schemas.evaluation_run_config import EvaluationRunConfig
from karotte.schemas.http_mcp_server_config import HttpMcpServerConfig

linux_only = pytest.mark.skipif(
    sys.platform != "linux", reason="depends on glibc dynamic loader semantics"
)

POISONED_LD = "/usr/local/nvidia/lib64:"  # trailing colon => empty component => cwd
CLEAN_LD = "/usr/local/nvidia/lib64"


@pytest.fixture
def plantdir(tmp_path: Path) -> Path:
    """A world-writable dir holding a planted `liblate.so.1`, like a student /workdir."""
    cc = shutil.which("cc") or shutil.which("gcc")
    if cc is None:
        pytest.skip("no C compiler to build the planted library")
    d = tmp_path / "workdir"
    d.mkdir()
    d.chmod(0o1777)
    src = tmp_path / "late.c"
    src.write_text("void planted(void) {}\n")
    subprocess.run(
        [cc, "-shared", "-fPIC", "-o", str(d / "liblate.so.1"), str(src)],
        check=True,
    )
    return d


def _run_worker(body: str, *, cwd: Path, ld_library_path: str) -> str:
    """Run a worker in a subprocess and return its LOADED/SAFE verdict."""
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = ld_library_path
    # Keep sanitize_paths() from taking its outside-a-container CI shortcut.
    env["KAROTTE_CONTAINERIZED"] = "1"
    env.pop("CI", None)
    src = str(Path(__file__).resolve().parent.parent / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    probe = (
        "import ctypes\n"
        "try:\n"
        "    ctypes.CDLL('liblate.so.1')\n"
        "    print('LOADED')\n"
        "except OSError:\n"
        "    print('SAFE')\n"
    )
    script = body + "\n" + probe
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip().splitlines()[-1]


def _record(order: list[str], label: str):
    def _side_effect(*_a: object, **_k: object) -> MagicMock:
        order.append(label)
        return MagicMock()

    return _side_effect


class TestRealLoaderProperty:
    @linux_only
    def test_poisoned_env_loads_from_cwd(self, plantdir: Path):
        """Control: the trailing colon lets cwd win, so the harness detects the hole."""
        assert _run_worker("", cwd=plantdir, ld_library_path=POISONED_LD) == "LOADED"

    @linux_only
    def test_build_layer_clean_env_is_sufficient(self, plantdir: Path):
        """Build fix alone: a clean image env is safe with no runtime help."""
        assert _run_worker("", cwd=plantdir, ld_library_path=CLEAN_LD) == "SAFE"

    @linux_only
    def test_runtime_layer_reexec_neutralizes_poisoned_env(self, plantdir: Path):
        """Runtime fix alone: re-exec makes a poisoned-born process safe."""
        body = "from karotte.run_helpers import sanitize_paths_and_reexec\nsanitize_paths_and_reexec()"
        assert _run_worker(body, cwd=plantdir, ld_library_path=POISONED_LD) == "SAFE"

    @linux_only
    def test_in_process_sanitize_alone_is_not_enough(self, plantdir: Path):
        """Regression guard: sanitizing os.environ without re-execing stays exposed."""
        body = "from karotte.run_helpers import sanitize_paths\nsanitize_paths()"
        assert _run_worker(body, cwd=plantdir, ld_library_path=POISONED_LD) == "LOADED"


class TestReexecHelper:
    @linux_only
    def test_reexec_is_idempotent(self, plantdir: Path):
        """The guard stops an exec loop: the second call returns."""
        body = (
            "from karotte.run_helpers import sanitize_paths_and_reexec\n"
            "sanitize_paths_and_reexec()\n"
            "sanitize_paths_and_reexec()"
        )
        assert _run_worker(body, cwd=plantdir, ld_library_path=POISONED_LD) == "SAFE"


class TestEntrypointsReexecBeforeWork:
    """Each entrypoint must re-exec (or spawn a sanitized child) before any dlopen."""

    def test_run_cli_reexecs_on_the_in_container_path(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        from karotte.cli.run import run

        monkeypatch.setenv("KAROTTE_CONTAINERIZED", "1")
        run_config = EvaluationRunConfig(
            run_id="test",
            task_id="example-task",
            model="vertex_ai/gemini",
            model_api_key=None,
            mcp_server_config=HttpMcpServerConfig(),
        )
        order: list[str] = []
        with (
            patch(
                "karotte.cli.run.sanitize_paths_and_reexec",
                side_effect=lambda: order.append("reexec"),
            ) as mock_reexec,
            patch("karotte.cli.run.harden_filesystem"),
            patch("karotte.cli.run.parse_config", return_value=run_config),
            patch(
                "karotte.cli.run.apply_run_config_preprocessors",
                return_value=run_config,
            ),
            patch(
                "karotte.cli.run.load_task",
                side_effect=_record(order, "load_task"),
            ),
            patch(
                "karotte.mcp_servers.http_mcp_server.run_server",
                return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()),
            ),
            patch("karotte.cli.run.anyio", run=MagicMock(return_value=None)),
        ):
            run(config="{}", containerized=True, no_ui=True)

        mock_reexec.assert_called_once()
        assert order and order[0] == "reexec", "must re-exec before loading the task"

    def test_http_child_reexecs_before_serving(self, monkeypatch: pytest.MonkeyPatch):
        import karotte.mcp_servers.http_mcp_server as mod

        monkeypatch.setattr(
            sys, "argv", ["prog", "127.0.0.1", "8123", "False", "False"]
        )
        order: list[str] = []
        with (
            patch(
                "karotte.run_helpers.sanitize_paths_and_reexec",
                side_effect=lambda: order.append("reexec"),
            ) as mock_reexec,
            patch("karotte.container.is_containerized", return_value=True),
            patch.object(mod, "HttpMcpServer", side_effect=_record(order, "serve")),
        ):
            mod._subprocess_entrypoint()  # pyright: ignore[reportPrivateUsage]

        mock_reexec.assert_called_once()
        assert order[0] == "reexec", "must re-exec before building the server"
