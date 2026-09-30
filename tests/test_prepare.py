"""Tests for --prepare-only mode: set up an env up to the step loop and hold it open."""

import os
import shutil
import subprocess
import sys
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, final, override
from unittest.mock import patch

import anyio
import pytest

import karotte.run_helpers
from karotte import Step, Task
from karotte.cli.run import _run_without_ui  # pyright: ignore[reportPrivateUsage]
from karotte.evaluation_runner import EvaluationRunner
from karotte.judges.always_pass_judge import AlwaysPassJudge
from karotte.mcp_servers.http_mcp_server import HttpMcpServer
from karotte.run_helpers import (
    get_container_run_command,
    hold_prepared_env,
    prepare_non_containerized,
    prepared_sentinel_path,
    run_containerized,
)
from karotte.schemas.evaluation_run_config import EvaluationRunConfig
from karotte.schemas.http_mcp_server_config import HttpMcpServerConfig
from karotte.schemas.transcript import (
    Event,
    MessageAddedEvent,
    ScoringEvent,
    StepStartedEvent,
    TaskPreHookCompletedEvent,
    TaskStartedEvent,
)


class _PrepareStep(Step):
    @property
    @override
    def instructions(self) -> str:
        return "This step must never run in prepare mode."

    @property
    @override
    def judge(self) -> AlwaysPassJudge:
        return AlwaysPassJudge()


@final
class _PrepareTask(Task):
    id = "prepare-task"

    @property
    @override
    def system_prompt(self) -> str:
        return "Prepare-mode system prompt."

    @property
    @override
    def tools(self) -> list[str]:
        return ["echo"]

    @property
    @override
    def steps(self) -> list[Step]:
        return [_PrepareStep(config=self.config)]

    @override
    def pre_hook(self) -> dict[str, Any]:
        return {"prepared": True}


@final
class _FailingTask(Task):
    @property
    @override
    def system_prompt(self) -> str | None:
        return None

    id = "failing-task"

    @property
    @override
    def tools(self) -> list[str]:
        return ["echo"]

    @property
    @override
    def steps(self) -> list[Step]:
        return [_PrepareStep(config=self.config)]

    @override
    def pre_hook(self) -> dict[str, Any]:
        raise RuntimeError("pre_hook boom")


def _make_config(
    sample_config: EvaluationRunConfig,
    mcp_server: HttpMcpServer,
    run_id: str,
    task_id: str,
) -> EvaluationRunConfig:
    return sample_config.model_copy(
        update={
            "run_id": run_id,
            "task_id": task_id,
            "transcript_file": None,
            "mcp_server_config": HttpMcpServerConfig(port=mcp_server.config.port),
        }
    )


@pytest.mark.asyncio
async def test_prepare_runs_setup_but_not_step_loop(
    characterization_mcp_server: HttpMcpServer, sample_config: EvaluationRunConfig
):
    """prepare() does everything up to the step loop — tool loading, pre_hook,
    system prompt — and stops: no step, no judge, no scoring."""
    config = _make_config(
        sample_config, characterization_mcp_server, "prepare-run", "prepare-task"
    )
    task = _PrepareTask(config)
    runner = EvaluationRunner(config, task)

    events: list[Event] = [event async for event in runner.prepare()]

    assert [type(e) for e in events] == [
        TaskStartedEvent,
        TaskPreHookCompletedEvent,
        MessageAddedEvent,
    ]
    pre_hook_event = events[1]
    assert isinstance(pre_hook_event, TaskPreHookCompletedEvent)
    assert pre_hook_event.metadata == {"prepared": True}

    assert [tool["function"]["name"] for tool in runner.tools] == ["echo"]

    system_message = events[2]
    assert isinstance(system_message, MessageAddedEvent)
    assert system_message.message.role == "system"
    assert system_message.message.content == "Prepare-mode system prompt."

    assert not any(isinstance(e, StepStartedEvent) for e in events)
    assert not any(isinstance(e, ScoringEvent) for e in events)


@pytest.mark.asyncio
async def test_prepare_preserves_existing_transcript_file(
    characterization_mcp_server: HttpMcpServer,
    sample_config: EvaluationRunConfig,
    tmp_path: Path,
):
    """prepare() is inspection-only: a transcript file left by an earlier real
    run is left untouched, and prepare writes nothing to disk."""
    transcript_file = tmp_path / "transcript.json"
    transcript_file.write_text('{"old": true}')
    config = _make_config(
        sample_config,
        characterization_mcp_server,
        "prepare-transcript-run",
        "prepare-task",
    ).model_copy(update={"transcript_file": str(transcript_file)})
    runner = EvaluationRunner(config, _PrepareTask(config))

    async for _ in runner.prepare():
        pass

    assert transcript_file.read_text() == '{"old": true}'


@pytest.mark.asyncio
async def test_prepare_propagates_pre_hook_failure(
    characterization_mcp_server: HttpMcpServer, sample_config: EvaluationRunConfig
):
    """A failing pre_hook is not swallowed: prepare() raises instead of idling."""
    config = _make_config(
        sample_config, characterization_mcp_server, "prepare-fail-run", "failing-task"
    )
    runner = EvaluationRunner(config, _FailingTask(config))

    with pytest.raises(RuntimeError, match="pre_hook boom"):
        async for _ in runner.prepare():
            pass


@pytest.mark.asyncio
async def test_prepare_non_containerized_failure_writes_no_sentinel(
    characterization_mcp_server: HttpMcpServer, sample_config: EvaluationRunConfig
):
    """Setup failure exits (raises) without writing the readiness marker or idling."""
    run_id = "prepare-fail-nc-run"
    config = _make_config(
        sample_config, characterization_mcp_server, run_id, "failing-task"
    )
    sentinel = prepared_sentinel_path(run_id)
    sentinel.unlink(missing_ok=True)

    try:
        with pytest.raises(RuntimeError, match="pre_hook boom"):
            await prepare_non_containerized(config, _FailingTask(config))
        assert not sentinel.exists()
    finally:
        sentinel.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_prepare_non_containerized_writes_sentinel_and_holds(
    characterization_mcp_server: HttpMcpServer, sample_config: EvaluationRunConfig
):
    """A stale marker is removed during setup, the run-id-scoped sentinel
    appears once prepared, the process idles until cancelled, and the sentinel
    is cleaned up on the way out."""
    run_id = "prepare-hold-run"
    config = _make_config(
        sample_config, characterization_mcp_server, run_id, "prepare-task"
    )
    task = _PrepareTask(config)
    sentinel = prepared_sentinel_path(run_id)
    # Leftover from a previous, killed hold: must not read as ready.
    sentinel.write_text("stale")

    returned = False

    async def _run_prepare():
        nonlocal returned
        await prepare_non_containerized(config, task)
        returned = True

    def _ready() -> bool:
        try:
            return sentinel.read_text() == run_id
        except FileNotFoundError:
            return False

    try:
        with anyio.fail_after(30):
            async with anyio.create_task_group() as tg:
                tg.start_soon(_run_prepare)

                while not _ready():
                    await anyio.sleep(0.05)

                # Still holding: give it a moment and confirm it hasn't returned.
                await anyio.sleep(0.2)
                assert not returned

                tg.cancel_scope.cancel()

        assert not returned
        assert not sentinel.exists()
    finally:
        sentinel.unlink(missing_ok=True)


def test_container_command_forwards_prepare_only(sample_config: EvaluationRunConfig):
    """A containerized `karotte run --prepare-only` forwards the flag into the
    in-container invocation and skips publishing the websocket port, which a
    held env never uses."""
    command, _ = get_container_run_command(
        sample_config, "podman", dev=False, keep_container=False, prepare_only=True
    )
    assert command.index("--prepare-only") > command.index("--no-containerized")
    assert "--publish" not in command

    command, _ = get_container_run_command(
        sample_config, "podman", dev=False, keep_container=False
    )
    assert "--prepare-only" not in command
    assert "--publish" in command


def test_run_containerized_tolerates_nonzero_exit_when_prepare_only(
    sample_config: EvaluationRunConfig,
):
    """A held container is torn down by an external stop, so `podman run` exits
    non-zero; run_containerized must not treat that as a failure (check=False)."""
    captured: dict[str, Any] = {}

    def _capture(_cmd: list[str], **kwargs: Any) -> None:
        captured.update(kwargs)

    with patch("karotte.run_helpers.subprocess.run", side_effect=_capture):
        run_containerized(sample_config, runtime="podman", dev=False, prepare_only=True)
    assert captured["check"] is False

    captured.clear()
    with patch("karotte.run_helpers.subprocess.run", side_effect=_capture):
        run_containerized(sample_config, runtime="podman", dev=False)
    assert captured["check"] is True


def test_run_without_ui_tears_down_containers_on_interrupt_when_prepare_only(
    sample_config: EvaluationRunConfig,
):
    """An interrupt while holding a prepare-only container stops the containers
    (so the blocking `podman run` returns) and exits cleanly instead of
    propagating."""
    with (
        patch("karotte.cli.run.clean_up_old_containers"),
        patch("karotte.cli.run.run_containerized", side_effect=KeyboardInterrupt),
        patch("karotte.cli.run.stop_containers") as mock_stop,
    ):
        _run_without_ui(
            [sample_config],
            "podman",
            dev=True,
            build_context=".",
            keep_containers=False,
            prepare_only=True,
        )

    mock_stop.assert_called_once_with("podman", [sample_config.run_id])


def test_run_without_ui_stops_containers_then_reraises_for_normal_run(
    sample_config: EvaluationRunConfig,
):
    """A normal run still surfaces the interrupt, but only after stopping the
    containers so shutdown doesn't hang on the worker."""
    with (
        patch("karotte.cli.run.clean_up_old_containers"),
        patch("karotte.cli.run.run_containerized", side_effect=KeyboardInterrupt),
        patch("karotte.cli.run.stop_containers") as mock_stop,
        pytest.raises(KeyboardInterrupt),
    ):
        _run_without_ui(
            [sample_config],
            "podman",
            dev=True,
            build_context=".",
            keep_containers=False,
        )

    mock_stop.assert_called_once_with("podman", [sample_config.run_id])


def test_stop_signal_ends_hold_cleanly(
    sample_config: EvaluationRunConfig, monkeypatch: pytest.MonkeyPatch
):
    """A KeyboardInterrupt (Ctrl-C, or SIGTERM via the CLI's handler) during
    the hold ends the session without propagating, so the process exits 0."""

    async def _interrupted_hold(_config: EvaluationRunConfig, _task: Task) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(
        karotte.run_helpers, "prepare_non_containerized", _interrupted_hold
    )

    hold_prepared_env(sample_config, _PrepareTask(sample_config))


_INTEGRATION_ENV_INIT = """\
from typing import Any

from karotte import Step, Task
from karotte.judges.always_pass_judge import AlwaysPassJudge


class _HoldStep(Step):
    @property
    def instructions(self) -> str:
        return "This step never runs in prepare mode."

    @property
    def judge(self) -> AlwaysPassJudge:
        return AlwaysPassJudge()


class PrepareHoldTask(Task):
    id = "prepare-hold-task"

    @property
    def system_prompt(self) -> str | None:
        return None

    @property
    def tools(self) -> list[str]:
        return ["echo"]

    @property
    def steps(self) -> list[Step]:
        return [_HoldStep(config=self.config)]

    def pre_hook(self) -> dict[str, Any]:
        return {"prepared": True}


def get_tasks() -> list[type[Task]]:
    return [PrepareHoldTask]
"""


# The held env itself runs with host semantics; only the CLI's image check is lifted.
_KAROTTE_WITHOUT_IMAGE_GUARD = (
    "import sys; from karotte.cli import entry; "
    + "sys.modules['karotte.cli.run'].is_containerized = lambda: True; "
    + "sys.exit(entry())"
)


def _write_integration_env(env_dir: Path, echo_tool: Path) -> None:
    """A minimal importable `environment` package with one task, reusing the
    checked-in echo tool."""
    package = env_dir / "environment"
    tools = package / "tools"
    tools.mkdir(parents=True)
    (package / "__init__.py").write_text(_INTEGRATION_ENV_INIT)
    (tools / "__init__.py").write_text("")
    shutil.copy(echo_tool, tools / "echo.py")


def test_prepare_only_cli_writes_marker_holds_and_stops_on_sigterm(
    tmp_path: Path,
    resource_dir: Path,
    unused_tcp_port_factory: Callable[[], int],
):
    """End to end through the real CLI: `karotte run --config … --no-containerized
    --prepare-only` writes the ready marker, stays alive, and exits 0 on
    SIGTERM (what a pod delete or `podman stop` sends)."""
    _write_integration_env(
        tmp_path,
        resource_dir / "characterization_env" / "environment" / "tools" / "echo.py",
    )

    run_id = f"prepare-cli-{uuid.uuid4().hex[:8]}"
    # No model_api_key: --prepare-only must not require one, since the held env
    # never calls the model. Built via the prepare-only validation context so
    # the config object itself can omit the key.
    config = EvaluationRunConfig.model_validate(
        {
            "run_id": run_id,
            "task_id": "prepare-hold-task",
            "model": "test_model",
            "mcp_server_config": {"port": unused_tcp_port_factory()},
            "websocket_config": {"port": unused_tcp_port_factory()},
        },
        context={"prepare_only": True},
    )
    assert config.model_api_key is None
    config_file = tmp_path / "run_config.json"
    config_file.write_text(config.model_dump_json())

    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{tmp_path}{os.pathsep}{existing}" if existing else str(tmp_path)
    )

    sentinel = prepared_sentinel_path(run_id)
    log_path = tmp_path / "karotte.log"
    with open(log_path, "w") as log_file:
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                _KAROTTE_WITHOUT_IMAGE_GUARD,
                "run",
                "--config",
                str(config_file),
                "--no-containerized",
                "--prepare-only",
            ],
            env=env,
            cwd=tmp_path,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

    try:
        deadline = time.monotonic() + 60
        while not sentinel.exists():
            assert process.poll() is None, (
                f"karotte exited before preparing:\n{log_path.read_text()}"
            )
            assert time.monotonic() < deadline, (
                f"ready marker never appeared:\n{log_path.read_text()}"
            )
            time.sleep(0.1)

        assert sentinel.read_text() == run_id

        time.sleep(1.0)
        assert process.poll() is None, "process must hold after preparing"

        process.terminate()
        process.wait(timeout=10)
        assert process.returncode == 0, (
            f"stopping a held env must exit 0:\n{log_path.read_text()}"
        )
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        sentinel.unlink(missing_ok=True)
