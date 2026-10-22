"""Tests for what every CLI agent gets from `CliAgent`: its home, environment
and per-step process handling, driven by a stand-in tool script."""

import asyncio
import gzip
import hashlib
import platform
import stat
import subprocess
import sys
import threading
import time
import tomllib
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import ClassVar, Literal, final
from unittest.mock import MagicMock

import pytest

from karotte.agents.agent import CliAgentExitedError, StepTimeLimitReachedError
from karotte.agents.cli_agent import (
    AGENTS_BIN_DIR,
    CliAgent,
    install_release,
    toml_string,
)
from karotte.model_forwarder import GeminiForwarder
from karotte.providers import PROXY_PLACEHOLDER_KEY
from karotte.schemas.chat import Message
from karotte.schemas.evaluation_run_config import EvaluationRunConfig
from karotte.schemas.transcript import Event, MessageAddedEvent

_FAKE_TOOL = """
import os, pathlib, sys, time
gate = pathlib.Path(sys.argv[2])
for i, line in enumerate(pathlib.Path(sys.argv[1]).read_text().splitlines()):
    print(line, flush=True)
    if i == 0:
        while not gate.exists():
            time.sleep(0.01)
exec(sys.argv[3])
if int(sys.argv[4]):
    print("boom", file=sys.stderr)
sys.exit(int(sys.argv[4]))
"""


@final
class _EchoAgent(CliAgent):
    name: ClassVar[str] = "echo"
    version: ClassVar[str] = "1"
    provider_url: ClassVar[str] = "https://example.invalid"
    providers: ClassVar[tuple[str, ...]] = ("example",)
    binary: ClassVar[str] = f"{AGENTS_BIN_DIR}/echo"
    home_dir: ClassVar[str] = ".echo"
    home_env: ClassVar[str | None] = "ECHO_HOME"
    prompt_input: ClassVar[Literal["argv", "file", "stdin"]] = "file"

    after_output: str = ""
    exit_code: int = 0
    held: str = ""
    reported: str | None = None

    @classmethod
    def install(cls) -> list[str]:
        return []

    def _config_files(self) -> dict[str, str]:
        return {"config.toml": 'model = "m"\n'}

    def _key_env(self) -> str:
        return "ECHO_API_KEY"

    def _extra_env(self) -> dict[str, str]:
        return {"ECHO_TELEMETRY": "0"}

    def _argv(self, instructions: str, prompt_file: Path | None) -> list[str]:
        script = self._home / "fake_tool.py"
        script.write_text(_FAKE_TOOL)
        return [
            sys.executable,
            str(script),
            "/dev/stdin" if self.prompt_input == "stdin" else str(prompt_file),
            str(self._home / "gate"),
            self.after_output,
            str(self.exit_code),
        ]

    def _line_parser(self, instructions: str) -> Callable[[str, int], Iterable[Event]]:
        def parse(line: str, i: int) -> Iterable[Event]:
            if line == "slow":
                time.sleep(1.5)
            if line:
                yield MessageAddedEvent(
                    message=Message(role="assistant", content=f"{i}:{line}")
                )

        return parse

    def _output_ended(self) -> Iterable[Event]:
        if self.held:
            yield MessageAddedEvent(
                message=Message(role="assistant", content=self.held)
            )

    def _reported_error(self) -> str | None:
        return self.reported


def _agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **config: object
) -> _EchoAgent:
    monkeypatch.setenv("KAROTTE_WORKDIR", str(tmp_path))
    monkeypatch.delenv("KAROTTE_PROXY_URL", raising=False)
    monkeypatch.setattr("karotte.agents.cli_agent.demoted_uid_gid", lambda: None)
    monkeypatch.setattr(
        "karotte.agents.cli_agent.make_preexec",
        lambda *_fds: None,  # pyright: ignore[reportUnknownLambdaType]
    )
    agent = _EchoAgent(
        EvaluationRunConfig.model_validate(
            {
                "run_id": "r",
                "task_id": "t",
                "model": "mistral/m",
                "model_api_key": "sk-real",
                "save_artifacts": False,
                **config,
            }
        )
    )
    agent._write_config()  # pyright: ignore[reportPrivateUsage]
    return agent


def _step_dir(agent: _EchoAgent) -> Path:
    steps = agent._step_dir  # pyright: ignore[reportPrivateUsage]
    assert steps is not None
    return steps


def _contents(events: list[Event]) -> list[object]:
    return [
        e.message.content
        for e in events
        if isinstance(e, MessageAddedEvent) and e.message.role == "assistant"
    ]


class TestHomeAndEnv:
    def test_config_files_land_in_the_home(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        _agent(tmp_path, monkeypatch)
        assert (tmp_path / ".echo" / "config.toml").read_text() == 'model = "m"\n'

    def test_env_points_the_tool_at_its_home_as_the_student(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        env = _agent(tmp_path, monkeypatch)._env()  # pyright: ignore[reportPrivateUsage]
        assert env["ECHO_HOME"] == str(tmp_path / ".echo")
        assert env["HOME"] == str(tmp_path)
        assert env["USER"] == "student"
        assert env["ECHO_TELEMETRY"] == "0"
        assert env["ECHO_API_KEY"] == "sk-real"
        assert AGENTS_BIN_DIR in env["PATH"].split(":")

    def test_the_student_owns_every_config_dir_and_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        agent = _agent(tmp_path, monkeypatch)
        monkeypatch.setattr(
            agent, "_config_files", lambda: {"a/b/settings.json": "{}", "c.toml": ""}
        )
        monkeypatch.setattr("karotte.agents.cli_agent.demoted_uid_gid", lambda: 1000)
        owned: list[Path] = []
        monkeypatch.setattr(
            "karotte.agents.cli_agent.os.chown",
            lambda path, uid, gid: owned.append(Path(path)),  # pyright: ignore[reportUnknownLambdaType]
        )
        agent._write_config()  # pyright: ignore[reportPrivateUsage]
        home = tmp_path / ".echo"
        assert set(owned) == {
            home,
            home / "a",
            home / "a" / "b",
            home / "a" / "b" / "settings.json",
            home / "c.toml",
        }

    def test_a_tool_without_a_home_env_gets_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        agent = _agent(tmp_path, monkeypatch)
        monkeypatch.setattr(agent, "home_env", None)
        env = agent._env()  # pyright: ignore[reportPrivateUsage]
        assert "ECHO_HOME" not in env
        assert all(isinstance(name, str) for name in env)


class TestForwarder:
    @pytest.mark.asyncio
    async def test_without_a_proxy_the_agent_picks_the_forwarder(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        agent = _agent(tmp_path, monkeypatch)
        own = GeminiForwarder("https://example.invalid", "sk-real")
        monkeypatch.setattr(agent, "_make_forwarder", lambda: own)
        await agent.start(MagicMock())
        try:
            assert agent.model_url == own.url
            assert own.url.startswith("http://127.0.0.1:")
            env = agent._env()  # pyright: ignore[reportPrivateUsage]
            assert env["ECHO_API_KEY"] == PROXY_PLACEHOLDER_KEY
        finally:
            await agent.stop()

    @pytest.mark.asyncio
    async def test_with_a_proxy_there_is_no_forwarder(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        agent = _agent(tmp_path, monkeypatch)
        monkeypatch.setenv("KAROTTE_PROXY_URL", "http://proxy:4000/")
        made: list[object] = []
        monkeypatch.setattr(agent, "_make_forwarder", lambda: made.append(1))
        await agent.start(MagicMock())
        await agent.stop()
        assert made == []
        assert agent.model_url == "http://proxy:4000"

    @pytest.mark.asyncio
    async def test_a_proxy_via_the_forwarder_keeps_the_key_from_the_cli(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        agent = _agent(tmp_path, monkeypatch)
        monkeypatch.setattr(_EchoAgent, "proxy_via_forwarder", True)
        monkeypatch.setenv("KAROTTE_PROXY_URL", "http://proxy:4000/")
        await agent.start(MagicMock())
        try:
            forwarder = agent._forwarder  # pyright: ignore[reportPrivateUsage]
            assert forwarder is not None
            assert forwarder.upstream == "http://proxy:4000"
            assert agent.model_url == forwarder.url
            env = agent._env()  # pyright: ignore[reportPrivateUsage]
            assert env["ECHO_API_KEY"] == PROXY_PLACEHOLDER_KEY
        finally:
            await agent.stop()

    @pytest.mark.asyncio
    async def test_the_forwarder_is_made_off_the_event_loop(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        agent = _agent(tmp_path, monkeypatch)
        threads: list[int] = []
        monkeypatch.setattr(
            agent, "_make_forwarder", lambda: threads.append(threading.get_ident())
        )
        await agent.start(MagicMock())
        await agent.stop()
        assert threads
        assert threads != [threading.get_ident()]


class TestRunStep:
    @pytest.mark.asyncio
    async def test_yields_each_line_before_the_tool_exits(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        agent = _agent(tmp_path, monkeypatch)
        events = agent.run_step("first\nsecond")
        assert isinstance(await anext(events), MessageAddedEvent)
        first = await asyncio.wait_for(anext(events), 10)
        assert _contents([first]) == ["0:first"]
        assert not (tmp_path / ".echo" / "gate").exists()

        (tmp_path / ".echo" / "gate").touch()
        assert _contents([e async for e in events]) == ["1:second"]
        steps = _step_dir(agent)
        assert (steps / "step_0.prompt").read_text() == "first\nsecond"
        assert (steps / "step_0.ndjson").read_text().splitlines() == ["first", "second"]

    @pytest.mark.asyncio
    async def test_each_step_gets_its_own_prompt_and_log(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        agent = _agent(tmp_path, monkeypatch)
        (tmp_path / ".echo" / "gate").touch()
        _ = [e async for e in agent.run_step("one")]
        _ = [e async for e in agent.run_step("two")]
        assert (_step_dir(agent) / "step_1.prompt").read_text() == "two"
        assert (_step_dir(agent) / "step_1.ndjson").read_text() == "two\n"

    @pytest.mark.asyncio
    async def test_nonzero_exit_raises_after_yielding(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        agent = _agent(tmp_path, monkeypatch)
        agent.exit_code = 3
        (tmp_path / ".echo" / "gate").touch()
        seen: list[Event] = []
        with pytest.raises(CliAgentExitedError, match="echo exited 3 on step 0: boom"):
            async for event in agent.run_step("hi"):
                seen.append(event)
        assert _contents(seen) == ["0:hi"]

    @pytest.mark.asyncio
    async def test_nonzero_exit_quotes_the_clis_own_error_over_stderr(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        agent = _agent(tmp_path, monkeypatch)
        agent.exit_code = 1
        agent.reported = "API Error: 400 bad request"
        (tmp_path / ".echo" / "gate").touch()
        with pytest.raises(CliAgentExitedError) as raised:
            _ = [e async for e in agent.run_step("hi")]
        assert (
            str(raised.value) == "echo exited 1 on step 0: API Error: 400 bad request"
        )

    @pytest.mark.asyncio
    async def test_time_limit_score_keeps_output_so_far(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        agent = _agent(tmp_path, monkeypatch)
        agent.after_output = "time.sleep(30)"
        (tmp_path / ".echo" / "gate").touch()
        events = [
            e
            async for e in agent.run_step(
                "hi", time_limit_seconds=1, on_time_limit="score"
            )
        ]
        assert _contents(events) == ["0:hi"]

    @pytest.mark.asyncio
    async def test_time_limit_still_yields_what_the_parser_held(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        agent = _agent(tmp_path, monkeypatch)
        agent.after_output = "time.sleep(30)"
        agent.held = "held"
        (tmp_path / ".echo" / "gate").touch()
        events = [
            e
            async for e in agent.run_step(
                "hi", time_limit_seconds=1, on_time_limit="score"
            )
        ]
        assert _contents(events) == ["0:hi", "held"]

    @pytest.mark.asyncio
    async def test_time_limit_applies_after_the_tool_closes_its_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        agent = _agent(tmp_path, monkeypatch)
        agent.after_output = "os.close(1); time.sleep(30)"
        (tmp_path / ".echo" / "gate").touch()
        with pytest.raises(StepTimeLimitReachedError):
            _ = [
                e
                async for e in agent.run_step(
                    "hi", time_limit_seconds=1, on_time_limit="error"
                )
            ]

    @pytest.mark.asyncio
    async def test_finishing_as_the_deadline_passes_is_not_a_timeout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        agent = _agent(tmp_path, monkeypatch)
        agent.after_output = "sys.stdout.write('slow')"
        (tmp_path / ".echo" / "gate").touch()
        events = [
            e
            async for e in agent.run_step(
                "hi", time_limit_seconds=1, on_time_limit="error"
            )
        ]
        assert _contents(events) == ["0:hi", "1:slow"]

    @pytest.mark.asyncio
    async def test_a_child_holding_stderr_does_not_hold_up_the_step(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr("karotte.agents.cli_agent._DRAIN_TIMEOUT_SECONDS", 0.5)
        agent = _agent(tmp_path, monkeypatch)
        agent.after_output = (
            "import subprocess; subprocess.Popen([sys.executable, '-c', "
            "'import time; time.sleep(10)'], stdout=subprocess.DEVNULL, "
            "start_new_session=True); "
            "os.dup2(os.open(os.devnull, os.O_WRONLY), 1); time.sleep(0.5)"
        )
        (tmp_path / ".echo" / "gate").touch()
        started = time.monotonic()
        _ = [e async for e in agent.run_step("hi")]
        assert time.monotonic() - started < 5


class TestStepFiles:
    @pytest.mark.asyncio
    async def test_root_never_writes_through_links_the_student_plants(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        agent = _agent(tmp_path, monkeypatch)
        target = tmp_path / "grader.py"
        target.write_text("original")
        for name in ("step_0.prompt", "step_0.ndjson"):
            (tmp_path / ".echo" / name).symlink_to(target)
        (tmp_path / ".echo" / "gate").touch()
        _ = [e async for e in agent.run_step("hi")]
        assert target.read_text() == "original"

    @pytest.mark.asyncio
    async def test_the_student_can_read_but_not_change_them(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        agent = _agent(tmp_path, monkeypatch)
        (tmp_path / ".echo" / "gate").touch()
        _ = [e async for e in agent.run_step("hi")]
        steps = _step_dir(agent)
        assert stat.S_IMODE(steps.stat().st_mode) == 0o711
        assert stat.S_IMODE((steps / "step_0.prompt").stat().st_mode) == 0o644

    @pytest.mark.asyncio
    async def test_only_an_agent_that_reads_a_prompt_file_gets_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        agent = _agent(tmp_path, monkeypatch)
        monkeypatch.setattr(agent, "prompt_input", "argv")
        (tmp_path / ".echo" / "gate").touch()
        with pytest.raises(CliAgentExitedError):
            _ = [e async for e in agent.run_step("hi")]
        assert not (_step_dir(agent) / "step_0.prompt").exists()

    @pytest.mark.asyncio
    async def test_the_prompt_can_go_to_stdin(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        agent = _agent(tmp_path, monkeypatch)
        monkeypatch.setattr(agent, "prompt_input", "stdin")
        (tmp_path / ".echo" / "gate").touch()
        events = [e async for e in agent.run_step("from\nstdin")]
        assert _contents(events) == ["0:from", "1:stdin"]

    @pytest.mark.asyncio
    async def test_the_agent_can_wrap_the_prompt_it_reads(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        agent = _agent(tmp_path, monkeypatch)
        monkeypatch.setattr(agent, "prompt_input", "stdin")

        def wrap(text: str) -> str:
            return f"<{text}>"

        monkeypatch.setattr(agent, "_prompt_text", wrap)
        (tmp_path / ".echo" / "gate").touch()
        events = [e async for e in agent.run_step("hi")]
        assert _contents(events) == ["0:<hi>"]

    @pytest.mark.asyncio
    async def test_stop_removes_them(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        agent = _agent(tmp_path, monkeypatch)
        (tmp_path / ".echo" / "gate").touch()
        _ = [e async for e in agent.run_step("hi")]
        steps = _step_dir(agent)
        await agent.stop()
        assert not steps.exists()


class TestTomlString:
    @pytest.mark.parametrize(
        "value",
        ["plain", 'q"uote\\back', "new\nline\ttab", "emoji 😀", "del\x7f", "nul\x00"],
    )
    def test_round_trips_through_a_toml_parser(self, value: str):
        assert tomllib.loads(f"v = {toml_string(value)}")["v"] == value


def _install(tmp_path: Path, payload: bytes, sha: str, **kwargs: object) -> Path:
    release = tmp_path / "release"
    release.write_bytes(payload)
    target = tmp_path / "bin" / "tool"
    script = install_release(
        "tool",
        f"file://{release}",
        dict.fromkeys(("x86_64", "aarch64"), sha),
        str(target),
        **kwargs,  # pyright: ignore[reportArgumentType]
    )
    for command in script:
        _ = subprocess.run(command, shell=True, check=True, capture_output=True)
    return target


class TestInstallRelease:
    def test_installs_a_verified_binary(self, tmp_path: Path):
        payload = b"#!/bin/sh\necho hi\n"
        target = _install(tmp_path, payload, hashlib.sha256(payload).hexdigest())
        assert target.read_bytes() == payload
        assert stat.S_IMODE(target.stat().st_mode) == 0o755

    def test_refuses_a_download_that_fails_its_checksum(self, tmp_path: Path):
        with pytest.raises(subprocess.CalledProcessError):
            _ = _install(tmp_path, b"tampered", "0" * 64)
        assert not (tmp_path / "bin" / "tool").exists()

    def test_can_verify_the_unpacked_binary(self, tmp_path: Path):
        binary = b"unpacked"
        target = _install(
            tmp_path,
            gzip.compress(binary),
            hashlib.sha256(binary).hexdigest(),
            unpack='gunzip -c "$tmp/download" > "$tmp/bin"',
            verify_unpacked=True,
        )
        assert target.read_bytes() == binary

    def test_downloads_each_architectures_own_url(self, tmp_path: Path):
        payloads = {"x86_64": b"x86 build", "aarch64": b"arm build"}
        urls: dict[str, str] = {}
        for arch, payload in payloads.items():
            (tmp_path / arch).write_bytes(payload)
            urls[arch] = f"file://{tmp_path / arch}"
        target = tmp_path / "bin" / "tool"
        script = install_release(
            "tool",
            urls,
            {arch: hashlib.sha256(p).hexdigest() for arch, p in payloads.items()},
            str(target),
        )
        for command in script:
            _ = subprocess.run(command, shell=True, check=True, capture_output=True)
        machine = {"arm64": "aarch64", "amd64": "x86_64"}.get(
            platform.machine(), platform.machine()
        )
        assert target.read_bytes() == payloads[machine]
