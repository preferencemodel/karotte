"""Unit tests for CLI-agent plumbing: model resolution, the registry, the
`agents install` command, and MistralVibeAgent's (subprocess-free) config."""

import asyncio
import os
from pathlib import Path
from typing import final
from unittest.mock import MagicMock

import pytest

from karotte.agents import cli_agent_types, get_cli_agent_type
from karotte.agents.agent import StepTimeLimitReachedError
from karotte.agents.cli_agent import AGENTS_BIN_DIR
from karotte.agents.mistral_vibe import MistralVibeAgent
from karotte.agents.models import resolve_model
from karotte.providers import PROXY_PLACEHOLDER_KEY
from karotte.schemas.evaluation_run_config import EvaluationRunConfig
from karotte.schemas.transcript import Event, MessageAddedEvent


class TestResolveModel:
    def test_splits_provider_and_places_api_key(self):
        resolved = resolve_model("mistral/mistral-medium-3.5", "sk-abc")
        assert resolved.provider == "mistral"
        assert resolved.model == "mistral-medium-3.5"
        assert resolved.env == {"MISTRAL_API_KEY": "sk-abc"}

    def test_no_prefix_yields_bare_model_and_no_env(self):
        resolved = resolve_model("gpt-4o", "sk-abc")
        assert resolved.provider == ""
        assert resolved.model == "gpt-4o"
        assert resolved.env == {}

    def test_unknown_provider_has_no_key_env(self):
        resolved = resolve_model("acme/model", "sk-abc")
        assert resolved.provider == "acme"
        assert resolved.model == "model"
        assert resolved.env == {}

    def test_missing_api_key_omits_env(self):
        resolved = resolve_model("mistral/m", None)
        assert resolved.env == {}


class TestRegistry:
    def test_mistral_vibe_is_registered(self):
        assert cli_agent_types()["mistral-vibe"] is MistralVibeAgent

    def test_get_unknown_agent_raises(self):
        with pytest.raises(ValueError, match="Unknown CLI agent 'nope'"):
            get_cli_agent_type("nope")


class TestMistralVibeAgent:
    def _agent(self) -> MistralVibeAgent:
        config = EvaluationRunConfig(
            run_id="r",
            task_id="t",
            model="mistral/mistral-medium-3.5",
            model_api_key="sk-abc",
            turn_limit=7,
        )
        return MistralVibeAgent(config)

    def test_allows_student_mcp_access(self):
        assert MistralVibeAgent.allows_student_mcp_access is True

    def test_declares_native_bash_and_file_tools(self):
        assert "bash" in MistralVibeAgent.native_tool_names

    def test_install_pins_version(self):
        cmds = MistralVibeAgent.install()
        assert any(f"mistral-vibe=={MistralVibeAgent.version}" in c for c in cmds)
        assert any(AGENTS_BIN_DIR in c for c in cmds)

    def test_argv_passes_instructions_and_turn_limit(self):
        argv = self._agent()._argv("do the thing")  # pyright: ignore[reportPrivateUsage]
        assert argv[0] == f"{AGENTS_BIN_DIR}/vibe"
        assert "do the thing" in argv
        assert "--max-turns" in argv and "7" in argv
        assert "--auto-approve" in argv

    def test_argv_omits_turn_limit_when_unset(self):
        config = EvaluationRunConfig(
            run_id="r", task_id="t", model="mistral/m", model_api_key="k"
        )
        argv = MistralVibeAgent(config)._argv("x")  # pyright: ignore[reportPrivateUsage]
        assert "--max-turns" not in argv

    def test_env_sets_credentials_and_vibe_home(self):
        env = self._agent()._env()  # pyright: ignore[reportPrivateUsage]
        assert env["MISTRAL_API_KEY"] == "sk-abc"
        assert env["VIBE_HOME"].endswith(".vibe")
        assert AGENTS_BIN_DIR in env["PATH"]
        # Runs as the student, so HOME must not point at (unreadable) /root.
        assert env["HOME"] != "/root"
        assert env["USER"] == "student"

    def test_env_restores_venv_bin_on_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The bin of the Containerfile-set VIRTUAL_ENV is put back on PATH."""
        venv = tmp_path / ".venv"
        (venv / "bin").mkdir(parents=True)
        (venv / "pyvenv.cfg").write_text("")
        monkeypatch.setenv("VIRTUAL_ENV", str(venv))
        env = self._agent()._env()  # pyright: ignore[reportPrivateUsage]
        # venv bin must precede system dirs so `python` resolves to the venv.
        assert env["PATH"].startswith(str(venv / "bin"))

    def test_env_skips_venv_when_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """A VIRTUAL_ENV without a pyvenv.cfg is not added to PATH."""
        monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path / ".venv"))
        env = self._agent()._env()  # pyright: ignore[reportPrivateUsage]
        assert str(tmp_path / ".venv" / "bin") not in env["PATH"].split(":")

    def _write_config(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
        agent = self._agent()
        agent._vibe_home = tmp_path / ".vibe"  # pyright: ignore[reportPrivateUsage]
        monkeypatch.setattr(
            "karotte.agents.mistral_vibe.agent.demoted_uid_gid", lambda: None
        )
        agent._write_config()  # pyright: ignore[reportPrivateUsage]
        return (tmp_path / ".vibe" / "config.toml").read_text()

    def test_write_config_contains_model_and_mcp_endpoint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.delenv("KAROTTE_PROXY_URL", raising=False)
        config = self._write_config(tmp_path, monkeypatch)
        assert 'active_model = "mistral-medium-3.5"' in config
        assert "streamable-http" in config
        assert self._agent().mcp_url in config

    def test_write_config_routes_through_proxy_when_set(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("KAROTTE_PROXY_URL", "https://proxy.example")
        config = self._write_config(tmp_path, monkeypatch)
        assert 'api_base = "https://proxy.example/v1"' in config
        assert 'api_key_env_var = "MISTRAL_API_KEY"' in config
        assert 'api_style = "openai"' in config

    def test_write_config_direct_provider_without_proxy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.delenv("KAROTTE_PROXY_URL", raising=False)
        config = self._write_config(tmp_path, monkeypatch)
        assert 'api_base = "https://api.mistral.ai/v1"' in config

    @pytest.mark.asyncio
    async def test_start_without_proxy_forwards_and_withholds_the_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.delenv("KAROTTE_PROXY_URL", raising=False)
        agent = self._agent()
        agent._vibe_home = tmp_path / ".vibe"  # pyright: ignore[reportPrivateUsage]
        monkeypatch.setattr(
            "karotte.agents.mistral_vibe.agent.demoted_uid_gid", lambda: None
        )
        await agent.start(MagicMock())
        try:
            assert agent.model_url.startswith("http://127.0.0.1:")
            config = (tmp_path / ".vibe" / "config.toml").read_text()
            assert f'api_base = "{agent.model_url}/v1"' in config
            env = agent._env()  # pyright: ignore[reportPrivateUsage]
            assert env["MISTRAL_API_KEY"] == PROXY_PLACEHOLDER_KEY
            assert "sk-abc" not in env.values()
        finally:
            await agent.stop()
        assert agent.model_url == "https://api.mistral.ai"

    @pytest.mark.asyncio
    async def test_start_with_proxy_does_not_forward(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("KAROTTE_PROXY_URL", "https://proxy.example/")
        agent = self._agent()
        agent._vibe_home = tmp_path / ".vibe"  # pyright: ignore[reportPrivateUsage]
        monkeypatch.setattr(
            "karotte.agents.mistral_vibe.agent.demoted_uid_gid", lambda: None
        )
        await agent.start(MagicMock())
        assert agent.model_url == "https://proxy.example"
        env = agent._env()  # pyright: ignore[reportPrivateUsage]
        assert env["MISTRAL_API_KEY"] == "sk-abc"
        await agent.stop()

    def test_env_gets_placeholder_key_through_keyless_proxy(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("KAROTTE_PROXY_URL", "https://proxy.example")
        config = EvaluationRunConfig(run_id="r", task_id="t", model="mistral/m")
        env = MistralVibeAgent(config)._env()  # pyright: ignore[reportPrivateUsage]
        assert env["MISTRAL_API_KEY"] == PROXY_PLACEHOLDER_KEY


@final
class _FakeReader:
    def __init__(self, data: bytes) -> None:
        self._data = data

    async def read(self, _n: int = -1) -> bytes:
        return self._data


@final
class _FakeProc:
    """Stands in for an asyncio subprocess in run_step tests.

    `hang=True` makes wait() block forever until the process is 'killed',
    exercising the time-limit path.
    """

    def __init__(
        self,
        stdout: bytes = b"",
        stderr: bytes = b"",
        exit_code: int = 0,
        hang: bool = False,
    ) -> None:
        self.pid = 987654
        self.returncode: int | None = None
        self._exit_code = exit_code
        self._hang = hang
        self.killed = False
        self.stdout = _FakeReader(stdout)
        self.stderr = _FakeReader(stderr)

    async def wait(self) -> int:
        if self._hang and not self.killed:
            await asyncio.Event().wait()  # released only once killed
        self.returncode = -9 if self.killed else self._exit_code
        return self.returncode

    def kill(self) -> None:
        self.killed = True


class TestMistralVibeRunStep:
    def _agent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **config_kwargs: object
    ) -> MistralVibeAgent:
        monkeypatch.setenv("KAROTTE_WORKDIR", str(tmp_path))
        (tmp_path / ".vibe").mkdir()
        config = EvaluationRunConfig(
            run_id="r",
            task_id="t",
            model="mistral/m",
            model_api_key="k",
            save_artifacts=False,
            **config_kwargs,  # pyright: ignore[reportArgumentType]
        )
        return MistralVibeAgent(config)

    def _patch_proc(
        self, monkeypatch: pytest.MonkeyPatch, proc: _FakeProc
    ) -> list[int]:
        """Wire the fake process in and record killpg targets."""

        async def fake_create(*_args: object, **_kwargs: object) -> _FakeProc:
            return proc

        killed_pgids: list[int] = []

        def fake_killpg(pgid: int, _sig: int) -> None:
            killed_pgids.append(pgid)
            proc.killed = True

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create)
        monkeypatch.setattr(os, "killpg", fake_killpg)
        return killed_pgids

    async def _run(self, agent: MistralVibeAgent, **kwargs: object) -> list[Event]:
        return [
            event
            async for event in agent.run_step("do the thing", **kwargs)  # pyright: ignore[reportArgumentType]
        ]

    @pytest.mark.asyncio
    async def test_completes_without_time_limit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        agent = self._agent(tmp_path, monkeypatch)
        self._patch_proc(monkeypatch, _FakeProc(exit_code=0))

        events = await self._run(agent)

        # The instruction message is always emitted; no error is raised.
        assert any(
            isinstance(e, MessageAddedEvent) and e.message.role == "user"
            for e in events
        )

    @pytest.mark.asyncio
    async def test_hands_the_pipes_it_opened_to_the_demoted_agent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """stdout and stderr are pipes this process opens, so vibe and whatever
        it runs can reopen them by path. stdin is inherited, so it is not ours
        to give."""
        agent = self._agent(tmp_path, monkeypatch)
        self._patch_proc(monkeypatch, _FakeProc(exit_code=0))
        named: list[tuple[int, ...]] = []
        monkeypatch.setattr(
            "karotte.agents.mistral_vibe.agent.make_preexec",
            lambda *fds: named.append(fds),  # pyright: ignore[reportUnknownLambdaType]
        )

        await self._run(agent)

        assert named == [(1, 2)]

    @pytest.mark.asyncio
    async def test_timeout_error_raises_after_killing_process(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        agent = self._agent(tmp_path, monkeypatch)
        proc = _FakeProc(hang=True)
        killed_pgids = self._patch_proc(monkeypatch, proc)

        with pytest.raises(StepTimeLimitReachedError, match="0.01"):
            await self._run(agent, time_limit_seconds=0.01, on_time_limit="error")

        assert proc.pid in killed_pgids

    @pytest.mark.asyncio
    async def test_timeout_score_kills_process_without_raising(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        agent = self._agent(tmp_path, monkeypatch)
        proc = _FakeProc(hang=True)
        killed_pgids = self._patch_proc(monkeypatch, proc)

        events = await self._run(agent, time_limit_seconds=0.01, on_time_limit="score")

        assert proc.pid in killed_pgids
        assert any(
            isinstance(e, MessageAddedEvent) and e.message.role == "user"
            for e in events
        )

    @pytest.mark.asyncio
    async def test_killpg_fallback_to_process_kill(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """When there is no process group, run_step falls back to proc.kill()."""
        agent = self._agent(tmp_path, monkeypatch)
        proc = _FakeProc(hang=True)

        async def fake_create(*_args: object, **_kwargs: object) -> _FakeProc:
            return proc

        def raising_killpg(_pgid: int, _sig: int) -> None:
            raise ProcessLookupError

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create)
        monkeypatch.setattr(os, "killpg", raising_killpg)

        await self._run(agent, time_limit_seconds=0.01, on_time_limit="score")

        assert proc.killed

    @pytest.mark.asyncio
    async def test_context_window_limit_is_not_enforced(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Vibe accepts the context-window limit but never enforces it: usage is
        only known after the run, so the process runs to completion."""
        agent = self._agent(tmp_path, monkeypatch)
        proc = _FakeProc(exit_code=0)
        killed_pgids = self._patch_proc(monkeypatch, proc)

        events = await self._run(
            agent, context_window_limit=1, on_context_window_limit="error"
        )

        assert not proc.killed
        assert killed_pgids == []
        assert any(
            isinstance(e, MessageAddedEvent) and e.message.role == "user"
            for e in events
        )
