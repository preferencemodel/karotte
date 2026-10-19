import asyncio
import io
import json
import socket
import subprocess
import sys
import tarfile
import threading
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, final
from unittest.mock import patch

import pytest
import typer
import websockets
from websockets.asyncio.server import ServerConnection, serve

from karotte import modal_runtime
from karotte.build import get_container_build_command, require_builder
from karotte.forwarded_env import EXIT_ON_RUN_ERROR_ENV_VAR
from karotte.hardware import HardwareLimits
from karotte.modal_runtime import (
    CONFIG_PATH,
    NOCAPS_STEP,
    ModalError,
    WebsocketRelay,
    containerfile_for_modal,
    modal_problems,
    run_modal,
)
from karotte.schemas.evaluation_run_config import EvaluationRunConfig
from tests.conftest import register_hardware_plugins

GIB = 1 << 30


def _config(**update: object) -> EvaluationRunConfig:
    return EvaluationRunConfig.model_validate(
        {
            "run_id": "modal-test",
            "task_id": "example-task",
            "model": "claude-sonnet-5",
            "model_api_key": "sk-secret",
            **update,
        }
    )


def _gitkeep(path: Path) -> bool:
    return path.name == ".gitkeep"


class TestContainerfile:
    def test_dnf_sets_no_file_capabilities_in_every_stage(self, tmp_path: Path):
        text = (
            "FROM a AS builder\nRUN dnf install -y x\nFROM scratch\nFROM b\nRUN true\n"
        )

        rewritten, _ = containerfile_for_modal(text, tmp_path, _gitkeep)

        assert rewritten.splitlines() == [
            "FROM a AS builder",
            NOCAPS_STEP,
            "RUN dnf install -y x",
            "FROM scratch",
            "FROM b",
            NOCAPS_STEP,
            "RUN true",
        ]

    def test_a_folder_of_ignored_placeholders_is_copied_and_emptied(
        self, tmp_path: Path
    ):
        (tmp_path / "student_data").mkdir()
        (tmp_path / "student_data" / ".gitkeep").touch()
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").touch()
        (tmp_path / "src" / ".gitkeep").touch()
        text = (
            "FROM a\n"
            + "RUN one && \\\n"
            + "  # COPY not/an/instruction /x\n"
            + "  two\n"
            + "COPY --chown=student student_data/ ${STUDENT_WORKDIR}/data/\n"
            + "COPY src/ /root/src/\n"
        )

        rewritten, placeholders = containerfile_for_modal(text, tmp_path, _gitkeep)

        assert placeholders == {Path("student_data/.gitkeep")}
        assert rewritten.splitlines()[-3:] == [
            "COPY --chown=student student_data/ ${STUDENT_WORKDIR}/data/",
            'RUN rm -f "${STUDENT_WORKDIR}/data/.gitkeep"',
            "COPY src/ /root/src/",
        ]

    def test_an_empty_folder_is_refused(self, tmp_path: Path):
        (tmp_path / "shared_data").mkdir()

        with pytest.raises(ModalError, match="shared_data"):
            containerfile_for_modal(
                "FROM a\nCOPY shared_data/ /workdir/shared/\n", tmp_path, _gitkeep
            )


class TestWiring:
    def test_build_runs_the_modal_builder(self):
        assert get_container_build_command("modal", "/env") == [
            sys.executable,
            "-m",
            "karotte.modal_runtime",
            "build",
            "/env",
        ]

    def test_build_secrets_are_refused(self):
        with pytest.raises(typer.BadParameter, match="modal"):
            get_container_build_command("modal", "/env", build_secrets=["uv=f"])

    def test_no_builder_binary_needed(self):
        require_builder("modal")

    def test_run_containerized_starts_a_sandbox(self):
        from karotte.run_helpers import run_containerized

        config = _config()
        with patch("karotte.modal_runtime.run_modal") as run:
            run_containerized(
                config, "modal", dev=False, build_context="/env", parallel_runs=2
            )

        run.assert_called_once()
        assert run.call_args.args == (config,)
        assert run.call_args.kwargs["build_context"] == "/env"

    def test_clean_up_and_stop(self):
        from karotte.run_helpers import clean_up_old_containers, stop_containers

        with patch("karotte.modal_runtime.clean_up_sandboxes") as clean_up:
            clean_up_old_containers("modal", ["batch-0", "batch-1"])
        clean_up.assert_called_once_with("batch-")
        with patch("karotte.modal_runtime.stop_sandboxes") as stop:
            stop_containers("modal", ["batch-0"])
        stop.assert_called_once_with(["batch-0"])


def _gpu_passthrough(hardware: str) -> HardwareLimits | None:
    return HardwareLimits(passthrough=True) if hardware == "gpu" else None


class TestProblems:
    @pytest.fixture(autouse=True)
    def _modal_installed(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(modal_runtime, "modal_installed", lambda: True)
        register_hardware_plugins(
            monkeypatch,
            limits={"a": _gpu_passthrough},
        )

    def _problems(self, **kwargs: Any) -> list[str]:
        args: dict[str, Any] = {
            "required_hardware": None,
            "mounts": None,
            "dev": False,
            "keep_containers": False,
            "prepare_only": False,
            "proxy_url": None,
            **kwargs,
        }
        return modal_problems(**args)

    def test_a_plain_run_has_none(self):
        assert self._problems() == []

    @pytest.mark.parametrize(
        ("kwargs", "named"),
        [
            ({"required_hardware": "gpu"}, "gpu"),
            ({"mounts": ["a:/b"]}, "--mount"),
            ({"dev": True}, "--dev"),
            ({"keep_containers": True}, "--keep-containers"),
            ({"prepare_only": True}, "--prepare-only"),
        ],
    )
    def test_unsupported_options(self, kwargs: dict[str, Any], named: str):
        problems = self._problems(**kwargs)
        assert len(problems) == 1 and named in problems[0]

    def test_without_modal_installed(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(modal_runtime, "modal_installed", lambda: False)
        assert "karotte[modal]" in self._problems()[0]

    @pytest.mark.parametrize(
        ("addresses", "reachable"),
        [
            (["127.0.0.1"], False),
            (["10.1.2.3"], False),
            (["100.101.102.103"], False),
            (["fe80::1"], False),
            (["203.0.114.5", "192.168.1.1"], False),
            (["8.8.8.8"], True),
        ],
    )
    def test_a_proxy_must_be_reachable_from_the_cloud(
        self,
        monkeypatch: pytest.MonkeyPatch,
        addresses: list[str],
        reachable: bool,
    ):
        monkeypatch.setattr(modal_runtime, "_resolve", lambda *_: addresses)  # pyright: ignore[reportUnknownLambdaType]

        problems = self._problems(proxy_url="https://proxy.example:8443/v1")

        assert (problems == []) is reachable
        if not reachable:
            assert "--no-proxy" in problems[0]


@final
class FakeProcess:
    def __init__(self, lines: list[str], returncode: int) -> None:
        self.stdout = iter(lines)
        self.returncode = returncode

    def wait(self) -> int:
        return self.returncode


@final
class FakeFilesystem:
    def __init__(self, sandbox: "FakeSandbox") -> None:
        self.sandbox = sandbox

    def make_directory(self, path: str) -> None:
        self.sandbox.dirs.append(path)

    def write_text(self, data: str, path: str) -> None:
        self.sandbox.files[path] = data.encode()

    def copy_to_local(self, remote: str, local: Path) -> None:
        Path(local).write_bytes(self.sandbox.files[remote])


@final
class FakeSandbox:
    def __init__(self, modal: "FakeModal", kwargs: dict[str, Any]) -> None:
        self.modal = modal
        self.kwargs = kwargs
        self.tags: dict[str, str] = dict(kwargs.get("tags") or {})
        self.files: dict[str, bytes] = {}
        self.dirs: list[str] = []
        self.execs: list[tuple[tuple[str, ...], dict[str, Any]]] = []
        self.terminated = False
        self.filesystem = FakeFilesystem(self)

    def exec(self, *argv: str, **kwargs: Any) -> FakeProcess:
        if self.modal.exec_error is not None:
            raise self.modal.exec_error
        self.execs.append((argv, kwargs))
        if argv[0] == "tar":
            self.files[argv[argv.index("-cf") + 1]] = self.modal.out_tar
            return FakeProcess([], 0)
        return FakeProcess(self.modal.output, self.modal.returncode)

    def create_connect_token(self, port: int) -> SimpleNamespace:
        self.modal.token_ports.append(port)
        return SimpleNamespace(url="https://sandbox.example", token="tok")

    def get_tags(self) -> dict[str, str]:
        return self.tags

    def terminate(self) -> None:
        self.terminated = True


class FakeError(Exception):
    pass


def _match_nothing(*_patterns: str) -> Callable[[Path], bool]:
    def matches(_path: Path) -> bool:
        return False

    return matches


@final
class FakeModal(ModuleType):
    """Records what karotte asks of Modal; ``Sandbox`` and friends are its methods."""

    def __init__(self) -> None:
        super().__init__("modal")
        self.sandboxes: list[FakeSandbox] = []
        self.output = ["line one\n", "line two\n"]
        self.returncode = 0
        self.out_tar = _out_tar(_empty)
        self.exec_error: Exception | None = None
        self.token_ports: list[int] = []
        self.images: list[dict[str, Any]] = []
        self.App = SimpleNamespace(lookup=self._lookup)
        self.Image = SimpleNamespace(from_dockerfile=self._from_dockerfile)
        self.Sandbox = SimpleNamespace(create=self.create, list=self._list)
        self.FilePatternMatcher = _match_nothing
        self.exception = SimpleNamespace(Error=FakeError)

    def _lookup(self, name: str, **_kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(name=name, app_id="ap-1")

    def _from_dockerfile(self, path: Path, **kwargs: Any) -> SimpleNamespace:
        self.images.append({"path": path, **kwargs})
        return SimpleNamespace(dockerfile=Path(path).read_text())

    def create(self, *argv: str, **kwargs: Any) -> FakeSandbox:
        sandbox = FakeSandbox(self, {"argv": argv, **kwargs})
        self.sandboxes.append(sandbox)
        return sandbox

    def _list(self, tags: dict[str, str], **_kwargs: Any) -> list[FakeSandbox]:
        return [
            s
            for s in self.sandboxes
            if not s.terminated and all(s.tags.get(k) == v for k, v in tags.items())
        ]


def _empty(_tar: tarfile.TarFile) -> None:
    pass


def _out_tar(build: Callable[[tarfile.TarFile], None]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        build(tar)
    return buffer.getvalue()


def _add_file(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


class TestRun:
    @pytest.fixture
    def fake(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeModal:
        fake = FakeModal()
        monkeypatch.setitem(sys.modules, "modal", fake)
        task = SimpleNamespace(required_hardware="cpu")
        monkeypatch.setattr(modal_runtime, "load_task", lambda *_: task)  # pyright: ignore[reportUnknownLambdaType]
        limits = HardwareLimits(memory_bytes=8 * GIB, cpus=4, disk_bytes=50 * GIB)
        register_hardware_plugins(monkeypatch, limits={"a": lambda *_: limits})  # pyright: ignore[reportUnknownLambdaType]
        monkeypatch.setattr(modal_runtime, "_owner", lambda: "me@laptop")
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
        monkeypatch.delenv("LOGURU_LEVEL", raising=False)
        return fake

    @pytest.fixture
    def env_dir(self, tmp_path: Path) -> Path:
        env = tmp_path / "env"
        env.mkdir()
        (env / "Containerfile").write_text("FROM a\nRUN true\n")
        return env

    def _run(self, env_dir: Path, tmp_path: Path, **kwargs: Any) -> Path:
        transcript = tmp_path / "out" / "transcript.json"
        run_modal(
            _config(
                transcript_file=str(transcript),
                websocket_config={"port": _free_port()},
            ),
            log_file=tmp_path / "run.log",
            build_context=str(env_dir),
            **kwargs,
        )
        return transcript

    def test_the_sandbox_is_a_vm_sized_for_the_task(
        self, fake: FakeModal, env_dir: Path, tmp_path: Path
    ):
        self._run(env_dir, tmp_path)

        [sandbox] = fake.sandboxes
        kwargs = sandbox.kwargs
        assert kwargs["runtime"] == "vm"
        assert kwargs["cpu"] == 4
        assert kwargs["memory"] == 9 * 1024
        assert kwargs["tags"] == {
            "karotte_run_id": "modal-test",
            "karotte_owner": "me@laptop",
        }
        assert kwargs["image"].dockerfile.splitlines()[1] == NOCAPS_STEP
        assert fake.images[0]["context_dir"] == env_dir

    def test_the_run_reads_its_config_from_a_file(
        self, fake: FakeModal, env_dir: Path, tmp_path: Path
    ):
        """The API key stays out of the exec's command line."""
        self._run(env_dir, tmp_path, proxy_url="https://proxy.example")

        [sandbox] = fake.sandboxes
        written = json.loads(sandbox.files[CONFIG_PATH])
        assert written["model_api_key"] == "sk-secret"
        assert written["transcript_file"] == "/out/transcript.json"
        argv, kwargs = sandbox.execs[0]
        assert "sk-secret" not in " ".join(argv)
        assert argv[-2:] == ("--config", CONFIG_PATH)
        assert "/out" in sandbox.dirs
        env = kwargs["env"]
        assert env["KAROTTE_SANDBOX"] == "vm"
        assert env["KAROTTE_VM_LAUNCHER"] == "modal"
        assert env["KAROTTE_SANDBOX_MEMORY_BYTES"] == str(8 * GIB)
        assert env["KAROTTE_DISK_BUDGET_BYTES"] == str(50 * GIB)
        assert env["ANTHROPIC_BASE_URL"] == "https://proxy.example"
        assert env["KAROTTE_PROXY_URL"] == "https://proxy.example"
        assert env[EXIT_ON_RUN_ERROR_ENV_VAR] == "1"

    @pytest.mark.usefixtures("fake")
    def test_output_goes_to_the_log_file(self, env_dir: Path, tmp_path: Path):
        self._run(env_dir, tmp_path)

        assert (tmp_path / "run.log").read_text() == "line one\nline two\n"

    def test_out_comes_back_without_links(
        self, fake: FakeModal, env_dir: Path, tmp_path: Path
    ):
        def build(tar: tarfile.TarFile) -> None:
            _add_file(tar, "./transcript.json", b"{}")
            _add_file(tar, "./modal-test_artifacts/a.txt", b"artifact")
            link = tarfile.TarInfo("./escape")
            link.type = tarfile.SYMTYPE
            link.linkname = "/etc/passwd"
            tar.addfile(link)

        fake.out_tar = _out_tar(build)

        transcript = self._run(env_dir, tmp_path)

        out = transcript.parent
        assert transcript.read_text() == "{}"
        assert (out / "modal-test_artifacts" / "a.txt").read_text() == "artifact"
        assert not (out / "escape").exists()

    def test_a_failed_run_still_copies_out_and_terminates(
        self, fake: FakeModal, env_dir: Path, tmp_path: Path
    ):
        fake.returncode = 1

        def build(tar: tarfile.TarFile) -> None:
            _add_file(tar, "./transcript.json", b"{}")

        fake.out_tar = _out_tar(build)

        with pytest.raises(subprocess.CalledProcessError) as raised:
            self._run(env_dir, tmp_path)

        assert raised.value.returncode == 1
        assert (tmp_path / "out" / "transcript.json").exists()
        assert fake.sandboxes[0].terminated

    def test_the_sandbox_is_terminated_when_the_run_breaks(
        self, fake: FakeModal, env_dir: Path, tmp_path: Path
    ):
        fake.exec_error = FakeError("lost the sandbox")

        with pytest.raises(ModalError, match="lost the sandbox"):
            self._run(env_dir, tmp_path)

        assert fake.sandboxes[0].terminated

    def test_stop_and_clean_up_touch_only_our_runs(self, fake: FakeModal):
        def sandbox(run_id: str, owner: str = "me@laptop") -> FakeSandbox:
            return fake.create(tags={"karotte_run_id": run_id, "karotte_owner": owner})

        mine = sandbox("batch-0")
        other_batch = sandbox("other-0")
        not_mine = sandbox("batch-1", owner="you@desktop")

        modal_runtime.stop_sandboxes(["batch-0"])
        assert mine.terminated and not other_batch.terminated

        mine_again = sandbox("batch-2")
        modal_runtime.clean_up_sandboxes("batch-")
        assert mine_again.terminated
        assert not other_batch.terminated and not not_mine.terminated


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestWebsocketRelay:
    def _upstream(self, port: int, seen: list[tuple[str, str | None]]):
        async def handler(ws: ServerConnection) -> None:
            assert ws.request is not None
            seen.append((ws.request.path, ws.request.headers.get("Authorization")))
            await ws.send("event")

        stop = threading.Event()

        def serve_forever() -> None:
            async def main() -> None:
                async with serve(handler, "127.0.0.1", port):
                    while not stop.is_set():
                        await asyncio.sleep(0.05)

            asyncio.run(main())

        thread = threading.Thread(target=serve_forever, daemon=True)
        return thread, stop

    def test_a_client_waits_for_the_sandbox_and_gets_its_events(self):
        """The client connects before the sandbox exists, as the TUI does."""
        local, remote = _free_port(), _free_port()
        seen: list[tuple[str, str | None]] = []
        relay = WebsocketRelay(local)
        received: list[str] = []

        async def client() -> None:
            async with websockets.connect(f"ws://127.0.0.1:{local}/events") as ws:
                received.append(str(await ws.recv()))

        client_thread = threading.Thread(target=lambda: asyncio.run(client()))
        client_thread.start()
        thread, stop = self._upstream(remote, seen)
        try:
            thread.start()
            relay.connect_to(f"http://127.0.0.1:{remote}", "tok")
            client_thread.join(timeout=10)
        finally:
            relay.close()
            stop.set()

        assert received == ["event"]
        assert seen == [("/events", "Bearer tok")]

    def test_closing_frees_the_port(self):
        port = _free_port()
        WebsocketRelay(port).close()
        WebsocketRelay(port).close()

    def test_port_in_use(self):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            s.listen()
            with pytest.raises(ModalError, match="websocket"):
                WebsocketRelay(s.getsockname()[1])
