import fnmatch
import json
import shutil
import socket
import stat
import subprocess
from collections.abc import Callable, Iterator
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, final

import pytest
import typer

from karotte import modal_sandbox, run_helpers
from karotte.apple_container import STAGED_MOUNTS_DIR
from karotte.build import build_container, require_runtime
from karotte.firecracker.vm import KAROTTE_BIN
from karotte.hardware import HardwareLimits, VmSize
from karotte.hide_run_config import RUN_CONFIG_PATH
from karotte.modal_sandbox import (
    IDLE_TIMEOUT_SECONDS,
    LAUNCHER_TAG,
    RUN_TAG,
    TIMEOUT_SECONDS,
    ModalError,
    guest_env,
    modal_problems,
    run_modal,
)
from karotte.schemas.evaluation_run_config import EvaluationRunConfig
from karotte.staged_mounts import STAGED_MOUNTS_ENV_VAR
from karotte.templates import TEMPLATES_DIR
from karotte.terminal import app as terminal_app
from tests.conftest import register_hardware_plugins

GIB = 1 << 30


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _config(**update: object) -> EvaluationRunConfig:
    return EvaluationRunConfig.model_validate(
        {
            "run_id": "modal-test",
            "task_id": "example-task",
            "model": "claude-sonnet-5",
            "model_api_key": "dummy",
            "websocket_config": {"port": _free_port()},
            **update,
        }
    )


class FakeStream(list[str]):
    def read(self) -> str:
        return "".join(self)


def _proc(code: int, stdout: str = "", stderr: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        stdout=FakeStream(stdout.splitlines(keepends=True)),
        stderr=FakeStream(stderr.splitlines(keepends=True)),
        wait=lambda: code,
    )


@final
class FakeSandbox:
    """A sandbox whose filesystem is a directory on the host; commands other
    than karotte run there, with absolute paths moved under it."""

    def __init__(
        self, root: Path, tags: dict[str, str], on_run: Callable[["FakeSandbox"], Any]
    ) -> None:
        self.root = root
        self.tags = tags
        self.on_run = on_run
        self.terminated = False
        self.object_id = "sb-fake"
        self.execs: list[tuple[str, ...]] = []
        self.ports: list[int] = []
        self.filesystem = SimpleNamespace(
            copy_from_local=self._copy_from_local,
            copy_to_local=self._copy_to_local,
            write_text=self._write_text,
        )

    def path(self, remote: str) -> Path:
        return self.root / remote.lstrip("/")

    def _write_text(self, data: str, remote: str) -> None:
        dest = self.path(remote)
        dest.parent.mkdir(parents=True, exist_ok=True)
        _ = dest.write_text(data)

    def _copy_from_local(self, local: Path, remote: str) -> None:
        self._write_text("", remote)
        _ = shutil.copyfile(local, self.path(remote))

    def _copy_to_local(self, remote: str, local: Path) -> None:
        _ = shutil.copyfile(self.path(remote), local)

    def exec(self, *args: str, env: dict[str, str] | None = None) -> Any:
        _ = env
        self.execs.append(args)
        if args[0] == KAROTTE_BIN:
            return self.on_run(self)
        mapped = [str(self.path(a)) if a.startswith("/") else a for a in args]
        result = subprocess.run(mapped, capture_output=True, text=True, check=False)
        return _proc(result.returncode, result.stdout, result.stderr)

    def tunnels(self) -> dict[int, SimpleNamespace]:
        return {
            p: SimpleNamespace(tls_socket=("tunnel.example", 443)) for p in self.ports
        }

    def get_tags(self) -> dict[str, str]:
        return self.tags

    def terminate(self) -> None:
        self.terminated = True


@final
class FakeModal:
    def __init__(self, root: Path, on_run: Callable[[FakeSandbox], Any]) -> None:
        self.root = root
        self.on_run = on_run
        self.created: list[dict[str, Any]] = []
        self.sandboxes: list[FakeSandbox] = []
        self.enable_output = nullcontext
        self.App = SimpleNamespace(lookup=self._lookup)
        self.Image = SimpleNamespace(
            from_dockerfile=self._from_dockerfile, from_id=self._from_id
        )
        self.Sandbox = SimpleNamespace(create=self._create, list=self._list)

    @staticmethod
    def _lookup(name: str, create_if_missing: bool) -> SimpleNamespace:
        _ = (name, create_if_missing)
        return SimpleNamespace(app_id="ap-1")

    @staticmethod
    def _from_dockerfile(path: Path, **kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(path=path, **kwargs)

    @staticmethod
    def _from_id(image_id: str) -> SimpleNamespace:
        return SimpleNamespace(image_id=image_id)

    @staticmethod
    def FilePatternMatcher(*patterns: str) -> Callable[[Path], bool]:  # noqa: N802 - Modal's name
        return lambda path: any(fnmatch.fnmatch(path.as_posix(), p) for p in patterns)

    def _create(self, *args: str, **kwargs: Any) -> FakeSandbox:
        self.created.append({"args": args, **kwargs})
        sb = FakeSandbox(self.root, kwargs["tags"], self.on_run)
        sb.ports = list(kwargs["encrypted_ports"])
        self.sandboxes.append(sb)
        return sb

    def _list(self, *, app_id: str, tags: dict[str, str]) -> Iterator[FakeSandbox]:
        assert app_id == "ap-1"
        for sb in self.sandboxes:
            if all(sb.tags.get(k) == v for k, v in tags.items()):
                yield sb


def _use(monkeypatch: pytest.MonkeyPatch, fake: FakeModal) -> FakeModal:
    monkeypatch.setattr(modal_sandbox, "import_modal", lambda: fake)
    return fake


def _without_modal(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing() -> None:
        raise ModalError("needs the modal package")

    monkeypatch.setattr(modal_sandbox, "import_modal", missing)


class TestProblems:
    @pytest.fixture(autouse=True)
    def _fake(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        register_hardware_plugins(monkeypatch)
        _ = _use(monkeypatch, FakeModal(tmp_path, lambda sb: _proc(0)))

    @pytest.mark.parametrize(
        ("proxy", "blocked"),
        [
            (None, False),
            ("https://proxy.example.com", False),
            ("http://8.8.8.8:4000", False),
            ("http://localhost:4000", True),
            ("http://10.0.0.5:8080", True),
            ("http://my-mac.local:4000", True),
        ],
    )
    def test_proxy(self, proxy: str | None, blocked: bool):
        assert bool(modal_problems(None, proxy)) == blocked

    def test_missing_package(self, monkeypatch: pytest.MonkeyPatch):
        _without_modal(monkeypatch)
        assert modal_problems(None, None) == ["needs the modal package"]

    def test_prepare_only_and_keep_containers(self):
        problems = modal_problems(None, None, prepare_only=True, keep_containers=True)
        assert len(problems) == 2

    def test_passthrough_hardware(self, monkeypatch: pytest.MonkeyPatch):
        def limits(hardware: str) -> HardwareLimits:
            return HardwareLimits(passthrough=hardware == "gpu")

        register_hardware_plugins(monkeypatch, limits={"gpu": limits})
        assert any("--runtime docker" in p for p in modal_problems("gpu", None))

    def test_require_runtime_names_the_extra(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ):
        _without_modal(monkeypatch)
        with pytest.raises(typer.Exit):
            require_runtime("modal")
        assert capsys.readouterr().err.strip() == "needs the modal package"


def test_guest_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LOGURU_LEVEL", "DEBUG")
    size = VmSize(
        cpus=2, sandbox_memory_bytes=4 * GIB, vm_memory_bytes=5 * GIB, disk_bytes=None
    )
    env = guest_env(size, "https://proxy.example.com", [])
    assert env["KAROTTE_SANDBOX"] == "vm"
    assert env["KAROTTE_SANDBOX_MEMORY_BYTES"] == str(4 * GIB)
    assert env["KAROTTE_DISK_BUDGET_BYTES"] == str(32 * GIB)
    assert env["KAROTTE_VM_LAUNCHER"] == "modal"
    assert env["KAROTTE_FIREWALL_BACKEND"] == "nft"
    assert env["KAROTTE_PROXY_URL"] == "https://proxy.example.com"
    assert env["LOGURU_LEVEL"] == "DEBUG"


@pytest.mark.usefixtures("built")
class TestRun:
    @pytest.fixture
    def built(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """No hardware plugin, and what `karotte build --runtime modal` leaves."""
        register_hardware_plugins(monkeypatch)

        def load_task(config: EvaluationRunConfig) -> SimpleNamespace:
            _ = config
            return SimpleNamespace(required_hardware=None)

        monkeypatch.setattr(modal_sandbox, "load_task", load_task)
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
        record = modal_sandbox.image_record(str(tmp_path))
        record.parent.mkdir(parents=True)
        _ = record.write_text("im-1\n")

    @pytest.fixture
    def mounts(self, tmp_path: Path) -> dict[str, Path]:
        rw = tmp_path / "host" / "rw"
        rw.mkdir(parents=True)
        _ = (rw / "keep.txt").write_text("old")
        _ = (rw / "gone.txt").write_text("delete me")
        # The sandbox copies no links back; that must not read as deleted.
        (rw / "link").symlink_to("/etc/hostname")
        _ = (rw / "locked.txt").write_text("untouched")
        (rw / "locked.txt").chmod(0o444)
        (tmp_path / "host" / "rw-link").symlink_to(rw)
        ro = tmp_path / "host" / "ro.txt"
        _ = ro.write_text("read only")
        return {"rw": rw, "ro": ro, "rw-link": tmp_path / "host" / "rw-link"}

    @staticmethod
    def _student(code: int) -> Callable[[FakeSandbox], Any]:
        """What the in-sandbox run leaves: a transcript, and the student's
        changes, which its copy-back put in the staging directory."""

        def run(sb: FakeSandbox) -> Any:
            staging = sb.path(f"{STAGED_MOUNTS_DIR}/0")
            _ = (staging / "keep.txt").write_text("new")
            (staging / "gone.txt").unlink()
            _ = (staging / "added.txt").write_text("added")
            if (read_only := sb.path(f"{STAGED_MOUNTS_DIR}/1/ro.txt")).exists():
                _ = read_only.write_text("changed")
            _ = sb.path("/out/transcript.json").write_text('{"events": []}')
            return _proc(code, "run output\n", "run log\n")

        return run

    def test_a_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mounts: dict[str, Path]
    ):
        fake = _use(monkeypatch, FakeModal(tmp_path / "remote", self._student(0)))
        log = tmp_path / "run.log"
        run_modal(
            _config(transcript_file=str(tmp_path / "results" / "transcript.json")),
            log_file=log,
            build_context=str(tmp_path),
            mounts=[
                f"{mounts['rw-link']}:/workdir/rw",
                f"{mounts['ro']}:/etc/ro.txt:ro",
            ],
            proxy_url="https://proxy.example.com",
        )

        [created] = fake.created
        assert created["args"] == ("sleep", "infinity")
        assert created["runtime"] == "vm"
        assert created["timeout"] == TIMEOUT_SECONDS
        assert created["idle_timeout"] == IDLE_TIMEOUT_SECONDS
        assert (created["cpu"], created["memory"]) == (2.0, 5 * 1024)
        assert created["tags"][RUN_TAG] == "modal-test"
        assert created["image"].image_id == "im-1"
        assert json.loads(created["env"][STAGED_MOUNTS_ENV_VAR])[0]["writable"]

        [sb] = fake.sandboxes
        assert sb.terminated
        [run] = [e for e in sb.execs if e[0] == KAROTTE_BIN]
        assert run[-2:] == (
            "--config",
            RUN_CONFIG_PATH,
        )  # the API key stays out of argv
        config_file = sb.path(RUN_CONFIG_PATH)
        assert stat.S_IMODE(config_file.stat().st_mode) == 0o600
        assert (
            json.loads(config_file.read_text())["transcript_file"]
            == "/out/transcript.json"
        )

        assert (
            tmp_path / "results" / "transcript.json"
        ).read_text() == '{"events": []}'
        assert (mounts["rw"] / "keep.txt").read_text() == "new"
        assert (mounts["rw"] / "added.txt").read_text() == "added"
        assert not (mounts["rw"] / "gone.txt").exists()
        assert (mounts["rw"] / "link").is_symlink()
        assert stat.S_IMODE((mounts["rw"] / "locked.txt").stat().st_mode) == 0o444
        assert mounts["rw-link"].is_symlink()
        assert mounts["ro"].read_text() == "read only"
        assert "run output" in log.read_text()
        assert "run log" in log.read_text()

    def test_a_failed_run_raises_and_terminates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mounts: dict[str, Path]
    ):
        fake = _use(monkeypatch, FakeModal(tmp_path / "remote", self._student(3)))
        with pytest.raises(subprocess.CalledProcessError) as info:
            run_modal(
                _config(),
                build_context=str(tmp_path),
                mounts=[f"{mounts['rw']}:/workdir/rw"],
            )
        assert info.value.returncode == 3
        assert fake.sandboxes[0].terminated
        # The student's work still comes back from a failed run.
        assert (mounts["rw"] / "keep.txt").read_text() == "new"

    def test_no_built_image_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "elsewhere"))
        fake = _use(monkeypatch, FakeModal(tmp_path / "remote", lambda sb: _proc(0)))
        with pytest.raises(ModalError, match="karotte build --runtime modal"):
            run_modal(_config(), build_context=str(tmp_path))
        assert fake.created == []

    def test_an_unsafe_entry_is_skipped_on_copy_back(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        def run(sb: FakeSandbox) -> Any:
            sb.path("/out/link").symlink_to("/etc/passwd")
            sb.path("/out/inside").symlink_to("transcript.json")
            _ = sb.path("/out/transcript.json").write_text("{}")
            return _proc(0)

        _ = _use(monkeypatch, FakeModal(tmp_path / "remote", run))
        results = tmp_path / "results"
        run_modal(
            _config(transcript_file=str(results / "transcript.json")),
            build_context=str(tmp_path),
        )
        assert (results / "transcript.json").read_text() == "{}"
        assert not (results / "link").is_symlink()
        assert not (results / "inside").exists()

    def test_a_failed_mount_still_brings_the_transcript_back(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mounts: dict[str, Path]
    ):
        copy = FakeSandbox._copy_to_local  # pyright: ignore[reportPrivateUsage]
        calls: list[str] = []

        def mount_fails_first(sb: FakeSandbox, remote: str, local: Path) -> None:
            calls.append(remote)
            if len(calls) == 1:  # mounts come back before /out
                raise RuntimeError("file too large")
            copy(sb, remote, local)

        monkeypatch.setattr(FakeSandbox, "_copy_to_local", mount_fails_first)
        _ = _use(monkeypatch, FakeModal(tmp_path / "remote", self._student(0)))
        results = tmp_path / "results"
        run_modal(
            _config(transcript_file=str(results / "transcript.json")),
            build_context=str(tmp_path),
            mounts=[f"{mounts['rw']}:/workdir/rw"],
        )
        assert (results / "transcript.json").read_text() == '{"events": []}'


def test_cleanup_spares_other_launchers_and_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    fake = _use(monkeypatch, FakeModal(tmp_path, lambda sb: _proc(0)))
    me = modal_sandbox.launcher_id()
    for run_id, launcher in [
        ("job-0", me),
        ("job-1", me),
        ("other-0", me),
        ("job-0", "someone@elsewhere"),
    ]:
        fake.sandboxes.append(
            FakeSandbox(
                tmp_path, {RUN_TAG: run_id, LAUNCHER_TAG: launcher}, fake.on_run
            )
        )

    def terminated() -> list[str]:
        return [sb.tags[RUN_TAG] for sb in fake.sandboxes if sb.terminated]

    run_helpers.stop_containers("modal", ["job-1"])
    assert terminated() == ["job-1"]
    run_helpers.clean_up_old_containers("modal", ["job-0", "job-1"])
    assert terminated() == ["job-0", "job-1"]


def test_tui_builds_on_modal_quietly(monkeypatch: pytest.MonkeyPatch):
    """The TUI builds through its own path, not `build_container`."""
    calls: list[tuple[str, bool]] = []

    def build_image(
        build_context: str, build_secrets: object, show_output: bool
    ) -> None:
        _ = build_secrets
        calls.append((build_context, show_output))

    monkeypatch.setattr(modal_sandbox, "build_image", build_image)
    output: list[str] = []
    app = terminal_app.KarotteApp(configs=[_config()])
    app._build_container("modal", "env", output_callback=output.append)  # pyright: ignore[reportPrivateUsage]
    assert calls == [("env", False)]
    assert output[-1] == "Environment container image built successfully.\n"


def test_the_default_template_builds_with_the_builder_arg(tmp_path: Path):
    fake = FakeModal(tmp_path, lambda sb: _proc(0))
    image = modal_sandbox.modal_image(fake, str(TEMPLATES_DIR / "default"))  # pyright: ignore[reportArgumentType]
    assert image.build_args == {"KAROTTE_IMAGE_BUILDER": "modal"}


def test_an_older_environment_is_refused_before_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    _ = (tmp_path / "Containerfile").write_text(
        "FROM x\n# ARG KAROTTE_IMAGE_BUILDER comes later\n"
    )
    _ = _use(monkeypatch, FakeModal(tmp_path, lambda sb: _proc(0)))
    with pytest.raises(typer.Exit) as info:
        build_container("modal", str(tmp_path))
    assert info.value.exit_code == 1
    assert "ARG KAROTTE_IMAGE_BUILDER" in capsys.readouterr().err


def test_context_ignore_keeps_placeholders_of_otherwise_empty_dirs(tmp_path: Path):
    for path, text in [
        ("student_data/.gitkeep", ""),
        ("root_data/.gitkeep", ""),
        ("root_data/real.csv", ""),
        ("answers/answers.json", "42"),
    ]:
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        _ = (tmp_path / path).write_text(text)
    _ = (tmp_path / ".dockerignore").write_text(
        "student_data/.gitkeep\nroot_data/.gitkeep\nanswers/answers.json\n"
    )
    fake = FakeModal(tmp_path, lambda sb: _proc(0))
    ignore = modal_sandbox.context_ignore(fake, tmp_path)  # pyright: ignore[reportArgumentType]

    # Docker keeps student_data/ as an empty directory; Modal needs a file.
    assert not ignore(Path("student_data/.gitkeep"))
    assert not ignore(tmp_path / "student_data/.gitkeep")
    # root_data/ has a real file, so its placeholder stays ignored.
    assert ignore(Path("root_data/.gitkeep"))
    # An ignored file with content never stands in for its directory.
    assert ignore(Path("answers/answers.json"))
