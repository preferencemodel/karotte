import json
import os
import shutil
import signal
import subprocess
from pathlib import Path
from typing import Any, final

import pytest
import typer

from karotte import apple_container
from karotte.apple_container import (
    LivenessWatchdog,
    Probe,
    assign_host_ports,
    check_build_context,
    clean_up_old_containers,
    disk_budget_bytes,
    export_hint,
    file_mounts_dir,
    finish_file_shares,
    karotte_container_ids,
    kill_vm,
    share_file,
    stop_containers,
    validate_container_runtime,
    vm_resources,
)
from karotte.build import get_container_build_command
from karotte.confinement import GIB, SANDBOX_MEMORY_ENV_VAR
from karotte.hardware import HardwareLimits
from karotte.run_helpers import copy_hint, get_container_run_command
from karotte.schemas.evaluation_run_config import EvaluationRunConfig
from karotte.staged_mounts import STAGED_MOUNTS_ENV_VAR
from tests.conftest import register_hardware_plugins


def completed(
    args: list[str], stdout: str = "", returncode: int = 0, stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args, returncode, stdout, stderr)


@final
class FakeRun:
    """Stands in for ``subprocess.run``: answers by the command's first words
    and records every call."""

    def __init__(self, answers: dict[tuple[str, ...], Any]) -> None:
        self.answers = answers
        self.calls: list[list[str]] = []

    def __call__(self, args: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(args))
        for prefix, answer in sorted(self.answers.items(), key=lambda a: -len(a[0])):
            if tuple(args[: len(prefix)]) == prefix:
                if isinstance(answer, BaseException):
                    raise answer
                if isinstance(answer, subprocess.CompletedProcess):
                    return answer
                return completed(args, answer)
        return completed(args)


def env_value(command: list[str], name: str) -> str | None:
    for flag, value in zip(command, command[1:], strict=False):
        if flag == "--env" and value.startswith(f"{name}="):
            return value.removeprefix(f"{name}=")
    return None


def flag_values(command: list[str], flag: str) -> list[str]:
    return [v for f, v in zip(command, command[1:], strict=False) if f == flag]


@pytest.fixture
def config(sample_config: EvaluationRunConfig, tmp_path: Path) -> EvaluationRunConfig:
    return sample_config.model_copy(
        update={"transcript_file": str(tmp_path / "out" / "transcript.json")}
    )


def _hardware(hardware: str) -> HardwareLimits:
    if hardware == "gpu-1":
        return HardwareLimits(passthrough=True)
    return HardwareLimits(memory_bytes=5 * GIB, disk_bytes=80 * GIB, cpus=2)


@pytest.fixture(autouse=True)
def hardware_plugin(monkeypatch: pytest.MonkeyPatch) -> None:
    """2 vCPUs, 5 GiB and an 80 GiB disk for any hardware but `gpu-1`, which
    needs passthrough."""
    register_hardware_plugins(monkeypatch, limits={"a": _hardware})


@pytest.fixture(autouse=True)
def native_image(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(apple_container, "image_platform", lambda _: None)  # pyright: ignore[reportUnknownLambdaType]


def run_command(config: EvaluationRunConfig, **kwargs: Any) -> list[str]:
    kwargs.setdefault("dev", False)
    kwargs.setdefault("keep_container", False)
    command, _ = get_container_run_command(config, "apple-container", **kwargs)
    return command


@pytest.fixture(autouse=True)
def private_file_mounts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(apple_container, "FILE_MOUNTS_DIR", tmp_path / "file-mounts")


class TestShareFile:
    def test_a_writable_file_is_linked_so_writes_reach_it(self, tmp_path: Path):
        source = tmp_path / "data.csv"
        source.write_text("a")

        shared = share_file(source, "r", 0, writable=True)

        assert [p.name for p in shared.iterdir()] == ["data.csv"]
        assert (shared / "data.csv").stat().st_ino == source.stat().st_ino

    def test_a_linked_source_is_shared_as_its_target_under_its_own_name(
        self, tmp_path: Path
    ):
        """The link itself would point at nothing inside the share."""
        real = tmp_path / "real.csv"
        real.write_text("a")
        link = tmp_path / "data.csv"
        link.symlink_to(real)

        shared = share_file(link, "r", 0, writable=False) / "data.csv"

        assert not shared.is_symlink()
        assert shared.read_text() == "a"

    def test_what_the_guest_copied_back_replaces_the_original_after_the_run(
        self, tmp_path: Path
    ):
        source = tmp_path / "data.csv"
        source.write_text("old")
        shared = share_file(source, "r", 0, writable=True) / "data.csv"
        # The guest's copy-back renames a new file over the shared one.
        replacement = shared.with_name(".new")
        replacement.write_text("new")
        os.replace(replacement, shared)

        finish_file_shares("r")

        assert source.read_text() == "new"
        assert not file_mounts_dir("r").exists()

    def test_a_share_with_nothing_copied_back_leaves_the_original(self, tmp_path: Path):
        source = tmp_path / "data.csv"
        source.write_text("old")
        inode = source.stat().st_ino
        _ = share_file(source, "r", 0, writable=True)

        finish_file_shares("r")

        assert source.read_text() == "old"
        assert source.stat().st_ino == inode

    def test_without_a_link_a_read_only_file_is_copied(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        def no_link(_src: object, _dst: object) -> None:
            raise OSError(18, "Cross-device link")

        monkeypatch.setattr("karotte.apple_container.os.link", no_link)
        source = tmp_path / "data.csv"
        source.write_text("a")

        shared = share_file(source, "r", 0, writable=False)

        assert (shared / "data.csv").read_text() == "a"
        with pytest.raises(typer.Abort):
            _ = share_file(source, "r", 1, writable=True)


class TestRunCommand:
    def test_a_run_error_exits_non_zero(self, config: EvaluationRunConfig):
        """As under docker: otherwise a crashed task reports success."""
        assert env_value(run_command(config), "KAROTTE_EXIT_ON_RUN_ERROR") == "1"

    def test_runs_a_vm_sandbox_with_a_launcher_disk_budget(
        self, config: EvaluationRunConfig, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(
            apple_container,
            "disk_budget_bytes",
            lambda *_, **__: 12345,  # pyright: ignore[reportUnknownLambdaType]
        )

        command = run_command(config)

        assert command[:2] == ["container", "run"]
        assert env_value(command, "KAROTTE_SANDBOX") == "vm"
        assert env_value(command, "KAROTTE_DISK_BUDGET_BYTES") == "12345"
        assert "--name" in command
        assert command[command.index("--name") + 1] == "karotte_run_test_run"

    def test_adds_the_capabilities_karotte_needs_and_no_security_opt(
        self, config: EvaluationRunConfig
    ):
        command = run_command(config)

        assert flag_values(command, "--cap-add") == [
            "CAP_NET_ADMIN",
            "CAP_SYS_ADMIN",
            "CAP_SYS_RESOURCE",
        ]
        assert "--security-opt" not in command
        assert not any(arg.startswith("--runtime") for arg in command)

    def test_a_task_without_hardware_gets_the_default_vm(
        self, config: EvaluationRunConfig
    ):
        command = run_command(config)

        # 2 vCPUs, and the default 4 GiB sandbox + 1 GiB of headroom.
        assert flag_values(command, "--cpus") == ["2"]
        assert flag_values(command, "--memory") == ["5120M"]
        assert f"{SANDBOX_MEMORY_ENV_VAR}={4 * GIB}" in flag_values(command, "--env")
        # Tells `karotte check confinement` this VM holds the headroom.
        assert env_value(command, "KAROTTE_VM_LAUNCHER") == "apple-container"

    def test_publishes_the_websocket_port_on_localhost_only(
        self, config: EvaluationRunConfig
    ):
        port = config.websocket_config.port

        command = run_command(config)

        assert flag_values(command, "--publish") == [f"127.0.0.1:{port}:{port}"]

    def test_prepare_only_publishes_nothing(self, config: EvaluationRunConfig):
        command = run_command(config, prepare_only=True)

        assert "--publish" not in command
        assert "--prepare-only" in command

    @pytest.mark.parametrize(
        ("platform", "expected"),
        [(None, []), ("linux/amd64", ["linux/amd64"])],
    )
    def test_platform_follows_the_image(
        self,
        config: EvaluationRunConfig,
        monkeypatch: pytest.MonkeyPatch,
        platform: str | None,
        expected: list[str],
    ):
        monkeypatch.setattr(apple_container, "image_platform", lambda _: platform)  # pyright: ignore[reportUnknownLambdaType]

        assert flag_values(run_command(config), "--platform") == expected

    @pytest.mark.parametrize(("keep", "has_rm"), [(False, True), (True, False)])
    def test_rm_unless_kept(
        self, config: EvaluationRunConfig, keep: bool, has_rm: bool
    ):
        assert ("--rm" in run_command(config, keep_container=keep)) is has_rm

    def test_transcript_dir_is_mounted_under_root(
        self, config: EvaluationRunConfig, tmp_path: Path
    ):
        command, updated = get_container_run_command(
            config, "apple-container", dev=False, keep_container=False
        )

        assert f"type=bind,source={tmp_path / 'out'},target=/root/out" in flag_values(
            command, "--mount"
        )
        assert updated.transcript_file == "/root/out/transcript.json"
        assert json.loads(command[-1])["transcript_file"] == "/root/out/transcript.json"

    def test_dev_source_stays_under_root(
        self, config: EvaluationRunConfig, tmp_path: Path
    ):
        command = run_command(config, dev=True, build_context=str(tmp_path))

        assert (
            f"type=bind,source={tmp_path}/src/environment,"
            + "target=/root/.venv/lib/python3.12/site-packages/environment/"
        ) in flag_values(command, "--mount")

    def test_user_mounts_are_staged_under_root(
        self, config: EvaluationRunConfig, tmp_path: Path
    ):
        data = tmp_path / "data"
        data.mkdir()
        results = tmp_path / "results"
        results.mkdir()
        single = tmp_path / "config.yaml"
        single.write_text("x")

        command = run_command(
            config,
            mounts=[
                f"{data}:/workdir/data:ro",
                f"{results}:/results",
                f"{single}:/etc/app.yaml:ro",
            ],
        )

        mounts = flag_values(command, "--mount")
        assert (
            f"type=bind,source={data},target=/root/.karotte_mounts/0,readonly" in mounts
        )
        assert f"type=bind,source={results},target=/root/.karotte_mounts/1" in mounts
        shared = file_mounts_dir(config.run_id) / "2"
        assert (
            f"type=bind,source={shared},target=/root/.karotte_mounts/2,readonly"
            in mounts
        )
        # Only the file, not its siblings.
        assert [p.name for p in shared.iterdir()] == ["config.yaml"]
        assert not any("target=/workdir" in m or "target=/results" in m for m in mounts)
        staged = json.loads(env_value(command, STAGED_MOUNTS_ENV_VAR) or "")
        assert staged == [
            {
                "source": "/root/.karotte_mounts/0",
                "target": "/workdir/data",
                "writable": False,
            },
            {
                "source": "/root/.karotte_mounts/1",
                "target": "/results",
                "writable": True,
            },
            {
                "source": "/root/.karotte_mounts/2/config.yaml",
                "target": "/etc/app.yaml",
                "writable": False,
            },
        ]

    def test_no_staged_mounts_without_user_mounts(self, config: EvaluationRunConfig):
        assert env_value(run_command(config), STAGED_MOUNTS_ENV_VAR) is None

    def test_other_runtimes_stage_nothing(
        self, config: EvaluationRunConfig, tmp_path: Path
    ):
        command, _ = get_container_run_command(
            config,
            "podman",
            dev=False,
            keep_container=False,
            mounts=[f"{tmp_path}:/data"],
        )

        assert env_value(command, STAGED_MOUNTS_ENV_VAR) is None
        assert env_value(command, "KAROTTE_DISK_BUDGET_BYTES") is None

    def test_ends_with_the_in_guest_karotte_run(self, config: EvaluationRunConfig):
        command = run_command(config, proxy_url="http://proxy:4000")

        image = command.index("karotte")
        assert command[image + 1 : image + 5] == [
            "/root/.venv/bin/karotte",
            "run",
            "--no-containerized",
            "--config",
        ]
        assert env_value(command, "ANTHROPIC_BASE_URL") == "http://proxy:4000"

    def test_host_settings_are_forwarded(
        self, config: EvaluationRunConfig, monkeypatch: pytest.MonkeyPatch
    ):
        """As with docker: the guest reads them, e.g. the student firewall mode."""
        monkeypatch.setenv("KAROTTE_STUDENT_NETWORK", "internal")
        monkeypatch.setenv("LOGURU_LEVEL", "DEBUG")
        command, _ = get_container_run_command(config, "apple-container", False, False)
        assert env_value(command, "KAROTTE_STUDENT_NETWORK") == "internal"
        assert env_value(command, "LOGURU_LEVEL") == "DEBUG"


class TestVmResources:
    def test_sized_from_the_plugin_with_headroom(self):
        size = vm_resources("cpu-2.6gb")

        assert (size.cpus, size.vm_memory_bytes) == (2, 6 * GIB)

    def test_passthrough_hardware_is_refused(self):
        with pytest.raises(typer.Abort):
            _ = vm_resources("gpu-1")


class TestDiskBudget:
    def test_capped_at_most_of_the_host_free_space(self):
        assert disk_budget_bytes("cpu-2.6gb", free_bytes=10 * GIB) == int(8 * GIB)

    def test_capped_at_the_plugins_budget(self):
        assert disk_budget_bytes("cpu-2.6gb", free_bytes=1000 * GIB) == 80 * GIB

    def test_runs_launched_together_split_the_free_space(self):
        """The rootfs is sparse: four runs each promised 80 GiB of 150 GiB
        free could fill the disk together."""
        budget = disk_budget_bytes("cpu-2.6gb", free_bytes=150 * GIB, runs=4)

        assert budget == int(150 * GIB * 0.8) // 4
        assert 4 * budget <= 150 * GIB * 0.8

    def test_parallel_runs_reach_the_guest_budget(
        self,
        config: EvaluationRunConfig,
        monkeypatch: pytest.MonkeyPatch,
    ):
        usage = shutil.disk_usage(Path.home())
        monkeypatch.setattr(
            "karotte.apple_container.shutil.disk_usage",
            lambda _: usage._replace(free=150 * GIB),  # pyright: ignore[reportUnknownLambdaType]
        )

        command = run_command(config, parallel_runs=4)

        assert env_value(command, "KAROTTE_DISK_BUDGET_BYTES") == str(
            int(150 * GIB * 0.8) // 4
        )


class TestAssignHostPorts:
    def test_keeps_free_configured_ports(self, sample_config: EvaluationRunConfig):
        configs = assign_host_ports(
            [sample_config], is_free=lambda _: True, free_port=lambda: 1
        )

        assert configs[0].websocket_config.port == sample_config.websocket_config.port

    def test_replaces_busy_and_repeated_ports(self, sample_config: EvaluationRunConfig):
        port = sample_config.websocket_config.port
        spare = iter([port, 40001, 40002])

        configs = assign_host_ports(
            [sample_config, sample_config.model_copy(update={"run_id": "other"})],
            is_free=lambda p: p != port,
            free_port=lambda: next(spare),
        )

        assert [c.websocket_config.port for c in configs] == [40001, 40002]


def container(id_: str, state: str = "running") -> dict[str, Any]:
    return {"configuration": {"id": id_}, "status": {"state": state}}


LISTING = [
    container("buildkit"),
    container("karotte_run_abc-0"),
    container("karotte_run_abc-1", "stopped"),
    container("karotte_run_other"),
    container("3f2b7c1e-8a4d-4c55-9d0b-6c1a2f3e4d5b", "stopped"),
    container("my-db"),
]


class TestCleanup:
    def test_only_karotte_containers_match(self):
        assert karotte_container_ids(LISTING) == [
            "karotte_run_abc-0",
            "karotte_run_abc-1",
            "karotte_run_other",
        ]

    def test_a_prefix_outside_karotte_names_is_refused(self):
        with pytest.raises(AssertionError):
            _ = karotte_container_ids(LISTING, "3f2b")

    def test_deletes_this_invocations_containers_only(self):
        run = FakeRun({("container", "ls"): json.dumps(LISTING)})

        clean_up_old_containers("abc-", run=run)

        assert run.calls[-1] == [
            "container",
            "delete",
            "--force",
            "karotte_run_abc-0",
            "karotte_run_abc-1",
        ]

    def test_deletes_nothing_when_nothing_matches(self):
        run = FakeRun({("container", "ls"): json.dumps(LISTING)})

        clean_up_old_containers("zzz", run=run)

        assert [c[:2] for c in run.calls] == [["container", "ls"]]

    def test_a_failed_listing_deletes_nothing(self):
        run = FakeRun({("container", "ls"): completed([], returncode=1)})

        clean_up_old_containers("", run=run)

        assert len(run.calls) == 1

    def test_stop_targets_exact_run_ids_that_exist(self):
        run = FakeRun({("container", "ls"): json.dumps(LISTING)})

        stop_containers(["abc-0", "missing"], run=run)

        assert run.calls[-1] == ["container", "stop", "karotte_run_abc-0"]


class TestCopyHint:
    def test_container_uses_export_not_cp(self):
        hint = copy_hint("apple-container", "abc")

        assert hint == export_hint("abc")
        assert "container export -o karotte_run_abc.tar karotte_run_abc" in hint
        assert "--no-same-owner --no-same-permissions" in hint
        assert " cp " not in hint

    def test_other_runtimes_keep_cp(self):
        assert (
            copy_hint("docker", "abc") == "docker cp karotte_run_abc:/workdir/ ./out/"
        )


class TestBuild:
    def test_build_command(self, tmp_path: Path):
        command = get_container_build_command(
            "apple-container",
            str(tmp_path),
            cache_from=["type=registry,ref=x"],
            cache_to=["type=registry,ref=y"],
            build_secrets=["uv_env=/s"],
        )

        assert command == [
            "container",
            "build",
            "--secret=id=uv_env,src=/s",
            "--file",
            "Containerfile",
            "--tag",
            "karotte",
            str(tmp_path),
        ]

    def test_a_context_through_a_symlink_is_refused(self, tmp_path: Path):
        real = tmp_path / "real"
        real.mkdir()
        link = tmp_path / "link"
        link.symlink_to(real)

        with pytest.raises(typer.Abort):
            check_build_context(str(link / "."))

    def test_private_tmp_is_refused(self):
        with pytest.raises(typer.Abort):
            check_build_context("/private/tmp/env")

    def test_a_plain_directory_is_accepted(self, tmp_path: Path):
        real = Path(tmp_path.resolve()) / "env"
        real.mkdir()

        check_build_context(str(real))
        check_build_context(str(real / ".." / "env"))


STATUS_RUNNING = "FIELD   VALUE\nstatus              running\nclient.version  1.4.1\n"


def preflight_run(
    version: str = "container CLI version 1.4.1 (build: release, commit: 9a8917c)",
    status: str = STATUS_RUNNING,
    status_rc: int = 0,
) -> FakeRun:
    return FakeRun(
        {
            ("container", "--version"): version,
            ("container", "system", "status"): completed([], status, status_rc),
        }
    )


def validate(run: FakeRun | None = None, **kwargs: Any) -> None:
    kwargs.setdefault("mac_version", "26.7")
    kwargs.setdefault("machine", "arm64")
    kwargs.setdefault("host_memory_bytes", 64 * GIB)
    validate_container_runtime("cpu-2.6gb", run=run or preflight_run(), **kwargs)


class TestPreflight:
    def test_passes_on_a_supported_mac(self):
        validate()

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"mac_version": "15.6"},
            {"mac_version": ""},
            {"machine": "x86_64"},
            {"host_memory_bytes": 4 * GIB},
        ],
    )
    def test_refuses_an_unsupported_host(self, kwargs: dict[str, Any]):
        with pytest.raises(typer.Abort):
            validate(**kwargs)

    @pytest.mark.parametrize(
        "run",
        [
            preflight_run(version="container CLI version 1.4.0 (build: release)"),
            preflight_run(version="container CLI version 0.9.12"),
            preflight_run(version="garbage"),
            preflight_run(status="status  stopped\n"),
            preflight_run(status_rc=1),
            FakeRun({("container", "--version"): FileNotFoundError()}),
        ],
    )
    def test_refuses_a_missing_old_or_stopped_container(self, run: FakeRun):
        with pytest.raises(typer.Abort):
            validate(run)

    def test_a_refusal_names_the_alternative(self, capsys: pytest.CaptureFixture[str]):
        with pytest.raises(typer.Abort):
            validate(machine="x86_64")

        assert "--runtime docker" in capsys.readouterr().err

    def test_a_stuck_service_is_reported(self):
        run = preflight_run()
        run.answers[("container", "system", "status")] = subprocess.TimeoutExpired(
            ["container"], 60
        )

        with pytest.raises(typer.Abort):
            validate(run)

    def test_accepts_a_newer_version(self):
        validate(preflight_run(version="container CLI version 1.10.0 (build: x)"))


@final
class FakeProbes:
    def __init__(
        self, exec_results: list[Probe], websocket: bool | list[bool] = False
    ) -> None:
        """``websocket`` is one answer for every ping, or one per ping."""
        self.exec_results = exec_results
        self.websocket = websocket
        self.websocket_calls = 0
        self.killed: list[str] = []

    def exec_probe(self, name: str, _timeout: float) -> Probe:
        assert name == "karotte_run_r"
        return self.exec_results.pop(0) if self.exec_results else "answered"

    def websocket_probe(self, port: int, _timeout: float) -> bool:
        assert port == 8001
        self.websocket_calls += 1
        if isinstance(self.websocket, bool):
            return self.websocket
        return self.websocket.pop(0)

    def kill(self, name: str) -> None:
        self.killed.append(name)


def watchdog(probes: FakeProbes, port: int | None = 8001, **kwargs: Any):
    kwargs.setdefault("interval_s", 0)
    return LivenessWatchdog(
        "karotte_run_r",
        port,
        probe_timeout_s=1,
        max_unanswered=3,
        exec_probe=probes.exec_probe,
        websocket_probe=probes.websocket_probe,
        kill=probes.kill,
        **kwargs,
    )


class TestLivenessWatchdog:
    def test_answering_guest_is_left_alone(self):
        probes = FakeProbes(["answered", "failed", "answered"], websocket=True)
        dog = watchdog(probes)

        assert all(dog.check() for _ in range(3))
        # Asked once, to learn the guest has a websocket to fall back on.
        assert probes.websocket_calls == 1

    def test_a_timed_out_exec_counts_when_a_websocket_that_answered_stops(self):
        assert watchdog(FakeProbes(["timed_out"], websocket=True)).check()
        dog = watchdog(FakeProbes(["answered", "timed_out"], websocket=[True, False]))
        assert dog.check()
        assert not dog.check()

    def test_a_timeout_with_no_websocket_answer_yet_is_not_counted(self):
        """The `container` service may be stuck on another VM; nothing here
        says it's this one. A held env and a run whose server isn't up yet."""
        assert watchdog(FakeProbes(["timed_out"] * 3), port=None).check()
        dog = watchdog(FakeProbes(["timed_out"] * 3, websocket=False))
        assert all(dog.check() for _ in range(3))

    def test_kills_after_consecutive_unanswered_probes(self):
        probes = FakeProbes(
            ["answered", "timed_out", "timed_out", "timed_out"],
            websocket=[True, False, False, False],
        )
        on_kill: list[bool] = []
        dog = watchdog(probes, on_kill=lambda: on_kill.append(True))

        dog.start()
        dog._thread.join(timeout=5)  # pyright: ignore[reportPrivateUsage]

        assert probes.killed == ["karotte_run_r"]
        assert dog.killed
        assert on_kill == [True]

    def test_an_answer_resets_the_count(self):
        probes = FakeProbes(
            [
                "answered",
                "timed_out",
                "timed_out",
                "answered",
                "timed_out",
                "timed_out",
            ],
            websocket=[True, False, False, False, False],
        )
        dog = watchdog(probes)

        results = [dog.check() for _ in range(6)]

        assert results == [True, False, False, True, False, False]
        assert probes.killed == []

    def test_stopped_watchdog_never_kills(self):
        probes = FakeProbes(["timed_out"] * 100)
        dog = watchdog(probes, interval_s=3600)

        dog.start()
        dog.stop()
        dog._thread.join(timeout=5)  # pyright: ignore[reportPrivateUsage]

        assert probes.killed == []

    def test_only_karotte_containers_are_watched(self):
        with pytest.raises(AssertionError):
            _ = LivenessWatchdog("my-db", None)


APP_ROOT = "/Users/u/Library/Application Support/com.apple.container"
HELPER = "/usr/local/libexec/container/plugins/container-runtime-linux/bin/container-runtime-linux"
VM = "/System/Library/Frameworks/Virtualization.framework/Versions/A/XPCServices/com.apple.Virtualization.VirtualMachine.xpc/Contents/MacOS/com.apple.Virtualization.VirtualMachine"
PS = f"""\
  101 {HELPER} start --root {APP_ROOT}/containers/buildkit --uuid buildkit
  102 {VM}
  201 {HELPER} start --root {APP_ROOT}/containers/karotte_run_r --uuid karotte_run_r
  202 {VM}
  301 {HELPER} start --root {APP_ROOT}/containers/karotte_run_r2 --uuid karotte_run_r2
  302 {VM}
  400 /usr/bin/vim --uuid karotte_run_r
"""


class TestKillVm:
    def run(self) -> FakeRun:
        return FakeRun(
            {
                ("ps", "-axww"): PS,
                (
                    "lsof",
                    "-t",
                    "--",
                    f"{APP_ROOT}/containers/karotte_run_r/rootfs.ext4",
                ): "202\n201\n",
                ("ps", "-o", "comm=", "-p", "202"): VM + "\n",
                ("ps", "-o", "comm=", "-p", "201"): HELPER + "\n",
            }
        )

    def test_kills_this_vms_helper_and_vm_process_then_deletes_it(self):
        run = self.run()
        killed: list[tuple[int, int]] = []

        kill_vm(
            "karotte_run_r", run=run, kill=lambda pid, sig: killed.append((pid, sig))
        )

        assert killed == [(202, signal.SIGKILL), (201, signal.SIGKILL)]
        assert run.calls[-1] == ["container", "delete", "--force", "karotte_run_r"]

    def test_already_gone_processes_are_fine(self):
        def kill(_pid: int, _sig: int) -> None:
            raise ProcessLookupError

        kill_vm("karotte_run_r", run=self.run(), kill=kill)

    def test_refuses_a_container_karotte_did_not_start(self):
        with pytest.raises(AssertionError):
            kill_vm("buildkit", run=self.run(), kill=lambda _pid, _sig: None)
