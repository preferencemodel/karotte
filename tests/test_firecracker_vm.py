import fcntl
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest

from karotte.firecracker import drives, network, preflight, rootfs, vm
from karotte.firecracker.drives import MountDrive, copy_plain
from karotte.firecracker.rootfs import (
    GUEST_DIR,
    ImageConfig,
    RootfsError,
    base_drive_path,
    check_image_env,
    debugfs_script,
    parse_image_config,
)
from karotte.firecracker.vm import (
    GuestMount,
    HostRelay,
    RunDir,
    VmError,
    Watchdog,
    boot_args,
    clean_up_vms,
    device_name,
    disk_budget,
    firecracker_config,
    guest_argv,
    guest_env,
    jailer_argv,
    parse_mounts,
    vm_resources,
    vsock_connect,
    write_inputs,
)
from karotte.forwarded_env import FORWARDED_ENV_VARS
from karotte.hardware import HardwareLimits, VmSize
from karotte.runtime import get_engine
from karotte.schemas.evaluation_run_config import EvaluationRunConfig
from tests.conftest import register_hardware_plugins

GIB = 1 << 30


def _config(**update: object) -> EvaluationRunConfig:
    return EvaluationRunConfig.model_validate(
        {
            "run_id": "fc-test",
            "task_id": "example-task",
            "model": "claude-sonnet-5",
            "model_api_key": "dummy",
            **update,
        }
    )


def _load_agent() -> ModuleType:
    spec = importlib.util.spec_from_file_location("fc_agent", GUEST_DIR / "agent.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestSizing:
    SIZE: VmSize = VmSize(
        cpus=2,
        sandbox_memory_bytes=5 * GIB,
        vm_memory_bytes=6 * GIB,
        disk_bytes=80 * GIB,
    )

    def test_memory_and_vcpus_come_from_the_size(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr("karotte.firecracker.vm.os.cpu_count", lambda: 8)

        assert vm_resources(self.SIZE) == (2, 6 * 1024)

    def test_vcpus_capped_by_host(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr("karotte.firecracker.vm.os.cpu_count", lambda: 4)
        size = VmSize(
            cpus=16, sandbox_memory_bytes=GIB, vm_memory_bytes=2 * GIB, disk_bytes=None
        )

        assert vm_resources(size)[0] == 4

    def test_disk_budget_is_min_of_the_plugins_and_free_space(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        usage = shutil.disk_usage(tmp_path)
        free = 1000 * GIB

        def disk_usage(_path: Path):
            return usage._replace(free=free)

        monkeypatch.setattr("karotte.firecracker.vm.shutil.disk_usage", disk_usage)
        assert disk_budget(self.SIZE, tmp_path) == 80 * GIB

        free = 10 * GIB
        assert disk_budget(self.SIZE, tmp_path) == int(10 * GIB * 0.8)

    def test_runs_launched_together_split_the_free_space(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        """Scratch drives are sparse: four runs each handed 80 GiB of 150 GiB
        free could fill the disk together."""
        usage = shutil.disk_usage(tmp_path)
        monkeypatch.setattr(
            "karotte.firecracker.vm.shutil.disk_usage",
            lambda _: usage._replace(free=150 * GIB),  # pyright: ignore[reportUnknownLambdaType]
        )

        budget = disk_budget(self.SIZE, tmp_path, runs=4)

        assert budget == int(150 * GIB * 0.8) // 4
        assert 4 * budget <= 150 * GIB * 0.8


class TestConfig:
    def test_boot_args(self):
        net = network.pasta_network()

        args = boot_args(net)

        for part in (
            "console=ttyS0",
            "reboot=k",
            "panic=1",
            "cgroup_no_v1=all",
            "root=/dev/vda ro",
            "init=/.karotte/init",
            "ip=192.0.2.2::192.0.2.1:255.255.255.252::eth0:off",
        ):
            assert part in args
        assert "ip=" not in boot_args(None)

    def test_drives_in_guest_device_order(self, tmp_path: Path):
        config = firecracker_config(
            kernel=tmp_path / "vmlinux",
            base=tmp_path / "base.ext4",
            scratch=tmp_path / "scratch.ext4",
            io=tmp_path / "io.ext4",
            mounts=[tmp_path / "mount0.ext4", tmp_path / "mount1.ext4"],
            vcpus=2,
            mem_mib=6144,
            vsock_uds=tmp_path / "v.sock",
            net=None,
        )

        drive_list = config["drives"]
        assert isinstance(drive_list, list)
        assert [(d["drive_id"], d["is_read_only"]) for d in drive_list] == [
            ("base", True),
            ("scratch", False),
            ("io", False),
            ("mount0", True),
            ("mount1", True),
        ]
        assert drive_list[0]["is_root_device"] is True
        assert [device_name(i) for i in range(5)] == ["vda", "vdb", "vdc", "vdd", "vde"]
        assert config["machine-config"] == {"vcpu_count": 2, "mem_size_mib": 6144}
        assert config["vsock"] == {"guest_cid": 3, "uds_path": str(tmp_path / "v.sock")}
        assert "network-interfaces" not in config

    def test_network_interface(self, tmp_path: Path):
        net = network.pasta_network()

        config = firecracker_config(
            kernel=tmp_path / "vmlinux",
            base=tmp_path / "base.ext4",
            scratch=tmp_path / "scratch.ext4",
            io=tmp_path / "io.ext4",
            mounts=[],
            vcpus=1,
            mem_mib=1024,
            vsock_uds=tmp_path / "v.sock",
            net=net,
        )

        assert config["network-interfaces"] == [
            {
                "iface_id": "eth0",
                "guest_mac": network.GUEST_MAC,
                "host_dev_name": "fc-tap0",
            }
        ]
        boot = config["boot-source"]
        assert isinstance(boot, dict)
        assert "ip=192.0.2.2::192.0.2.1:" in boot["boot_args"]

    def test_jailer_argv(self, tmp_path: Path):
        argv = jailer_argv(
            tmp_path / "jailer",
            tmp_path / "firecracker",
            tmp_path / "run",
            "abc123",
            1003,
            1005,
            6144,
            ["--no-api", "--config-file", "/fc.json"],
        )

        assert argv[0] == str(tmp_path / "jailer")
        assert argv[argv.index("--uid") + 1] == "1003"
        assert argv[argv.index("--chroot-base-dir") + 1] == str(
            tmp_path / "run" / "jail"
        )
        assert argv[argv.index("--cgroup") + 1] == f"memory.max={(6144 + 256) << 20}"
        assert argv[argv.index("--") + 1 :] == ["--no-api", "--config-file", "/fc.json"]
        assert "--netns" not in argv

    def test_a_jailed_vmm_joins_the_pasta_namespace(self, tmp_path: Path):
        argv = jailer_argv(
            tmp_path / "jailer",
            tmp_path / "firecracker",
            tmp_path / "run",
            "abc123",
            1003,
            1005,
            6144,
            ["--no-api"],
            netns="/proc/42/ns/net",
        )

        assert argv[argv.index("--netns") + 1] == "/proc/42/ns/net"
        assert argv.index("--netns") < argv.index("--")


class TestGuestInputs:
    def test_argv_runs_karotte_uncontainerized(self):
        config = _config(transcript_file="/out/t.json")

        argv = guest_argv(config, prepare_only=False)

        assert argv[:3] == ["/root/.venv/bin/karotte", "run", "--no-containerized"]
        assert argv[3] == "--config"
        assert (
            EvaluationRunConfig.model_validate_json(argv[4]).transcript_file
            == "/out/t.json"
        )
        assert "--prepare-only" in guest_argv(config, prepare_only=True)

    def test_env_keeps_image_env_and_marks_the_vm(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        for var in FORWARDED_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        env = guest_env(
            [
                "KAROTTE_CONTAINERIZED=1",
                "KAROTTE_DEMOTE_ID=1000",
                "PATH=/a:/b",
                "X=a=b",
            ],
            disk_budget_bytes=12345,
            sandbox_memory_bytes=4096,
            proxy_url="https://proxy.example",
        )

        assert env == [
            "KAROTTE_CONTAINERIZED=1",
            "KAROTTE_DEMOTE_ID=1000",
            "PATH=/a:/b",
            "X=a=b",
            "HOME=/root",
            "KAROTTE_SANDBOX=vm",
            "KAROTTE_DISK_BUDGET_BYTES=12345",
            "KAROTTE_SANDBOX_MEMORY_BYTES=4096",
            "KAROTTE_VM_LAUNCHER=firecracker",
            "ANTHROPIC_BASE_URL=https://proxy.example",
            "KAROTTE_PROXY_URL=https://proxy.example",
            "KAROTTE_EXIT_ON_RUN_ERROR=1",
        ]

    def test_env_forwards_host_settings(self, monkeypatch: pytest.MonkeyPatch):
        """As with docker: the guest reads them, e.g. the student firewall mode."""
        monkeypatch.setenv("KAROTTE_STUDENT_NETWORK", "internal")
        env = guest_env([], disk_budget_bytes=1, sandbox_memory_bytes=1, proxy_url=None)
        assert "KAROTTE_STUDENT_NETWORK=internal" in env

    def test_mounts(self, tmp_path: Path):
        mounts = parse_mounts(
            [f"{tmp_path}:/data:ro"], dev=True, build_context=str(tmp_path)
        )

        assert mounts == [
            GuestMount(tmp_path / "src" / "environment", vm.DEV_TARGET),
            GuestMount(tmp_path, "/data"),
        ]

    def test_read_write_mounts_are_refused(self, tmp_path: Path):
        with pytest.raises(VmError, match="read-only mounts only"):
            parse_mounts([f"{tmp_path}:/data"], dev=False, build_context=".")

    def test_write_inputs(self, tmp_path: Path):
        write_inputs(
            tmp_path,
            argv=["/bin/karotte", "run", "--config", '{"a": "b c"}'],
            env=["A=1", "B=x y"],
            workdir="/workdir",
            relay_port=8001,
            hosts=["10.0.0.5 proxy.example"],
            nameservers=["1.1.1.1"],
            mounts=[
                ("vdd", MountDrive(tmp_path / "m0", "dir", "."), "/data"),
                ("vde", MountDrive(tmp_path / "m1", "file", "f.txt"), "/etc/f.txt"),
            ],
        )

        assert (
            tmp_path / "argv"
        ).read_bytes() == b'/bin/karotte\0run\0--config\0{"a": "b c"}\0'
        assert (tmp_path / "env").read_bytes() == b"A=1\0B=x y\0"
        assert (tmp_path / "guest.conf").read_text() == (
            "WORKDIR=/workdir\nHEARTBEAT_PORT=52\nRELAY_PORT=8001\n"
        )
        assert (tmp_path / "hosts").read_text() == "10.0.0.5 proxy.example\n"
        assert (tmp_path / "resolv.conf").read_text() == "nameserver 1.1.1.1\n"
        assert (tmp_path / "mounts").read_text() == (
            "vdd\tdir\t.\t/data\nvde\tfile\tf.txt\t/etc/f.txt\n"
        )

    def test_no_network_no_resolv_conf(self, tmp_path: Path):
        write_inputs(
            tmp_path,
            argv=["x"],
            env=[],
            workdir="/",
            relay_port=None,
            hosts=[],
            nameservers=None,
            mounts=[],
        )

        assert not (tmp_path / "resolv.conf").exists()
        assert "RELAY_PORT" not in (tmp_path / "guest.conf").read_text()


class TestRootfs:
    def test_parse_image_config(self):
        config = parse_image_config(
            {
                "Id": "sha256:abc",
                "Architecture": "arm64",
                "Config": {
                    "Env": ["KAROTTE_CONTAINERIZED=1"],
                    "WorkingDir": "/workdir",
                },
            }
        )

        assert config == ImageConfig(
            "sha256:abc", ("KAROTTE_CONTAINERIZED=1",), "/workdir", "aarch64"
        )
        assert parse_image_config({"Id": "x", "Config": None}).working_dir == "/"

    def test_image_must_enable_confinement(self):
        config = ImageConfig("sha256:abc", ("KAROTTE_CONTAINERIZED=1",), "/", "x86_64")

        with pytest.raises(RootfsError, match="KAROTTE_DEMOTE_ID"):
            check_image_env(config)
        check_image_env(
            ImageConfig(
                "x",
                ("KAROTTE_CONTAINERIZED=1", "KAROTTE_DEMOTE_ID=1000"),
                "/",
                "x86_64",
            )
        )

    def test_base_drive_is_keyed_by_image_and_guest_files(self, tmp_path: Path):
        config = ImageConfig("sha256:" + "a" * 64, (), "/", "x86_64")

        path = base_drive_path(config, tmp_path)

        assert path.parent == tmp_path
        assert path.name.startswith("a" * 20 + "-")
        with patch.object(rootfs, "guest_files_digest", return_value="f" * 64):
            assert base_drive_path(config, tmp_path) != path

    def _fake_image(self, monkeypatch: pytest.MonkeyPatch, built: list[Path]) -> None:
        config = ImageConfig(
            "sha256:" + "a" * 64,
            ("KAROTTE_CONTAINERIZED=1", "KAROTTE_DEMOTE_ID=1000"),
            "/",
            "x86_64",
        )
        monkeypatch.setattr(rootfs, "inspect_image", lambda _i, _e: config)  # pyright: ignore[reportUnknownLambdaType]

        def build(
            _engine: str, _image: str, _config: ImageConfig, directory: Path, path: Path
        ) -> None:
            directory.mkdir(parents=True, exist_ok=True)
            path.write_text("drive")
            built.append(path)

        monkeypatch.setattr(rootfs, "_build", build)

    def test_a_run_gets_its_own_link_to_the_base_drive(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Another run evicting old drives can't take it away before boot."""
        built: list[Path] = []
        self._fake_image(monkeypatch, built)
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        base, _ = rootfs.build_base_drive(
            "img", tmp_path / "cache", link_to=run_dir / "base.ext4"
        )
        cached = built[0]
        cached.unlink()  # evicted

        assert base == run_dir / "base.ext4"
        assert base.read_text() == "drive"

    def test_a_drive_evicted_before_the_link_is_built_again(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        built: list[Path] = []
        self._fake_image(monkeypatch, built)
        real_link = os.link
        evicted: list[bool] = []

        def link(src: Path, dst: Path) -> None:
            if not evicted:
                evicted.append(True)
                Path(src).unlink()
            real_link(src, dst)

        monkeypatch.setattr("karotte.firecracker.rootfs.os.link", link)

        base, _ = rootfs.build_base_drive(
            "img", tmp_path / "cache", link_to=tmp_path / "base.ext4"
        )

        assert len(built) == 2
        assert base.read_text() == "drive"

    def test_debugfs_script_writes_root_owned_guest_files(self):
        script = debugfs_script().splitlines()

        assert script[:2] == ["mkdir /.karotte", "cd /.karotte"]
        assert "write /guest/init init" in script
        assert "sif init mode 0100755" in script
        assert "sif agent.py mode 0100644" in script
        assert "sif init uid 0" in script
        assert script[-3:] == [
            "sif /.karotte mode 040755",
            "sif /.karotte uid 0",
            "sif /.karotte gid 0",
        ]

    def test_guest_init_is_bash_and_executable(self):
        init = GUEST_DIR / "init"

        assert init.read_text().startswith("#!/bin/bash\n")
        result = subprocess.run(
            ["bash", "-n", str(init)], capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr


class TestAgent:
    def test_sockaddr_vm_layout(self):
        agent = _load_agent()

        addr = agent.sockaddr_vm(52, cid=2)

        assert len(addr) == 16
        assert agent.peer_cid(addr) == 2
        assert agent.peer_cid(agent.sockaddr_vm(8001)) == 0xFFFFFFFF


needs_e2fsprogs = pytest.mark.skipif(
    shutil.which("mkfs.ext4", path=os.environ.get("PATH", "") + ":/usr/sbin:/sbin")
    is None
    or shutil.which("debugfs", path=os.environ.get("PATH", "") + ":/usr/sbin:/sbin")
    is None,
    reason="needs e2fsprogs",
)


@needs_e2fsprogs
class TestDrives:
    def test_scratch_drive_is_sparse(self, tmp_path: Path):
        path = tmp_path / "scratch.ext4"

        drives.make_scratch_drive(path, 8 * GIB)

        assert path.stat().st_size == 8 * GIB
        assert path.stat().st_blocks * 512 < 64 * (1 << 20)

    def test_out_comes_back_as_plain_files(self, tmp_path: Path):
        content = tmp_path / "io"
        (content / "out" / "run_artifacts").mkdir(parents=True)
        (content / "out" / "transcript.json").write_text('{"events": []}')
        (content / "out" / "run_artifacts" / "a.txt").write_text("a")
        (content / "out" / "link").symlink_to("/etc/passwd")
        (content / "status").mkdir()
        (content / "status" / "exit_code").write_text("3\n")
        io_drive = tmp_path / "io.ext4"
        drives.make_io_drive(io_drive, content, size=64 << 20)
        dest = tmp_path / "dest"

        drives.copy_out(io_drive, dest, tmp_path)

        assert (dest / "transcript.json").read_text() == '{"events": []}'
        assert (dest / "run_artifacts" / "a.txt").read_text() == "a"
        assert not (dest / "link").exists() and not (dest / "link").is_symlink()
        assert drives.read_exit_code(io_drive) == 3

    def test_missing_exit_code(self, tmp_path: Path):
        content = tmp_path / "io"
        (content / "out").mkdir(parents=True)
        io_drive = tmp_path / "io.ext4"
        drives.make_io_drive(io_drive, content, size=64 << 20)

        assert drives.read_exit_code(io_drive) is None

    def test_mount_drives(self, tmp_path: Path):
        src = tmp_path / "src"
        (src / "sub").mkdir(parents=True)
        (src / "sub" / "a.txt").write_text("a")
        file = tmp_path / "f.txt"
        file.write_text("f")

        dir_drive = drives.make_mount_drive(tmp_path / "m0.ext4", src)
        file_drive = drives.make_mount_drive(tmp_path / "m1.ext4", file)

        assert (dir_drive.kind, dir_drive.name) == ("dir", ".")
        assert (file_drive.kind, file_drive.name) == ("file", "f.txt")
        listing = subprocess.run(
            [drives.e2fs_tool("debugfs"), "-R", "ls -p /", str(dir_drive.path)],
            capture_output=True,
            text=True,
        ).stdout
        assert "/sub/" in listing and "lost+found" not in listing
        content = subprocess.run(
            [drives.e2fs_tool("debugfs"), "-R", "cat /f.txt", str(file_drive.path)],
            capture_output=True,
            text=True,
        ).stdout
        assert content == "f"

    def test_the_e2fsprogs_1_47_3_large_file_failure_says_why(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """1.47.3's ``mkfs.ext4 -d <dir>`` fails on files over 2 GiB."""
        src = tmp_path / "src"
        src.mkdir()

        def run(argv: list[str]) -> None:
            raise drives.DriveError(
                f"{argv[0]} failed: mkfs.ext4: Ext2 file too big while populating file system"
            )

        monkeypatch.setattr(drives, "_run", run)

        with pytest.raises(drives.DriveError, match="1.47.3"):
            _ = drives.make_mount_drive(tmp_path / "m.ext4", src)

    def test_a_mounted_file_keeps_its_mode_but_not_setuid(self, tmp_path: Path):
        """A 0755 script must still run in the guest."""
        script = tmp_path / "run.sh"
        script.write_text("#!/bin/sh\n")
        script.chmod(0o4755)

        drive = drives.make_mount_drive(tmp_path / "m.ext4", script)

        stat_out = subprocess.run(
            [drives.e2fs_tool("debugfs"), "-R", "stat /run.sh", str(drive.path)],
            capture_output=True,
            text=True,
        ).stdout
        assert "Mode:  0755" in stat_out or "Mode:  00755" in stat_out


def test_copy_plain_writes_nothing_through_links_on_the_host(tmp_path: Path):
    """A link already in the output directory, at a name the guest also
    writes, must not carry the copy outside it."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "victim").write_text("keep")
    src = tmp_path / "src"
    (src / "sub").mkdir(parents=True)
    (src / "victim").write_text("guest")
    (src / "sub" / "f").write_text("guest")
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "victim").symlink_to(outside / "victim")
    (dest / "sub").symlink_to(outside)

    copy_plain(src, dest)

    assert (outside / "victim").read_text() == "keep"
    assert not (outside / "f").exists()
    assert not (dest / "victim").is_symlink()
    assert (dest / "victim").read_text() == "guest"


def test_copy_plain_skips_links_and_special_files(tmp_path: Path):
    src = tmp_path / "src"
    (src / "d").mkdir(parents=True)
    (src / "d" / "f").write_text("x")
    (src / "dirlink").symlink_to(tmp_path)
    os.mkfifo(src / "fifo")
    dest = tmp_path / "dest"

    copy_plain(src, dest)

    assert (dest / "d" / "f").read_text() == "x"
    assert sorted(p.name for p in dest.iterdir()) == ["d"]


class FakeFirecrackerVsock:
    """Answers ``CONNECT <port>`` like Firecracker, then echoes."""

    server: socket.socket
    accept_ports: set[int]

    def __init__(self, path: Path, accept_ports: set[int]) -> None:
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server.bind(str(path))
        self.server.listen(8)
        self.accept_ports = accept_ports
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self) -> None:
        while True:
            try:
                conn, _ = self.server.accept()
            except OSError:
                return
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn: socket.socket) -> None:
        with conn:
            line = b""
            while not line.endswith(b"\n"):
                line += conn.recv(1)
            port = int(line.split()[1])
            if port not in self.accept_ports:
                return
            conn.sendall(b"OK 1073741824\n")
            while data := conn.recv(4096):
                conn.sendall(data.upper())


class TestVsock:
    def test_connect_and_relay(self, tmp_path: Path):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        uds = tmp_path / "v.sock"
        FakeFirecrackerVsock(uds, {8001, port})

        with vsock_connect(uds, 8001, timeout=5) as conn:
            conn.sendall(b"hi")
            assert conn.recv(10) == b"HI"
        with pytest.raises(ConnectionError):
            vsock_connect(uds, 52, timeout=5)

        relay = HostRelay(port, uds)
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=5) as client:
                client.sendall(b"events")
                assert client.recv(10) == b"EVENTS"
        finally:
            relay.close()

    def test_closing_the_relay_frees_its_port(self, tmp_path: Path):
        """A thread blocked in accept() kept the listener, so the next run on
        the same websocket port failed with EADDRINUSE."""
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        relay = HostRelay(port, tmp_path / "v.sock")

        relay.close()

        assert not relay.thread.is_alive()
        HostRelay(port, tmp_path / "v.sock").close()

    def test_relay_port_in_use(self, tmp_path: Path):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            s.listen()
            with pytest.raises(VmError, match="websocket"):
                HostRelay(s.getsockname()[1], tmp_path / "v.sock")


class TestWatchdog:
    def _run(self, answers: list[bool], **kwargs: float) -> tuple[Watchdog, list[str]]:
        killed: list[str] = []
        replies = iter(answers)
        dog = Watchdog(
            Path("/nonexistent"),
            lambda: killed.append("kill"),
            probe=lambda _uds: next(replies, False),
            interval=0.01,
            **kwargs,
        )
        dog.start()
        dog.thread.join(timeout=5)
        dog.stop()
        return dog, killed

    def test_kills_a_guest_that_stops_answering(self):
        dog, killed = self._run([True, True], timeout=0.05, boot_grace=10)

        assert dog.fired and killed == ["kill"]

    def test_kills_a_guest_that_never_answers(self):
        dog, killed = self._run([], timeout=10, boot_grace=0.05)

        assert dog.fired and killed == ["kill"]

    def test_leaves_an_answering_guest_alone(self):
        killed: list[str] = []
        dog = Watchdog(
            Path("/nonexistent"),
            lambda: killed.append("kill"),
            timeout=0.05,
            interval=0.01,
            probe=lambda _uds: True,
        )
        dog.start()
        time.sleep(0.2)
        dog.stop()
        dog.thread.join(timeout=5)

        assert not dog.fired and killed == []


class TestRunDirs:
    def _run_dir(self, root: Path, run_id: str, **meta: object) -> RunDir:
        d = RunDir.for_run(run_id, root)
        d.path.mkdir(parents=True)
        (d.path / "lock").touch()
        (d.path / "meta.json").write_text(json.dumps({"run_id": run_id, **meta}))
        return d

    def test_short_names_for_socket_paths(self, tmp_path: Path):
        d = RunDir.for_run("x" * 200, tmp_path)

        assert len(d.path.name) == 12

    def test_clean_up(self, tmp_path: Path):
        mine = self._run_dir(tmp_path, "batch-1")
        other_live = self._run_dir(tmp_path, "other-live")
        other_stale = self._run_dir(tmp_path, "other-stale")
        other_kept = self._run_dir(tmp_path, "other-kept", keep=True)
        # Our own pid can't be mistaken for a VMM: its command line doesn't
        # name the run dir.
        stray = self._run_dir(tmp_path, "other-stray", vmm_pgid=os.getpid())
        with (other_live.path / "lock").open("wb") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)

            clean_up_vms("batch-", tmp_path)

            assert other_live.path.exists()
        assert not mine.path.exists()
        assert not other_stale.path.exists()
        assert other_kept.path.exists()
        assert not stray.path.exists()

    def test_live_run_dir(self, tmp_path: Path):
        d = RunDir.for_run("r1", tmp_path)
        d.create()

        assert d.is_live()
        d.remove()
        assert not d.path.exists()


def _pasta_ok(*_args: object, **_kwargs: object) -> SimpleNamespace:
    return SimpleNamespace(returncode=0, stderr="")


class TestPreflight:
    @pytest.fixture(autouse=True)
    def _host(self, monkeypatch: pytest.MonkeyPatch):
        def which(name: str, path: str | None = None) -> str:
            _ = path
            return f"/usr/bin/{name}"

        def e2fs_tool(name: str) -> str:
            return f"/usr/sbin/{name}"

        monkeypatch.setattr("karotte.firecracker.preflight.sys.platform", "linux")
        monkeypatch.setattr(preflight, "host_arch", lambda: "x86_64")
        monkeypatch.setattr("karotte.firecracker.preflight.shutil.which", which)
        monkeypatch.setattr(preflight, "e2fs_tool", e2fs_tool)
        monkeypatch.setattr(network, "network_mode", lambda *_args: "pasta")  # pyright: ignore[reportUnknownLambdaType]
        monkeypatch.setattr(preflight, "ensure_artifacts", lambda: None)
        monkeypatch.setattr("karotte.firecracker.preflight.subprocess.run", _pasta_ok)

    def test_ready(self, tmp_path: Path):
        kvm = tmp_path / "kvm"
        kvm.touch()

        assert preflight.firecracker_problems("cpu-2.6gb", None, kvm=kvm) == []

    def test_pasta_is_started_once_to_see_it_works(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        kvm = tmp_path / "kvm"
        kvm.touch()
        runs: list[list[str]] = []

        def recording(argv: list[str], **_: object) -> SimpleNamespace:
            runs.append(argv)
            return SimpleNamespace(returncode=0, stderr="")

        monkeypatch.setattr("karotte.firecracker.preflight.subprocess.run", recording)

        assert preflight.firecracker_problems("cpu-2.6gb", None, kvm=kvm) == []
        assert ["pasta", "--config-net", "--quiet", "--", "true"] in runs

    def test_under_the_jailer_the_jailed_namespace_is_probed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """As root, the unprivileged `pasta -- true` says nothing about a
        jailed run; set up its namespace instead."""
        kvm = tmp_path / "kvm"
        kvm.touch()
        monkeypatch.setenv("KAROTTE_FIRECRACKER_JAILER", "1")
        monkeypatch.setattr(preflight, "jailer_ids", lambda: (1003, 1005))
        started: list[tuple[int, int]] = []

        class FakeNetns:
            @classmethod
            def start(
                cls, _allow: object, _blocked: object, uid: int, gid: int
            ) -> "FakeNetns":
                started.append((uid, gid))
                return cls()

            def stop(self) -> None:
                pass

        monkeypatch.setattr(preflight, "JailNetns", FakeNetns)

        assert preflight.firecracker_problems("cpu-2.6gb", None, kvm=kvm) == []
        assert started == [(1003, 1005)]

    def test_a_pasta_that_cannot_run_points_at_the_apparmor_fix(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Ubuntu 24.04: AppArmor denies its older passt a user namespace."""
        kvm = tmp_path / "kvm"
        kvm.touch()

        def refused(argv: list[str], **_kwargs: object) -> SimpleNamespace:
            if argv[0] != "pasta":
                return SimpleNamespace(returncode=0, stderr="")
            return SimpleNamespace(
                returncode=1,
                stderr="No routable interface for IPv6: IPv6 is disabled\n"
                + "Couldn't write to /proc/self/uid_map: Operation not permitted\n"
                + "Couldn't configure user mappings\n",
            )

        monkeypatch.setattr("karotte.firecracker.preflight.subprocess.run", refused)

        (problem,) = preflight.firecracker_problems("cpu-2.6gb", None, kvm=kvm)
        assert "uid_map: Operation not permitted" in problem
        assert "apparmor_restrict_unprivileged_userns" in problem

    def test_docker_without_buildkit_is_reported(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Ubuntu's docker.io ships without buildx; the Containerfile's
        `RUN --mount` needs it."""
        kvm = tmp_path / "kvm"
        kvm.touch()

        def no_buildx(argv: list[str], **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(
                returncode=1 if argv[:2] == ["docker", "buildx"] else 0, stderr=""
            )

        monkeypatch.setattr("karotte.firecracker.preflight.subprocess.run", no_buildx)

        (problem,) = preflight.firecracker_problems("cpu-2.6gb", None, kvm=kvm)
        assert "docker-buildx" in problem

    def test_reports_what_to_fix(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        register_hardware_plugins(
            monkeypatch,
            limits={"a": lambda _hw: HardwareLimits(passthrough=True)},  # pyright: ignore[reportUnknownLambdaType]
        )
        problems = preflight.firecracker_problems(
            "gpu-1", [f"{tmp_path}:/data"], kvm=tmp_path / "missing"
        )

        assert any("has no KVM" in p for p in problems)
        assert any("gpu-1 needs devices a VM can't pass through" in p for p in problems)
        assert any("read-only mounts only" in p for p in problems)

    def test_kvm_not_writable(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        kvm = tmp_path / "kvm"
        kvm.touch()

        def access(_path: Path, _mode: int) -> bool:
            return False

        monkeypatch.setattr("karotte.firecracker.preflight.os.access", access)

        (problem,) = preflight.firecracker_problems("cpu-2.6gb", None, kvm=kvm)
        assert "usermod -aG kvm" in problem

    def test_abort_suggests_docker(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ):
        import typer

        def problems(_hardware: str, _mounts: list[str] | None) -> list[str]:
            return ["no KVM"]

        monkeypatch.setattr(preflight, "firecracker_problems", problems)

        with pytest.raises(typer.Abort):
            preflight.validate_firecracker_runtime("cpu-2.6gb", None)
        err = capsys.readouterr().err
        assert "no KVM" in err and "--runtime docker" in err


class TestWiring:
    def test_engine_builds_with_docker(self):
        assert get_engine("firecracker") == "docker"

    def test_run_containerized_boots_a_vm(self):
        from karotte.run_helpers import run_containerized

        config = _config()
        with patch("karotte.firecracker.vm.run_firecracker") as run_firecracker:
            run_containerized(config, "firecracker", dev=True, mounts=["a:/b:ro"])

        run_firecracker.assert_called_once()
        assert run_firecracker.call_args.args == (config,)
        assert run_firecracker.call_args.kwargs["dev"] is True
        assert run_firecracker.call_args.kwargs["mounts"] == ["a:/b:ro"]
        assert run_firecracker.call_args.kwargs["parallel_runs"] == 1

    def test_clean_up_and_stop(self):
        from karotte.run_helpers import clean_up_old_containers, stop_containers

        with patch("karotte.firecracker.vm.clean_up_vms") as clean_up:
            clean_up_old_containers("firecracker", ["batch-0", "batch-1"])
        clean_up.assert_called_once_with("batch-")
        with patch("karotte.firecracker.vm.stop_vms") as stop:
            stop_containers("firecracker", ["batch-0"])
        stop.assert_called_once_with(["batch-0"])
