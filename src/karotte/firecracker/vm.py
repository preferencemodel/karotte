"""Run one karotte evaluation in a Firecracker microVM.

The VM boots the image (as a read-only base drive) under an overlay on a
sparse scratch drive, with the guest init as PID 1. Inputs go in on an io
drive and /out comes back from it after power-off. The websocket is relayed
over vsock to 127.0.0.1 on the host, like ``--publish 127.0.0.1:port:port``.
The VMM runs as the user, with Firecracker's seccomp filters on.
"""

import contextlib
import ctypes
import fcntl
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from loguru import logger

from karotte.confinement import DISK_BUDGET_ENV_VAR, GIB, SANDBOX_MEMORY_ENV_VAR
from karotte.firecracker import FirecrackerError, drives, network
from karotte.firecracker.artifacts import cache_dir, ensure_artifacts, host_arch
from karotte.firecracker.rootfs import build_base_drive
from karotte.forwarded_env import sandbox_env
from karotte.hardware import VmSize, vm_size
from karotte.load_tasks import load_task
from karotte.schemas.evaluation_run_config import EvaluationRunConfig

GUEST_CID = 3
HEARTBEAT_PORT = 52
"""Below 1024, so only guest root can listen on it."""

KAROTTE_BIN = "/root/.venv/bin/karotte"
DEV_TARGET = "/root/.venv/lib/python3.12/site-packages/environment/"

_MAX_VCPUS = 32
_FREE_DISK_FRACTION = 0.8

WATCHDOG_ENV_VAR = "KAROTTE_FIRECRACKER_WATCHDOG_SECONDS"
"""How long the guest may go without answering before its VMM is killed; 0 disables."""
_DEFAULT_WATCHDOG_SECONDS = 120
_WATCHDOG_INTERVAL_SECONDS = 5
_BOOT_GRACE_SECONDS = 180

JAILER_ENV_VAR = "KAROTTE_FIRECRACKER_JAILER"
"""Set to 1 to start Firecracker through its jailer; needs root."""

BOOT_ARGS = (
    "console=ttyS0 8250.nr_uarts=1 loglevel=3 reboot=k panic=1 pci=off random.trust_cpu=on "
    + "i8042.noaux i8042.nomux i8042.dumbkbd cgroup_no_v1=all "
    + "root=/dev/vda ro rootfstype=ext4 init=/.karotte/init"
)


class VmError(FirecrackerError):
    pass


def runs_dir() -> Path:
    return cache_dir() / "runs"


def vm_resources(size: VmSize) -> tuple[int, int]:
    """vCPUs (at most the host's) and guest memory in MiB."""
    vcpus = max(1, min(size.cpus, os.cpu_count() or 1, _MAX_VCPUS))
    return vcpus, size.vm_memory_bytes >> 20


def disk_budget(size: VmSize, directory: Path) -> int:
    """min(the hardware plugin's disk budget, 80% of the host's free space)."""
    budget = int(shutil.disk_usage(directory).free * _FREE_DISK_FRACTION)
    if size.disk_bytes is not None:
        budget = min(budget, size.disk_bytes)
    return budget


def device_name(index: int) -> str:
    """Guest block device of the drive at ``index`` in the config."""
    if index >= 26:
        raise VmError("Too many drives")
    return "vd" + chr(ord("a") + index)


def boot_args(net: network.GuestNetwork | None) -> str:
    return BOOT_ARGS + (f" {net.kernel_ip_arg()}" if net else "")


def firecracker_config(
    *,
    kernel: Path,
    base: Path,
    scratch: Path,
    io: Path,
    mounts: Sequence[Path],
    vcpus: int,
    mem_mib: int,
    vsock_uds: Path,
    net: network.GuestNetwork | None,
) -> dict[str, object]:
    drive_list: list[dict[str, str | bool]] = [
        {
            "drive_id": "base",
            "path_on_host": str(base),
            "is_root_device": True,
            "is_read_only": True,
        },
        {
            "drive_id": "scratch",
            "path_on_host": str(scratch),
            "is_root_device": False,
            "is_read_only": False,
        },
        {
            "drive_id": "io",
            "path_on_host": str(io),
            "is_root_device": False,
            "is_read_only": False,
        },
    ]
    drive_list += [
        {
            "drive_id": f"mount{i}",
            "path_on_host": str(p),
            "is_root_device": False,
            "is_read_only": True,
        }
        for i, p in enumerate(mounts)
    ]
    config: dict[str, object] = {
        "boot-source": {"kernel_image_path": str(kernel), "boot_args": boot_args(net)},
        "drives": drive_list,
        "machine-config": {"vcpu_count": vcpus, "mem_size_mib": mem_mib},
        "vsock": {"guest_cid": GUEST_CID, "uds_path": str(vsock_uds)},
    }
    if net is not None:
        config["network-interfaces"] = [
            {
                "iface_id": "eth0",
                "guest_mac": network.GUEST_MAC,
                "host_dev_name": net.tap,
            }
        ]
    return config


def guest_argv(run_config: EvaluationRunConfig, prepare_only: bool) -> list[str]:
    return [
        KAROTTE_BIN,
        "run",
        "--no-containerized",
        *(["--prepare-only"] if prepare_only else []),
        "--config",
        run_config.model_dump_json(),
    ]


def guest_env(
    image_env: Sequence[str],
    *,
    disk_budget_bytes: int,
    sandbox_memory_bytes: int,
    proxy_url: str | None,
) -> list[str]:
    env: dict[str, str] = {}
    for entry in image_env:
        key, _, value = entry.partition("=")
        env[key] = value
    env.setdefault("HOME", "/root")
    env.setdefault(
        "PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    )
    env["KAROTTE_SANDBOX"] = "vm"
    env[DISK_BUDGET_ENV_VAR] = str(disk_budget_bytes)
    env[SANDBOX_MEMORY_ENV_VAR] = str(sandbox_memory_bytes)
    if proxy_url:
        env["ANTHROPIC_BASE_URL"] = proxy_url
        env["KAROTTE_PROXY_URL"] = proxy_url
    env.update(sandbox_env())
    return [f"{k}={v}" for k, v in env.items()]


@dataclass(frozen=True)
class GuestMount:
    source: Path
    target: str


def parse_mounts(
    mounts: Sequence[str] | None, dev: bool, build_context: str
) -> list[GuestMount]:
    """``--dev`` and ``--mount`` specs as read-only guest mounts. A VM can't
    share a writable directory with the host, so read-write mounts are refused."""
    result: list[GuestMount] = []
    if dev:
        src = Path(build_context).absolute() / "src" / "environment"
        result.append(GuestMount(src, DEV_TARGET))
    for spec in mounts or []:
        parts = spec.split(":")
        if len(parts) != 3 or parts[2] != "ro":
            raise VmError(
                f"The firecracker runtime supports read-only mounts only; append :ro to {spec!r}"
            )
        result.append(GuestMount(Path(parts[0]).absolute(), parts[1]))
    return result


def write_inputs(
    in_dir: Path,
    *,
    argv: Sequence[str],
    env: Sequence[str],
    workdir: str,
    relay_port: int | None,
    hosts: Sequence[str],
    nameservers: Sequence[str] | None,
    mounts: Sequence[tuple[str, drives.MountDrive, str]],
) -> None:
    """The io drive's ``in/`` as the guest init reads it."""
    in_dir.mkdir(parents=True, exist_ok=True)
    (in_dir / "argv").write_bytes(b"".join(a.encode() + b"\0" for a in argv))
    (in_dir / "env").write_bytes(b"".join(e.encode() + b"\0" for e in env))
    conf = [f"WORKDIR={workdir}", f"HEARTBEAT_PORT={HEARTBEAT_PORT}"]
    if relay_port is not None:
        conf.append(f"RELAY_PORT={relay_port}")
    (in_dir / "guest.conf").write_text("\n".join(conf) + "\n")
    (in_dir / "hosts").write_text("".join(f"{h}\n" for h in hosts))
    if nameservers is not None:
        (in_dir / "resolv.conf").write_text(network.resolv_conf(nameservers))
    (in_dir / "mounts").write_text(
        "".join(f"{dev}\t{d.kind}\t{d.name}\t{target}\n" for dev, d, target in mounts)
    )


# --- vsock ---


def vsock_connect(uds: Path, port: int, timeout: float) -> socket.socket:
    """A stream to guest ``port`` through Firecracker's vsock socket."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(str(uds))
        s.sendall(f"CONNECT {port}\n".encode())
        line = b""
        while not line.endswith(b"\n"):
            byte = s.recv(1)
            if not byte or len(line) > 64:
                raise ConnectionError(f"vsock port {port}: no answer")
            line += byte
        if not line.startswith(b"OK "):
            raise ConnectionError(f"vsock port {port}: {line!r}")
    except BaseException:
        s.close()
        raise
    return s


def probe_guest(uds: Path, timeout: float = 5) -> bool:
    try:
        with vsock_connect(uds, HEARTBEAT_PORT, timeout) as s:
            return s.recv(3) == b"ok\n"
    except OSError:
        return False


def _pipe(src: socket.socket, dst: socket.socket) -> None:
    try:
        while chunk := src.recv(65536):
            dst.sendall(chunk)
    except OSError:
        pass
    finally:
        for s in (src, dst):
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


_ACCEPT_POLL_SECONDS = 0.2


class HostRelay:
    """Listens on 127.0.0.1:port and forwards each connection to the guest's
    port over vsock."""

    port: int
    uds: Path
    server: socket.socket
    thread: threading.Thread
    stopping: threading.Event

    def __init__(self, port: int, uds: Path) -> None:
        self.port = port
        self.uds = uds
        try:
            self.server = socket.create_server(("127.0.0.1", port))
        except OSError as e:
            raise VmError(
                f"Cannot listen on 127.0.0.1:{port} for the websocket: {e}"
            ) from e
        # Closing a listener doesn't wake an accept() blocked on it (Linux),
        # so accept waits in short slices and checks for close().
        self.server.settimeout(_ACCEPT_POLL_SECONDS)
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self) -> None:
        while not self.stopping.is_set():
            try:
                conn, _ = self.server.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            conn.settimeout(None)
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn: socket.socket) -> None:
        try:
            guest = vsock_connect(self.uds, self.port, timeout=5)
        except OSError:
            conn.close()
            return
        guest.settimeout(None)
        threading.Thread(target=_pipe, args=(guest, conn), daemon=True).start()
        _pipe(conn, guest)
        conn.close()
        guest.close()

    def close(self) -> None:
        self.stopping.set()
        self.thread.join(timeout=_ACCEPT_POLL_SECONDS * 4)
        self.server.close()


class Watchdog:
    """Kills the VMM when the guest stops answering the heartbeat, so a
    livelocked guest ends the run instead of hanging it."""

    uds: Path
    kill: Callable[[], None]
    timeout: float
    interval: float
    boot_grace: float
    probe: Callable[[Path], bool]
    fired: bool
    _stop: threading.Event
    thread: threading.Thread

    def __init__(
        self,
        uds: Path,
        kill: Callable[[], None],
        timeout: float,
        interval: float = _WATCHDOG_INTERVAL_SECONDS,
        boot_grace: float = _BOOT_GRACE_SECONDS,
        probe: Callable[[Path], bool] = probe_guest,
    ) -> None:
        self.uds = uds
        self.kill = kill
        self.timeout = timeout
        self.interval = interval
        self.boot_grace = boot_grace
        self.probe = probe
        self.fired = False
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        start = time.monotonic()
        last_ok: float | None = None
        while not self._stop.wait(self.interval):
            now = time.monotonic()
            if self.probe(self.uds):
                if last_ok is None:
                    logger.info(f"The guest answered {now - start:.1f}s after start")
                last_ok = now
                continue
            if last_ok is None and now - start < self.boot_grace:
                continue
            if last_ok is not None and now - last_ok < self.timeout:
                continue
            if self._stop.is_set():
                return
            self.fired = True
            logger.error(
                f"The guest has not answered for {now - (last_ok or start):.0f}s; killing its VMM"
            )
            self.kill()
            return

    def stop(self) -> None:
        self._stop.set()


def watchdog_seconds() -> float:
    value = os.environ.get(WATCHDOG_ENV_VAR)
    if not value:
        return _DEFAULT_WATCHDOG_SECONDS
    try:
        return float(value)
    except ValueError:
        logger.warning(f"Ignoring {WATCHDOG_ENV_VAR}={value!r}: not a number")
        return _DEFAULT_WATCHDOG_SECONDS


# --- run directories ---


@dataclass
class RunDir:
    """``runs/<hash of run id>``: drives, config, sockets and ``meta.json``.

    Held under an flock while its launcher lives, so cleanup can tell a live
    run from a leftover. Short names keep the vsock socket paths under the
    108-byte limit.
    """

    path: Path
    run_id: str
    lock: IO[bytes] | None = None

    @classmethod
    def for_run(cls, run_id: str, root: Path | None = None) -> "RunDir":
        root = root or runs_dir()
        name = hashlib.sha256(run_id.encode()).hexdigest()[:12]
        return cls(root / name, run_id)

    def create(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        self.path.chmod(0o700)
        self.lock = (self.path / "lock").open("wb")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.write_meta()

    def write_meta(self, **extra: object) -> None:
        meta = {**self.meta(), "run_id": self.run_id, **extra}
        (self.path / "meta.json").write_text(json.dumps(meta))

    def meta(self) -> dict[str, object]:
        try:
            return json.loads((self.path / "meta.json").read_text())
        except (OSError, ValueError):
            return {}

    def is_live(self) -> bool:
        """Whether a launcher holds the run."""
        try:
            with (self.path / "lock").open("rb") as f:
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    return True
                return False
        except FileNotFoundError:
            return False

    def kill_vmm(self) -> None:
        """SIGKILL the VMM's process group, if it's still this run's."""
        pgid = self.meta().get("vmm_pgid")
        if not isinstance(pgid, int) or not _is_our_vmm(pgid, self.path):
            return
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def remove(self) -> None:
        if self.lock is not None:
            self.lock.close()
            self.lock = None
        shutil.rmtree(self.path, ignore_errors=True)


def _is_our_vmm(pid: int, run_path: Path) -> bool:
    # A jailed VMM's command line has only paths inside its chroot, and the
    # run dir's name as its id.
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return False
    return run_path.name.encode() in cmdline


def existing_run_dirs(root: Path | None = None) -> list[RunDir]:
    root = root or runs_dir()
    result: list[RunDir] = []
    for path in sorted(root.glob("*")) if root.is_dir() else []:
        d = RunDir(path, "")
        run_id = d.meta().get("run_id")
        result.append(RunDir(path, run_id if isinstance(run_id, str) else ""))
    return result


def stop_vms(run_ids: Sequence[str], root: Path | None = None) -> None:
    """Kill the VMMs of these runs; their launchers then clean up."""
    for d in existing_run_dirs(root):
        if d.run_id in run_ids:
            d.kill_vmm()


def clean_up_vms(prefix: str, root: Path | None = None) -> None:
    """Remove runs of this invocation's prefix (live or not), and leftovers
    of launchers that died, killing any VMM still running."""
    for d in existing_run_dirs(root):
        live = d.is_live()
        if d.run_id.startswith(prefix) or not (live or d.meta().get("keep")):
            d.kill_vmm()
            d.remove()


# --- the run ---


def _set_pdeathsig() -> None:
    # The VMM dies with the thread that started it.
    ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG


def run_firecracker(
    run_config: EvaluationRunConfig,
    *,
    dev: bool = False,
    log_file: Path | None = None,
    keep_vm: bool = False,
    build_context: str = ".",
    mounts: Sequence[str] | None = None,
    proxy_url: str | None = None,
    prepare_only: bool = False,
    image: str = "karotte",
    engine: str = "docker",
) -> None:
    """Run one evaluation in a Firecracker VM, like ``docker run`` of the
    image. Raises ``CalledProcessError`` when the run fails."""
    task = load_task(run_config)
    guest_mounts = parse_mounts(mounts, dev, build_context)
    jailer = jailer_ids() if os.environ.get(JAILER_ENV_VAR) == "1" else None
    mode = network.network_mode()
    artifacts = ensure_artifacts()

    run = RunDir.for_run(run_config.run_id)
    if run.path.exists():
        clean_up_vms(run_config.run_id)
    run.create()
    net: network.GuestNetwork | None = None
    relay: HostRelay | None = None
    watchdog: Watchdog | None = None
    proc: subprocess.Popen[bytes] | None = None
    jail_netns: JailNetns | None = None
    started = time.monotonic()
    try:
        # The run's own link to the drive: old drives may be evicted from the
        # cache while this run is still starting.
        base, image_config = build_base_drive(
            image, cache_dir() / "rootfs", engine, link_to=run.path / "base.ext4"
        )
        if image_config.architecture != host_arch():
            raise VmError(
                f"The image is {image_config.architecture}; a Firecracker VM here runs {host_arch()}"
            )
        # Everything the VMM opens lives in vm_root: the run dir, or the
        # jailer's chroot, where the config names files relative to it.
        vm_root = run.path
        if jailer is not None:
            vm_root = jail_root(run.path, artifacts.firecracker, run.path.name)
            vm_root.mkdir(parents=True)
            for src in (artifacts.kernel, base):
                os.link(src, vm_root / src.name)

        def in_vm(path: Path) -> Path:
            return Path("/", path.name) if jailer is not None else path

        try:
            size = vm_size(task.required_hardware)
        except ValueError as e:
            raise VmError(str(e)) from e
        budget = disk_budget(size, run.path)
        drives.make_scratch_drive(vm_root / "scratch.ext4", budget)

        mount_drives = [
            (g, drives.make_mount_drive(vm_root / f"mount{i}.ext4", g.source))
            for i, g in enumerate(guest_mounts)
        ]

        host_out: Path | None = None
        if run_config.transcript_file:
            transcript = Path(run_config.transcript_file).absolute()
            host_out = transcript.parent
            host_out.mkdir(parents=True, exist_ok=True)
            run_config = run_config.model_copy(
                update={"transcript_file": f"/out/{transcript.name}"}
            )

        allows = network.proxy_allows(proxy_url) if mode != "none" else []
        if mode == "pasta":
            network.check_pasta_allows(allows)
            net = network.pasta_network()
        if net is not None:
            network.log_filter(
                mode, [a for a in allows if network.is_blocked(a.address)]
            )

        in_dir = run.path / "io" / "in"
        write_inputs(
            in_dir,
            argv=guest_argv(run_config, prepare_only),
            env=guest_env(
                image_config.env,
                disk_budget_bytes=budget,
                sandbox_memory_bytes=size.sandbox_memory_bytes,
                proxy_url=proxy_url,
            ),
            workdir=image_config.working_dir,
            relay_port=None if prepare_only else run_config.websocket_config.port,
            hosts=network.proxy_hosts_lines(proxy_url, allows),
            nameservers=network.guest_nameservers() if net else None,
            mounts=[
                (device_name(3 + i), d, g.target)
                for i, (g, d) in enumerate(mount_drives)
            ],
        )
        io_drive = vm_root / "io.ext4"
        drives.make_io_drive(io_drive, run.path / "io")
        shutil.rmtree(run.path / "io")

        vcpus, mem_mib = vm_resources(size)
        vsock_uds = vm_root / "v.sock"
        host_vsock = vsock_uds
        if jailer is not None:
            # The path inside the jail can pass the 108-byte limit on a unix
            # socket address (it does under /root/.cache on aarch64); the
            # host connects through a short link instead.
            host_vsock = run.path / "v.sock"
            host_vsock.symlink_to(vsock_uds)
        config = firecracker_config(
            kernel=in_vm(artifacts.kernel),
            base=in_vm(base),
            scratch=in_vm(vm_root / "scratch.ext4"),
            io=in_vm(io_drive),
            mounts=[in_vm(d.path) for _, d in mount_drives],
            vcpus=vcpus,
            mem_mib=mem_mib,
            vsock_uds=in_vm(vsock_uds),
            net=net,
        )
        config_path = vm_root / "fc.json"
        config_path.write_text(json.dumps(config, indent=2))

        argv = [
            str(artifacts.firecracker),
            "--no-api",
            "--config-file",
            str(in_vm(config_path)),
            "--level",
            "Warning",
        ]
        blocked = (
            network.blocked_destinations(network.host_ipv4_addresses())
            if mode == "pasta"
            else []
        )
        if jailer is not None:
            uid, gid = jailer
            for path in vm_root.iterdir():
                if path.name not in (artifacts.kernel.name, base.name):
                    os.chown(path, uid, gid)
            if mode == "pasta":
                jail_netns = JailNetns.start(allows, blocked, uid, gid)
            argv = jailer_argv(
                artifacts.jailer,
                artifacts.firecracker,
                run.path,
                run.path.name,
                uid,
                gid,
                mem_mib,
                argv[1:],
                netns=jail_netns.path if jail_netns is not None else None,
            )
        elif mode == "pasta":
            argv = network.pasta_command(argv, run.path / "vmm.pid", allows, blocked)

        if not prepare_only:
            relay = HostRelay(run_config.websocket_config.port, host_vsock)

        logger.info(
            f"Starting Firecracker VM karotte_run_{run_config.run_id}: {vcpus} vCPUs, {mem_mib} MiB, "
            + f"{budget / GIB:.1f} GiB scratch, network {mode}"
            + (", jailed" if jailer else "")
        )
        with open(log_file, "wb") if log_file else contextlib.nullcontext() as out:
            proc = subprocess.Popen(
                argv,
                # pasta_command's script watches its stdin for the launcher.
                stdin=subprocess.PIPE
                if mode == "pasta" and jailer is None
                else subprocess.DEVNULL,
                stdout=out,
                stderr=subprocess.STDOUT if out else None,
                cwd=run.path,
                start_new_session=True,
                preexec_fn=_set_pdeathsig,
            )
            run.write_meta(vmm_pgid=proc.pid, keep=keep_vm)
            timeout = watchdog_seconds()
            if timeout > 0:
                pgid = proc.pid
                watchdog = Watchdog(host_vsock, lambda: _killpg(pgid), timeout)
                watchdog.start()
            try:
                vmm_rc = proc.wait()
            except KeyboardInterrupt:
                _killpg(proc.pid)
                proc.wait()
                raise
        if watchdog is not None:
            watchdog.stop()

        if host_out is not None:
            drives.copy_out(io_drive, host_out, run.path)
        exit_code = drives.read_exit_code(io_drive)
        logger.info(
            f"Firecracker VM karotte_run_{run_config.run_id} exited after {time.monotonic() - started:.1f}s "
            + f"(VMM exit {vmm_rc}, run exit {exit_code})"
        )
        if watchdog is not None and watchdog.fired:
            raise VmError("The guest stopped answering and its VM was killed")
        if not prepare_only and exit_code != 0:
            raise subprocess.CalledProcessError(
                exit_code if exit_code is not None else (vmm_rc or 1), argv
            )
    finally:
        if watchdog is not None:
            watchdog.stop()
        if relay is not None:
            relay.close()
        if proc is not None:
            if proc.poll() is None:
                _killpg(proc.pid)
                proc.wait()
            if proc.stdin is not None:
                proc.stdin.close()
        if jail_netns is not None:
            jail_netns.stop()
        if jailer is not None:
            remove_jail_cgroup(run.path.name)
        if keep_vm:
            logger.info(
                f"Kept the VM's drives in {run.path}; the run's changes to the filesystem are under upper/ in scratch.ext4"
            )
            if run.lock is not None:
                run.lock.close()
        else:
            run.remove()


# --- the jailer (opt-in, root only) ---

JAIL_CGROUP_PARENT = "karotte-firecracker"


def jailer_ids() -> tuple[int, int]:
    """The uid and gid the jailed VMM drops to: the user who ran sudo."""
    uid, gid = os.environ.get("SUDO_UID"), os.environ.get("SUDO_GID")
    if os.geteuid() != 0 or not uid or not gid:
        raise VmError(f"{JAILER_ENV_VAR}=1 needs karotte run under sudo")
    return int(uid), int(gid)


def jail_root(run_path: Path, firecracker: Path, vm_id: str) -> Path:
    """Where the jailer chroots the VMM."""
    return run_path / "jail" / firecracker.name / vm_id / "root"


def jailer_argv(
    jailer: Path,
    firecracker: Path,
    run_path: Path,
    vm_id: str,
    uid: int,
    gid: int,
    mem_mib: int,
    firecracker_args: Sequence[str],
    netns: str | None = None,
) -> list[str]:
    """The jailer chroots, caps the VMM's memory in its own cgroup, drops to
    ``uid``:``gid`` and execs Firecracker, in network namespace ``netns`` if
    given."""
    return [
        str(jailer),
        "--id",
        vm_id,
        "--exec-file",
        str(firecracker),
        "--uid",
        str(uid),
        "--gid",
        str(gid),
        "--chroot-base-dir",
        str(run_path / "jail"),
        "--cgroup-version",
        "2",
        "--parent-cgroup",
        JAIL_CGROUP_PARENT,
        "--cgroup",
        f"memory.max={(mem_mib + _VMM_OVERHEAD_MIB) << 20}",
        *(["--netns", netns] if netns is not None else []),
        "--",
        *firecracker_args,
    ]


_JAIL_NETNS_TIMEOUT_S = 15.0


@dataclass
class JailNetns:
    """The network namespace a jailed VMM runs in under pasta: a holder
    process owns it (see :func:`network.jail_netns_holder_command`) and pasta
    gives it the host's network from outside."""

    holder: subprocess.Popen[bytes]
    pasta: subprocess.Popen[bytes] | None
    path: str

    @classmethod
    def start(
        cls, allow: Sequence[network.Allow], blocked: Sequence[str], uid: int, gid: int
    ) -> "JailNetns":
        holder = subprocess.Popen(
            network.jail_netns_holder_command(allow, blocked, uid, gid),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=_set_pdeathsig,
        )
        assert holder.stdout is not None
        line = holder.stdout.readline().decode().split()
        if len(line) != 2 or line[0] != "ready":
            holder.kill()
            _, err = holder.communicate()
            raise VmError(
                f"Could not set up the jailed VM's network namespace: {err.decode().strip()}"
            )
        netns = cls(holder, None, f"/proc/{line[1]}/ns/net")
        try:
            # Once more if pasta quits while starting: seen once on a CI
            # runner, with nothing in its output to say why.
            for attempt in range(2):
                netns.pasta = subprocess.Popen(
                    network.pasta_attach_command(netns.path),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    preexec_fn=_set_pdeathsig,
                )
                try:
                    netns._wait_for_pasta()
                    break
                except VmError as e:
                    if attempt == 1 or netns.pasta.poll() is None:
                        raise
                    logger.warning(f"{e}; trying once more")
        except BaseException:
            netns.stop()
            raise
        return netns

    def _wait_for_pasta(self) -> None:
        """Until pasta has given the namespace a default route."""
        assert self.pasta is not None
        deadline = time.monotonic() + _JAIL_NETNS_TIMEOUT_S
        while time.monotonic() < deadline:
            if self.pasta.poll() is not None:
                assert self.pasta.stderr is not None
                raise VmError(
                    f"pasta could not attach to the jailed VM's network namespace (exit {self.pasta.returncode}): {self.pasta.stderr.read().decode().strip()}"
                )
            route = subprocess.run(
                [
                    "nsenter",
                    f"--net={self.path}",
                    "ip",
                    "-4",
                    "route",
                    "show",
                    "default",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if route.stdout.strip():
                return
            time.sleep(0.1)
        raise VmError("pasta did not set up the jailed VM's network in time")

    def stop(self) -> None:
        if self.pasta is not None and self.pasta.poll() is None:
            self.pasta.kill()
            _ = self.pasta.wait()
        if self.holder.poll() is None:
            assert self.holder.stdin is not None
            self.holder.stdin.close()
            try:
                _ = self.holder.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.holder.kill()
                _ = self.holder.wait()


_VMM_OVERHEAD_MIB = 256


def remove_jail_cgroup(vm_id: str) -> None:
    """The jailer leaves the VMM's cgroup behind; remove it, and the parent
    once no other VM uses it."""
    parent = Path("/sys/fs/cgroup", JAIL_CGROUP_PARENT)
    for path in (parent / vm_id, parent):
        try:
            path.rmdir()
        except OSError:
            pass


def _killpg(pgid: int) -> None:
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
