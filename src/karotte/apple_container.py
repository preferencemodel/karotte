"""The `apple-container` runtime: each run in its own Apple `container` VM (macOS 26+,
Apple silicon).

The guest has its own kernel, so karotte runs it as the `vm` sandbox. What
differs from docker/podman:
- Bind mounts are virtiofs, which ignores guest ownership and mode, so every
  mount sits under root's 0700 home: the transcript at ``/root/out``, user
  mounts at ``/root/.karotte_mounts/<n>`` (copied into place by karotte in the
  guest, see ``karotte.staged_mounts``).
- The rootfs is a sparse 512 GiB file, so the launcher passes the student's
  disk budget, sized from the host's real free space.
- A guest that livelocks blocks every `container` command on the Mac, so a
  watchdog kills that VM's host processes if it stops answering.
- `container cp` restores ownership and setuid bits on the host, so data is
  copied out with `container export` and a tar extraction that drops them.
"""

import json
import os
import platform
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal, Never, final

import typer
from loguru import logger

from karotte.confinement import (
    _FREE_DISK_FRACTION,  # pyright: ignore[reportPrivateUsage]
    DISK_BUDGET_ENV_VAR,
    GIB,
    SANDBOX_ENV_VAR,
    SANDBOX_MEMORY_ENV_VAR,
    VM_LAUNCHER_ENV_VAR,
    Sandbox,
)
from karotte.forwarded_env import sandbox_env
from karotte.hardware import VmSize, vm_size
from karotte.schemas.evaluation_run_config import EvaluationRunConfig
from karotte.staged_mounts import STAGED_MOUNTS_ENV_VAR, StagedMount, encode
from karotte.task import Task

CONTAINER = "container"
IMAGE = "karotte"
NAME_PREFIX = "karotte_run_"
BUILDER_ID = "buildkit"
"""Apple's image builder, listed next to our containers."""

MIN_MACOS_MAJOR = 26
MIN_VERSION = (1, 4, 1)


TRANSCRIPT_DIR = "/root/out"
STAGED_MOUNTS_DIR = "/root/.karotte_mounts"
DEV_SRC_TARGET = "/root/.venv/lib/python3.12/site-packages/environment/"

APP_ROOT = Path.home() / "Library" / "Application Support" / "com.apple.container"
VM_PROCESS_NAME = "com.apple.Virtualization.VirtualMachine"

COMMAND_TIMEOUT_S = 60


def get_run_command(
    run_config: EvaluationRunConfig,
    task: Task,
    dev: bool,
    keep_container: bool,
    build_context: str = ".",
    mounts: list[str] | None = None,
    proxy_url: str | None = None,
    prepare_only: bool = False,
    parallel_runs: int = 1,
) -> tuple[list[str], EvaluationRunConfig]:
    """The `container run` command for one run, and the config as the guest
    sees it (transcript path moved under ``/root/out``). ``parallel_runs`` is
    how many runs are launched together, which share the host's free disk."""
    size = vm_resources(task.required_hardware)
    command = [
        CONTAINER,
        "run",
        "--name",
        f"{NAME_PREFIX}{run_config.run_id}",
        "--env",
        f"{SANDBOX_ENV_VAR}={Sandbox.VM}",
        "--env",
        f"{SANDBOX_MEMORY_ENV_VAR}={size.sandbox_memory_bytes}",
        "--env",
        f"{VM_LAUNCHER_ENV_VAR}=apple-container",
        "--env",
        f"{DISK_BUDGET_ENV_VAR}={disk_budget_bytes(task.required_hardware, runs=parallel_runs)}",
        "--cap-add",
        "CAP_NET_ADMIN",
        "--cap-add",
        "CAP_SYS_ADMIN",
        # Without it the student could lower its own oom_score_adj.
        "--cap-add",
        "CAP_SYS_RESOURCE",
        "--cpus",
        str(size.cpus),
        "--memory",
        f"{size.vm_memory_bytes // 1024**2}M",
    ]

    if platform_flag := image_platform(IMAGE):
        command.extend(["--platform", platform_flag])

    # `--publish` binds every interface unless given an address.
    if not prepare_only:
        port = run_config.websocket_config.port
        command.extend(["--publish", f"127.0.0.1:{port}:{port}"])

    if not keep_container:
        command.append("--rm")

    if dev:
        src_path = Path(build_context).absolute() / "src" / "environment"
        command.extend(
            ["--mount", f"type=bind,source={src_path},target={DEV_SRC_TARGET}"]
        )

    staged: list[StagedMount] = []
    for index, spec in enumerate(mounts or []):
        parts = spec.split(":")
        source = Path(parts[0]).absolute()
        writable = not (len(parts) == 3 and parts[2] == "ro")
        staging = f"{STAGED_MOUNTS_DIR}/{index}"
        # virtiofs shares directories; a file comes in through a directory of
        # its own, not its parent, which would share its siblings too.
        if source.is_dir():
            mount_source, staged_source = source, staging
        else:
            mount_source = share_file(source, run_config.run_id, index, writable)
            staged_source = f"{staging}/{source.name}"
        mount_arg = f"type=bind,source={mount_source},target={staging}"
        if not writable:
            mount_arg += ",readonly"
        command.extend(["--mount", mount_arg])
        staged.append(StagedMount(staged_source, parts[1], writable))
    if staged:
        command.extend(["--env", f"{STAGED_MOUNTS_ENV_VAR}={encode(staged)}"])

    if run_config.transcript_file:
        file_path = Path(run_config.transcript_file).absolute()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        command.extend(
            [
                "--mount",
                f"type=bind,source={file_path.parent},target={TRANSCRIPT_DIR}",
            ]
        )
        run_config = run_config.model_copy(
            update={"transcript_file": f"{TRANSCRIPT_DIR}/{file_path.name}"}
        )

    if proxy_url:
        command.extend(["--env", f"ANTHROPIC_BASE_URL={proxy_url}"])
        command.extend(["--env", f"KAROTTE_PROXY_URL={proxy_url}"])

    for var, value in sandbox_env().items():
        command.extend(["--env", f"{var}={value}"])

    command.append(IMAGE)
    command.extend(
        [
            "/root/.venv/bin/karotte",
            "run",
            "--no-containerized",
            *(["--prepare-only"] if prepare_only else []),
            "--config",
            run_config.model_dump_json(),
        ]
    )
    return command, run_config


FILE_MOUNTS_DIR = Path.home() / ".cache" / "karotte" / "container-file-mounts"


def file_mounts_dir(run_id: str) -> Path:
    return FILE_MOUNTS_DIR / run_id


_SHARES_FILE = "shares.json"


def share_file(source: Path, run_id: str, index: int, writable: bool) -> Path:
    """A directory holding only ``source``, to share with the VM in its place.

    The file is hard-linked in under its own name (a link is followed to its
    target first). The guest's copy-back replaces the shared file with a new
    one, and :func:`finish_file_shares` moves that over the original after the
    run. Where the file can't be linked (another volume), a read-only mount
    gets a copy; a writable one is refused, since moving it back wouldn't be
    atomic."""
    original = source.resolve()
    directory = file_mounts_dir(run_id) / str(index)
    directory.mkdir(parents=True, exist_ok=True)
    directory.chmod(0o700)
    shared = directory / source.name
    shared.unlink(missing_ok=True)
    try:
        os.link(original, shared)
    except OSError as exc:
        if writable:
            _abort(
                f"Can't share {source} with the VM on its own ({exc}). Mount its"
                + " directory instead, or append :ro."
            )
        _ = shutil.copy2(original, shared)
    if writable:
        record = file_mounts_dir(run_id) / _SHARES_FILE
        shares = json.loads(record.read_text()) if record.exists() else []
        shares.append({"shared": str(shared), "original": str(original)})
        _ = record.write_text(json.dumps(shares))
    return directory


def finish_file_shares(run_id: str) -> None:
    """Move what the guest copied back into each writable single-file share
    over its original, then remove the run's shares. A share still holding the
    original (same inode) had nothing copied back, or a copy cut short."""
    directory = file_mounts_dir(run_id)
    record = directory / _SHARES_FILE
    try:
        shares = json.loads(record.read_text()) if record.exists() else []
        for share in shares:
            shared, original = Path(share["shared"]), Path(share["original"])
            try:
                if shared.stat().st_ino != original.stat().st_ino:
                    os.replace(shared, original)
            except OSError as exc:
                logger.warning(
                    f"Could not copy {original.name} back to {original}: {exc}"
                )
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def vm_resources(required_hardware: str | None) -> VmSize:
    """The VM's size for a hardware type (see :func:`karotte.hardware.vm_size`).
    The VM adds a vCPU of its own."""
    try:
        return vm_size(required_hardware)
    except ValueError as e:
        _abort(f"The apple-container runtime can't run this task: {e}.")


def disk_budget_bytes(
    required_hardware: str | None, free_bytes: int | None = None, runs: int = 1
) -> int:
    """The student's disk budget: the hardware plugin's, capped at this run's
    share of the host's free space now. The rootfs is sparse, so ``runs``
    launched together split that share instead of each being promised all of
    it. The guest's own `df` sees the sparse rootfs."""
    if free_bytes is None:
        root = APP_ROOT if APP_ROOT.is_dir() else Path.home()
        free_bytes = shutil.disk_usage(root).free
    budget = int(free_bytes * _FREE_DISK_FRACTION) // max(1, runs)
    class_budget = vm_resources(required_hardware).disk_bytes
    if class_budget is not None:
        budget = min(budget, class_budget)
    return budget


def image_platform(image: str) -> str | None:
    """``linux/amd64`` for an image with only an amd64 variant (it runs under
    Rosetta), else ``None`` for the native default."""
    result = subprocess.run(
        [CONTAINER, "image", "inspect", image],
        capture_output=True,
        text=True,
        timeout=COMMAND_TIMEOUT_S,
    )
    if result.returncode != 0:
        return None
    try:
        images: list[dict[str, Any]] = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    archs = {
        variant.get("config", {}).get("architecture")
        for image_info in images
        for variant in image_info.get("variants", [])
    }
    if "arm64" not in archs and "amd64" in archs:
        return "linux/amd64"
    return None


def assign_host_ports(
    run_configs: list[EvaluationRunConfig],
    is_free: Callable[[int], bool] | None = None,
    free_port: Callable[[], int] | None = None,
) -> list[EvaluationRunConfig]:
    """Give each run a websocket port that is free on the host, keeping the
    configured one where it is. Each VM has its own network, so the guest
    listens on the same port."""
    is_free = is_free or _port_is_free
    free_port = free_port or _free_port
    taken: set[int] = set()
    assigned: list[EvaluationRunConfig] = []
    for config in run_configs:
        port = config.websocket_config.port
        while port in taken or not is_free(port):
            port = free_port()
        taken.add(port)
        assigned.append(
            config.model_copy(
                update={
                    "websocket_config": config.websocket_config.model_copy(
                        update={"port": port}
                    )
                }
            )
        )
    return assigned


def _port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


Runner = Callable[..., subprocess.CompletedProcess[str]]


def list_containers(run: Runner = subprocess.run) -> list[dict[str, Any]] | None:
    """Every container `container ls -a` reports, or ``None`` if it failed."""
    try:
        result = run(
            [CONTAINER, "ls", "-a", "--format", "json"],
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        logger.warning("Listing containers timed out")
        return None
    if result.returncode != 0:
        logger.warning(
            f"Failed to list containers: {result.stderr.strip() or 'unknown error'}"
        )
        return None
    try:
        return json.loads(result.stdout or "[]")
    except json.JSONDecodeError:
        logger.warning("Failed to parse the container list")
        return None


def karotte_container_ids(
    containers: list[dict[str, Any]], prefix: str = NAME_PREFIX
) -> list[str]:
    """IDs of karotte's own containers starting with ``prefix``. Karotte names
    every container it starts ``karotte_run_<run_id>``; anything else, UUID
    names included, may be the user's."""
    assert prefix.startswith(NAME_PREFIX)
    ids: list[str] = []
    for container in containers:
        container_id = container.get("configuration", {}).get("id")
        if (
            isinstance(container_id, str)
            and container_id != BUILDER_ID
            and container_id.startswith(prefix)
        ):
            ids.append(container_id)
    return ids


def stop_containers(run_ids: list[str], run: Runner = subprocess.run) -> None:
    """Stop the containers of these exact run IDs that exist."""
    containers = list_containers(run)
    if containers is None:
        return
    wanted = {f"{NAME_PREFIX}{run_id}" for run_id in run_ids}
    names = [c for c in karotte_container_ids(containers) if c in wanted]
    if names:
        _run_logged(run, [CONTAINER, "stop", *names], "stop containers")


def clean_up_old_containers(prefix: str, run: Runner = subprocess.run) -> None:
    """Delete karotte's containers whose names start with
    ``karotte_run_<prefix>``, stopped ones included: a `run --rm` that didn't
    finish leaves one behind."""
    containers = list_containers(run)
    if containers is None:
        return
    ids = karotte_container_ids(containers, f"{NAME_PREFIX}{prefix}")
    if ids:
        _run_logged(run, [CONTAINER, "delete", "--force", *ids], "remove containers")


def _run_logged(run: Runner, command: list[str], what: str) -> bool:
    try:
        result = run(command, capture_output=True, text=True, timeout=COMMAND_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        logger.warning(f"Failed to {what}: timed out")
        return False
    if result.returncode != 0:
        logger.warning(f"Failed to {what}: {result.stderr.strip() or 'unknown error'}")
        return False
    return True


def export_hint(run_id: str) -> str:
    """How to copy a kept container's workdir to the host. Not `container cp`:
    it restores ownership and setuid bits from the guest on the host."""
    name = f"{NAME_PREFIX}{run_id}"
    return (
        f"{CONTAINER} export -o {name}.tar {name} && mkdir -p out && "
        + f"tar -xf {name}.tar -C out --no-same-owner --no-same-permissions workdir"
    )


def check_build_context(build_context: str) -> None:
    """Refuse a build context whose path goes through a symlink (``/tmp``,
    ``/var``) or lies under ``/private/tmp``: Apple's builder then copies
    directories empty."""
    absolute = Path(os.path.abspath(build_context))
    resolved = Path(os.path.realpath(build_context))
    if absolute != resolved or resolved.is_relative_to("/private/tmp"):
        _abort(
            f"Apple `container` can't build from {absolute} (resolves to {resolved}): "
            + "directories in a build context under a symlinked path or /private/tmp "
            + "come out empty. Move the environment under your home directory."
        )


def validate_container_runtime(
    required_hardware: str | None = None,
    run: Runner = subprocess.run,
    mac_version: str | None = None,
    machine: str | None = None,
    host_memory_bytes: int | None = None,
) -> None:
    """Abort unless Apple `container` can run this task here: macOS 26 or
    later on Apple silicon, the `container` service running, version 1.4.1 or
    later, and enough RAM for the VM of ``required_hardware`` if given."""
    if mac_version is None:
        mac_version = platform.mac_ver()[0] if sys.platform == "darwin" else ""
    machine = machine or platform.machine()
    if not mac_version:
        _abort("The apple-container runtime needs macOS (Apple `container`).")
    if int(mac_version.split(".")[0]) < MIN_MACOS_MAJOR:
        _abort(
            f"The apple-container runtime needs macOS {MIN_MACOS_MAJOR} or later, found {mac_version}. "
            + "Apple `container`'s networking is broken on earlier versions."
        )
    if machine != "arm64":
        _abort(f"The apple-container runtime needs Apple silicon, found {machine}.")

    install = "Install it from https://github.com/apple/container/releases."
    version = _query(run, [CONTAINER, "--version"], install)
    found = re.search(r"version (\d+)\.(\d+)\.(\d+)", version.stdout)
    if found is None:
        _abort(f"Could not read the Apple `container` version from {version.stdout!r}.")
    if tuple(int(part) for part in found.groups()) < MIN_VERSION:
        _abort(
            f"The apple-container runtime needs Apple `container` {'.'.join(map(str, MIN_VERSION))} "
            + f"or later, found {'.'.join(found.groups())}. {install}"
        )

    status = _query(run, [CONTAINER, "system", "status"], install)
    if status.returncode != 0 or not re.search(
        r"^status\s+running\s*$", status.stdout, re.MULTILINE
    ):
        _abort(
            "The Apple `container` service is not running. Start it with `container system start`."
        )

    memory_bytes = vm_resources(required_hardware).vm_memory_bytes
    if host_memory_bytes is None:
        host_memory_bytes = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    if memory_bytes > host_memory_bytes:
        _abort(
            f"This task needs a {memory_bytes / GIB:.0f} GiB VM, "
            + f"more than this machine's {host_memory_bytes / GIB:.0f} GiB."
        )


def _query(
    run: Runner, command: list[str], install: str
) -> subprocess.CompletedProcess[str]:
    try:
        return run(command, capture_output=True, text=True, timeout=COMMAND_TIMEOUT_S)
    except FileNotFoundError:
        _abort(f"Apple `container` is not installed. {install}")
    except subprocess.TimeoutExpired:
        _abort(
            f"`{shlex.join(command)}` did not answer within {COMMAND_TIMEOUT_S}s; "
            + "the Apple `container` service may be stuck on a VM."
        )


def _abort(message: str) -> Never:
    message += "\nTo run without a VM, use `--runtime docker` (or `--runtime podman`)."
    typer.secho(message, fg=typer.colors.RED, err=True)
    raise typer.Abort()


WATCHDOG_INTERVAL_S = 30.0
PROBE_TIMEOUT_S = 60.0
MAX_UNANSWERED_PROBES = 3
"""Consecutive probes the guest may leave unanswered before it is killed:
about six minutes with the timings above."""

Probe = Literal["answered", "failed", "timed_out"]


@final
class LivenessWatchdog:
    """Kills one run's VM if its guest stops answering.

    A livelocked guest (e.g. memory reclaim thrashing) doesn't die by itself,
    and while it spins, `container exec`, `kill`, `delete --force` and new
    runs hang for every container on the Mac. Every ``interval_s`` the
    watchdog runs `true` in the guest. A probe that fails quickly (the
    container isn't up yet, or is gone) doesn't count. A timed-out one only
    counts when the run's websocket has answered a ping before and doesn't
    now: the `container` service can be stuck on another VM while this one is
    fine, and only the websocket reaches the guest without it. With no
    websocket to ask (``--prepare-only``, or a run whose server isn't up yet),
    a timeout can't be pinned on this VM and isn't counted. After
    ``max_unanswered`` counted probes in a row, it kills this VM's host
    processes (see ``kill_vm``) and calls ``on_kill``.
    """

    def __init__(
        self,
        name: str,
        websocket_port: int | None,
        on_kill: Callable[[], None] = lambda: None,
        *,
        interval_s: float = WATCHDOG_INTERVAL_S,
        probe_timeout_s: float = PROBE_TIMEOUT_S,
        max_unanswered: int = MAX_UNANSWERED_PROBES,
        exec_probe: Callable[[str, float], Probe] | None = None,
        websocket_probe: Callable[[int, float], bool] | None = None,
        kill: Callable[[str], None] | None = None,
    ) -> None:
        assert name.startswith(NAME_PREFIX)
        self.name = name
        self.websocket_port = websocket_port
        self.on_kill = on_kill
        self.interval_s = interval_s
        self.probe_timeout_s = probe_timeout_s
        self.max_unanswered = max_unanswered
        self.exec_probe = exec_probe or _exec_probe
        self.websocket_probe = websocket_probe or _websocket_answers
        self.kill = kill or kill_vm
        self.killed = False
        self._websocket_seen = False
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._watch, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        """Stop watching. Doesn't wait for a probe in flight: its result is
        dropped."""
        self._stopped.set()

    def _watch(self) -> None:
        unanswered = 0
        while not self._stopped.wait(self.interval_s):
            answered = self.check()
            if self._stopped.is_set():
                return
            if answered:
                unanswered = 0
                continue
            unanswered += 1
            logger.warning(
                f"{self.name}: guest did not answer within {self.probe_timeout_s:.0f}s "
                + f"({unanswered}/{self.max_unanswered})"
            )
            if unanswered >= self.max_unanswered:
                logger.error(f"{self.name}: guest stopped answering; killing its VM")
                self.kill(self.name)
                self.killed = True
                self.on_kill()
                return

    def check(self) -> bool:
        """One probe round: whether the guest counts as answering."""
        port = self.websocket_port
        if self.exec_probe(self.name, self.probe_timeout_s) != "timed_out":
            if port is not None and not self._websocket_seen:
                self._websocket_seen = self.websocket_probe(port, self.probe_timeout_s)
            return True
        if port is not None and self.websocket_probe(port, self.probe_timeout_s):
            self._websocket_seen = True
            return True
        if not self._websocket_seen:
            logger.warning(
                f"{self.name}: `container exec` timed out, and with no websocket"
                + " answer to compare against it may be another VM holding up"
                + " the service; not counting it"
            )
            return True
        return False


def _exec_probe(name: str, timeout_s: float) -> Probe:
    try:
        result = subprocess.run(
            [CONTAINER, "exec", name, "true"],
            capture_output=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return "timed_out"
    return "answered" if result.returncode == 0 else "failed"


def _websocket_answers(port: int, timeout_s: float) -> bool:
    from websockets.exceptions import WebSocketException
    from websockets.sync.client import connect

    try:
        with connect(
            f"ws://127.0.0.1:{port}", open_timeout=timeout_s, close_timeout=1
        ) as ws:
            return ws.ping().wait(timeout_s)
    except (OSError, TimeoutError, WebSocketException):
        return False


def kill_vm(
    name: str,
    run: Runner = subprocess.run,
    kill: Callable[[int, int], None] = os.kill,
) -> None:
    """SIGKILL the host processes of the VM named ``name``, then delete it.

    `container kill` and `delete --force` hang on a livelocked guest; what
    frees it is killing both its `container-runtime-linux ... --uuid <name>`
    helper and the Virtualization.framework process that has its rootfs open.
    Only processes matched to this container are touched."""
    assert name.startswith(NAME_PREFIX)
    helpers, root = _runtime_helpers(name, run)
    vm_pids = _vm_processes(root / "rootfs.ext4", run)
    if not helpers and not vm_pids:
        logger.warning(f"{name}: found no host processes to kill")
    for pid in [*vm_pids, *helpers]:
        try:
            kill(pid, signal.SIGKILL)
            logger.info(f"{name}: killed host process {pid}")
        except ProcessLookupError:
            pass
    _ = _run_logged(run, [CONTAINER, "delete", "--force", name], f"delete {name}")


_HELPER_RE = re.compile(
    r"^\s*(\d+)\s+\S*container-runtime-linux\s+start\s+--root\s+(.+?)\s+--uuid\s+(\S+)\s*$"
)


def _runtime_helpers(name: str, run: Runner) -> tuple[list[int], Path]:
    """PIDs of the runtime helpers started for ``name`` and its root
    directory."""
    root = APP_ROOT / "containers" / name
    result = run(["ps", "-axww", "-o", "pid=,command="], capture_output=True, text=True)
    pids: list[int] = []
    for line in result.stdout.splitlines():
        match = _HELPER_RE.match(line)
        if match and match.group(3) == name:
            pids.append(int(match.group(1)))
            root = Path(match.group(2))
    return pids, root


def _vm_processes(rootfs: Path, run: Runner) -> list[int]:
    """PIDs of the Virtualization.framework processes that have ``rootfs``
    open."""
    result = run(["lsof", "-t", "--", str(rootfs)], capture_output=True, text=True)
    pids: list[int] = []
    for pid in result.stdout.split():
        comm = run(["ps", "-o", "comm=", "-p", pid], capture_output=True, text=True)
        if comm.stdout.strip().endswith(VM_PROCESS_NAME):
            pids.append(int(pid))
    return pids


def run_with_watchdog(
    command: Sequence[str],
    run_config: EvaluationRunConfig,
    check: bool,
    websocket_port: int | None,
    **popen_kwargs: Any,
) -> int:
    """Run a `container run` command under a ``LivenessWatchdog``."""
    name = f"{NAME_PREFIX}{run_config.run_id}"
    with subprocess.Popen(command, **popen_kwargs) as process:
        watchdog = LivenessWatchdog(name, websocket_port, on_kill=process.kill)
        watchdog.start()
        try:
            returncode = process.wait()
        except BaseException:
            process.kill()
            raise
        finally:
            watchdog.stop()
            finish_file_shares(run_config.run_id)
    if watchdog.killed:
        logger.error(f"{name}: the VM was killed after its guest stopped answering")
    if check and returncode != 0:
        raise subprocess.CalledProcessError(returncode, list(command))
    return returncode
