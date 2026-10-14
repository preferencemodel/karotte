"""Run one karotte evaluation in a Modal Sandbox.

The launcher starts the image as an idle sandbox, copies the run config and
the ``--mount`` paths in, and runs ``karotte run --no-containerized`` in it
with ``exec``. The websocket is relayed from 127.0.0.1 to the sandbox's TLS
tunnel. ``/out`` and the writable mounts come back as tarballs, and the
sandbox is terminated however the run ends.

``modal`` is the optional ``karotte[modal]`` extra, imported only here.
"""

import contextlib
import getpass
import hashlib
import ipaddress
import os
import re
import socket
import ssl
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import IO, Any
from urllib.parse import urlparse

from loguru import logger

from karotte.apple_container import STAGED_MOUNTS_DIR
from karotte.confinement import (
    DISK_BUDGET_ENV_VAR,
    FIREWALL_BACKEND_ENV_VAR,
    GIB,
    SANDBOX_MEMORY_ENV_VAR,
    VM_LAUNCHER_ENV_VAR,
)
from karotte.firecracker.vm import (
    DEV_TARGET,
    KAROTTE_BIN,
    HostRelay,
    _pipe,  # pyright: ignore[reportPrivateUsage]
)
from karotte.forwarded_env import sandbox_env
from karotte.hardware import VmSize, vm_size
from karotte.hide_run_config import RUN_CONFIG_PATH
from karotte.load_tasks import load_task
from karotte.schemas.evaluation_run_config import EvaluationRunConfig
from karotte.staged_mounts import (
    STAGED_MOUNTS_ENV_VAR,
    StagedMount,
    _copy_back,  # pyright: ignore[reportPrivateUsage]
    _copy_file_back,  # pyright: ignore[reportPrivateUsage]
    _entries,  # pyright: ignore[reportPrivateUsage]
    _remove_deleted,  # pyright: ignore[reportPrivateUsage]
    encode,
)

TIMEOUT_SECONDS = 24 * 3600
"""Modal's maximum; its default, 300 s, is too short for a run."""
IDLE_TIMEOUT_SECONDS = 30 * 60
"""Stops a sandbox whose launcher died, instead of billing for a day."""
DEFAULT_DISK_BUDGET_BYTES = 32 * GIB
"""The student's disk quota when no hardware plugin sets one."""

RUN_TAG = "karotte_run"
LAUNCHER_TAG = "karotte_launcher"
"""The app is shared by everyone in the Modal workspace, so cleanup only
touches sandboxes this user started on this machine."""

TRANSFER_DIR = "/root/.karotte_transfer"
OUT_DIR = "/out"

_LOCAL_HOSTS = ("localhost", "host.docker.internal", "host.containers.internal")
_BUILDER_ARG = re.compile(r"^\s*ARG\s+KAROTTE_IMAGE_BUILDER\b", re.MULTILINE)


class ModalError(RuntimeError):
    pass


def import_modal() -> ModuleType:
    try:
        import modal
    except ImportError as e:
        raise ModalError(
            "The modal runtime needs the modal package. Install it: uv pip install 'karotte[modal]'"
        ) from e
    return modal


def modal_problems(
    required_hardware: str | None,
    proxy_url: str | None,
    prepare_only: bool = False,
    keep_containers: bool = False,
) -> list[str]:
    """What stops a Modal run; empty when nothing does."""
    try:
        _ = import_modal()
    except ModalError as e:
        return [str(e)]
    problems: list[str] = []
    try:
        _ = vm_size(required_hardware)
    except ValueError:
        # TODO: GPU tasks: pass a `gpu` field of HardwareLimits to
        # Sandbox.create(gpu=...). Modal runs GPU sandboxes on gVisor only.
        problems.append(
            f"{required_hardware} needs devices passed through, which the modal runtime doesn't do yet; use --runtime docker."
        )
    if proxy_url and _unreachable(urlparse(proxy_url).hostname or ""):
        problems.append(
            f"A Modal sandbox can't reach the proxy at {proxy_url}. Use a proxy with a public address, or --no-proxy."
        )
    if prepare_only:
        problems.append("--prepare-only isn't supported on the modal runtime.")
    if keep_containers:
        problems.append("--keep-containers isn't supported on the modal runtime.")
    return problems


def _unreachable(host: str) -> bool:
    """Only this machine or its private network can reach ``host``."""
    if host in _LOCAL_HOSTS or host.endswith(".local"):
        return True
    try:
        return ipaddress.ip_address(host).is_private
    except ValueError:
        return False


def launcher_id() -> str:
    return f"{getpass.getuser()}@{socket.gethostname()}"


def guest_env(
    size: VmSize, proxy_url: str | None, staged: Sequence[StagedMount]
) -> dict[str, str]:
    env = {
        "KAROTTE_SANDBOX": "vm",
        DISK_BUDGET_ENV_VAR: str(size.disk_bytes or DEFAULT_DISK_BUDGET_BYTES),
        SANDBOX_MEMORY_ENV_VAR: str(size.sandbox_memory_bytes),
        VM_LAUNCHER_ENV_VAR: "modal",
        FIREWALL_BACKEND_ENV_VAR: "nft",
    }
    if proxy_url:
        env["ANTHROPIC_BASE_URL"] = proxy_url
        env["KAROTTE_PROXY_URL"] = proxy_url
    if staged:
        env[STAGED_MOUNTS_ENV_VAR] = encode(list(staged))
    return env | sandbox_env()


def _app(modal: ModuleType) -> Any:
    return modal.App.lookup("karotte", create_if_missing=True)


def modal_image(
    modal: ModuleType, build_context: str, build_secrets: Sequence[str] = ()
) -> Any:
    """The image Modal builds from the Containerfile. Each ``name=path``
    build secret is a dotenv file whose variables the build sees."""
    containerfile = Path(build_context) / "Containerfile"
    if not _BUILDER_ARG.search(containerfile.read_text()):
        raise ModalError(
            f"{containerfile} has no `ARG KAROTTE_IMAGE_BUILDER`, which a Modal build needs. "
            + "Copy the KAROTTE_IMAGE_BUILDER lines from the default template's Containerfile "
            + "(see the modal section of the Runtimes docs), then build again."
        )
    secrets = [
        modal.Secret.from_dotenv(path.parent, filename=path.name)
        for path in (Path(s.partition("=")[2]).absolute() for s in build_secrets)
    ]
    return modal.Image.from_dockerfile(
        containerfile,
        context_dir=build_context,
        secrets=secrets,
        build_args={"KAROTTE_IMAGE_BUILDER": "modal"},
        ignore=context_ignore(modal, Path(build_context)),
    )


def context_ignore(modal: ModuleType, context: Path) -> Callable[[Path], bool]:
    """``.dockerignore``, except that a directory keeps one empty ignored file,
    such as a ``.gitkeep``, when it would otherwise be empty: Docker copies an
    empty directory, but Modal uploads files only, so ``COPY student_data/``
    would fail. An ignored file with content is never uploaded."""
    context = context.absolute()
    dockerignore = context / ".dockerignore"
    patterns = dockerignore.read_text().splitlines() if dockerignore.is_file() else []
    ignored: Callable[[Path], bool] = modal.FilePatternMatcher(*patterns)
    kept: set[Path] = set()
    for dirpath, dirnames, filenames in os.walk(context):
        rel = Path(dirpath).relative_to(context)
        dirnames[:] = [d for d in dirnames if not ignored(rel / d)]
        files = sorted(rel / f for f in filenames)
        if rel != Path(".") and not dirnames and files and all(map(ignored, files)):
            kept.update([f for f in files if _empty_file(context / f)][:1])

    def ignore(path: Path) -> bool:
        if path.is_absolute():
            path = path.relative_to(context)
        return ignored(path) and path not in kept

    return ignore


def _empty_file(path: Path) -> bool:
    st = path.lstat()
    return stat.S_ISREG(st.st_mode) and not st.st_size


def image_record(build_context: str) -> Path:
    """Where ``karotte build`` notes the image it built, as docker notes a tag."""
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    key = hashlib.sha256(str(Path(build_context).resolve()).encode()).hexdigest()
    return Path(base) / "karotte" / "modal" / "images" / key[:16]


def build_image(
    build_context: str, build_secrets: Sequence[str] = (), show_output: bool = True
) -> None:
    """Build on Modal and note the image for ``karotte run``. The TUI turns
    ``show_output`` off, since it owns the screen."""
    modal = import_modal()
    with modal.enable_output() if show_output else contextlib.nullcontext():
        image = modal_image(modal, build_context, build_secrets).build(_app(modal))
    record = image_record(build_context)
    record.parent.mkdir(parents=True, exist_ok=True)
    _ = record.write_text(image.object_id)
    logger.info(f"Built image {image.object_id} on Modal")


@dataclass
class Upload:
    """A ``--mount`` path and the directory it is unpacked into."""

    source: Path
    target: str
    writable: bool
    staging: str
    entries: list[Path] = field(default_factory=list)
    """What a writable directory held, to tell what the student deleted from
    what the host added. Only regular files and directories: the sandbox
    copies nothing else back, so anything else would read as deleted."""

    @property
    def staged(self) -> StagedMount:
        source = (
            self.staging
            if self.source.is_dir()
            else f"{self.staging}/{self.source.name}"
        )
        return StagedMount(source, self.target, self.writable)


def plan_uploads(mounts: Sequence[str] | None) -> list[Upload]:
    uploads: list[Upload] = []
    for index, spec in enumerate(mounts or []):
        parts = spec.split(":")
        source = Path(parts[0]).resolve()  # a bind mount follows a link
        upload = Upload(
            source, parts[1], parts[2:] != ["ro"], f"{STAGED_MOUNTS_DIR}/{index}"
        )
        if upload.writable and source.is_dir():
            upload.entries = [
                e
                for e in _entries(source)
                if stat.S_ISREG(mode := (source / e).lstat().st_mode)
                or stat.S_ISDIR(mode)
            ]
        uploads.append(upload)
    return uploads


def exec_checked(sb: Any, *args: str) -> None:
    proc = sb.exec(*args)
    stderr = proc.stderr.read()  # before the wait, so a full stream can't stall it
    if code := proc.wait():
        raise ModalError(
            f"`{' '.join(args)}` failed in the sandbox (exit {code}): {stderr[-2000:]}"
        )


def put_tree(sb: Any, local: Path, remote_dir: str) -> None:
    """Copy a directory's contents, or a file, into ``remote_dir``."""
    remote = f"{TRANSFER_DIR}/in.tar"
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "in.tar"
        with tarfile.open(archive, "w") as tar:
            tar.add(local, arcname="." if local.is_dir() else local.name)
        sb.filesystem.copy_from_local(archive, remote)
    exec_checked(sb, "mkdir", "-p", remote_dir)
    exec_checked(sb, "tar", "-xf", remote, "-C", remote_dir, "--no-same-owner")


def get_tree(
    sb: Any, remote_dir: str, local_dir: Path, keep_mode: bool = False
) -> None:
    """Copy the regular files and directories in ``remote_dir`` into
    ``local_dir``, refusing paths out of it. ``keep_mode`` keeps the files'
    permission bits, which mount copy-back writes onto the host files."""
    remote = f"{TRANSFER_DIR}/out.tar"
    exec_checked(sb, "tar", "-cf", remote, "-C", remote_dir, ".")
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "out.tar"
        sb.filesystem.copy_to_local(remote, archive)
        with tarfile.open(archive) as tar:
            tar.extractall(local_dir, filter=lambda m, d: _safe_member(m, d, keep_mode))


def _safe_member(
    member: tarfile.TarInfo, dest: str, keep_mode: bool
) -> tarfile.TarInfo | None:
    """Regular files and directories only, as firecracker copies them back."""
    try:
        if member.isfile() or member.isdir():
            safe = tarfile.data_filter(member, dest)
            if keep_mode and member.isfile():
                # Readable by us, so copy-back can read it.
                safe = safe.replace(mode=member.mode & 0o777 | 0o400, deep=False)
            return safe
        reason = "not a file or directory"
    except tarfile.FilterError as e:
        reason = str(e)
    logger.warning(f"Not copying {member.name!r} back from the sandbox: {reason}")
    return None


def stage_inputs(
    sb: Any,
    run_config: EvaluationRunConfig,
    uploads: Sequence[Upload],
    dev: bool,
    build_context: str,
) -> None:
    exec_checked(sb, "mkdir", "-p", "-m", "0700", TRANSFER_DIR)
    exec_checked(sb, "mkdir", "-p", OUT_DIR)
    sb.filesystem.write_text(run_config.model_dump_json(), RUN_CONFIG_PATH)
    exec_checked(sb, "chmod", "0600", RUN_CONFIG_PATH)
    if dev:
        # Over the installed package: a file deleted on the host stays.
        put_tree(sb, Path(build_context).absolute() / "src" / "environment", DEV_TARGET)
    for upload in uploads:
        put_tree(sb, upload.source, upload.staging)


def bring_back(sb: Any, upload: Upload) -> None:
    """Copy a writable mount back over its host path, as a bind mount would
    have left it; the sandbox has already copied the student's changes to
    the staging directory."""
    with tempfile.TemporaryDirectory() as tmp:
        back = Path(tmp).resolve()  # copy-back distrusts links above it
        get_tree(sb, upload.staging, back, keep_mode=True)
        if upload.source.is_dir():
            _ = _copy_back(back, upload.source)
            _remove_deleted(back, upload.source, set(), upload.entries)
        elif (back / upload.source.name).is_file():
            _copy_file_back(back / upload.source.name, upload.source)


def run_modal(
    run_config: EvaluationRunConfig,
    *,
    dev: bool = False,
    log_file: Path | None = None,
    build_context: str = ".",
    mounts: Sequence[str] | None = None,
    proxy_url: str | None = None,
) -> None:
    """Run one evaluation in a Modal sandbox, like ``docker run`` of the
    image. Raises ``CalledProcessError`` when the run fails."""
    modal = import_modal()
    try:
        size = vm_size(load_task(run_config).required_hardware)
    except ValueError as e:
        raise ModalError(str(e)) from e
    record = image_record(build_context)
    if not record.is_file():
        raise ModalError(
            "No image built for this environment on Modal; run `karotte build --runtime modal`, or run without --dev"
        )
    uploads = plan_uploads(mounts)
    host_out = None
    if run_config.transcript_file:
        transcript = Path(run_config.transcript_file).absolute()
        host_out = transcript.parent
        host_out.mkdir(parents=True, exist_ok=True)
        run_config = run_config.model_copy(
            update={"transcript_file": f"{OUT_DIR}/{transcript.name}"}
        )
    port = run_config.websocket_config.port
    env = guest_env(size, proxy_url, [u.staged for u in uploads])
    sb = modal.Sandbox.create(
        "sleep",
        "infinity",
        app=_app(modal),
        image=modal.Image.from_id(record.read_text().strip()),
        tags={RUN_TAG: run_config.run_id, LAUNCHER_TAG: launcher_id()},
        cpu=float(size.cpus),
        memory=size.vm_memory_bytes >> 20,
        timeout=TIMEOUT_SECONDS,
        idle_timeout=IDLE_TIMEOUT_SECONDS,
        runtime="vm",  # a guest kernel of its own, so cgroups are real
        env=env,
        encrypted_ports=[port],
    )
    logger.info(f"Modal sandbox karotte_run_{run_config.run_id} is {sb.object_id}")
    try:
        # Listen before staging: the TUI connects as soon as the run starts.
        relay = _TunnelRelay(port, sb.tunnels()[port].tls_socket)
        try:
            stage_inputs(sb, run_config, uploads, dev, build_context)
            code = _exec_run(sb, env, log_file)
        finally:
            relay.close()
        for upload in uploads:
            if upload.writable:
                try:
                    bring_back(sb, upload)
                except Exception as e:  # noqa: BLE001 - one mount must not cost the rest
                    logger.warning(f"Could not copy mount {upload.target} back: {e}")
        if host_out is not None:
            get_tree(sb, OUT_DIR, host_out)
        if code != 0:
            raise subprocess.CalledProcessError(code, [KAROTTE_BIN, "run"])
    finally:
        sb.terminate()


class _TunnelRelay(HostRelay):
    """firecracker's relay from 127.0.0.1:port, to the sandbox's TLS tunnel
    instead of a vsock."""

    target: tuple[str, int]
    tls: ssl.SSLContext

    def __init__(self, port: int, target: tuple[str, int]) -> None:
        self.target = target
        self.tls = ssl.create_default_context()
        super().__init__(port, Path())

    def _handle(self, conn: socket.socket) -> None:
        try:
            raw = socket.create_connection(self.target, timeout=10)
            upstream = self.tls.wrap_socket(raw, server_hostname=self.target[0])
        except OSError:
            conn.close()
            return
        upstream.settimeout(None)
        threading.Thread(target=_pipe, args=(upstream, conn), daemon=True).start()
        _pipe(conn, upstream)
        conn.close()
        upstream.close()


def _exec_run(sb: Any, env: dict[str, str], log_file: Path | None) -> int:
    proc = sb.exec(
        KAROTTE_BIN, "run", "--no-containerized", "--config", RUN_CONFIG_PATH, env=env
    )
    lock = threading.Lock()

    def stream(lines: Iterable[str], out: IO[str]) -> None:
        for line in lines:
            with lock:
                _ = out.write(line)
                out.flush()

    with (
        open(log_file, "w", encoding="utf-8", errors="replace")
        if log_file
        else contextlib.nullcontext(sys.stdout)
    ) as out:
        threads = [
            threading.Thread(target=stream, args=(lines, out), daemon=True)
            for lines in (proc.stdout, proc.stderr)
        ]
        for thread in threads:
            thread.start()
        code = proc.wait()
        for thread in threads:
            thread.join()
    return code


def stop_sandboxes(run_ids: Sequence[str]) -> None:
    """Terminate this launcher's sandboxes for these exact run ids."""
    _terminate(lambda run_id: run_id in run_ids)


def clean_up_sandboxes(prefix: str) -> None:
    """Terminate this launcher's sandboxes left from an earlier run of this
    invocation, by run id prefix."""
    _terminate(lambda run_id: run_id.startswith(prefix))


def _terminate(matches: Callable[[str], bool]) -> None:
    try:
        modal = import_modal()
        app_id = _app(modal).app_id
        for sb in modal.Sandbox.list(app_id=app_id, tags={LAUNCHER_TAG: launcher_id()}):
            run_id = sb.get_tags().get(RUN_TAG, "")
            if run_id and matches(run_id):
                logger.info(f"Terminating Modal sandbox karotte_run_{run_id}")
                sb.terminate()
    except Exception as e:  # noqa: BLE001 - cleanup must not fail the run
        logger.warning(f"Could not list Modal sandboxes: {e}")
