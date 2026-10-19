"""Run one karotte evaluation in a Modal VM sandbox, relaying its websocket to
127.0.0.1 through a connect token and copying /out back at the end."""

import asyncio
import contextlib
import getpass
import hashlib
import importlib.util
import ipaddress
import os
import socket
import subprocess
import sys
import tarfile
import tempfile
import threading
from collections.abc import Callable, Sequence
from pathlib import Path, PurePosixPath
from typing import IO, Any, final
from urllib.parse import urlsplit

import websockets
from loguru import logger
from websockets.asyncio.server import ServerConnection, serve

from karotte.confinement import (
    DISK_BUDGET_ENV_VAR,
    SANDBOX_ENV_VAR,
    SANDBOX_MEMORY_ENV_VAR,
    VM_LAUNCHER_ENV_VAR,
    Sandbox,
)
from karotte.firecracker.drives import copy_plain
from karotte.forwarded_env import sandbox_env
from karotte.hardware import vm_size
from karotte.load_tasks import load_task
from karotte.schemas.evaluation_run_config import EvaluationRunConfig

APP_NAME = "karotte"
RUN_ID_TAG = "karotte_run_id"
OWNER_TAG = "karotte_owner"

KAROTTE_BIN = "/root/.venv/bin/karotte"
CONFIG_PATH = "/root/.karotte_run_config.json"
OUT_TAR = "/tmp/karotte_out.tar"

TIMEOUT_SECONDS = 24 * 60 * 60
"""Modal's longest sandbox lifetime."""
IDLE_TIMEOUT_SECONDS = 5 * 60
"""Ends a sandbox whose launcher died soon after its run does."""

NOCAPS_STEP = (
    "RUN if [ -f /etc/dnf/dnf.conf ]; then"
    + " sed -i '/^\\[main\\]/a tsflags=nocaps' /etc/dnf/dnf.conf; fi"
)
"""Modal's image builder can't set file capabilities, so installing shadow-utils fails without it."""

INSTALL_HINT = "Install the Modal client with `uv add 'karotte[modal]'`, then log in with `modal token new`."


class ModalError(Exception):
    pass


def modal_installed() -> bool:
    return importlib.util.find_spec("modal") is not None


def _modal() -> Any:
    try:
        import modal
    except ImportError as e:
        raise ModalError(
            f"The modal runtime needs the Modal client. {INSTALL_HINT}"
        ) from e
    return modal


def containerfile_for_modal(
    text: str, context: Path, ignored: Callable[[Path], bool]
) -> tuple[str, set[Path]]:
    """``text`` rewritten for Modal's builder, and the ignored placeholders (``.gitkeep``) it must upload:
    Modal can't COPY a folder holding no uploaded files, so they're copied and then removed."""
    out: list[str] = []
    placeholders: set[Path] = set()
    instruction: list[str] = []
    continued = False
    for line in text.splitlines():
        out.append(line)
        stripped = line.strip()
        if continued and (not stripped or stripped.startswith("#")):
            continue
        if not continued:
            instruction = []
        instruction.append(line)
        continued = line.rstrip().endswith("\\")
        if continued or not stripped or stripped.startswith("#"):
            continue
        words = " ".join(part.rstrip("\\") for part in instruction).split()
        keyword = words[0].upper()
        if keyword == "FROM" and words[1].lower() != "scratch":
            out.append(NOCAPS_STEP)
        elif keyword == "COPY":
            found = _placeholders(words[1:], context, ignored)
            if found:
                source, destination, files = found
                placeholders.update(source / f for f in files)
                out.append(_remove_step(destination, files))
    return "\n".join(out) + "\n", placeholders


def _placeholders(
    args: list[str], context: Path, ignored: Callable[[Path], bool]
) -> tuple[Path, str, list[Path]] | None:
    paths = [a for a in args if not a.startswith("--")]
    if len(paths) != 2 or paths[0].startswith("["):
        return None
    source = Path(paths[0].lstrip("/"))
    if not (context / source).is_dir():
        return None
    files = sorted(
        p.relative_to(context / source)
        for p in (context / source).rglob("*")
        if p.is_file()
    )
    if not files:
        raise ModalError(
            f"Modal can't COPY the empty folder {source}: put a file in it, such as a .gitkeep the .dockerignore leaves out."
        )
    if any(not ignored(source / f) for f in files):
        return None
    return source, paths[1], files


def _remove_step(destination: str, files: list[Path]) -> str:
    targets = " ".join(f'"{PurePosixPath(destination, f.as_posix())}"' for f in files)
    return f"RUN rm -f {targets}"


def _dockerignore_lines(context: Path) -> list[str]:
    for name in ("Containerfile.dockerignore", ".dockerignore"):
        path = context / name
        if path.is_file():
            return path.read_text().splitlines()
    return []


def _containerfile_cache(context: Path) -> Path:
    base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    name = hashlib.sha256(str(context).encode()).hexdigest()[:12]
    return base / "karotte" / "modal" / name / "Containerfile"


def _nothing_ignored(_path: Path) -> bool:
    return False


def modal_image(build_context: str) -> Any:
    """The environment's image; Modal caches its layers, so declaring it again is cheap once built."""
    modal = _modal()
    context = Path(build_context).absolute()
    lines = _dockerignore_lines(context)
    dockerignore: Callable[[Path], bool] = (
        modal.FilePatternMatcher(*lines) if lines else _nothing_ignored
    )
    text, placeholders = containerfile_for_modal(
        (context / "Containerfile").read_text(), context, dockerignore
    )
    containerfile = _containerfile_cache(context)
    containerfile.parent.mkdir(parents=True, exist_ok=True)
    # Parallel runs write it at once; Modal must never read a half-written one.
    with tempfile.NamedTemporaryFile(
        "w", dir=containerfile.parent, delete=False
    ) as tmp:
        tmp.write(text)
    os.replace(tmp.name, containerfile)

    def ignore(file: Path) -> bool:
        return dockerignore(file) and Path(file) not in placeholders

    return modal.Image.from_dockerfile(
        containerfile, context_dir=context, ignore=ignore
    )


def build_image(build_context: str) -> None:
    modal = _modal()
    app = modal.App.lookup(APP_NAME, create_if_missing=True)
    with modal.enable_output():
        modal_image(build_context).build(app)


def _resolve(host: str) -> list[str]:
    return [str(info[4][0]) for info in socket.getaddrinfo(host, None)]


def _proxy_problem(proxy_url: str) -> str | None:
    host = urlsplit(proxy_url).hostname or ""
    try:
        addresses = _resolve(host)
    except OSError as e:
        return f"Cannot resolve the proxy {host}: {e}"
    if all(ipaddress.ip_address(a.split("%")[0]).is_global for a in addresses):
        return None
    return f"The proxy {proxy_url} isn't on a public address, so a Modal sandbox can't reach it. Pass a public --proxy, or --no-proxy."


def modal_problems(
    *,
    required_hardware: str | None,
    mounts: Sequence[str] | None,
    dev: bool,
    keep_containers: bool,
    prepare_only: bool,
    proxy_url: str | None,
) -> list[str]:
    """What stops a Modal run; empty when nothing does."""
    if not modal_installed():
        return [f"The modal runtime needs the Modal client. {INSTALL_HINT}"]
    problems: list[str] = []
    try:
        _ = vm_size(required_hardware)
    except ValueError as e:
        problems.append(str(e))
    for flag, used in (
        ("--mount", bool(mounts)),
        ("--dev", dev),
        ("--keep-containers", keep_containers),
        ("--prepare-only", prepare_only),
    ):
        if used:
            problems.append(f"The modal runtime doesn't support {flag} yet.")
    if proxy_url and (problem := _proxy_problem(proxy_url)):
        problems.append(problem)
    return problems


_RELAY_RETRY_SECONDS = 1.0


@final
class WebsocketRelay:
    """Forwards websockets on 127.0.0.1:port to the sandbox named by :meth:`connect_to`;
    clients connecting earlier wait."""

    def __init__(self, port: int) -> None:
        self.port = port
        self._loop = asyncio.new_event_loop()
        self._upstream: tuple[str, str] | None = None
        self._ready = asyncio.Event()
        self._stop = asyncio.Event()
        started = threading.Event()
        self._error: OSError | None = None
        self._thread = threading.Thread(
            target=self._serve, args=(started,), daemon=True
        )
        self._thread.start()
        started.wait()
        if self._error is not None:
            raise ModalError(
                f"Cannot listen on 127.0.0.1:{port} for the websocket: {self._error}"
            )

    def _serve(self, started: threading.Event) -> None:
        async def main() -> None:
            try:
                server = await serve(self._handle, "127.0.0.1", self.port)
            except OSError as e:
                self._error = e
                started.set()
                return
            started.set()
            async with server:
                await self._stop.wait()

        self._loop.run_until_complete(main())

    def connect_to(self, url: str, token: str) -> None:
        websocket_url = "ws" + url.removeprefix("http")
        self._loop.call_soon_threadsafe(self._set_upstream, websocket_url, token)

    def _set_upstream(self, url: str, token: str) -> None:
        self._upstream = (url, token)
        self._ready.set()

    async def _handle(self, client: ServerConnection) -> None:
        await self._ready.wait()
        assert self._upstream is not None and client.request is not None
        url, token = self._upstream
        headers = {"Authorization": f"Bearer {token}"}
        while not self._stop.is_set():
            try:
                upstream = await websockets.connect(
                    url + client.request.path, additional_headers=headers
                )
            except (OSError, websockets.InvalidHandshake):
                await asyncio.sleep(_RELAY_RETRY_SECONDS)
                continue
            async with upstream:
                await _forward_both_ways(client, upstream)
            return

    def close(self) -> None:
        if self._thread.is_alive():
            self._loop.call_soon_threadsafe(self._stop.set)
            self._thread.join(timeout=5)
        self._loop.close()


async def _forward_both_ways(a: Any, b: Any) -> None:
    async def forward(src: Any, dst: Any) -> None:
        with contextlib.suppress(websockets.ConnectionClosed):
            async for message in src:
                await dst.send(message)

    tasks = [asyncio.create_task(forward(a, b)), asyncio.create_task(forward(b, a))]
    _, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()


def _owner() -> str:
    return f"{getpass.getuser()}@{socket.gethostname()}"


def guest_env(
    *, sandbox_memory_bytes: int, disk_bytes: int | None, proxy_url: str | None
) -> dict[str, str]:
    env = {
        SANDBOX_ENV_VAR: Sandbox.VM.value,
        VM_LAUNCHER_ENV_VAR: "modal",
        SANDBOX_MEMORY_ENV_VAR: str(sandbox_memory_bytes),
    }
    if disk_bytes is not None:
        env[DISK_BUDGET_ENV_VAR] = str(disk_bytes)
    if proxy_url:
        env["ANTHROPIC_BASE_URL"] = proxy_url
        env["KAROTTE_PROXY_URL"] = proxy_url
    return {**env, **sandbox_env()}


def run_modal(
    run_config: EvaluationRunConfig,
    *,
    log_file: Path | None = None,
    build_context: str = ".",
    proxy_url: str | None = None,
) -> None:
    """Run one evaluation in a Modal VM sandbox. Raises ``CalledProcessError``
    when the run fails and ``ModalError`` when Modal does."""
    modal = _modal()
    task = load_task(run_config)
    try:
        size = vm_size(task.required_hardware)
    except ValueError as e:
        raise ModalError(str(e)) from e

    host_out: Path | None = None
    if run_config.transcript_file:
        transcript = Path(run_config.transcript_file).absolute()
        host_out = transcript.parent
        host_out.mkdir(parents=True, exist_ok=True)
        run_config = run_config.model_copy(
            update={"transcript_file": f"/out/{transcript.name}"}
        )

    port = run_config.websocket_config.port
    argv = [KAROTTE_BIN, "run", "--no-containerized", "--config", CONFIG_PATH]
    relay = WebsocketRelay(port)
    sandbox = None
    try:
        app = modal.App.lookup(APP_NAME, create_if_missing=True)
        sandbox = modal.Sandbox.create(
            "sleep",
            "infinity",
            app=app,
            image=modal_image(build_context),
            runtime="vm",
            cpu=size.cpus,
            memory=size.vm_memory_bytes >> 20,
            timeout=TIMEOUT_SECONDS,
            idle_timeout=IDLE_TIMEOUT_SECONDS,
            tags={RUN_ID_TAG: run_config.run_id, OWNER_TAG: _owner()},
        )
        logger.info(
            f"Started Modal sandbox karotte_run_{run_config.run_id}: {size.cpus} CPUs, {size.vm_memory_bytes >> 20} MiB"
        )
        sandbox.filesystem.make_directory("/out")
        sandbox.filesystem.write_text(run_config.model_dump_json(), CONFIG_PATH)
        token = sandbox.create_connect_token(port=port)
        relay.connect_to(token.url, token.token)

        env = guest_env(
            sandbox_memory_bytes=size.sandbox_memory_bytes,
            disk_bytes=size.disk_bytes,
            proxy_url=proxy_url,
        )
        process = sandbox.exec("sh", "-c", 'exec "$0" "$@" 2>&1', *argv, env=env)
        with (
            open(log_file, "w")
            if log_file
            else contextlib.nullcontext(sys.stdout) as out
        ):
            _stream(process.stdout, out)
        returncode = process.wait()

        if host_out is not None:
            _copy_out(sandbox, host_out)
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, argv)
    except modal.exception.Error as e:
        raise ModalError(f"Modal: {e}") from e
    finally:
        relay.close()
        if sandbox is not None:
            with contextlib.suppress(Exception):
                sandbox.terminate()


def _stream(lines: Any, out: IO[str]) -> None:
    for line in lines:
        out.write(line)
        out.flush()


def _copy_out(sandbox: Any, dest: Path) -> None:
    """Copy the sandbox's /out into ``dest``: regular files and directories only."""
    tar = sandbox.exec("tar", "-C", "/out", "-cf", OUT_TAR, ".")
    if tar.wait() != 0:
        raise ModalError("Could not pack /out in the sandbox to copy it back")
    with tempfile.TemporaryDirectory(prefix="karotte-modal-out-") as tmp:
        local_tar = Path(tmp) / "out.tar"
        sandbox.filesystem.copy_to_local(OUT_TAR, local_tar)
        extracted = Path(tmp) / "out"
        with tarfile.open(local_tar) as archive:
            archive.extractall(
                extracted,
                filter=lambda member, path: (
                    member if member.isfile() or member.isdir() else None
                ),
            )
        copy_plain(extracted, dest)


def _our_sandboxes(modal: Any) -> list[Any]:
    app = modal.App.lookup(APP_NAME, create_if_missing=True)
    return list(modal.Sandbox.list(app_id=app.app_id, tags={OWNER_TAG: _owner()}))


def stop_sandboxes(run_ids: Sequence[str]) -> None:
    """Terminate the sandboxes of these runs; their launchers then return."""
    modal = _modal()
    for sandbox in _our_sandboxes(modal):
        if sandbox.get_tags().get(RUN_ID_TAG) in run_ids:
            sandbox.terminate()


def clean_up_sandboxes(prefix: str) -> None:
    """Terminate this machine's sandboxes left from runs of this invocation's prefix."""
    modal = _modal()
    for sandbox in _our_sandboxes(modal):
        if sandbox.get_tags().get(RUN_ID_TAG, "").startswith(prefix):
            sandbox.terminate()


def _main(args: list[str]) -> None:
    if len(args) != 2 or args[0] != "build":
        sys.exit("usage: python -m karotte.modal_runtime build CONTEXT")
    try:
        build_image(args[1])
    except Exception as e:  # noqa: BLE001 - shown to whoever ran the build
        sys.exit(f"Modal image build failed: {e}")


if __name__ == "__main__":
    _main(sys.argv[1:])
