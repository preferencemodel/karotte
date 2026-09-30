"""Shared helpers for running containerized evaluations."""

import json
import os
import socket
import stat
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Never

import anyio
import pydantic
import typer
from loguru import logger

from karotte import Runtime
from karotte.check_paths import split_loader_path
from karotte.confinement import Confinement, current_sandbox
from karotte.container import is_containerized
from karotte.hardware import container_run_args
from karotte.load_tasks import load_task
from karotte.providers import SERVICE_TIER_ENV
from karotte.runtime import get_engine
from karotte.save_artifact import local_artifact_dir
from karotte.schemas.evaluation_run_config import EvaluationRunConfig
from karotte.schemas.transcript import ErrorEvent, TaskCompletedEvent
from karotte.task import Task
from karotte.transcript_streaming.stream_transcript_to_stdout import (
    stream_transcript_to_stdout,
)
from karotte.transcript_streaming.stream_transcript_to_websocket import (
    stream_transcript_to_websocket,
)

if TYPE_CHECKING:
    from karotte.evaluation_runner import EvaluationRunner


EXIT_ON_RUN_ERROR_ENV_VAR = "KAROTTE_EXIT_ON_RUN_ERROR"
"""Set by the host on its containers so the inner run exits non-zero when the run ends in an error."""


def parse_config(config: str, prepare_only: bool = False) -> EvaluationRunConfig:
    """Parse an EvaluationRunConfig from a JSON string or file path.

    When ``prepare_only`` is set, the model API key requirement is skipped: a
    prepared env is held for inspection and never calls the model.
    """
    context = {"prepare_only": prepare_only}
    json_validation_error: pydantic.ValidationError | None = None
    try:
        return _resolve_model_api_key(
            EvaluationRunConfig.model_validate_json(config, context=context),
            prepare_only,
        )
    except pydantic.ValidationError as e:
        json_validation_error = e

    if not Path(config).is_file():
        error_msg = "Failed to load evaluation run configuration.\n"
        error_msg += f"Input: {config!r}\n\n"
        error_msg += "Validation errors:\n"
        for error in json_validation_error.errors():
            loc = ".".join(str(x) for x in error["loc"]) or "(root)"
            error_msg += f"  - {loc}: {error['msg']}\n"
        _print_and_abort(error_msg)

    try:
        return _resolve_model_api_key(
            EvaluationRunConfig.model_validate_json(
                Path(config).read_text(), context=context
            ),
            prepare_only,
        )
    except pydantic.ValidationError:
        _print_and_abort(
            "Failed to load evaluation run configuration. "
            + f"{config!r} does not contain a valid configuration."
        )


def _resolve_model_api_key(
    run_config: EvaluationRunConfig, prepare_only: bool = False
) -> EvaluationRunConfig:
    """Resolve $ENV_VAR references in the config."""
    if run_config.model_api_key and run_config.model_api_key.startswith("$"):
        env_var = run_config.model_api_key[1:]
        try:
            value = os.environ[env_var]
        except KeyError:
            if run_config.use_fake_model or prepare_only:
                return run_config
            _print_and_abort(
                f"The run config references {env_var!r} as the model API key, but it's not set as an environment variable."
            )
        return run_config.model_copy(update={"model_api_key": value})
    return run_config


def build_configs(run_config: EvaluationRunConfig, n: int) -> list[EvaluationRunConfig]:
    """Create n variations of a run configuration for parallel runs."""
    if n == 1:
        return [run_config]

    # Create n variations of the run configuration.
    # If the original websocket port is x, the first
    # run will use x, the second one x+1, and so on.
    return [
        run_config.model_copy(
            update={
                "run_id": f"{run_config.run_id}-{i}",
                "websocket_config": run_config.websocket_config.model_copy(
                    update={
                        "port": run_config.websocket_config.port + i,
                    }
                ),
                "transcript_file": f"{run_config.transcript_file.removesuffix('.json')}_{i}.json"
                if run_config.transcript_file
                else None,
            }
        )
        for i in range(n)
    ]


def run_containerized(
    run_config: EvaluationRunConfig,
    runtime: Runtime,
    dev: bool,
    log_file: Path | None = None,
    keep_container: bool = False,
    build_context: str = ".",
    mounts: list[str] | None = None,
    proxy_url: str | None = None,
    prepare_only: bool = False,
) -> None:
    """Run a single containerized evaluation.

    Args:
        run_config: The evaluation run configuration.
        runtime: Container runtime to use (podman or docker).
        dev: Whether to mount source code for development.
        log_file: If provided, redirect output to this file. Otherwise output to stdout.
        keep_container: If True, keep the container after run completes.
        build_context: Path to the build context directory.
        mounts: List of bind mount specs in "source:target[:ro]" format.
        proxy_url: If set, route API calls through this proxy URL.
        prepare_only: If True, the container prepares the env and holds it open
            instead of running the task.
    """
    run_command, _ = get_container_run_command(
        run_config,
        runtime,
        dev,
        keep_container,
        build_context,
        mounts,
        proxy_url,
        prepare_only,
    )

    # Only attach TTY when outputting to stdout (not when redirecting to log file)
    # This avoids TTY issues when running in the TUI with multiprocessing
    # Insert TTY flags right after "run" subcommand
    if log_file is None and sys.stdin.isatty():
        run_index = run_command.index("run") + 1
        run_command.insert(run_index, "--interactive")
        run_command.insert(run_index, "--tty")

    # A held env is ended by an external stop (podman stop / pod delete), which
    # makes the container — and thus `podman run` — exit non-zero; that is the
    # expected way to end a hold, not a failure. For a real run a non-zero exit
    # is a genuine failure and must propagate.
    check = not prepare_only

    if log_file:
        with open(log_file, "w") as f:
            subprocess.run(run_command, check=check, stdout=f, stderr=subprocess.STDOUT)
    else:
        subprocess.run(run_command, check=check)


def get_container_run_command(
    run_config: EvaluationRunConfig,
    runtime: Runtime,
    dev: bool,
    keep_container: bool,
    build_context: str = ".",
    mounts: list[str] | None = None,
    proxy_url: str | None = None,
    prepare_only: bool = False,
) -> tuple[list[str], EvaluationRunConfig]:
    """Build the command to run a container.

    Args:
        run_config: The evaluation run configuration.
        runtime: Container runtime to use (podman or docker).
        dev: Whether to mount environment source code from the build context for development.
        keep_container: If True, keep the container after run completes.
        build_context: Path to the build context directory.
        mounts: List of bind mount specs in "source:target[:ro]" format.
        proxy_url: If set, route API calls through this proxy URL.
        prepare_only: If True, the in-container invocation prepares the env and
            holds it open instead of running the task.

    Returns:
        A tuple of (command, updated_config). The config may be modified
        if transcript_file path needs to be transformed for container mount.
    """
    task = load_task(run_config)
    engine = get_engine(runtime)

    command: list[str] = []

    # CI runners need sudo for the container runtime.
    if os.environ.get("CI"):
        command.append("sudo")

    command.extend(
        [
            engine,
            "run",
        ]
    )

    if runtime == "docker:gvisor":
        command.extend(["--runtime=runsc", "--env", "KAROTTE_GVISOR=1"])
        command.extend(["--env", "KAROTTE_SANDBOX=gvisor"])
        # SYS_ADMIN lets student sessions be put in a PID namespace, which is
        # how they get reaped race-free before grading. Safe to add under
        # gVisor, which services syscalls in userspace rather than passing them
        # to the host kernel; eval pods add it for the same reason.
        command.append("--cap-add=SYS_ADMIN")
        # Without it, root can't read the student's /proc/<pid>/smaps under
        # gVisor, so the memory watchdog weighs nothing and never reaps.
        command.append("--cap-add=SYS_PTRACE")
    else:
        command.extend(["--env", "KAROTTE_SANDBOX=runc"])

    command.extend(["--env", f"{EXIT_ON_RUN_ERROR_ENV_VAR}=1"])
    command.extend(["--security-opt", "seccomp=unconfined"])

    command.extend(
        [
            "--cap-add=NET_ADMIN",
            "--name",
            f"karotte_run_{run_config.run_id}",
        ]
    )

    # A held env streams no transcript, so don't pin the host websocket port
    # for the (indefinite) lifetime of the hold.
    if not prepare_only:
        command.extend(
            [
                "--publish",
                f"{run_config.websocket_config.port}:{run_config.websocket_config.port}",
            ]
        )

    if not keep_container:
        command.append("--rm")

    if dev:
        src_path = Path(build_context).absolute() / "src" / "environment"
        command.extend(
            [
                "--mount",
                f"type=bind,source={src_path},target=/root/.venv/lib/python3.12/site-packages/environment/",
            ]
        )

    for mount_spec in mounts or []:
        parts = mount_spec.split(":")
        source = str(Path(parts[0]).absolute())
        mount_arg = f"type=bind,source={source},target={parts[1]}"
        if len(parts) == 3 and parts[2] == "ro":
            mount_arg += ",readonly"
        command.extend(["--mount", mount_arg])

    command.extend(container_run_args(task, runtime))

    if run_config.transcript_file:
        file_path = Path(run_config.transcript_file).absolute()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        _give_to_sudo_user(file_path.parent)
        command.extend(
            [
                "--mount",
                f"type=bind,source={file_path.parent},target=/out",
            ]
        )
        run_config = run_config.model_copy(
            update={"transcript_file": f"/out/{file_path.name}"}
        )

    if proxy_url:
        command.extend(["--env", f"ANTHROPIC_BASE_URL={proxy_url}"])
        # Generic proxy endpoint for CLI agents, which route their provider
        # through the same proxy (see MistralVibeAgent) rather than the
        # Anthropic-specific base URL the builtin loop uses.
        command.extend(["--env", f"KAROTTE_PROXY_URL={proxy_url}"])

    for var in ("LOGURU_LEVEL", SERVICE_TIER_ENV):
        if value := os.environ.get(var):
            command.extend(["--env", f"{var}={value}"])

    command.append("localhost/karotte" if engine == "podman" else "karotte")

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


def _give_to_sudo_user(path: Path) -> None:
    """Under sudo, hand the host output dir to the invoking user, since the container chowns outputs to its owner."""
    if os.geteuid() != 0 or "SUDO_UID" not in os.environ:
        return
    uid = int(os.environ["SUDO_UID"])
    gid = int(os.environ.get("SUDO_GID", uid))
    os.chown(path, uid, gid, follow_symlinks=False)


def chown_outputs(run_config: EvaluationRunConfig) -> None:
    """Give the transcript and local artifacts to the owner of the directory they were written to.

    A rootful container would otherwise leave them owned by root on the host.
    """
    if not is_containerized():
        return
    outputs = [Path(run_config.transcript_file)] if run_config.transcript_file else []
    if (artifact_dir := local_artifact_dir()) is not None:
        outputs.append(artifact_dir)
    for output in outputs:
        try:
            _chown_tree(output)
        except OSError as e:
            logger.warning("Could not hand {} to the host user: {}", output, e)


def _chown_tree(path: Path) -> None:
    if not os.path.lexists(path):
        return
    owner = path.parent.stat()
    uid, gid = owner.st_uid, owner.st_gid
    os.chown(path, uid, gid, follow_symlinks=False)
    # fwalk holds directory fds, so swapping a directory for a symlink mid-walk can't redirect it.
    for _, dirs, files, dir_fd in os.fwalk(path, follow_symlinks=False):
        for name in dirs + files:
            os.chown(name, uid, gid, dir_fd=dir_fd, follow_symlinks=False)


def _set_up_runner(run_config: EvaluationRunConfig, task: Task) -> "EvaluationRunner":
    """Construct the evaluation runner and apply the in-container firewall.

    The websocket port is always blocked for the student user (ScoringEvents
    leak scoring/hints there). The MCP port is blocked too unless the selected
    agent runs in-container as the student and makes native MCP calls, in which
    case student access is required.
    """
    from karotte.evaluation_runner import EvaluationRunner

    runner = EvaluationRunner(run_config, task)

    blocked_ports = [run_config.websocket_config.port]
    allowed_hosts: list[str] = []
    if not runner.allows_student_mcp_access:
        blocked_ports.append(run_config.mcp_server_config.port)
    else:
        # A CLI agent runs as the student and reaches its model provider only
        # through the proxy. Allow the student to reach exactly that
        # endpoint.
        if proxy := os.environ.get("KAROTTE_PROXY_URL"):
            allowed_hosts = _resolve_host_ips(proxy)

    runner.network_firewall = _maybe_block_internet(
        blocked_ports=blocked_ports, allowed_ips=allowed_hosts
    )

    return runner


def _resolve_host_ips(url: str) -> list[str]:
    """IPv4 addresses the URL's host resolves to, for firewall allow rules."""
    from urllib.parse import urlparse

    host = urlparse(url).hostname
    if not host:
        return []
    try:
        infos = socket.getaddrinfo(host, None, family=socket.AF_INET)
    except OSError:
        logger.warning("Could not resolve proxy host {} for firewall allow", host)
        return []
    return sorted({str(info[4][0]) for info in infos})


async def run_non_containerized(
    run_config: EvaluationRunConfig, task: Task
) -> ErrorEvent | None:
    """Run the task in this process and return the error the run ended in, if any."""
    from karotte.fake_model import setup_fake_model
    from karotte.schemas.transcript import Event as TaskEvent
    from karotte.transcript_streaming.stream_transcript_to_backend import (
        stream_transcript_to_backend,
    )

    runner = _set_up_runner(run_config, task)

    if run_config.use_fake_model:
        setup_fake_model(runner, run_config)

    last_error: ErrorEvent | None = None
    final_status: str | None = None

    async with anyio.create_task_group() as tg:
        stdout_send, stdout_recv = anyio.create_memory_object_stream[TaskEvent]()
        websocket_send, websocket_recv = anyio.create_memory_object_stream[TaskEvent]()
        backend_send, backend_recv = None, None

        tg.start_soon(
            stream_transcript_to_websocket, run_config.websocket_config, websocket_recv
        )
        tg.start_soon(stream_transcript_to_stdout, stdout_recv)

        if run_config.backend_uri is not None:
            logger.info("Sending transcripts to the backend")
            backend_send, backend_recv = anyio.create_memory_object_stream[TaskEvent]()
            tg.start_soon(stream_transcript_to_backend, backend_recv, run_config)

        # Stream events to all outputs
        async for event in runner.run():
            if isinstance(event, ErrorEvent):
                last_error = event
            elif isinstance(event, TaskCompletedEvent):
                final_status = event.status
            await stdout_send.send(event)
            await websocket_send.send(event)
            if backend_send is not None:
                await backend_send.send(event)

        await stdout_send.aclose()
        await websocket_send.aclose()
        if backend_send is not None:
            await backend_send.aclose()

    return last_error if final_status == "error" else None


def prepared_sentinel_path(run_id: str) -> Path:
    """Path of the readiness marker written once an env is prepared and held.

    The marker lives in the env's own filesystem: for a containerized run it
    exists only inside the container (check it with e.g.
    ``podman exec karotte_run_<run_id> test -f /tmp/karotte_prepared_<run_id>``),
    not on the host.
    """
    return Path(f"/tmp/karotte_prepared_{run_id}")


async def prepare_non_containerized(
    run_config: EvaluationRunConfig, task: Task
) -> None:
    """Set up an env up to the step loop, then hold it open.

    Skips the agentic loop and all scoring, and produces no transcript /
    backend output. Once prepared, writes a readiness sentinel and
    idles until cancelled or killed; the sentinel is removed on the way out.
    """
    sentinel = prepared_sentinel_path(run_config.run_id)
    # A marker left behind by an earlier, killed hold must not signal
    # readiness while this run is still setting up.
    sentinel.unlink(missing_ok=True)

    runner = _set_up_runner(run_config, task)

    async for event in runner.prepare():
        logger.info("prepare: {}", type(event).__name__)

    sentinel.write_text(run_config.run_id)
    logger.info("Env prepared and held open. Ready marker: {}", sentinel)

    try:
        await anyio.sleep_forever()
    finally:
        sentinel.unlink(missing_ok=True)


def hold_prepared_env(run_config: EvaluationRunConfig, task: Task) -> None:
    """Prepare an env and hold it open until stopped.

    A ``KeyboardInterrupt`` (raised on Ctrl-C or SIGTERM) is the intended way to
    end a held session, so it exits cleanly (0) instead of propagating. Anything
    else, e.g. a setup failure, still exits non-zero.
    """
    try:
        anyio.run(prepare_non_containerized, run_config, task)
    except KeyboardInterrupt:
        logger.info("Held env stopped.")


def common_run_id_prefix(run_ids: list[str]) -> str:
    """Longest prefix shared by all run IDs in one invocation."""
    if not run_ids:
        return ""
    prefix = run_ids[0]
    for run_id in run_ids[1:]:
        while prefix and not run_id.startswith(prefix):
            prefix = prefix[:-1]
    return prefix


def stop_containers(runtime: Runtime, run_ids: list[str]) -> None:
    """Stop the containers for these exact run IDs, ignoring any already gone.

    Used to tear down foregrounded runs — including indefinitely-held
    ``--prepare-only`` envs — when the launching process is interrupted, so the
    blocking ``podman run`` returns instead of hanging.
    """
    engine = get_engine(runtime)
    names = [f"karotte_run_{run_id}" for run_id in run_ids]
    result = subprocess.run([engine, "stop", *names], capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning(
            f"Failed to stop containers: {result.stderr.strip() or 'unknown error'}"
        )


def clean_up_old_containers(runtime: Runtime, run_ids: list[str]) -> None:
    """Stop and remove containers from a previous run of this invocation.

    Only containers whose names start with ``karotte_run_<prefix>`` are removed,
    where ``prefix`` is the longest common prefix of ``run_ids``. That lets
    separate invocations (different run-id prefixes) run in parallel without
    killing each other's containers.

    Args:
        runtime: Container runtime to use (podman or docker).
        run_ids: Run IDs for this invocation (e.g. from ``run_configs``).
    """
    from loguru import logger

    engine = get_engine(runtime)
    prefix = common_run_id_prefix(run_ids)
    name_filter = f"karotte_run_{prefix}"

    result = subprocess.run(
        [engine, "ps", "-a", "--filter", f"name={name_filter}", "-q"],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.warning(
            f"Failed to list containers: {result.stderr.strip() or 'unknown error'}"
        )
        return

    container_ids = result.stdout.strip().split()

    if container_ids:
        rm_result = subprocess.run(
            [engine, "rm", "--force", *container_ids], capture_output=True, text=True
        )
        if rm_result.returncode != 0:
            logger.warning(
                f"Failed to remove containers: {rm_result.stderr.strip() or 'unknown error'}"
            )


def _maybe_block_internet(
    blocked_ports: list[int] | None = None,
    allowed_ips: list[str] | None = None,
) -> bool:
    return Confinement(current_sandbox()).restrict_to_internal_network(
        "student", blocked_ports=blocked_ports, allowed_ips=allowed_ips
    )


_DAEMON_JSON_PATH = Path("/etc/docker/daemon.json")

_BASE_RUNSC_ARGS = [
    "-net-raw",
    "--systrap-disable-syscall-patching",
    "-overlay2=none",
    "-file-access=shared",
    "-network=sandbox",
    "--net-disconnect-ok",
]

_GVISOR_SETUP_INSTRUCTIONS = """\
The docker:gvisor runtime requires runsc to be registered with specific \
arguments in the Docker daemon configuration.

Run the following commands to set it up:

```
sudo tee /etc/docker/daemon.json <<'EOF'
{{
    "runtimes": {{
        "runsc": {{
            "path": "{runsc}",
            "runtimeArgs": ["{args}"]
        }}
    }}
}}
EOF
```

Then reload the Docker daemon configuration:
```
sudo systemctl reload docker
```

runsc needs the gvisor-bin/ directory from its release next to it.
"""


def validate_gvisor_runtime(daemon_json_path: Path = _DAEMON_JSON_PATH) -> None:
    """Validate that runsc is installed and the Docker daemon is configured correctly.

    Aborts with setup instructions if the configuration is missing or incomplete.
    """
    import shutil

    required_args = _BASE_RUNSC_ARGS
    runsc = shutil.which("runsc")
    if not runsc:
        _print_and_abort(
            "runsc (gVisor) is not installed. Install gvisor first:\n\n  https://gvisor.dev/docs/user_guide/install/"
        )
    instructions = _GVISOR_SETUP_INSTRUCTIONS.format(
        runsc=runsc, args='", "'.join(required_args)
    )

    if not daemon_json_path.exists():
        _print_and_abort(f"{daemon_json_path} not found.\n\n" + instructions)

    try:
        config = json.loads(daemon_json_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        _print_and_abort(f"Failed to read {daemon_json_path}: {e}\n\n" + instructions)

    runsc_config = config.get("runtimes", {}).get("runsc")
    if runsc_config is None:
        _print_and_abort(
            f"runsc runtime is not registered in {daemon_json_path}.\n\n" + instructions
        )

    runtime_args = runsc_config.get("runtimeArgs", [])
    missing_args = [arg for arg in required_args if arg not in runtime_args]
    if missing_args:
        _print_and_abort(
            f"runsc runtime in {daemon_json_path} is missing required arguments: {', '.join(missing_args)}\n\n"
            + instructions
        )


def _print_and_abort(message: str) -> Never:
    typer.secho(
        message,
        fg=typer.colors.RED,
        err=True,
    )
    raise typer.Abort()


def harden_filesystem() -> list[Path]:
    """Take group and world write off the mount points this runtime left open.
    Returns the ones that changed.
    """
    return Confinement(current_sandbox()).harden_filesystem()


_SANITIZED_PATH_VARS = ("PATH", "LD_LIBRARY_PATH", "LD_PRELOAD", "LD_AUDIT")

_REEXEC_GUARD = "KAROTTE_LOADER_PATHS_SANITIZED"


def sanitize_paths_and_reexec() -> None:
    """Sanitize loader paths, then re-exec so glibc re-reads the cleaned values (it reads LD_LIBRARY_PATH only at exec).

    The guard, inherited by children, skips the redundant re-exec once an ancestor already cleaned the env.
    """
    if os.environ.get(_REEXEC_GUARD):
        return
    sanitize_paths()
    os.environ[_REEXEC_GUARD] = "1"
    os.execv(sys.executable, [sys.executable, *sys.orig_argv[1:]])


def sanitize_paths() -> None:
    """
    Make sure there are no student-writable paths in PATH/LD_LIBRARY_PATH/LD_PRELOAD/LD_AUDIT env vars.

    Procedure:
    1) Split each env var into its constituent paths
    2) For each path, if it's NOT owned by root:root, or if it is world-writable, reject it. If it's a symlink, follow
       the link and check the target with the same rules. Then, check all parents of the path too with the same rules.
    3) Set the env var to the remaining paths.
    4) If the procedure actaully resulted in a different value, print a info log that shows what it was before, what
       it is after, and what was removed.
    """
    # Skip only for a local test runner, whose PATH some tests depend on. Never
    # skip inside a container: an eval run that happens to set CI must still have
    # student-writable entries stripped from the (root) server's PATH.
    if "CI" in os.environ and not is_containerized():
        logger.warning("Not running sanitize_paths() in CI (outside a container)")
        return

    for var in _SANITIZED_PATH_VARS:
        _sanitize_env_var(var)


def _sanitize_env_var(var: str) -> None:
    original_value = os.environ.get(var, "")
    if not original_value:
        return

    entries = split_loader_path(var, original_value)
    accepted: list[str] = []
    rejected: list[str] = []
    for entry in entries:
        if _is_safe_path_entry(entry):
            accepted.append(entry)
        else:
            rejected.append(entry)

    new_value = os.pathsep.join(accepted)
    if new_value == original_value:
        return

    if var == "PATH" and not new_value:
        # An empty PATH has implementation-defined behavior: execvp falls back
        # to a built-in default search path, and some shells/libcs treat the
        # empty string as a single empty component (i.e. the cwd) — the exact
        # thing this function exists to prevent. If every entry was rejected,
        # something is badly wrong with the container; refuse to continue.
        _print_and_abort(
            f"PATH sanitization rejected every entry. Original: {original_value!r}; rejected: {os.pathsep.join(rejected)!r}"
        )

    os.environ[var] = new_value
    logger.debug(
        "Sanitized {}: removed {} student-writable entr{}.\n"
        + "  before:  {}\n"
        + "  after:   {}\n"
        + "  removed: {}",
        var,
        len(rejected),
        "y" if len(rejected) == 1 else "ies",
        original_value,
        new_value,
        os.pathsep.join(rejected),
    )


_MAX_SYMLINKS = 40  # Linux ELOOP threshold.


def _is_safe_path_entry(entry: str) -> bool:
    """Verify every directory traversed while resolving ``entry`` is root-owned and not world-writable.

    Walks the unresolved path component-by-component with ``lstat`` rather than calling
    ``Path.resolve()`` up front. The resolve-first approach misses cases like
    ``/student/bin -> /usr/local/bin``: the resolved chain (``/usr/local/bin`` and
    parents) looks safe, but ``/student`` is student-writable so they can repoint the
    symlink between our check and any subsequent ``exec``. Walking unresolved catches it.

    Symlinks are followed without checking the link's own metadata: on Linux the mode
    bits of a symlink are ignored and ``lchown``'d uid/gid don't gate access. What
    governs whether a symlink can be swapped is the *containing* directory, which we've
    already verified by the time we reach the link.
    """
    if not entry or not entry.startswith("/"):
        # Empty or relative entries (like "" or ".") let students inject binaries via cwd.
        return False

    pending: list[str] = [c for c in entry.split("/") if c and c != "."]
    current = "/"
    symlinks_followed = 0

    try:
        if not _is_safe_dir_stat(os.lstat(current)):
            return False
    except OSError:
        return False

    while pending:
        part = pending.pop(0)
        if part == "..":
            current = os.path.dirname(current) or "/"
            continue
        candidate = "/" + part if current == "/" else current + "/" + part
        try:
            st = os.lstat(candidate)
        except OSError:
            return False
        if stat.S_ISLNK(st.st_mode):
            symlinks_followed += 1
            if symlinks_followed > _MAX_SYMLINKS:
                return False
            try:
                target = os.readlink(candidate)
            except OSError:
                return False
            target_parts = [c for c in target.split("/") if c and c != "."]
            if target.startswith("/"):
                current = "/"
            pending = target_parts + pending
        else:
            if not _is_safe_dir_stat(st):
                return False
            current = candidate

    return True


def _is_safe_dir_stat(st: os.stat_result) -> bool:
    return st.st_uid == 0 and st.st_gid == 0 and not (st.st_mode & stat.S_IWOTH)
