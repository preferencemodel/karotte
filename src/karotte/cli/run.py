import glob
import json
import os
import subprocess
from collections.abc import Sequence
from importlib.metadata import entry_points
from pathlib import Path
from typing import Annotated, Never, get_args

import anyio
import typer
from loguru import logger

from karotte import Runtime, staged_mounts
from karotte.build import build_container, require_buildx, require_runtime
from karotte.container import is_containerized
from karotte.forwarded_env import EXIT_ON_RUN_ERROR_ENV_VAR
from karotte.hardware import container_run_args, default_runtime
from karotte.judges import RubricJudge
from karotte.load_tasks import load_task, require_environment
from karotte.run_config_preprocessors import apply_run_config_preprocessors
from karotte.run_helpers import (
    build_configs,
    chown_outputs,
    clean_up_old_containers,
    copy_hint,
    harden_filesystem,
    parse_config,
    run_containerized,
    sanitize_paths_and_reexec,
    stop_containers,
    validate_gvisor_runtime,
)
from karotte.schemas.evaluation_run_config import EvaluationRunConfig

PROXY_ENTRY_POINT_GROUP = "karotte.default_proxy_url"


def default_proxy_url() -> str | None:
    """The proxy an installed package registers, first by entry point name."""
    for ep in sorted(
        entry_points(group=PROXY_ENTRY_POINT_GROUP), key=lambda ep: ep.name
    ):
        try:
            return ep.load()
        except Exception as e:  # noqa: BLE001 - a broken plugin must not break karotte
            logger.warning("Ignoring default proxy from {!r}: {}", ep.name, e)
    return None


def _referenced_key_env(config: str) -> str | None:
    """The env var a JSON config or config file names as its `$VAR` model API key."""
    try:
        data = json.loads(config)
    except ValueError:
        try:
            data = json.loads(Path(config).read_text())
        except (OSError, ValueError):
            return None
    key = data.get("model_api_key") if isinstance(data, dict) else None
    return key[1:] if isinstance(key, str) and key.startswith("$") else None


def _export_proxy(proxy_url: str) -> None:
    """Route this process's model calls through ``proxy_url``, unless the environment already routes them."""
    if "KAROTTE_PROXY_URL" in os.environ or "ANTHROPIC_BASE_URL" in os.environ:
        return
    os.environ["ANTHROPIC_BASE_URL"] = proxy_url
    os.environ["KAROTTE_PROXY_URL"] = proxy_url


def run(
    config: Annotated[
        str,
        typer.Option(
            "--config",
            "-c",
            help="JSON-serialized EvaluationRunConfig, or a path to a file with one.",
        ),
    ],
    containerized: Annotated[
        bool,
        typer.Option(
            help="Whether to execute the run inside a docker/podman container. "
            + "--no-containerized only works inside a karotte container image."
        ),
    ] = True,
    runtime: Annotated[
        Runtime | None,
        typer.Option(
            metavar="NAME",
            help=f"Container runtime: {', '.join(get_args(Runtime))}. Default: the"
            + " OS's VM (`apple-container` on macOS, `firecracker` on Linux); docker for"
            + " accelerator hardware and other platforms.",
        ),
    ] = None,
    build_context: Annotated[
        str, typer.Option(help="Path to the build context for the container.")
    ] = ".",
    n_parallel: Annotated[
        int,
        typer.Option(
            "--n-parallel",
            "-n",
            metavar="N",
            help="Number of parallel runs. Can only be >1 when running containerized.",
        ),
    ] = 1,
    dev: Annotated[
        bool,
        typer.Option(
            help="Skip the image build and bind the environment's src/environment "
            + "folder into the container, to iterate quickly on environment code. "
            + "Can only be used when running containerized. "
            + "Changed dependencies or env/scoring data still need a rebuild."
        ),
    ] = False,
    mount: Annotated[
        list[str] | None,
        typer.Option(
            "--mount",
            help="Bind-mount a host path into the container. "
            + "Format: source:target[:ro]. "
            + "Bind mounts are read-write by default; append :ro for read-only. "
            + "Can be specified multiple times. "
            + "Use @file.txt to read mount specs from a file (one per line; "
            + "blank lines and lines starting with # are ignored). "
            + "When running parallel containers, all containers share the same bind mounts. "
            + "Can only be used when running containerized.",
        ),
    ] = None,
    no_ui: Annotated[
        bool,
        typer.Option(
            "--no-ui",
            help="Disable the TUI and print output from all runs directly to the terminal.",
        ),
    ] = False,
    keep_containers: Annotated[
        bool,
        typer.Option(
            help="Keep containers after runs instead of removing them. "
            + "They are named karotte_run_<id>, and existing containers with the "
            + "same names get removed before the runs start. "
            + "Copy data out of them with `<runtime> cp`.",
        ),
    ] = False,
    cache_from: Annotated[
        list[str] | None,
        typer.Option(help="Cache source passed to the container runtime."),
    ] = None,
    cache_to: Annotated[
        list[str] | None,
        typer.Option(help="Cache destination passed to the container runtime."),
    ] = None,
    build_secret: Annotated[
        list[str] | None,
        typer.Option(
            help="name=path of a file the Containerfile can mount as a build secret "
            + "with type=secret and id=<name>. Repeatable."
        ),
    ] = None,
    proxy: Annotated[
        str | None,
        typer.Option(
            "--proxy",
            help="Proxy URL for API calls. Sets ANTHROPIC_BASE_URL and KAROTTE_PROXY_URL for the run; with --no-containerized only if neither is set yet. Defaults to the one an installed plugin registers.",
        ),
    ] = None,
    no_proxy: Annotated[
        bool,
        typer.Option(
            "--no-proxy",
            help="Disable the proxy.",
        ),
    ] = False,
    prepare_only: Annotated[
        bool,
        typer.Option(
            "--prepare-only",
            help="Set up the env like a real run, including tools and the task "
            + "pre-hook, then hold it open without running any steps, so you can "
            + "exec in and inspect it. Readiness is signaled by a marker file in "
            + "/tmp named karotte_prepared_ plus the run ID. Containerized, the env "
            + "is held in its container (karotte_run_<id>); with --no-containerized "
            + "the current process holds it. Cannot be combined with --n-parallel > 1.",
        ),
    ] = False,
) -> None:
    """Execute an evaluation run."""

    from karotte.mcp_servers.http_mcp_server import run_server as run_mcp_server

    if n_parallel > 1 and not containerized:
        _print_and_abort("Cannot run multiple runs without containerization.")

    if prepare_only and n_parallel > 1:
        _print_and_abort("Cannot use --prepare-only with --n-parallel > 1.")

    if dev and not containerized:
        _print_and_abort("Cannot use the `--dev` option without containerization.")

    proxy_url = None if no_proxy else proxy or default_proxy_url()

    if mount and not containerized:
        _print_and_abort("Cannot use the `--mount` option without containerization.")

    if mount:
        mount = _expand_mount_file_references(mount)
        _validate_mount_specs(mount)

    if keep_containers and not containerized:
        _print_and_abort(
            "Cannot use the `--keep-containers` option without containerization."
        )

    if not containerized and not is_containerized():
        _print_and_abort(
            "--no-containerized only runs inside a karotte container image; "
            + "drop the flag to run in a container."
        )

    if proxy_url:
        # The rubric judge reads ANTHROPIC_API_KEY itself, so it gets one too.
        for env_var in ("ANTHROPIC_API_KEY", _referenced_key_env(config)):
            if env_var and not os.environ.get(env_var):
                os.environ[env_var] = "model_api_key"

    require_environment()
    # An explicit runtime is checked before anything else; the default needs
    # the task's hardware, so it is checked once that is known.
    on_host = containerized and not is_containerized()
    if on_host and runtime is not None:
        require_runtime(runtime)
        if not dev:
            require_buildx(runtime)

    run_config = parse_config(config, prepare_only=prepare_only)
    run_config = apply_run_config_preprocessors(run_config)

    if runtime is None:
        runtime = default_runtime(
            load_task(run_config).required_hardware if on_host else None
        )
        if on_host:
            logger.info(f"Runtime: {runtime} (default; pass --runtime to change)")
            require_runtime(runtime)
            if not dev:
                require_buildx(runtime)

    if runtime == "docker:gvisor":
        validate_gvisor_runtime()

    if containerized and not is_containerized():
        try:
            _ = container_run_args(load_task(run_config), runtime)
        except Exception as e:  # noqa: BLE001 - a plugin refuses the launch by raising
            _print_and_abort(str(e))

    if runtime == "apple-container" and containerized and not is_containerized():
        from karotte.apple_container import validate_container_runtime

        validate_container_runtime(load_task(run_config).required_hardware)
    if runtime == "firecracker" and containerized and not is_containerized():
        from karotte.firecracker.preflight import validate_firecracker_runtime

        validate_firecracker_runtime(load_task(run_config).required_hardware, mount)

    if run_config.rubric_judge_api_key is not None:
        RubricJudge.default_api_key = run_config.rubric_judge_api_key

    # If we are already inside a container or should run uncontainerized, don't launch the TUI.
    # Just execute the run and stream to websocket
    if is_containerized() or containerized is False:
        from karotte.subprocess import chdir_to_workdir

        sanitize_paths_and_reexec()
        harden_filesystem()
        staged_mounts.copy_in()

        chdir_to_workdir()

        # The host already chose the proxy, so a plugin default must not undo --no-proxy.
        if proxy_url and proxy:
            _export_proxy(proxy_url)

        task = load_task(run_config)

        error = None
        with run_mcp_server(run_config.mcp_server_config), staged_mounts.copied_back():
            from karotte.run_helpers import hold_prepared_env, run_non_containerized

            if prepare_only:
                hold_prepared_env(run_config, task)
            else:
                try:
                    error = anyio.run(run_non_containerized, run_config, task)
                finally:
                    chown_outputs(run_config)
        # The backend's pods rely on exit 0; only the host's containers ask for a non-zero exit.
        if error is not None and os.environ.get(EXIT_ON_RUN_ERROR_ENV_VAR):
            raise typer.Exit(130 if error.exception_type == "KeyboardInterrupt" else 1)
        return

    run_configs = build_configs(run_config, n_parallel)
    if runtime == "apple-container":
        from karotte.apple_container import assign_host_ports

        run_configs = assign_host_ports(run_configs)

    if n_parallel > 1 and mount:
        writable_mounts = [m for m in mount if not m.endswith(":ro")]
        if writable_mounts:
            typer.secho(
                f"Warning: {len(writable_mounts)} writable bind mount(s) will be shared across "
                + f"{n_parallel} parallel containers. Consider using :ro suffix for read-only access.",
                fg=typer.colors.YELLOW,
                err=True,
            )

    # Run without UI - build container and run with stdout output.
    # Prepare-only runs produce no event stream, so the TUI adds nothing.
    if no_ui or prepare_only:
        _run_without_ui(
            run_configs,
            runtime,
            dev,
            build_context,
            keep_containers,
            mounts=mount,
            cache_from=cache_from,
            cache_to=cache_to,
            build_secrets=build_secret or (),
            proxy_url=proxy_url,
            prepare_only=prepare_only,
        )
        return

    # Launch TUI
    from karotte.terminal.app import KarotteApp

    tui_app = KarotteApp(configs=run_configs)

    tui_app.runtime = runtime
    tui_app.dev = dev
    tui_app.build_context = build_context
    tui_app.build_secrets = build_secret or ()
    tui_app.cache_from = cache_from
    tui_app.cache_to = cache_to
    tui_app.containerized = containerized
    tui_app.keep_containers = keep_containers
    tui_app.mounts = mount
    tui_app.proxy_url = proxy_url

    # Launch TUI - this blocks until user quits
    tui_app.run()
    _print_output_paths(run_configs)
    if tui_app.run_failed:
        raise typer.Exit(1)


def _run_without_ui(
    run_configs: list[EvaluationRunConfig],
    runtime: Runtime,
    dev: bool,
    build_context: str,
    keep_containers: bool,
    mounts: list[str] | None = None,
    cache_from: list[str] | None = None,
    cache_to: list[str] | None = None,
    build_secrets: Sequence[str] = (),
    proxy_url: str | None = None,
    prepare_only: bool = False,
) -> None:
    """Run containerized evaluations without the TUI, outputting directly to stdout."""
    from concurrent.futures import ThreadPoolExecutor
    from functools import partial

    # Build container if needed
    if not dev:
        build_container(
            runtime,
            build_context,
            cache_from=cache_from,
            cache_to=cache_to,
            build_secrets=build_secrets,
        )

    clean_up_old_containers(runtime, [c.run_id for c in run_configs])

    # Log container names for easy reference
    typer.secho("\nStarting containers:", fg=typer.colors.BLUE)
    for config in run_configs:
        typer.secho(f"  karotte_run_{config.run_id}", fg=typer.colors.BLUE)
    typer.echo()

    # Run containers in parallel using threads
    run_fn = partial(
        run_containerized,
        runtime=runtime,
        dev=dev,
        log_file=None,
        keep_container=keep_containers,
        build_context=build_context,
        mounts=mounts,
        proxy_url=proxy_url,
        prepare_only=prepare_only,
        parallel_runs=len(run_configs),
    )

    # A held --prepare-only container blocks its `podman run` indefinitely, so a
    # stop signal must tear the containers down before the executor's exit waits
    # on the worker threads — otherwise the process hangs and orphans the
    # containers. Stopping is also the right response to interrupting a real run.
    def exit_code(config: EvaluationRunConfig) -> int:
        from karotte.firecracker import FirecrackerError

        try:
            run_fn(config)
        except subprocess.CalledProcessError as e:
            return e.returncode
        except FirecrackerError as e:
            # One VM that can't start fails its run, not every run in the
            # invocation.
            logger.error(f"Run {config.run_id}: {e}")
            return 1
        return 0

    executor = ThreadPoolExecutor(max_workers=len(run_configs))
    exit_codes = [0] * len(run_configs)
    try:
        exit_codes = list(executor.map(exit_code, run_configs))
    except KeyboardInterrupt:
        stop_containers(runtime, [c.run_id for c in run_configs])
        if not prepare_only:
            raise
    finally:
        executor.shutdown(wait=True)

    if not prepare_only:
        _print_output_paths(run_configs)

    # Log completion message with copy example if containers are preserved
    if keep_containers and runtime != "firecracker":
        typer.secho(
            f"\nContainers preserved. To copy data:\n  {copy_hint(runtime, run_configs[0].run_id)}",
            fg=typer.colors.BLUE,
        )

    failed = [
        (config.run_id, code)
        for config, code in zip(run_configs, exit_codes, strict=True)
        if code != 0
    ]
    if failed:
        for run_id, code in failed:
            typer.secho(
                f"karotte_run_{run_id} failed (exit code {code}).",
                fg=typer.colors.RED,
                err=True,
            )
        raise typer.Exit(failed[0][1] if len(run_configs) == 1 else 1)


def _print_output_paths(run_configs: list[EvaluationRunConfig]) -> None:
    """Print where each containerized run's transcript and artifacts are on the host."""
    for config in run_configs:
        if not config.transcript_file:
            continue
        transcript = Path(config.transcript_file).absolute()
        if transcript.is_file():
            typer.secho(f"\nTranscript: {transcript}", fg=typer.colors.BLUE)
        artifact_dirs = sorted(
            transcript.parent.glob(f"{glob.escape(config.run_id)}_artifacts*"),
            key=lambda p: p.stat().st_mtime,
        )
        if artifact_dirs:
            typer.secho(f"Artifacts:  {artifact_dirs[-1]}", fg=typer.colors.BLUE)


def _expand_mount_file_references(mounts: list[str]) -> list[str]:
    """Expand @file references in mount specs.

    If a mount spec starts with ``@``, the remainder is treated as a path to a
    text file containing one mount spec per line.  Blank lines and lines whose
    first non-whitespace character is ``#`` are skipped.
    """
    expanded: list[str] = []
    for spec in mounts:
        if spec.startswith("@"):
            file_path = Path(spec[1:])
            if not file_path.is_file():
                _print_and_abort(f"Mount file not found: {file_path}")
            for line in file_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                expanded.append(line)
        else:
            expanded.append(spec)
    return expanded


def _validate_mount_specs(mounts: list[str]) -> None:
    for spec in mounts:
        parts = spec.split(":")
        if len(parts) < 2 or len(parts) > 3:
            _print_and_abort(
                f"Invalid mount spec: {spec!r}. Expected format: source:target[:ro]"
            )
        if len(parts) == 3 and parts[2] != "ro":
            _print_and_abort(
                f"Unsupported mount option {parts[2]!r} in {spec!r}. "
                + "Only ':ro' (read-only) is supported."
            )
        target = parts[1]
        if not target or not target.startswith("/"):
            _print_and_abort(
                f"Mount target must be a non-empty absolute path, got {target!r} in {spec!r}"
            )
        source = Path(parts[0])
        if not source.exists():
            _print_and_abort(
                f"Mount source path does not exist: {parts[0]!r} in {spec!r}"
            )


def _print_and_abort(message: str) -> Never:
    typer.secho(
        message,
        fg=typer.colors.RED,
        err=True,
    )
    raise typer.Abort()
