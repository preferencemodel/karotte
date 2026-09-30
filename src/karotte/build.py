"""Helpers for building container images."""

import os
import shlex
import subprocess
from collections.abc import Sequence
from shutil import which
from typing import Never

import typer

from karotte.runtime import Runtime, get_engine


def require_runtime(runtime: Runtime) -> None:
    """Exit with a one-line error if the runtime's binary isn't on PATH."""
    engine = get_engine(runtime)
    if which(engine) is None:
        alternative = "docker" if engine != "docker" else "podman"
        _exit_with_error(
            f"{engine} not found. Install it or pass --runtime {alternative}."
        )


def require_buildx(runtime: Runtime) -> None:
    """Exit with a one-line error if docker lacks buildx, which `RUN --mount` needs."""
    if get_engine(runtime) == "docker" and not _buildx_available():
        _exit_with_error(
            "docker buildx not found, and building the image needs it. "
            + "Install the docker-buildx package or Docker's docker-buildx-plugin."
        )


def build_container(
    runtime: Runtime,
    build_context: str,
    tag: str = "karotte",
    cache_from: list[str] | None = None,
    cache_to: list[str] | None = None,
    build_secrets: Sequence[str] = (),
) -> None:
    """Build the container image."""
    build_command = get_container_build_command(
        runtime,
        build_context,
        tag,
        cache_from=cache_from,
        cache_to=cache_to,
        build_secrets=build_secrets,
    )

    typer.secho(
        "Building container image with command: " + repr(shlex.join(build_command)),
        fg=typer.colors.BLUE,
    )

    build_result = subprocess.run(build_command, check=False)

    if build_result.returncode != 0:
        error_msg = "Failed to build environment container image.\n"
        if build_result:
            if build_result.stdout:
                error_msg += f"Stdout: {build_result.stdout}\n"
            if build_result.stderr:
                error_msg += f"Stderr: {build_result.stderr}\n"
        _print_and_abort(error_msg)

    typer.secho("Environment container image built successfully.", fg=typer.colors.BLUE)


def get_container_build_command(
    runtime: Runtime,
    build_context: str,
    tag: str = "karotte",
    cache_from: list[str] | None = None,
    cache_to: list[str] | None = None,
    build_secrets: Sequence[str] = (),
) -> list[str]:
    """Build the command to construct a container image.

    Each ``build_secrets`` entry is ``name=path``, mounted in the build as
    ``/run/secrets/<name>``.
    """
    secret_flags = [_secret_flag(s) for s in build_secrets]

    engine = get_engine(runtime)

    if engine == "podman" and _podman_is_using_vm():
        _print_and_abort(
            "Building with podman is currently not supported when using the podman"
            + " VM, e.g., on macOS. Please use `--runtime docker` instead."
        )

    command: list[str] = []

    # CI runners need sudo for the container runtime.
    if os.environ.get("CI"):
        command.append("sudo")

    command.extend([engine, "build", *secret_flags])

    if engine in ("docker", "nerdctl"):
        # docker and nerdctl default to `Dockerfile`
        command.extend(["--file", "Containerfile"])

    for source in cache_from or []:
        command.extend(["--cache-from", source])
    for dest in cache_to or []:
        command.extend(["--cache-to", dest])

    command.extend(["--tag", tag, build_context])
    return command


def _secret_flag(spec: str) -> str:
    name, sep, path = spec.partition("=")
    if not sep or not name or not path:
        raise typer.BadParameter(
            f"Expected name=path, got {spec!r}.", param_hint="--build-secret"
        )
    return f"--secret=id={name},src={path}"


def _buildx_available() -> bool:
    # Without the plugin, `docker build` falls back to the legacy builder.
    command = ["docker", "buildx", "version"]
    if os.environ.get("CI"):
        command.insert(0, "sudo")
    return subprocess.run(command, capture_output=True, check=False).returncode == 0


def _podman_is_using_vm() -> bool:
    """Check if podman is routing through a VM (e.g., podman machine on macOS)."""
    try:
        result = subprocess.run(
            ["podman", "info", "--format", "{{.Host.ServiceIsRemote}}"],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode == 0 and result.stdout.strip().lower() == "true"
    except FileNotFoundError:
        return False


def _exit_with_error(message: str) -> Never:
    typer.secho(message, fg=typer.colors.RED, err=True)
    raise typer.Exit(1)


def _print_and_abort(message: str) -> None:
    typer.secho(
        message,
        fg=typer.colors.RED,
        err=True,
    )
    e = typer.Abort()
    e.add_note(message)
    raise e
