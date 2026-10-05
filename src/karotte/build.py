"""Helpers for building container images."""

import os
import shlex
import subprocess
from collections.abc import Sequence
from shutil import which
from typing import Never

import typer
from loguru import logger

from karotte.runtime import Runtime, get_engine


def require_runtime(runtime: Runtime) -> None:
    """Exit with a one-line error if the runtime's binary isn't on PATH."""
    engine = get_engine(runtime)
    if which(engine) is None:
        alternative = "docker" if engine != "docker" else "podman"
        _exit_with_error(
            f"{engine} not found. Install it or pass --runtime {alternative}."
        )


def require_builder(runtime: Runtime) -> None:
    """Exit with a one-line error if the engine can't build the image.

    docker needs buildx for `RUN --mount`; nerdctl builds through BuildKit's
    `buildctl`.
    """
    engine = get_engine(runtime)
    if engine == "docker" and not _buildx_available():
        _exit_with_error(
            "docker buildx not found, and building the image needs it. "
            + "Install the docker-buildx package or Docker's docker-buildx-plugin."
        )
    if engine == "nerdctl" and not _buildctl_available():
        _exit_with_error(
            "buildctl not found, and nerdctl needs BuildKit to build the image. "
            + "Install BuildKit and start buildkitd."
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

    if engine == "container":
        from karotte.apple_container import check_build_context

        check_build_context(build_context)
        if cache_from or cache_to:
            logger.warning(
                "Apple `container` build has no build cache options; ignoring them"
            )
        return [
            "container",
            "build",
            *secret_flags,
            "--file",
            "Containerfile",
            "--tag",
            tag,
            build_context,
        ]

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
    return _succeeds(["docker", "buildx", "version"])


def _buildctl_available() -> bool:
    # `nerdctl build` runs `buildctl`, so it has to be on the PATH the build sees.
    return _succeeds(["buildctl", "--version"])


def _succeeds(command: list[str]) -> bool:
    # CI builds run under sudo, which has its own PATH.
    if os.environ.get("CI"):
        command = ["sudo", *command]
    try:
        result = subprocess.run(command, capture_output=True, check=False)
    except FileNotFoundError:
        return False
    return result.returncode == 0


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
