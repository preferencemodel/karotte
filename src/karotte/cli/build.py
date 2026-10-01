from typing import Annotated, get_args

import typer

from karotte import Runtime
from karotte.build import build_container, require_buildx, require_runtime


def build(
    runtime: Annotated[
        Runtime,
        typer.Option(
            metavar="NAME",
            help=f"Container runtime: {', '.join(get_args(Runtime))}.",
        ),
    ] = "docker",
    tag: Annotated[
        str,
        typer.Option(help="Tag to assign to the built container image."),
    ] = "karotte",
    build_context: Annotated[
        str, typer.Option(help="Path to the build context.")
    ] = ".",
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
) -> None:
    """Build a container image for the environment."""
    require_runtime(runtime)
    require_buildx(runtime)
    if runtime == "apple-container":
        from karotte.apple_container import validate_container_runtime

        validate_container_runtime()
    build_container(
        runtime,
        build_context,
        tag,
        cache_from=cache_from,
        cache_to=cache_to,
        build_secrets=build_secret or (),
    )
