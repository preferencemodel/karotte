from typing import Annotated, get_args

import typer
from loguru import logger

from karotte import Runtime
from karotte.build import build_container, require_builder, require_runtime
from karotte.hardware import default_hardware, default_runtime


def _default_build_runtime() -> Runtime:
    """The default runtime `run` would pick for the environment's tasks, so
    `build` fills the image store `run` reads. `run` picks per task, by its
    hardware, and the tasks share one image: tasks that would pick different
    runtimes need ``--runtime``. Outside an environment, the plugins' default
    hardware decides."""
    from karotte.load_tasks import load_all_task_classes
    from karotte.schemas.evaluation_run_config import EvaluationRunConfig

    try:
        task_classes = load_all_task_classes()
    except ModuleNotFoundError:
        task_classes = []
    config = EvaluationRunConfig(run_id="", task_id="", model="", model_api_key="")
    by_runtime: dict[Runtime, list[str]] = {}
    for cls in task_classes:
        task = cls(config)
        by_runtime.setdefault(default_runtime(task.required_hardware), []).append(
            task.id
        )
    if not by_runtime:
        return default_runtime(default_hardware())
    if len(by_runtime) > 1:
        detail = "; ".join(
            f"{r}: {', '.join(ids)}" for r, ids in sorted(by_runtime.items())
        )
        typer.secho(
            f"The tasks run under different default runtimes ({detail}). Pass --runtime.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)
    return next(iter(by_runtime))


def build(
    runtime: Annotated[
        Runtime | None,
        typer.Option(
            metavar="NAME",
            help=f"Container runtime: {', '.join(get_args(Runtime))}. Default: the"
            + " OS's VM runtime (`apple-container` on macOS; `firecracker` on Linux, which"
            + " builds with docker); docker elsewhere.",
        ),
    ] = None,
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
    if runtime is None:
        runtime = _default_build_runtime()
        logger.info(f"Runtime: {runtime} (default; pass --runtime to change)")
    require_runtime(runtime)
    require_builder(runtime)
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
