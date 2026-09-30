import importlib.util
from collections.abc import Callable

import typer

from karotte.schemas.evaluation_run_config import EvaluationRunConfig
from karotte.task import Task


def require_environment() -> None:
    """Exit with a one-line error when no environment package is installed."""
    if not _environment_is_installed():
        typer.secho(
            "No karotte environment found. cd into one and run `uv run karotte ...` there.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)


def load_all_task_classes() -> list[type[Task]]:
    return sorted(_get_task_loader()(), key=lambda cls: cls.id)


def load_task(config: EvaluationRunConfig) -> Task:
    task_classes = load_all_task_classes()

    for cls in task_classes:
        if cls.id == config.task_id:
            return cls(config)

    raise ValueError(
        f"Task {config.task_id!r} not found. List existing tasks with `karotte tasks list`."
    )


def _environment_is_installed() -> bool:
    # find_spec locates the package without running its code.
    return importlib.util.find_spec("environment") is not None


def _get_task_loader() -> Callable[[], list[type[Task]]]:
    from environment import get_tasks  # pyright: ignore[reportMissingImports]

    return get_tasks
