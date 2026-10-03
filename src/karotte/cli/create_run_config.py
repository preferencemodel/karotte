import uuid
from pathlib import Path
from typing import Annotated

import rich
import typer

from karotte.load_tasks import load_all_task_classes, require_environment
from karotte.schemas.evaluation_run_config import EvaluationRunConfig
from karotte.schemas.http_mcp_server_config import HttpMcpServerConfig


def create_run_config(
    config_path: Annotated[
        str,
        typer.Argument(help="Path where the config file should be created."),
    ] = "run_config.json",
    model: Annotated[str, typer.Option(help="Model name.")] = "claude-fable-5",
    model_api_key: Annotated[
        str | None,
        typer.Option(
            help="Model API key, or a `$VAR` reference to one. Without it, karotte reads the key from the model provider's key variable, e.g. `OPENAI_API_KEY`, when the run starts.",
        ),
    ] = None,
    task: Annotated[
        str | None,
        typer.Option(help="Task ID. Defaults to the first task in the environment."),
    ] = None,
) -> None:
    """Create a default evaluation run configuration."""

    require_environment()
    task_classes = load_all_task_classes()
    if not task_classes:
        rich.print(
            "[red]No tasks available in this environment.[/red] All tasks may be "
            + "filtered out by INCLUDE_TASKS/EXCLUDE_TASKS in "
            + "environment/__init__.py, or none are defined yet."
        )
        raise typer.Exit(1)
    task_ids = [cls.id for cls in task_classes]
    if task is not None and task not in task_ids:
        rich.print(
            f"[red]Task {task!r} not found.[/red] Available tasks: {', '.join(task_ids)}"
        )
        raise typer.Exit(1)
    task_id = task or task_ids[0]

    config = EvaluationRunConfig(
        run_id=uuid.uuid4().hex[:8],
        task_id=task_id,
        model=model,
        model_api_key=model_api_key,
        mcp_server_config=HttpMcpServerConfig(),
        transcript_file="out/transcript.json",
        use_hints=True,
    )

    config_path_ = Path(config_path)
    config_path_.write_text(config.model_dump_json(indent=2))

    rich.print(f"[bold blue]Run config written to {config_path_}[/bold blue]")
