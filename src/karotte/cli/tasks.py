from typing import Annotated

import rich
import typer

app = typer.Typer(help="Inspect the tasks in this environment.")


@app.command("list")
def list_tasks(
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output tasks as JSON"),
    ] = False,
) -> None:
    """List all available tasks."""
    import json as json_lib

    from karotte.load_tasks import load_all_task_classes, require_environment
    from karotte.schemas.evaluation_run_config import EvaluationRunConfig

    require_environment()
    task_classes = load_all_task_classes()

    if json_output:
        sample_config = EvaluationRunConfig(
            run_id="",
            task_id="",
            model="",
            model_api_key="",
        )
        tasks = [cls(sample_config) for cls in task_classes]
        task_list = [
            {
                "id": task.id,
                "tools": task.tools,
                "required_hardware": task.required_hardware,
                "submission_paths": [str(path) for path in task.submission_paths],
                "data_mounts": [m.model_dump() for m in task.data_mounts],
            }
            for task in tasks
        ]
        # We are using normal print here because rich.print can introduce newlines which breaks json parsing
        print(json_lib.dumps(task_list, indent=2))
    else:
        rich.print("Available tasks:")
        for cls in task_classes:
            rich.print(f"  - {cls.id!r}")
