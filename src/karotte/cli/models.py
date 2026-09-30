from typing import Annotated

import typer

app = typer.Typer(help="Inspect the models this karotte release knows about.")


@app.command("list")
def list_models(
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output the catalog as JSON"),
    ] = False,
) -> None:
    """List the models karotte offers by name, with what each one supports.

    Capabilities only — who may run which model is decided by whoever
    schedules the run.
    """
    import json as json_lib

    import rich

    from karotte.model_catalog import catalog
    from karotte.model_spec import is_special_training_model

    known = catalog()

    if json_output:
        # Plain print: rich.print wraps long lines and breaks JSON parsing.
        print(json_lib.dumps(known.model_dump(mode="json"), indent=2))
        return

    rich.print("Available models:")
    # Training checkpoints need a backend that serves them.
    for spec in known.models:
        if is_special_training_model(spec.model):
            continue
        if spec.reasoning_effort_levels:
            effort = ", ".join(spec.reasoning_effort_levels)
        else:
            effort = "not adjustable"
        detail = f"{spec.max_output_tokens // 1000}k output, effort {effort}"
        names = f"{spec.model_display_name}, {spec.provider_display_name}"
        rich.print(f"  [bold]{spec.model}[/bold] ({names}) — {detail}")

    rich.print("\nAlso accepted, any model id under these prefixes:")
    for family in known.families:
        if is_special_training_model(family.prefix):
            continue
        rich.print(f"  [bold]{family.prefix}[/bold] — {family.description}")
