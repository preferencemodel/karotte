from typing import Annotated

import rich
import typer

app = typer.Typer(help="Inspect the environment templates this karotte version ships.")


@app.command("list")
def list_templates(
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output templates as JSON"),
    ] = False,
) -> None:
    """List all available templates."""
    import json as json_lib

    from karotte.templates import TemplatesMissingError, discover_templates

    try:
        installed = discover_templates().values()
    except TemplatesMissingError as e:
        typer.secho(str(e), fg="red", err=True)
        raise typer.Exit(1)

    if json_output:
        print(
            json_lib.dumps(
                [
                    {**t.template.model_dump(), "requirement": t.requirement}
                    for t in installed
                ],
                indent=2,
            )
        )
    else:
        rich.print("Available templates:")
        for entry in installed:
            template = entry.template
            source = f" (from {entry.requirement})" if entry.requirement else ""
            rich.print(f"  [bold]{template.id}[/bold] — {template.description}{source}")
            if template.requires:
                rich.print(f"    Requires: {', '.join(template.requires)}")
