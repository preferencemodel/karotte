import shlex
import subprocess
from pathlib import Path
from typing import Annotated

import typer


def create_env(
    output_dir: Annotated[
        Path,
        typer.Argument(
            help="Directory where the environment should be created.",
        ),
    ],
    template: Annotated[
        list[str] | None,
        typer.Option(
            help="""Templates to use for environment creation.
            Can specify multiple templates to be stacked on top of each other.
            For example, `--template default --template custom` overlay the
            `custom` template on top of the `default` template.

            Defaults to the `default` template if none is specified.
            """,
        ),
    ] = None,
    agent: Annotated[
        list[str] | None,
        typer.Option(
            help="""CLI agent(s) to bake into the image (e.g. `--agent mistral-vibe`).
            Repeatable. Recorded in the manifest; the installed karotte pins each
            agent's version. Defaults to none (builtin/external only).
            """,
        ),
    ] = None,
    vendor_karotte: Annotated[
        bool,
        typer.Option(
            "--vendor-karotte",
            help="Copy karotte source into the environment for local testing.",
        ),
    ] = False,
    no_lock: Annotated[
        bool,
        typer.Option(
            "--no-lock",
            hidden=True,
            help="Skip running uv lock after creation.",
        ),
    ] = False,
) -> None:
    """Create a new environment in the output directory."""
    from karotte.create_env import create_env as create_env_func

    templates = template or ["default"]

    try:
        create_env_func(
            templates=templates,
            output_dir=output_dir,
            agents=agent or [],
            vendor_karotte=vendor_karotte,
            no_lock=no_lock,
        )
    except subprocess.CalledProcessError as e:
        cmd = e.cmd if isinstance(e.cmd, str) else shlex.join(map(str, e.cmd))
        typer.secho(
            f"`{cmd}` failed with exit code {e.returncode}.", fg="red", err=True
        )
        raise typer.Exit(1)

    steps = [f"cd {shlex.quote(str(output_dir))}", "uv sync --extra dev"]
    if (output_dir / "setup_data.py").is_file():
        steps.append("uv run setup_data.py")
    steps += [
        "uv run karotte create-run-config --model anthropic/claude-fable-5",
        "export ANTHROPIC_API_KEY=...",
        "uv run karotte run --config run_config.json",
    ]
    typer.secho("\nNext steps:", bold=True)
    for step in steps:
        typer.echo(f"  {step}")
