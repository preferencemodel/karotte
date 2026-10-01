import json
import os
from typing import Annotated

import rich
import typer
from rich.markup import escape

from karotte.container import is_containerized
from karotte.load_tasks import load_all_task_classes, require_environment

app = typer.Typer(
    help="Check the environment (no subcommand), or what a sandbox does to the student.",
    invoke_without_command=True,
    no_args_is_help=False,
)


@app.callback()
def _default(ctx: typer.Context) -> None:  # pyright: ignore[reportUnusedFunction]
    """Check that the environment is properly set up."""
    if ctx.invoked_subcommand is None:
        check()


def check() -> None:
    """Check that the environment is properly set up."""
    require_environment()
    task_classes = load_all_task_classes()

    if len(task_classes) == 0:
        rich.print("[bold red]No tasks found in environment.[/bold red]")
        raise typer.Exit(1)

    MAX_TASK_NAME_LENGTH = 255
    too_long = [cls.id for cls in task_classes if len(cls.id) > MAX_TASK_NAME_LENGTH]
    if too_long:
        for name in too_long:
            rich.print(
                f"[bold red]Task name exceeds {MAX_TASK_NAME_LENGTH} characters ({len(name)}): {name!r}[/bold red]"
            )
        raise typer.Exit(1)

    rich.print(f"[bold blue]Available tasks: {len(task_classes)}[/bold blue]")

    from karotte.mcp_servers.discover_tools import discover_tools

    tool_names = sorted(discover_tools())

    rich.print(
        f"[bold blue]Available MCP tools:\n{'\n'.join(f'  - {t}' for t in tool_names)}[/bold blue]"
    )

    if is_containerized():
        from karotte.check_paths import UnsafeLoaderPath, check_paths

        try:
            check_paths()
        except UnsafeLoaderPath as e:
            rich.print(f"[bold red]{e}[/bold red]")
            raise typer.Exit(1)
        rich.print("[bold blue]Loader path check passed.[/bold blue]")

        from karotte.check_credentials import CredentialInImage, check_credentials

        try:
            check_credentials()
        except CredentialInImage as e:
            rich.print(f"[bold red]{e}[/bold red]")
            raise typer.Exit(1)
        rich.print("[bold blue]Credential check passed.[/bold blue]")

        from karotte.protected_store import ProtectedStore

        ProtectedStore().check_permissions()
        rich.print("[bold blue]Protected store permission check passed.[/bold blue]")

        try:
            from environment.check_permissions import (  # pyright: ignore[reportMissingImports]
                check_permissions,
            )

            check_permissions()
            rich.print("[bold blue]Environment permission checks passed.[/bold blue]")
        except ImportError:
            # No permission checks defined in this environment
            pass

    rich.print("[bold green]Check passed.[/bold green]")


@app.command(
    "confinement",
    short_help="Check what this sandbox does to the student.",
    help="Check what this sandbox does to the student. Run as root inside the"
    + " sandbox. Tries limits and firewall rules on an unused uid, mounts and"
    + " unmounts a small file quota, and starts one real student session under the"
    + " default limits; each is undone afterwards. It can leave the harness moved"
    + " into the karotte_harness cgroup with controllers delegated, which every run"
    + " does at start anyway; the report lists anything left behind. Exits 1 if"
    + " the sandbox gives the student less than it should.",
)
def confinement(
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print the report as JSON."),
    ] = False,
    hardware: Annotated[
        str | None,
        typer.Option(
            "--hardware",
            help="The hardware whose default student memory limit to apply. Default: the plugins' default hardware.",
        ),
    ] = None,
) -> None:
    """Check what this sandbox does to the student."""
    from karotte.confinement_check import (
        evaluate,
        gather,
        passed,
        render_table,
        report_json,
    )

    if os.geteuid() != 0:
        rich.print("[bold red]karotte check confinement must run as root.[/bold red]")
        raise typer.Exit(1)
    from karotte.hardware import default_hardware, hardware_limits

    if hardware is None:
        hardware = default_hardware()
    elif hardware_limits(hardware) is None:
        # A typo would otherwise check against a memory default nobody asked for.
        rich.print(
            f"[bold red]No installed hardware plugin knows {escape(hardware)!r}.[/bold red]"
        )
        raise typer.Exit(1)
    observations = gather(hardware)
    findings = evaluate(observations)
    if json_output:
        # Plain print: rich.print can wrap lines, which breaks JSON parsing.
        print(json.dumps(report_json(observations, findings), indent=2))
    else:
        rich.print(render_table(findings))
    if not passed(findings):
        raise typer.Exit(1)
