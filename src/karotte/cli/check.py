import rich
import typer

from karotte.container import is_containerized
from karotte.load_tasks import load_all_task_classes, require_environment


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
