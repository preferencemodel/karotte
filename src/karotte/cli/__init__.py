import signal
from importlib.metadata import entry_points
from types import FrameType
from typing import Annotated

import typer
from loguru import logger

from karotte.cli.agents import app as agents_app
from karotte.cli.build import build
from karotte.cli.check import app as check_app
from karotte.cli.create_env import create_env
from karotte.cli.create_run_config import create_run_config
from karotte.cli.dashboard import dashboard
from karotte.cli.models import app as models_app
from karotte.cli.run import run
from karotte.cli.tasks import app as tasks_app
from karotte.cli.templates import app as templates_app
from karotte.cli.update import update
from karotte.hide_run_config import hide_run_config_from_procfs
from karotte.init_shim import maybe_become_init
from karotte.log import configure_logging

app = typer.Typer(
    add_completion=False,
    pretty_exceptions_enable=False,
)


def _sigterm_handler(_signum: int, _frame: FrameType | None) -> None:
    raise KeyboardInterrupt


def _install_sigterm_handler() -> None:
    """Make SIGTERM raise KeyboardInterrupt, so an orchestrator-initiated
    shutdown (k8s, systemd) unwinds through the same teardown paths as Ctrl-C
    (SIGINT) — tearing down sandboxes and containers instead of orphaning them.
    Python only turns SIGINT into KeyboardInterrupt by default; SIGTERM
    otherwise kills the process without running any `finally`/context-manager
    cleanup.
    """
    signal.signal(signal.SIGTERM, _sigterm_handler)


def _version_callback(value: bool) -> None:
    if value:
        from importlib.metadata import version

        print(version("karotte"))
        raise typer.Exit()


@app.callback()
def main(
    _version: Annotated[
        bool | None,
        typer.Option("--version", callback=_version_callback, is_eager=True),
    ] = None,
) -> None:
    _install_sigterm_handler()


app.command()(create_env)
app.command()(update)
app.command()(create_run_config)
app.command()(dashboard)
app.command()(run)
app.command()(build)
app.add_typer(agents_app, name="agents")
app.add_typer(check_app, name="check")
app.add_typer(models_app, name="models")
app.add_typer(tasks_app, name="tasks")
app.add_typer(templates_app, name="templates")

PLUGIN_ENTRY_POINT_GROUP = "karotte.cli"


def add_plugin_commands(target: typer.Typer) -> None:
    """Adds the command groups installed packages register, never replacing a built-in."""
    taken = {
        info.name or (info.callback.__name__.replace("_", "-") if info.callback else "")
        for info in target.registered_commands
    } | {info.name for info in target.registered_groups}
    for ep in sorted(
        entry_points(group=PLUGIN_ENTRY_POINT_GROUP), key=lambda ep: ep.name
    ):
        if ep.name in taken:
            logger.warning("Ignoring plugin command {!r}: karotte has one", ep.name)
            continue
        try:
            target.add_typer(ep.load(), name=ep.name)
        except Exception as e:  # noqa: BLE001 - a broken plugin must not break karotte
            logger.warning("Ignoring plugin command {!r}: {}", ep.name, e)
            continue
        taken.add(ep.name)


add_plugin_commands(app)


def entry() -> None:
    """Console-script entry point. Secrets leave the command line before the
    init shim forks (the init parent never execs, so it would keep them), and
    the shim forks before the app starts threads or an event loop."""
    configure_logging()
    hide_run_config_from_procfs()
    maybe_become_init()
    app()


if __name__ == "__main__":
    entry()
