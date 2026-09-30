import subprocess
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger

app = typer.Typer(help="Inspect and manage the agents that can drive a run.")


@app.command("list")
def list_agents(
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output agents as JSON"),
    ] = False,
) -> None:
    """List the agents that can drive a run.

    CLI agents are only usable if the image was built with them; the others
    need nothing beyond the `agent` field in the run config.
    """
    import json as json_lib

    import rich

    from karotte.agents import cli_agent_types

    agents = [
        {"name": name, "kind": kind, "version": None, "requires_install": False}
        for name, kind in (("builtin", "builtin"), ("external", "external"))
    ]
    agents += [
        {
            "name": name,
            "kind": "cli",
            "version": agent_type.version,
            "requires_install": True,
        }
        for name, agent_type in sorted(cli_agent_types().items())
    ]

    if json_output:
        print(json_lib.dumps(agents, indent=2))
        return

    rich.print("Available agents:")
    for agent in agents:
        detail = (
            f"CLI agent v{agent['version']}, bake into the image with `karotte build`"
            if agent["requires_install"]
            else "usable on any image"
        )
        rich.print(f"  [bold]{agent['name']}[/bold] — {detail}")


def _manifest_path(project_dir: Path | None) -> Path:
    project_dir = project_dir or Path.cwd()
    path = project_dir / ".manifest.json"
    if not path.exists():
        raise typer.BadParameter(
            f"No .manifest.json found in {project_dir}. "
            + "Run this from an environment created with `karotte create-env`."
        )
    return path


@app.command()
def add(
    name: Annotated[str, typer.Argument(help="Name of the CLI agent to add.")],
    project_dir: Annotated[
        Path | None,
        typer.Option(help="Environment directory. Defaults to the current directory."),
    ] = None,
) -> None:
    """Add a CLI agent to the environment's manifest.

    Validates the name against the agents this karotte release knows about. The
    agent gets installed on the next `karotte build`.
    """
    from karotte.create_env import EnvManifest, validate_agents

    validate_agents([name])

    path = _manifest_path(project_dir)
    manifest = EnvManifest.model_validate_json(path.read_text())

    if name in manifest.agents:
        logger.info("Agent {} is already in the manifest.", name)
        return

    manifest.agents.append(name)
    path.write_text(manifest.model_dump_json(indent=2))
    logger.info("Added agent {}. Run `karotte build` to install it.", name)


@app.command()
def remove(
    name: Annotated[str, typer.Argument(help="Name of the CLI agent to remove.")],
    project_dir: Annotated[
        Path | None,
        typer.Option(help="Environment directory. Defaults to the current directory."),
    ] = None,
) -> None:
    """Remove a CLI agent from the environment's manifest.

    The agent is dropped on the next `karotte build`.
    """
    from karotte.create_env import EnvManifest

    path = _manifest_path(project_dir)
    manifest = EnvManifest.model_validate_json(path.read_text())

    if name not in manifest.agents:
        logger.info("Agent {} is not in the manifest.", name)
        return

    manifest.agents.remove(name)
    path.write_text(manifest.model_dump_json(indent=2))
    logger.info("Removed agent {}. Run `karotte build` to apply.", name)


@app.command(hidden=True)
def install(
    manifest: Annotated[
        Path,
        typer.Option(help="Env manifest JSON listing the agents to install."),
    ],
) -> None:
    """Install the pinned CLI agents named in the manifest (run at build time).

    Each agent class owns its install recipe and version; this just runs them.
    A no-op when the manifest lists no agents.
    """
    from karotte.agents import get_cli_agent_type
    from karotte.agents.cli_agent import AGENTS_DIR
    from karotte.container import is_containerized
    from karotte.create_env import EnvManifest

    if not is_containerized():
        typer.secho(
            "`karotte agents install` only runs inside a karotte image build.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Abort()

    names = EnvManifest.model_validate_json(manifest.read_text()).agents
    if not names:
        logger.info("No CLI agents to install.")
        return

    for name in names:
        agent_type = get_cli_agent_type(name)
        logger.info("Installing agent {} v{}", name, agent_type.version)
        for command in agent_type.install():
            logger.info("$ {}", command)
            subprocess.run(command, shell=True, check=True)

    # `uv tool install` leaves a mode-666 `.lock`; nothing under the agents dir may be student-writable.
    agents_dir = Path(AGENTS_DIR)
    if agents_dir.exists():
        logger.info("Removing group/other write from {}", AGENTS_DIR)
        subprocess.run(["chmod", "-R", "go-w", AGENTS_DIR], check=True)
