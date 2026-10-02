import os
from pathlib import Path
from typing import Annotated

import typer


def update(
    project_dir: Annotated[
        Path | None,
        typer.Argument(
            help="Directory of the project to update. Defaults to current directory.",
        ),
    ] = None,
    add_template: Annotated[
        list[str] | None,
        typer.Option(
            "--add-template",
            help=(
                "Template to add to the project (repeatable). Its files are"
                + " applied alongside the update and recorded in the manifest."
            ),
        ),
    ] = None,
    with_: Annotated[
        list[str] | None,
        typer.Option(
            "--with",
            help=(
                "Package to make available when rendering the templates (repeatable),"
                + " for templates the manifest does not yet record a package for."
            ),
        ),
    ] = None,
) -> None:
    """Update a project to the latest karotte templates using 3-way merge."""
    from karotte.update_env import (
        RELAUNCHED_ENV_VAR,
        missing_template_packages,
        relaunch_with,
    )
    from karotte.update_env import update_env as update_env_func

    project_dir = (project_dir or Path.cwd()).resolve()
    add_template = add_template or []
    with_ = with_ or []

    if not os.environ.get(RELAUNCHED_ENV_VAR):
        missing = missing_template_packages(project_dir, with_)
        if missing:
            args = ["update", str(project_dir)]
            args += [a for t in add_template for a in ("--add-template", t)]
            args += [a for w in with_ for a in ("--with", w)]
            raise typer.Exit(relaunch_with(project_dir, missing, args))

    update_env_func(
        project_dir=project_dir,
        add_templates=add_template,
        extra_with=with_,
    )
