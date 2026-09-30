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
    from karotte.update_env import update_env as update_env_func

    project_dir = project_dir or Path.cwd()

    update_env_func(
        project_dir=project_dir.resolve(),
        add_templates=add_template or (),
        extra_with=with_ or (),
    )
