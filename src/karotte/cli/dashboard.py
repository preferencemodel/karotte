from pathlib import Path
from typing import Annotated

import typer


def dashboard(
    transcript_dir: Annotated[
        Path, typer.Argument(help="Path to a directory containing transcript files.")
    ],
):
    """Visualize existing transcripts."""
    from karotte.terminal.app import run_static_dashboard

    run_static_dashboard(transcript_dir)
