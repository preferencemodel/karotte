"""Collect what the student handed in."""

from pathlib import Path
from typing import Any

from karotte import (
    EvaluationRunConfig,
    delete_files,
    kill_processes,
    save_artifact,
    save_submission,
)

from environment import STUDENT_UID


def collect_submission(
    config: EvaluationRunConfig,
    paths: tuple[Path, ...],
    save_submission_kwargs: dict[str, Any] | None = None,
) -> tuple[Path, ...]:
    """Stop the student, copy `paths` somewhere root-only, wipe the workdir, and
    save the copies as artifacts; returns the copies in order, and a copy the
    student never wrote does not exist. Grade the copies, never `paths`."""
    kill_processes(STUDENT_UID)
    delete_files(STUDENT_UID, extend_exclude=paths)
    saved = tuple(
        save_submission(path, **(save_submission_kwargs or {})) / path.name
        for path in paths
    )
    delete_files(STUDENT_UID)
    for path in saved:
        save_artifact(config, path)
    return saved
