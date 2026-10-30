"""Put the best copy that the `validate` tool graded as valid back over a
student's submission before a step is scored.

This is what the runner's `on_invalid_submission` and `on_final_submission`
options do (see `karotte.schemas.evaluation_run_config`).
"""

import asyncio
import os
import shutil
import stat
from pathlib import Path

from loguru import logger

from karotte.container import demoted_uid_gid
from karotte.process_utils import kill_processes
from karotte.student_misbehavior import StudentMisbehaviorError
from karotte.tools.validate import ValidateConfig, validate
from karotte.untrusted_paths import DIR_FLAGS, lstat_at, walk_to_parent
from karotte.validation import load_state


async def restore_best_submission(
    paths: tuple[Path, ...], *, only_if_invalid: bool
) -> bool:
    """Put the best copy a `validate` call graded as valid back over `paths`.

    Stops the student's processes first, so none can swap a path for a symlink
    or rewrite a file while it is being checked or replaced. Returns False when
    there is no copy, the student could not be stopped, or (with
    `only_if_invalid`) the submission on disk is to stay: see
    `_submission_stays`. Also returns False when no path could be put back; True
    means at least one was.
    """
    best = load_state().best_snapshot
    if best is None or not best.is_dir():
        logger.info("No valid copy was graded, so there is nothing to restore")
        return False
    config = validate.load_config()
    if not await asyncio.to_thread(_stop_student):
        return False
    if only_if_invalid and await _submission_stays(config, paths):
        return False
    return await asyncio.to_thread(_put_back, best, paths, config.submission_paths)


async def _submission_stays(config: ValidateConfig, paths: tuple[Path, ...]) -> bool:
    """Whether the submission on disk is left as it is: the dev scorer grades it
    valid, or cannot say. A timeout, a crash of the scorer itself or a result it
    never wrote is not evidence against the submission, and replacing a good
    submission because of one would lose it. The grade is not recorded and does
    not count as a call, and the student is already stopped, so nothing is
    reported or observable."""
    grade = await config.grade(paths)
    grade.discard()
    if grade.doubt is not None:
        logger.warning(
            "Keeping the submission as it is: the dev scorer could not say whether it is valid ({})",
            grade.reason,
        )
        return True
    return grade.tier == "valid"


def _stop_student() -> bool:
    uid = demoted_uid_gid()
    if uid is None:
        return True
    try:
        kill_processes(uid)
    except StudentMisbehaviorError as e:
        # Scoring runs the same kill and scores this as misbehavior.
        logger.warning("Not restoring, the student could not be stopped: {}", e)
        return False
    return True


def _put_back(
    snapshot: Path, paths: tuple[Path, ...], graded: tuple[Path, ...]
) -> bool:
    """Returns whether at least one path got its copy back. The copy holds each
    of the `graded` paths under its file name, so a path is restored only if it
    is one of them: another path with the same file name must not be given that
    content."""
    restored = False
    for path in paths:
        if path not in graded:
            logger.warning("{} is not a path the validate tool graded", path)
            continue
        source = snapshot / path.name
        if not os.path.lexists(source):
            logger.warning("The best valid copy holds nothing for {}", path)
            continue
        try:
            parent_fd, name = walk_to_parent(path)
        except OSError as e:
            logger.warning("Cannot restore {}: {}", path, e.strerror)
            continue
        try:
            _remove_at(parent_fd, name)
            _copy_into(source, parent_fd, name)
        finally:
            os.close(parent_fd)
        logger.info("Restored the best valid copy of {}", path)
        restored = True
    if not restored:
        logger.warning("None of the submission paths could be restored")
    return restored


def _remove_at(dir_fd: int, name: str) -> None:
    try:
        mode = lstat_at(dir_fd, name).st_mode
    except FileNotFoundError:
        return
    if stat.S_ISDIR(mode):
        shutil.rmtree(name, dir_fd=dir_fd)
    else:
        os.unlink(name, dir_fd=dir_fd)


def _copy_into(source: Path, dir_fd: int, name: str) -> None:
    """Copy the file or tree at `source`, which holds only regular files and
    directories, to `name` inside `dir_fd`, without following anything there."""
    if source.is_dir():
        os.mkdir(name, 0o755, dir_fd=dir_fd)
        child_fd = os.open(name, DIR_FLAGS, dir_fd=dir_fd)
        try:
            for entry in sorted(source.iterdir()):
                _copy_into(entry, child_fd, entry.name)
        finally:
            os.close(child_fd)
        return
    fd = os.open(
        name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644, dir_fd=dir_fd
    )
    with source.open("rb") as src, os.fdopen(fd, "wb") as dest:
        shutil.copyfileobj(src, dest)
