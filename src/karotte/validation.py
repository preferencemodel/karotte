"""Grade a student's submission in the middle of an episode, on data the final
judge does not use, and keep the best graded copy.

The `validate` tool (`karotte.tools.validate`) and the runner's
`on_invalid_submission` / `on_final_submission` options (see
`karotte.schemas.evaluation_run_config`) share this module.

A dev scorer is any executable. Karotte runs it as `dev_command` plus two
arguments, a directory holding a copy of each submission path under its own
name and the path of a JSON file to write. The JSON has the shape an
`ExecutableJudge` script writes (`score`, `metadata`) and may also set `tier`
(`valid`, `no_valid_output` or `crash`) and `reason`. Only `tier`, `score` and
`reason` ever reach the student; `metadata` is never read.
"""

import asyncio
import json
import math
import os
import shutil
import stat
import tempfile
import time
import unicodedata
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, ClassVar, Final, Literal, get_args

from loguru import logger
from pydantic import BaseModel, ConfigDict, model_validator

from karotte.container import demoted_uid_gid
from karotte.demoted import drain_bounded
from karotte.process_utils import kill_process_group, kill_processes
from karotte.protected_store import ProtectedStore, SecretNotFoundError
from karotte.save_submission import save_submission
from karotte.student_misbehavior import StudentMisbehaviorError
from karotte.text import escape_surrogates
from karotte.untrusted_paths import (
    DIR_FLAGS,
    FILE_FLAGS,
    lstat_at,
    open_at,
    walk_to_parent,
)

STATE_NAME: Final = "validate_state"

SNAPSHOTS_DIR: Final = Path("~/.config/karotte/validate_snapshots").expanduser()
"""Root-only copies of graded submissions. Not inside the protected store, which
holds files only."""

Tier = Literal["crash", "no_valid_output", "valid"]

Doubt = Literal["scorer_timed_out", "scorer_failed", "submission_changed"]

_MAX_REASON_CHARS: Final = 300
_MAX_RESULT_BYTES: Final = 1024 * 1024
_MAX_LOGGED_STDERR_BYTES: Final = 4096
_SCORER_NICENESS: Final = 10
_PIPE_GRACE_S: Final = 2.0
_COPY_ATTEMPTS: Final = 2
_COMPARE_CHUNK_BYTES: Final = 1 << 20


class SnapshotLimits(BaseModel):
    """How much of a submission to copy, as `save_submission` takes it. Unset
    limits keep its defaults."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    max_file_bytes: int | None = None
    max_total_bytes: int | None = None
    max_entries: int | None = None
    max_depth: int | None = None


class ValidateConfig(BaseModel):
    """Configures the `validate` tool for a task."""

    dev_command: list[str]
    """The dev scorer's argv. Karotte appends the snapshot directory and the
    path of the JSON file to write."""
    metric: str
    """Name of the dev metric, shown to the student."""
    higher_is_better: bool = True
    submission_paths: tuple[Path, ...]
    """What gets graded: the absolute paths the step's student hands in. Use
    the task's `submission_paths`."""
    max_calls: int = 5
    min_interval_s: float = 60.0
    """Least time between two graded calls."""
    timeout_s: float = 60.0
    """Longest a dev scorer may run on one submission."""
    snapshot_limits: SnapshotLimits = SnapshotLimits()
    """Overrides for the limits on copying a submission. Use the ones the
    step's final judge copies with."""

    @model_validator(mode="after")
    def _check(self) -> "ValidateConfig":
        if not self.dev_command:
            raise ValueError("dev_command must not be empty")
        if not self.submission_paths:
            raise ValueError("submission_paths must not be empty")
        for path in self.submission_paths:
            if not path.is_absolute():
                raise ValueError(f"submission_paths must be absolute, got '{path}'")
        names = [path.name for path in self.submission_paths]
        if len(set(names)) != len(names):
            raise ValueError(
                "submission_paths must have distinct file names: the dev scorer "
                + "finds each submission under its own name"
            )
        if self.max_calls < 0 or self.min_interval_s < 0 or self.timeout_s <= 0:
            raise ValueError(
                "max_calls and min_interval_s must not be negative and timeout_s must be positive"
            )
        return self


@dataclass(frozen=True)
class Grade:
    tier: Tier
    elapsed_s: float = 0.0
    dev_metric: float | None = None
    reason: str | None = None
    snapshot: Path | None = None
    """The copy that was graded, kept only for a `valid` grade."""
    doubt: Doubt | None = None
    """Set when `tier` may say nothing about the submission itself: the dev
    scorer ran out of time or failed on its own, or the submission changed while
    it was copied. A scorer that reports a bad submission is not in doubt. For
    `scorer_failed`, `reason` is for the log: it is not shown to the student."""

    def discard(self) -> None:
        if self.snapshot is not None:
            shutil.rmtree(self.snapshot, ignore_errors=True)


class ValidationState(BaseModel):
    """What the `validate` tool remembers between calls, in the protected store."""

    calls: int = 0
    """Graded calls so far."""
    last_call_at: float | None = None
    last_tier: Tier | None = None
    last_valid_metric: float | None = None
    best_metric: float | None = None
    best_snapshot: Path | None = None
    """The copy of the best valid submission."""


def load_state() -> ValidationState:
    try:
        return ProtectedStore().read(STATE_NAME, ValidationState)
    except SecretNotFoundError:
        return ValidationState()


def save_state(state: ValidationState) -> None:
    ProtectedStore().write(STATE_NAME, state)


def clear_snapshots() -> None:
    """Delete every kept copy. The state that points at them is in the protected
    store, which a run clears too."""
    shutil.rmtree(SNAPSHOTS_DIR, ignore_errors=True)


def record_grade(config: ValidateConfig, state: ValidationState, grade: Grade) -> None:
    """Record `grade` in `state`. A valid grade that beats the best so far takes
    over as the kept copy and the old one is deleted; any other copy is deleted."""
    state.last_tier = grade.tier
    if grade.dev_metric is None or grade.snapshot is None:
        return
    state.last_valid_metric = grade.dev_metric
    best = state.best_metric
    improved = best is None or (
        grade.dev_metric > best if config.higher_is_better else grade.dev_metric < best
    )
    if not improved:
        grade.discard()
        return
    if state.best_snapshot is not None:
        shutil.rmtree(state.best_snapshot, ignore_errors=True)
    state.best_metric = grade.dev_metric
    state.best_snapshot = grade.snapshot


async def grade_submission(config: ValidateConfig, paths: tuple[Path, ...]) -> Grade:
    """Copy `paths` somewhere root-only and run the dev scorer on the copy,
    leaving the student's processes alone."""
    started = time.monotonic()
    grade = await _grade(config, paths)
    return replace(grade, elapsed_s=time.monotonic() - started)


async def _grade(config: ValidateConfig, paths: tuple[Path, ...]) -> Grade:
    try:
        snapshot = _take_snapshot(paths, config.snapshot_limits)
    except _SubmissionRejectedError as e:
        return Grade("no_valid_output", reason=str(e))
    except _SubmissionChangedError:
        return Grade(
            "crash",
            reason="the submission kept changing while it was copied",
            doubt="submission_changed",
        )
    grade = await _run_scorer(config, snapshot)
    if grade.tier == "valid":
        return replace(grade, snapshot=snapshot)
    shutil.rmtree(snapshot, ignore_errors=True)
    return grade


class _SubmissionRejectedError(Exception):
    pass


class _SubmissionChangedError(Exception):
    pass


def _take_snapshot(paths: tuple[Path, ...], limits: SnapshotLimits) -> Path:
    """Copy `paths`, and check that they were not changing meanwhile.

    The student's processes keep running, and the paths are copied one after the
    other, so a copy can mix two states of the submission. Whatever it was
    graded as would then be a state the student never had, and could become the
    best copy. So the live files are compared with the copy once it is made. A
    mismatch means a copy again, once; a second one raises
    `_SubmissionChangedError`."""
    for _ in range(_COPY_ATTEMPTS):
        snapshot = _copy_paths(paths, limits)
        if _live_matches(snapshot, paths):
            return snapshot
        shutil.rmtree(snapshot, ignore_errors=True)
    raise _SubmissionChangedError


def _copy_paths(paths: tuple[Path, ...], limits: SnapshotLimits) -> Path:
    SNAPSHOTS_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    SNAPSHOTS_DIR.chmod(0o700)
    snapshot = Path(tempfile.mkdtemp(dir=SNAPSHOTS_DIR))
    try:
        for path in paths:
            _ = save_submission(
                path, snapshot / path.name, **limits.model_dump(exclude_none=True)
            )
            if not os.path.lexists(snapshot / path.name):
                raise _SubmissionRejectedError(f"{path} does not exist")
    except StudentMisbehaviorError as e:
        shutil.rmtree(snapshot, ignore_errors=True)
        raise _SubmissionRejectedError(_reason(str(e))) from e
    except BaseException:
        shutil.rmtree(snapshot, ignore_errors=True)
        raise
    return snapshot


def _live_matches(snapshot: Path, paths: tuple[Path, ...]) -> bool:
    """Whether every path still holds exactly what `snapshot` has for it."""
    for path in paths:
        try:
            parent_fd, name = walk_to_parent(path)
        except OSError:
            return False
        try:
            if not _same_as_copy(snapshot / path.name, parent_fd, name):
                return False
        finally:
            os.close(parent_fd)
    return True


def _same_as_copy(copy: Path, dir_fd: int, name: str) -> bool:
    """Whether `name` in `dir_fd` is, with everything below it, what `copy`
    holds. Opens nothing but regular files and directories and follows no
    symlink, whatever the student swapped in. Any error counts as a difference."""
    try:
        mode = lstat_at(dir_fd, name).st_mode
        if stat.S_ISDIR(mode):
            return _same_directory(copy, dir_fd, name)
        if stat.S_ISREG(mode):
            return _same_file(copy, dir_fd, name)
    except OSError:
        pass
    return False


def _same_directory(copy: Path, dir_fd: int, name: str) -> bool:
    if not copy.is_dir():
        return False
    fd = open_at(dir_fd, name, DIR_FLAGS)
    try:
        with os.scandir(fd) as entries:
            live = sorted(entry.name for entry in entries)
        if live != sorted(entry.name for entry in copy.iterdir()):
            return False
        return all(_same_as_copy(copy / child, fd, child) for child in live)
    finally:
        os.close(fd)


def _same_file(copy: Path, dir_fd: int, name: str) -> bool:
    if not copy.is_file():
        return False
    fd = open_at(dir_fd, name, FILE_FLAGS)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_size != copy.stat().st_size:
            return False
        with copy.open("rb") as saved:
            while chunk := os.read(fd, _COMPARE_CHUNK_BYTES):
                if saved.read(len(chunk)) != chunk:
                    return False
            return saved.read(1) == b""
    finally:
        os.close(fd)


async def _run_scorer(config: ValidateConfig, snapshot: Path) -> Grade:
    with tempfile.TemporaryDirectory(dir=SNAPSHOTS_DIR) as out_dir:
        result_file = Path(out_dir) / "result.json"
        # The stderr pipe is ours rather than asyncio's, whose `wait()` would not
        # return while a child that left the process group still held the pipe.
        read_fd, write_fd = os.pipe()
        try:
            proc = await asyncio.create_subprocess_exec(
                *config.dev_command,
                str(snapshot),
                str(result_file),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=write_fd,
                start_new_session=True,
                preexec_fn=_lower_priority,
            )
        except OSError as e:
            os.close(read_fd)
            return _scorer_failed(f"cannot start {config.dev_command}: {e}")
        finally:
            os.close(write_fd)
        stderr = asyncio.StreamReader()
        transport, _ = await asyncio.get_running_loop().connect_read_pipe(
            lambda: asyncio.StreamReaderProtocol(stderr),
            os.fdopen(read_fd, "rb", buffering=0),
        )
        stderr_task = asyncio.create_task(
            drain_bounded(stderr, _MAX_LOGGED_STDERR_BYTES)
        )
        timed_out = False
        try:
            async with asyncio.timeout(config.timeout_s):
                _ = await proc.wait()
        except TimeoutError:
            timed_out = True
        finally:
            kill_process_group(proc.pid)
            _ = await proc.wait()
            _, undrained = await asyncio.wait({stderr_task}, timeout=_PIPE_GRACE_S)
            _ = stderr_task.cancel()
            transport.close()
        if timed_out:
            return Grade(
                "crash",
                reason=f"the dev scorer did not finish within {config.timeout_s:g} s",
                doubt="scorer_timed_out",
            )
        if proc.returncode != 0:
            logged = b"" if undrained else stderr_task.result()
            return _scorer_failed(
                f"the dev scorer exited with code {proc.returncode}: {logged.decode(errors='replace')}"
            )
        return _interpret(result_file)


def _lower_priority() -> None:
    os.nice(_SCORER_NICENESS)


def _scorer_failed(reason: str) -> Grade:
    """The dev scorer itself is broken, which says nothing about the submission:
    it could not start, exited with an error or broke its contract. That is the
    environment's fault, so it is logged as an error, and `reason` is for the log
    only."""
    logger.error("The dev scorer failed: {}", reason)
    return Grade("crash", reason=reason, doubt="scorer_failed")


def _interpret(result_file: Path) -> Grade:
    """Read the scorer's result, taking only `tier`, `score` and `reason`."""
    try:
        raw: Any = json.loads(result_file.read_bytes()[:_MAX_RESULT_BYTES])
    except (OSError, ValueError):
        return _scorer_failed("the dev scorer produced no result")
    if not isinstance(raw, dict):
        return _scorer_failed("the dev scorer produced no result")
    result: dict[str, Any] = raw
    tier = result.get("tier", "valid")
    if tier not in get_args(Tier):
        return _scorer_failed("the dev scorer reported an unknown tier")
    if tier != "valid":
        reason = result.get("reason")
        default = (
            "submission rejected" if tier == "no_valid_output" else "submission failed"
        )
        return Grade(
            tier,
            reason=(_reason(reason) if isinstance(reason, str) else "") or default,
        )
    score = result.get("score")
    if isinstance(score, bool) or not isinstance(score, int | float):
        return _scorer_failed("the dev scorer produced no score")
    if not math.isfinite(score):
        return _scorer_failed("the dev scorer produced a non-finite score")
    return Grade("valid", dev_metric=float(score))


def _reason(text: str) -> str:
    """Make text that reaches the student safe to show: control characters go,
    runs of whitespace (line breaks included) become one space, and it is cut
    to `_MAX_REASON_CHARS`."""
    printable = "".join(
        char
        for char in escape_surrogates(text)
        if char.isspace() or unicodedata.category(char) != "Cc"
    )
    return " ".join(printable.split())[:_MAX_REASON_CHARS]


async def restore_best_submission(
    paths: tuple[Path, ...], config: ValidateConfig, *, only_if_invalid: bool
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
    grade = await grade_submission(config, paths)
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
