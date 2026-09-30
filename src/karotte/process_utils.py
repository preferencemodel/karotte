"""Utilities for process management."""

import os
import select
import signal
import time
from collections.abc import Iterator
from typing import NoReturn

import psutil
from loguru import logger

from karotte.cgroups import student_cgroup
from karotte.student_misbehavior import StudentMisbehaviorError

_MAX_KILL_WALKS = 10
"""Cap on re-walks within one kill sweep, so churn that truncates every walk
cannot stall it."""

_COHORT_KILL_ROUNDS = 100
"""``kill(-1)`` calls the helper issues per pass.

One is enough on Linux, where the kernel serialises ``kill(-1)`` against
``fork``. gVisor signals a copy of the thread-group list taken under a released
lock, so a process forked after the copy is missed; only sustained repetition
clears a fork chain there. Measured against parentless relays, 1 round left
6/60 alive and 100 left 0/60. A round costs ~250us against a 1500-process
table."""

_HELPER_FAILED = 1
"""Exit code the in-cohort kill helper uses when its syscalls did not go
through."""

_HELPER_DONE = b"\x01"
"""Byte the in-cohort kill helper writes once its ``kill(-1)`` rounds have gone
through. Only the helper holds the write end, and the parent closes its own copy
before waiting."""

_HELPER_TIMEOUT = 5.0
"""How long to wait for the in-cohort kill helper. It makes a handful of
syscalls, so overshooting this means it was interfered with."""


class UnreapableCohortError(StudentMisbehaviorError, RuntimeError):
    """A uid's processes could not be reaped, or the reap could not be verified."""


def kill_processes(
    uid: int,
    *,
    pass_interval: float = 0.05,
    deadline: float = 30.0,
) -> None:
    """
    Send SIGKILL to every process whose real UID matches ``uid``.

    A ``/proc`` sweep cannot see a "fork-and-die" chain: every generation is
    shorter-lived than a walk, so no walk observes one. Each pass therefore:

    1. Kills the cgroup, where a confinement built one (see
       :meth:`karotte.cgroups.StudentCgroup.kill_all`). Atomic against ``fork``.
    2. SIGKILLs the ``uid``-owned members of any PID namespace below ours, which
       drops everything inside it at once (see :func:`_kill_pidns_cohort`).
       Where sessions get their own namespace (gVisor), this alone suffices.
    3. Has a helper that has dropped to ``uid`` issue ``kill(-1, SIGKILL)`` (see
       :func:`_kill_cohort_from_within`). Covers the rest, which on runc is the
       whole cohort.
    4. SIGKILLs whatever ``/proc`` still shows, then sweeps again.

    A clean sweep only proves the cohort is gone when step 3 ran that pass. Step
    3 can be missing because the cohort saturated ``pids.max`` and left no free
    pid to fork the helper, which the next pass may fix, or because it could
    never run here at all (see :func:`_can_kill_from_within`).

    Raises :class:`UnreapableCohortError` for the first, at the deadline, which
    grading scores as 0. Raises a plain ``RuntimeError`` for the second, on the
    first pass, since no retry fixes a broken grader. Both sweep once more
    first; grading must fail rather than run beside a live student process.

    Every signal here is a syscall we make ourselves rather than a call out to a
    utility such as pkill, so a student replacing that binary has nothing to
    grab. Step 3 forks but never execs, so it resolves no binary either.
    """
    deadline_at = time.monotonic() + deadline
    could_kill_from_within = _can_kill_from_within(uid)
    while True:
        _kill_cgroup_cohort(uid)
        namespaced = _kill_pidns_cohort(uid)
        if namespaced:
            logger.info(
                f"Killed {len(namespaced)} process(es) owned by UID {uid} in foreign PID namespaces (pids {namespaced})"
            )
        cohort_killed = _kill_cohort_from_within(uid)
        # A pass that ran it settles what the check before the loop only predicts.
        could_kill_from_within = could_kill_from_within or cohort_killed
        killed = _kill_sweep(uid)
        if killed:
            logger.info(
                f"Killed {len(killed)} processes owned by UID {uid} (pids {killed})"
            )
        time.sleep(pass_interval)
        survivors, complete = _sweep(uid)
        if cohort_killed and complete and not survivors:
            return
        if survivors:
            pids = [proc.pid for proc in survivors]
            logger.warning(
                f"UID {uid} processes survived the kill sweep: {pids}; retrying"
            )
        if not cohort_killed and not could_kill_from_within:
            _kill_sweep(uid)
            raise RuntimeError(
                f"Cannot reap UID {uid} as euid {os.geteuid()}: the in-cohort kill could not run, and without it no sweep proves the cohort is gone"
            )
        if time.monotonic() >= deadline_at:
            _kill_sweep(uid)
            reason = (
                "processes were still alive"
                if cohort_killed
                else "the in-cohort kill never ran, so a clean sweep proves nothing"
            )
            raise UnreapableCohortError(
                f"Failed to reap all processes owned by UID {uid} within {deadline}s: {reason}"
            )


def _can_kill_from_within(uid: int) -> bool:
    """Whether this process could ever have run the in-cohort kill for ``uid``.

    Only root can: the helper's ``setgroups`` needs CAP_SETGID, and being ``uid``
    already is no substitute, because then ``kill(-1)`` would take this process
    down with the cohort. UID 0 has no in-cohort kill at all (see
    :func:`_kill_cohort_from_within`).
    """
    return uid != 0 and os.geteuid() == 0


def _kill_cgroup_cohort(uid: int) -> None:
    """SIGKILL the cohort in ``uid``'s student cgroup, where one is registered."""
    group = student_cgroup(uid)
    if group is None:
        return
    remaining = group.kill_all()
    if remaining:
        logger.warning(
            f"{remaining} process(es) owned by UID {uid} survived the cgroup kill"
        )


def _pid_namespace_of(pid: str) -> str | None:
    """The identity of a process's PID namespace, or ``None`` if it can't be
    read.

    gVisor does not report ``NSpid`` in ``/proc/<pid>/status``, so the namespace
    has to be identified by this symlink.
    """
    try:
        return os.readlink(f"/proc/{pid}/ns/pid")
    except OSError:
        return None


def _kill_pidns_cohort(uid: int) -> list[int]:
    """SIGKILL every process owned by ``uid`` that lives in a PID namespace
    below ours; returns their pids.

    Killing a PID namespace's init makes the kernel SIGKILL everything inside it
    in one step, so a fork chain dies with it even though no sweep ever saw a
    generation. The init does not have to be picked out: it is the session's
    shell, long-lived enough for a sweep to find, and the generations we miss go
    with it.

    Processes sharing our own namespace are left to the sweep; there is no
    atomic reap to be had for them.
    """
    own_ns = _pid_namespace_of("self")
    if own_ns is None:
        return []
    killed: list[int] = []
    procs, _complete = _sweep(uid)
    for proc in procs:
        if _pid_namespace_of(str(proc.pid)) in (None, own_ns):
            continue
        try:
            proc.kill()
            killed.append(proc.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return killed


def _kill_cohort_from_within(uid: int) -> bool:
    """SIGKILL everything owned by ``uid`` with ``kill(-1, SIGKILL)``, issued by
    a forked helper that has dropped to ``uid``. Returns whether that actually
    happened.

    The return value is load-bearing: sweeping cannot see a fork chain, so a
    caller may only trust a clean sweep from a pass where this returned True.
    The realistic way to get False is a cohort that has saturated ``pids.max``,
    leaving no free pid to fork the helper with.

    Confirmation comes over a pipe rather than from the helper's exit status,
    which an unrelated thread reaping our child first would make unknowable.

    The helper has to *be* the uid because ``kill(-1)`` picks its targets by
    signalling permission rather than from an argument. The kernel leaves the
    caller and pid 1 out of it, so the helper survives to be waited on.
    """
    if uid == 0:
        # kill(-1) as root would take down the whole container, us included.
        logger.warning("Refusing to reap UID 0 from within: kill(-1) as root")
        return False
    if os.geteuid() != 0:
        logger.warning(
            f"Refusing to reap UID {uid} from within as euid {os.geteuid()}: not root"
        )
        return False
    read_fd, write_fd = os.pipe()
    try:
        child = os.fork()
    except OSError as exc:
        os.close(read_fd)
        os.close(write_fd)
        logger.warning(f"Could not fork a helper to reap UID {uid} from within: {exc}")
        return False
    if child == 0:
        os.close(read_fd)
        _become_uid_and_kill_all(uid, write_fd)
    os.close(write_fd)
    try:
        return _await_helper(child, uid, read_fd)
    finally:
        os.close(read_fd)


def _become_uid_and_kill_all(uid: int, done_fd: int) -> NoReturn:
    """Body of the forked helper: become ``uid``, signal its whole cohort,
    report that it got that far, exit."""
    try:
        os.setgroups([])
        os.setuid(uid)
    except OSError:
        os._exit(_HELPER_FAILED)
    for _ in range(_COHORT_KILL_ROUNDS):
        try:
            os.kill(-1, signal.SIGKILL)
        except ProcessLookupError:
            break  # Nothing owned by uid is left alive to signal.
        except OSError:
            os._exit(_HELPER_FAILED)
    try:
        _ = os.write(done_fd, _HELPER_DONE)
    except OSError:
        os._exit(_HELPER_FAILED)
    os._exit(0)


def _await_helper(child: int, uid: int, done_fd: int) -> bool:
    """Wait for the helper's confirmation byte and reap it. Returns whether the
    in-cohort kill went through."""
    # poll() rather than select(), which cannot represent an fd past FD_SETSIZE
    poller = select.poll()
    poller.register(done_fd, select.POLLIN)
    try:
        ready = poller.poll(_HELPER_TIMEOUT * 1000)
        confirmed = bool(ready) and os.read(done_fd, 1) == _HELPER_DONE
    except OSError:
        confirmed = False
    if not confirmed:
        # Either its syscalls failed, or it never got to make them: the helper
        # shares the student's uid, so a student process can SIGSTOP it.
        logger.warning(
            f"Helper reaping UID {uid} from within did not confirm its kill(-1)"
        )
    # A confirmed helper is between its write and its _exit, so it is about to
    # go; an unconfirmed one has already had its timeout.
    _reap_helper(child, grace=_HELPER_TIMEOUT if confirmed else 0.0)
    return confirmed


def _reap_helper(child: int, grace: float) -> None:
    """Reap the helper, SIGKILLing it if it outstays ``grace``.

    Only signals a pid that ``waitpid`` has just said is still unreaped, so a
    recycled pid cannot be hit.
    """
    deadline_at = time.monotonic() + grace
    while True:
        try:
            reaped, _status = os.waitpid(child, os.WNOHANG)
        except ChildProcessError:
            return  # Someone else in this process reaped it.
        if reaped:
            return
        if time.monotonic() >= deadline_at:
            break
        time.sleep(0.005)
    try:
        os.kill(child, signal.SIGKILL)
        _ = os.waitpid(child, 0)
    except (OSError, ChildProcessError):
        pass


def _kill_sweep(uid: int) -> list[int]:
    """SIGKILL every non-zombie process owned by ``uid``; returns their pids.

    Walks again while the walk keeps truncating, up to ``_MAX_KILL_WALKS``: a
    sweep cut short never reached the tail of the cohort.
    """
    killed: list[int] = []
    signalled: set[int] = set()
    for _ in range(_MAX_KILL_WALKS):
        procs, complete = _sweep(uid)
        for proc in procs:
            if proc.pid in signalled:
                continue
            signalled.add(proc.pid)
            try:
                proc.kill()
                killed.append(proc.pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        if complete:
            break
    return killed


def _sweep(uid: int) -> tuple[list[psutil.Process], bool]:
    """Walk ``/proc`` and return the live (non-zombie) processes owned by ``uid``,
    plus whether the walk ran to completion.

    psutil re-raises the raw ``OSError`` when a ``/proc`` entry vanishes between
    the readdir and the read of its ``stat`` file, which is frequent under fork
    churn. It comes from the iterator itself, so the walk cannot resume; the
    partial result is a lower bound.
    """
    procs: list[psutil.Process] = []
    walk: Iterator[psutil.Process] = psutil.process_iter(["uids", "status"])
    while True:
        try:
            proc = next(walk)
        except StopIteration:
            return procs, True
        except OSError:
            return procs, False
        uids = proc.info["uids"]
        # ``uids`` is None when the /proc read raced a dying process or hit
        # AccessDenied; without an owner we cannot claim it for this uid.
        if uids is None or uids.real != uid:
            continue
        # Zombies are already dead: they can't fork, and signals are a no-op.
        if proc.info["status"] == psutil.STATUS_ZOMBIE:
            continue
        procs.append(proc)
