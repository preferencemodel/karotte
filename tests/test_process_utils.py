"""Tests for :mod:`karotte.process_utils`.

The interesting case is grading integrity: before scoring, the grader reaps
every process owned by the student uid so no leftover process can interfere
with the (timed) grading run. A student "fork-and-die" chain respawns faster
than a single non-atomic ``/proc`` sweep can observe, so sweeping alone cannot
tell a reaped cohort from one it merely failed to see. ``kill_processes``
counters this by reaping the student's PID namespace where there is one (gVisor
only), then having a helper that has dropped to the uid issue
``kill(-1, SIGKILL)``, which the kernel serialises against ``fork``, and only
then verifying by sweep.

The end-to-end tests exercise this with real churn in three shapes:

- a fork-and-die chain plus a SIGCONT spammer, anchored by a long-lived loop;
- a ring of mutual SIGCONT spammers, which no amount of SIGSTOPing converges on;
- a parentless relay, where no process lives long enough for any sweep to see
  it, so only the ``kill(-1)`` step reaps it.

They need Linux + root (to reap a separate uid without killing the test
process); everywhere else they skip.
"""

import os
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, NoReturn, final

import psutil
import pytest

from karotte.process_utils import (
    _COHORT_KILL_ROUNDS,  # pyright: ignore[reportPrivateUsage]
    _HELPER_DONE,  # pyright: ignore[reportPrivateUsage]
    _HELPER_FAILED,  # pyright: ignore[reportPrivateUsage]
    _await_helper,  # pyright: ignore[reportPrivateUsage]
    _become_uid_and_kill_all,  # pyright: ignore[reportPrivateUsage]
    _kill_cohort_from_within,  # pyright: ignore[reportPrivateUsage]
    _kill_pidns_cohort,  # pyright: ignore[reportPrivateUsage]
    _kill_sweep,  # pyright: ignore[reportPrivateUsage]
    kill_processes,
)
from karotte.student_misbehavior import StudentMisbehaviorError

_FD_SETSIZE = 1024
"""The ``fd_set`` bound ``select(2)`` is stuck with. ``poll(2)`` has none."""

NOBODY_UID = 65534
"""Spare uid we run the churn under so the reaper (running as root) never
matches the test process or anything else on the host."""

# Busy daemon + four aggressive fork-and-die chains + a SIGCONT spammer — the
# pattern that defeats a reaper which stops on its first clean sweep.
_CHURN = (
    "( while :; do :; done ) & "
    "for i in 1 2 3 4; do ( while :; do bash -c ':' & done ) & done; "
    "( while :; do kill -CONT -1 2>/dev/null; done ) & "
    "sleep 3600"
)

_RING_SIZE = 4
"""Ring members. Four is already enough to stall a freeze-based reaper
indefinitely; more only makes the test heavier."""

_PEERS_FILE = "peers"
_PID_PREFIX = "pid."

# One member of a mutual-SIGCONT ring: it publishes its pid, waits for the
# others, then SIGCONTs all of them in a tight loop. No sequence of SIGSTOPs
# converges on a ring like this, because any member a sweep hasn't reached yet
# thaws the ones it already passed. SIGKILL has no such problem, and this pins
# that the reaper does not depend on stopping the cohort first.
#
# Both handoffs publish via os.replace: a reader that catches a half-written
# file would silently end up with a smaller ring, and the test would still pass
# while no longer exercising the mutual thawing.
_RING_MEMBER = f"""
import os, signal, sys, time

pid_dir, index = sys.argv[1], sys.argv[2]
part_path = os.path.join(pid_dir, "part." + index)
with open(part_path, "w") as handle:
    handle.write(str(os.getpid()))
os.replace(part_path, os.path.join(pid_dir, "{_PID_PREFIX}" + index))

peers_file = os.path.join(pid_dir, "{_PEERS_FILE}")
while not os.path.exists(peers_file):
    time.sleep(0.01)
with open(peers_file) as handle:
    peers = [int(pid) for pid in handle.read().split() if int(pid) != os.getpid()]

while True:
    for peer in peers:
        try:
            os.kill(peer, signal.SIGCONT)
        except OSError:
            pass
"""

_MARKER_PREFIX = "relay_alive."

_RELAY_CHECK_EVERY = 512
"""Generations between a relay's marker touch and self-destruct check. Every
syscall a generation makes lengthens its life and so makes it easier to catch,
and the whole point of a generation is to be shorter-lived than a sweep."""

_RELAY_LIFETIME = 120.0
"""Wall-clock bound on the relays, so a regression cannot leave fork bombs
running on the host after the test has given up on them."""

_RELAYS = 3
"""Independent relay chains. Whether a sweep ever catches a given chain is a
coin weighted by how long its generations live against how long a walk of
``/proc`` takes, so one chain alone makes for a flaky test; the reaper has to
lose the race against all three for this to pass by luck."""

_BALLAST = 1500
"""Sleepers started alongside the relays, to make every ``/proc`` walk longer:
the longer a walk takes, the smaller a relay generation is against it.

They stand in for the speed we cannot give the relays. A generation forked from
CPython costs ~200us where a compiled one costs a fraction of that, so a real
relay is invisible against the walk of an ordinary ``/proc`` and these need the
walk stretched to match.

They are owned by the test (root), not by the cohort, because cohort-owned
ballast defeats the point: the first kill sweep reaps it, and every sweep after
that is quick enough to stand a real chance of catching a relay."""

_BALLAST_SOURCE = """
import os, sys, time

for _ in range(int(sys.argv[1])):
    if os.fork() == 0:
        time.sleep(3600)
        os._exit(0)
"""

# A parentless fork-and-die relay: every generation forks a successor and the
# parent exits at once, so no process in the cohort is ever long-lived. A sweep
# lists /proc once and then reads the entries one by one, so the generation in
# the listing is already dead by the time the walk reaches it, and its successor
# was born after the listing and isn't in it at all.
#
# The relays fork through libc rather than os.fork() to skip the fork handlers
# CPython runs in the child, which are most of what a generation would otherwise
# spend its life on.
_RELAY = f"""
import ctypes, os, sys, time

marker_dir, relays = sys.argv[1], int(sys.argv[2])

libc = ctypes.CDLL(None, use_errno=True)


def relay(marker):
    give_up_at = time.monotonic() + {_RELAY_LIFETIME}
    generation = 0
    while True:
        generation += 1
        if generation % {_RELAY_CHECK_EVERY} == 0:
            if time.monotonic() > give_up_at:
                os._exit(0)
            os.close(os.open(marker, os.O_CREAT | os.O_WRONLY, 0o666))
        if libc.fork() != 0:
            libc._exit(0)


for index in range(relays):
    if os.fork() == 0:
        relay(os.path.join(marker_dir, "{_MARKER_PREFIX}" + str(index)))
os._exit(0)
"""

real_cohort_kill = pytest.mark.real_cohort_kill
"""Opt back in to the real in-cohort kill (see :func:`_stub_cohort_kill`)."""


@pytest.fixture(autouse=True)
def _stub_cohort_kill(  # pyright: ignore[reportUnusedFunction]
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep the in-cohort kill out of the tests that fake ``/proc``.

    Those drive ``kill_processes`` with a made-up uid and a monkeypatched
    ``process_iter``, but the in-cohort kill is a real ``fork()`` +
    ``kill(-1, SIGKILL)`` that ignores ``process_iter`` entirely. Under a root
    test run it would SIGKILL everything genuinely owned by that uid on the
    host.
    """
    if "real_cohort_kill" in request.keywords:
        return

    def _ran(_uid: int) -> bool:
        return True

    monkeypatch.setattr("karotte.process_utils._kill_cohort_from_within", _ran)

    # Same reasoning for the namespace kill: the fake pids these tests use also
    # name real host processes, so let it read them all as sharing our
    # namespace. The tests that exercise it override this.
    def _same_ns(_path: str) -> str:
        return "pid:[1]"

    monkeypatch.setattr("karotte.process_utils.os.readlink", _same_ns)


def _preexec(uid: int):
    def fn():
        os.setsid()
        os.setgroups([])
        os.setgid(uid)
        os.setuid(uid)

    return fn


def _spawn_churn(uid: int) -> subprocess.Popen[bytes]:
    """Launch the fork-and-die churn as ``uid`` in its own session."""
    return subprocess.Popen(
        ["bash", "-c", _CHURN],
        preexec_fn=_preexec(uid),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _spawn_ring(uid: int, pid_dir: str) -> list[subprocess.Popen[bytes]]:
    """Start a fully wired mutual-SIGCONT ring as ``uid``: spawn the members,
    wait for all of them to publish their pid, then hand every member the full
    pid list, which releases them into their SIGCONT loops."""
    members = [
        subprocess.Popen(
            [sys.executable, "-c", _RING_MEMBER, pid_dir, str(index)],
            preexec_fn=_preexec(uid),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for index in range(_RING_SIZE)
    ]
    published: list[str] = []
    deadline = time.monotonic() + 30
    while len(published) < _RING_SIZE:
        assert time.monotonic() < deadline, f"ring never came up: {published}"
        time.sleep(0.01)
        published = [
            name for name in os.listdir(pid_dir) if name.startswith(_PID_PREFIX)
        ]
    pids = [Path(pid_dir, name).read_text() for name in published]
    assert all(pid.strip() for pid in pids), f"truncated pid handoff: {pids}"
    part = Path(pid_dir, "peers.part")
    part.write_text(" ".join(pids))
    part.replace(Path(pid_dir, _PEERS_FILE))
    return members


def _spawn_relays(uid: int, marker_dir: Path) -> subprocess.Popen[bytes]:
    """Launch the parentless fork-and-die relays (plus their ballast) as ``uid``.

    The returned handle exits once the relays are seeded, so it says nothing
    about whether they are still running — use :func:`_forking_relays` for
    that."""
    return subprocess.Popen(
        [sys.executable, "-c", _RELAY, str(marker_dir), str(_RELAYS)],
        preexec_fn=_preexec(uid),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _forking_relays(marker_dir: Path, window: float) -> list[str]:
    """Which relays are still running: delete every marker and report the ones
    that get re-created within ``window``.

    Sweeping ``/proc`` cannot answer this. Missing a relay is exactly the bug
    under test, so a sweep-based check would call a surviving relay reaped."""
    for marker in marker_dir.glob(f"{_MARKER_PREFIX}*"):
        marker.unlink(missing_ok=True)
    seen: set[str] = set()
    deadline = time.monotonic() + window
    while time.monotonic() < deadline and len(seen) < _RELAYS:
        seen.update(path.name for path in marker_dir.glob(f"{_MARKER_PREFIX}*"))
        time.sleep(0.01)
    return sorted(seen)


def _live_uid_procs(uid: int) -> list[int]:
    """PIDs of live (non-zombie) processes owned by ``uid``."""
    out: list[int] = []
    for proc in psutil.process_iter(["uids", "status"]):
        try:
            if proc.info["uids"].real != uid:
                continue
            if proc.info["status"] == psutil.STATUS_ZOMBIE:
                continue
            out.append(proc.pid)
        except psutil.NoSuchProcess:
            continue
    return out


def _force_cleanup(uid: int) -> None:
    """Guaranteed teardown so a broken run can't leak churn onto the host:
    freeze the cohort (SIGSTOP can't be dodged like SIGKILL), then SIGKILL.

    Sweeping alone cannot see a fork relay, so this leads with the in-cohort
    kill — otherwise a failing relay test would leave fork bombs behind."""
    _kill_cohort_from_within(uid)
    for _ in range(10_000):
        stopped_any = False
        for proc in psutil.process_iter(["uids", "status"]):
            try:
                if proc.info["uids"].real != uid:
                    continue
                if proc.info["status"] in (psutil.STATUS_ZOMBIE, psutil.STATUS_STOPPED):
                    continue
                proc.suspend()
                stopped_any = True
            except psutil.NoSuchProcess:
                continue
        if not stopped_any:
            break
    for proc in psutil.process_iter(["uids"]):
        try:
            if proc.info["uids"].real == uid:
                proc.send_signal(signal.SIGKILL)
        except psutil.NoSuchProcess:
            continue


@pytest.fixture
def ballast() -> Iterator[int]:
    """Run the relay test against a ``/proc`` big enough to sweep slowly.

    The sleepers get their own session so teardown is one ``killpg``, which
    matters because they are root-owned and so survive everything the reaper
    under test does.
    """
    launcher = subprocess.Popen(
        [sys.executable, "-c", _BALLAST_SOURCE, str(_BALLAST)],
        preexec_fn=os.setsid,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    launcher.wait(timeout=120)
    try:
        yield launcher.pid
    finally:
        try:
            os.killpg(launcher.pid, signal.SIGKILL)
        except OSError:
            pass


@pytest.fixture
def shared_dir() -> Iterator[str]:
    """A directory the ring members (running as ``NOBODY_UID``) can write to."""
    # Not under $TMPDIR: `nix develop` points it at a 0700 dir no other uid can enter.
    path = tempfile.mkdtemp(dir="/tmp")
    os.chmod(path, 0o777)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _assert_reaped(children: list[subprocess.Popen[bytes]]) -> None:
    """Reap ``NOBODY_UID`` and assert nothing of it is left running."""
    kill_processes(NOBODY_UID)
    # Wait on the children so they aren't counted as lingering (zombie) processes.
    for child in children:
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass

    survivors = _live_uid_procs(NOBODY_UID)
    assert survivors == [], (
        f"kill_processes left live uid={NOBODY_UID} processes: {survivors}"
    )


@pytest.mark.requires_root
@real_cohort_kill
def test_kill_processes_reaps_fork_and_die_chain() -> None:
    """``kill_processes`` leaves no live process behind, even against a
    fork-and-die chain that respawns faster than a single ``/proc`` sweep
    can observe."""
    launcher = _spawn_churn(NOBODY_UID)
    try:
        time.sleep(2)
        assert _live_uid_procs(NOBODY_UID), (
            "precondition failed: fork-and-die churn is not running"
        )

        _assert_reaped([launcher])
    finally:
        _force_cleanup(NOBODY_UID)


@pytest.mark.requires_root
@real_cohort_kill
def test_kill_processes_reaps_mutual_sigcont_ring(shared_dir: str) -> None:
    """The same, against a ring of processes that SIGCONT each other, which no
    sequence of SIGSTOPs ever gets fully stopped at once."""
    members = _spawn_ring(NOBODY_UID, shared_dir)
    try:
        time.sleep(1)
        assert len(_live_uid_procs(NOBODY_UID)) == _RING_SIZE, (
            "precondition failed: the SIGCONT ring is not running"
        )

        _assert_reaped(members)
    finally:
        _force_cleanup(NOBODY_UID)


@pytest.mark.requires_root
@real_cohort_kill
@pytest.mark.usefixtures("ballast")
def test_kill_processes_reaps_parentless_fork_relays(shared_dir: str) -> None:
    """The same, against relays with no long-lived process in them at all.

    The other two churn shapes are anchored by a loop that stays alive across
    sweeps, so a sweep always has something to find. Take the anchor away and
    every generation lives for microseconds, far less than a ``/proc`` walk
    takes, so sweep-based reaping never sees the cohort: it kills nothing,
    verifies clean, and returns success while the relays keep forking. Only the
    ``kill(-1)`` step reaps these."""
    marker_dir = Path(shared_dir)
    launcher = _spawn_relays(NOBODY_UID, marker_dir)
    try:
        started = _forking_relays(marker_dir, window=60)
        assert len(started) == _RELAYS, (
            f"precondition failed: only {len(started)}/{_RELAYS} relays started"
        )
        # Without the ballast in place the sweeps are quick enough to stand a
        # real chance of catching a relay, and the test stops discriminating.
        assert len(psutil.pids()) >= _BALLAST, (
            f"precondition failed: {len(psutil.pids())} processes, want >= {_BALLAST}"
        )

        kill_processes(NOBODY_UID)

        alive = _forking_relays(marker_dir, window=2)
        assert alive == [], (
            f"kill_processes returned success while relays {alive} were still running"
        )
        survivors = _live_uid_procs(NOBODY_UID)
        assert survivors == [], (
            f"kill_processes left live uid={NOBODY_UID} processes: {survivors}"
        )
    finally:
        launcher.wait(timeout=10)
        _force_cleanup(NOBODY_UID)


@final
class _FakeUids:
    def __init__(self, real: int):
        self.real = real


@final
class _FakeProc:
    """Minimal stand-in for a psutil.Process as seen by kill_processes."""

    def __init__(
        self,
        pid: int,
        uid: int | None,
        *,
        killable: bool = True,
        signal_exc: type[Exception] | None = None,
    ):
        self.pid = pid
        self.killable = killable
        self.signal_exc = signal_exc
        self.status = "running"
        self.kill_calls = 0
        # ``None`` models psutil.as_dict setting the field to its ad_value when
        # reading uids raises AccessDenied/ZombieProcess.
        self.info: dict[str, Any] = {
            "uids": _FakeUids(uid) if uid is not None else None
        }

    def kill(self) -> None:
        self.kill_calls += 1
        if self.signal_exc is not None:
            self.status = psutil.STATUS_ZOMBIE
            raise self.signal_exc(self.pid)
        # An unkillable process models uninterruptible sleep: the signal is
        # accepted but never delivered, so it stays live.
        if self.killable:
            self.status = psutil.STATUS_ZOMBIE


def _fake_process_iter(
    procs: list[_FakeProc],
    hidden_on_pass: dict[int, int] | None = None,
    abort_on_pass: set[int] | None = None,
    abort_after: int | None = None,
):
    """Yield ``procs`` on every sweep, except that a pid listed in
    ``hidden_on_pass`` is skipped on that (1-based) sweep — simulating a
    fork-and-die successor whose pid slot the ``/proc`` walk already passed.

    On a sweep listed in ``abort_on_pass``, and on every sweep past
    ``abort_after``, the walk dies after the first process with
    ``FileNotFoundError``, as psutil does when a ``/proc`` entry vanishes
    mid-read; the rest of ``procs`` is never reached."""
    pass_count = 0

    def process_iter(_attrs: list[str]):
        nonlocal pass_count
        pass_count += 1
        aborts = pass_count in (abort_on_pass or set()) or (
            abort_after is not None and pass_count > abort_after
        )
        for index, proc in enumerate(procs):
            if aborts and index:
                raise FileNotFoundError(f"/proc/{proc.pid}/stat")
            if hidden_on_pass and hidden_on_pass.get(proc.pid) == pass_count:
                continue
            proc.info["status"] = proc.status
            yield proc
        if aborts:
            raise FileNotFoundError("/proc")

    return process_iter


def test_kill_processes_kills_only_the_matching_cohort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every process owned by the uid is SIGKILLed, and nothing else is."""
    procs = [_FakeProc(pid, uid=1000) for pid in (10, 11, 12)]
    other = _FakeProc(99, uid=0)
    monkeypatch.setattr(
        "karotte.process_utils.psutil.process_iter", _fake_process_iter([*procs, other])
    )

    kill_processes(1000, pass_interval=0, deadline=5)

    for proc in procs:
        assert proc.kill_calls == 1
    assert other.kill_calls == 0


class TestKillProcessesKillsTheStudentCgroup:
    """Tasks reap through plain ``kill_processes``, so the atomic group kill
    has to live on that path, not only on the confinement object."""

    def test_the_group_dies_before_the_sweep_walks_proc(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from karotte import cgroups
        from karotte.cgroups import (
            _V2StudentCgroup,  # pyright: ignore[reportPrivateUsage]
        )

        order: list[str] = []
        group = _V2StudentCgroup(path=tmp_path, join_paths=[tmp_path])

        def group_kill() -> int:
            order.append("group")
            return 0

        monkeypatch.setattr(group, "kill_all", group_kill)
        monkeypatch.setitem(
            cgroups._student_groups,  # pyright: ignore[reportPrivateUsage]
            1000,
            group,
        )

        proc = _FakeProc(10, uid=1000)
        fake_iter = _fake_process_iter([proc])

        def sweeping_iter(attrs: list[str]):
            order.append("sweep")
            return fake_iter(attrs)

        monkeypatch.setattr("karotte.process_utils.psutil.process_iter", sweeping_iter)

        kill_processes(1000, pass_interval=0, deadline=5)

        assert order[0] == "group"
        assert "sweep" in order
        assert proc.kill_calls == 1

    def test_another_uids_group_is_left_alone(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from karotte import cgroups
        from karotte.cgroups import (
            _V2StudentCgroup,  # pyright: ignore[reportPrivateUsage]
        )

        group = _V2StudentCgroup(path=tmp_path, join_paths=[tmp_path])

        def boom() -> int:
            raise AssertionError("killed a group the uid does not own")

        monkeypatch.setattr(group, "kill_all", boom)
        monkeypatch.setitem(
            cgroups._student_groups,  # pyright: ignore[reportPrivateUsage]
            2000,
            group,
        )
        monkeypatch.setattr(
            "karotte.process_utils.psutil.process_iter", _fake_process_iter([])
        )

        kill_processes(1000, pass_interval=0, deadline=5)


def test_kill_processes_catches_process_hidden_from_one_sweep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact fork-and-die evasion: a sweep that sees only a zombie while
    the live successor is missing from that sweep's ``/proc`` snapshot. One
    false-clean sweep must not end the reaping."""
    zombie = _FakeProc(10, uid=1000)
    zombie.status = psutil.STATUS_ZOMBIE
    successor = _FakeProc(11, uid=1000)
    monkeypatch.setattr(
        "karotte.process_utils.psutil.process_iter",
        _fake_process_iter([zombie, successor], hidden_on_pass={successor.pid: 1}),
    )

    kill_processes(1000, pass_interval=0, deadline=5)

    assert successor.kill_calls == 1
    assert zombie.kill_calls == 0


def test_kill_processes_raises_when_cohort_cannot_be_reaped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A process that survives SIGKILL (uninterruptible sleep) must fail loudly,
    after a best-effort SIGKILL."""
    _as_euid(monkeypatch, 0)
    stuck = _FakeProc(10, uid=1000, killable=False)
    monkeypatch.setattr(
        "karotte.process_utils.psutil.process_iter", _fake_process_iter([stuck])
    )

    with pytest.raises(RuntimeError, match="1000"):
        kill_processes(1000, pass_interval=0, deadline=0.2)
    assert stuck.kill_calls >= 1


def test_kill_processes_recovers_from_aborted_proc_walk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """psutil re-raises the raw ``FileNotFoundError`` when a ``/proc`` entry
    vanishes mid-read, which ends the sweep at the ``for`` statement — outside
    the loop body's handlers. It must not abort the reap and let the processes
    the walk never reached survive into grading."""
    early = _FakeProc(10, uid=1000)
    unreached = _FakeProc(11, uid=1000)
    monkeypatch.setattr(
        "karotte.process_utils.psutil.process_iter",
        _fake_process_iter([early, unreached], abort_on_pass={1}),
    )

    kill_processes(1000, pass_interval=0, deadline=5)

    assert early.kill_calls == 1
    assert unreached.kill_calls == 1


def test_kill_processes_does_not_treat_aborted_walk_as_clean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sweep that died part-way saw only some of ``/proc``, so it says nothing
    about what is left; counting it as clean would let the cohort be declared
    reaped while unseen processes are still running."""
    _as_euid(monkeypatch, 0)
    seen = _FakeProc(10, uid=1000)
    never_seen = _FakeProc(11, uid=1000)
    monkeypatch.setattr(
        "karotte.process_utils.psutil.process_iter",
        _fake_process_iter([seen, never_seen], abort_after=0),
    )

    with pytest.raises(RuntimeError, match="1000"):
        kill_processes(1000, pass_interval=0, deadline=0.2)
    assert never_seen.kill_calls == 0


def test_kill_processes_does_not_declare_success_on_aborted_verification_sweep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The verification sweep after the kill sweep can abort too, and then it
    reports an empty survivor list only because it never got that far. Treating
    that as proof the cohort is gone would hand grading a live student process.

    The first walk completes, so the kill sweep signals both; ``stuck`` ignores
    the SIGKILL, and every walk after that aborts before reaching it."""
    _as_euid(monkeypatch, 0)
    reached = _FakeProc(10, uid=1000)
    stuck = _FakeProc(11, uid=1000, killable=False)
    monkeypatch.setattr(
        "karotte.process_utils.psutil.process_iter",
        _fake_process_iter([reached, stuck], abort_after=1),
    )

    with pytest.raises(RuntimeError, match="1000"):
        kill_processes(1000, pass_interval=0, deadline=0.2)

    assert stuck.status != psutil.STATUS_ZOMBIE


def test_kill_sweep_retries_a_truncated_walk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A kill sweep whose walk is cut short never reached the tail of the
    cohort. ``kill_processes`` re-verifies after the in-loop sweeps, but the
    final best-effort sweep before it gives up has no such backstop, so the
    sweep itself has to walk again until one walk completes."""
    early = _FakeProc(10, uid=1000)
    late = _FakeProc(11, uid=1000)
    monkeypatch.setattr(
        "karotte.process_utils.psutil.process_iter",
        _fake_process_iter([early, late], abort_on_pass={1}),
    )

    killed = _kill_sweep(1000)

    assert killed == [early.pid, late.pid]
    assert late.kill_calls == 1


def test_kill_processes_no_matching_processes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With nothing to reap, it completes without signalling anyone."""
    other = _FakeProc(99, uid=0)
    monkeypatch.setattr(
        "karotte.process_utils.psutil.process_iter", _fake_process_iter([other])
    )

    kill_processes(1000, pass_interval=0, deadline=5)

    assert other.kill_calls == 0


def test_kill_processes_skips_procs_with_unreadable_uids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A process whose ``uids`` came back as ``None`` (AccessDenied/ZombieProcess
    during the /proc read) is skipped, not dereferenced into an AttributeError
    that would abort the whole reap."""
    blind = _FakeProc(10, uid=None)
    target = _FakeProc(11, uid=1000)
    monkeypatch.setattr(
        "karotte.process_utils.psutil.process_iter",
        _fake_process_iter([blind, target]),
    )

    kill_processes(1000, pass_interval=0, deadline=5)

    assert blind.kill_calls == 0
    assert target.kill_calls == 1


@pytest.mark.parametrize("exc", [psutil.AccessDenied, psutil.ZombieProcess])
def test_kill_processes_tolerates_signal_race(
    monkeypatch: pytest.MonkeyPatch, exc: type[Exception]
) -> None:
    """A process that turns into a zombie between the sweep read and the signal
    (kill raising AccessDenied/ZombieProcess) must not crash the reap."""
    racer = _FakeProc(10, uid=1000, signal_exc=exc)
    monkeypatch.setattr(
        "karotte.process_utils.psutil.process_iter", _fake_process_iter([racer])
    )

    kill_processes(1000, pass_interval=0, deadline=5)

    assert racer.kill_calls >= 1


@final
class _Exited(Exception):
    """Stands in for the helper's ``os._exit``, which a test cannot survive."""

    def __init__(self, code: int):
        super().__init__(code)
        self.code = code


def _trap_helper_syscalls(
    monkeypatch: pytest.MonkeyPatch,
    *,
    setuid_exc: Exception | None = None,
    kill_exc: Exception | None = None,
) -> list[tuple[int, int]]:
    """Run the helper body in-process: record its ``kill`` calls and turn its
    ``os._exit`` into an exception so the test survives it."""
    kills: list[tuple[int, int]] = []

    def fake_kill(pid: int, sig: int) -> None:
        kills.append((pid, sig))
        if kill_exc is not None:
            raise kill_exc

    def fake_setuid(_uid: int) -> None:
        if setuid_exc is not None:
            raise setuid_exc

    def fake_exit(code: int) -> NoReturn:
        raise _Exited(code)

    def fake_setgroups(_groups: list[int]) -> None:
        return

    monkeypatch.setattr(os, "setgroups", fake_setgroups)
    monkeypatch.setattr(os, "setuid", fake_setuid)
    monkeypatch.setattr(os, "kill", fake_kill)
    monkeypatch.setattr(os, "_exit", fake_exit)
    return kills


@pytest.fixture
def devnull_fd() -> Iterator[int]:
    """A writable fd to stand in for the helper's confirmation pipe."""
    fd = os.open(os.devnull, os.O_WRONLY)
    try:
        yield fd
    finally:
        os.close(fd)


def test_cohort_kill_helper_repeats_kill_all(
    monkeypatch: pytest.MonkeyPatch, devnull_fd: int
) -> None:
    """gVisor's ``kill(-1)`` signals a snapshot of the thread-group list taken
    before the lock is dropped, so a process forked after the snapshot outlives
    the call — unlike on Linux, where the task-list walk holds ``tasklist_lock``
    throughout. Repeating the call is what closes that gap, so one round is not
    enough."""
    kills = _trap_helper_syscalls(monkeypatch)

    with pytest.raises(_Exited) as exit_info:
        _become_uid_and_kill_all(1000, devnull_fd)

    assert exit_info.value.code == 0
    assert kills == [(-1, signal.SIGKILL)] * _COHORT_KILL_ROUNDS
    assert _COHORT_KILL_ROUNDS > 1


def test_cohort_kill_helper_stops_once_the_cohort_is_gone(
    monkeypatch: pytest.MonkeyPatch, devnull_fd: int
) -> None:
    """ESRCH means nothing owned by the uid is left to signal, so the remaining
    rounds are pointless."""
    kills = _trap_helper_syscalls(monkeypatch, kill_exc=ProcessLookupError())

    with pytest.raises(_Exited) as exit_info:
        _become_uid_and_kill_all(1000, devnull_fd)

    assert exit_info.value.code == 0
    assert kills == [(-1, signal.SIGKILL)]


def test_cohort_kill_helper_reports_a_failed_privilege_drop(
    monkeypatch: pytest.MonkeyPatch, devnull_fd: int
) -> None:
    """A helper that could not become the uid never signalled the cohort, so it
    has to exit non-zero rather than let the parent assume the cohort is dead."""
    kills = _trap_helper_syscalls(monkeypatch, setuid_exc=PermissionError())

    with pytest.raises(_Exited) as exit_info:
        _become_uid_and_kill_all(1000, devnull_fd)

    assert exit_info.value.code == _HELPER_FAILED
    assert kills == []


@pytest.fixture
def high_fd_pipe() -> Iterator[tuple[int, int]]:
    """A pipe whose fds sit past ``FD_SETSIZE``, by padding with open fds first."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    want = _FD_SETSIZE + 64
    if hard != resource.RLIM_INFINITY and hard < want:
        pytest.skip(f"RLIMIT_NOFILE hard limit {hard} is below {want}")
    resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))
    padding: list[int] = []
    pipe: tuple[int, int] | None = None
    try:
        while not padding or padding[-1] < _FD_SETSIZE:
            padding.append(os.open(os.devnull, os.O_RDONLY))
        pipe = os.pipe()
        yield pipe
    finally:
        for fd in ([*pipe] if pipe else []) + padding:
            try:
                os.close(fd)
            except OSError:
                pass
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))


def test_await_helper_reads_a_confirmation_on_a_high_numbered_fd(
    high_fd_pipe: tuple[int, int],
) -> None:
    """A grader holding a thousand fds pushes the helper's pipe past
    ``FD_SETSIZE``, which ``select(2)`` cannot represent: it raises
    ``ValueError``, which is not an ``OSError`` and so escapes the handler here,
    replacing the ``RuntimeError`` grading expects and leaking the helper
    unreaped. ``poll(2)`` has no such bound."""
    read_fd, write_fd = high_fd_pipe
    assert read_fd >= _FD_SETSIZE, f"pipe landed at fd {read_fd}, wanted a high one"
    _ = os.write(write_fd, _HELPER_DONE)

    # Our own pid is not our child, so the reap short-circuits on ECHILD.
    assert _await_helper(os.getpid(), 1000, read_fd) is True


def _cohort_kill_returning(
    monkeypatch: pytest.MonkeyPatch, results: list[bool]
) -> list[int]:
    """Stub the in-cohort kill to report ``results`` in order (repeating the
    last), and record how many times it was called."""
    calls: list[int] = []

    def _ran(uid: int) -> bool:
        calls.append(uid)
        return results[min(len(calls) - 1, len(results) - 1)]

    monkeypatch.setattr("karotte.process_utils._kill_cohort_from_within", _ran)
    return calls


def _as_euid(monkeypatch: pytest.MonkeyPatch, euid: int) -> None:
    """Run the reap as ``euid``, which decides whether it could have run the
    in-cohort kill on the uid it is reaping at all."""
    monkeypatch.setattr("karotte.process_utils.os.geteuid", lambda: euid)


def test_kill_processes_does_not_trust_a_clean_sweep_without_the_in_cohort_kill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pid-exhaustion hole. A cohort that has saturated ``pids.max`` leaves
    no free pid to fork the helper with, so the in-cohort kill never runs -- and
    a relay that retries its own failed forks rides that out while every sweep
    comes back clean. Reading that sweep as success hands grading a live student
    process, which is exactly what it did before this was checked."""
    _as_euid(monkeypatch, 0)
    _ = _cohort_kill_returning(monkeypatch, [False])
    monkeypatch.setattr(
        "karotte.process_utils.psutil.process_iter", _fake_process_iter([])
    )

    with pytest.raises(RuntimeError, match="in-cohort kill never ran"):
        kill_processes(1000, pass_interval=0, deadline=0.2)


class TestKillProcessesFailsAsMisbehavior:
    """A root grader that cannot get the cohort down is looking at a student
    holding it open — surviving SIGKILL, or saturating the pid table the
    in-cohort kill needs a fork from. That scores 0 rather than erroring the
    run as an infra failure."""

    def test_an_unverifiable_pass_raises_student_misbehavior(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _as_euid(monkeypatch, 0)
        _ = _cohort_kill_returning(monkeypatch, [False])
        monkeypatch.setattr(
            "karotte.process_utils.psutil.process_iter", _fake_process_iter([])
        )

        with pytest.raises(
            StudentMisbehaviorError, match="in-cohort kill never ran"
        ) as raised:
            kill_processes(1000, pass_interval=0, deadline=0.2)

        assert isinstance(raised.value, RuntimeError)

    def test_a_surviving_process_raises_student_misbehavior(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _as_euid(monkeypatch, 0)
        stuck = _FakeProc(10, uid=1000, killable=False)
        monkeypatch.setattr(
            "karotte.process_utils.psutil.process_iter", _fake_process_iter([stuck])
        )

        with pytest.raises(StudentMisbehaviorError, match="still alive"):
            kill_processes(1000, pass_interval=0, deadline=0.2)


class TestKillProcessesBlamesTheStudentOnlyWhenItCould:
    """A reap that could never have run is a broken grader: uid 0, which
    ``kill(-1)`` is refused for, or a non-root grader, which cannot drop into
    the cohort to issue it. No student put it there and no retry fixes it, so it
    must fail loudly instead of scoring the student 0."""

    def test_uid_zero_is_not_misbehavior(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _as_euid(monkeypatch, 0)
        _ = _cohort_kill_returning(monkeypatch, [False])
        monkeypatch.setattr(
            "karotte.process_utils.psutil.process_iter", _fake_process_iter([])
        )

        with pytest.raises(RuntimeError) as raised:
            kill_processes(0, pass_interval=0, deadline=0.2)

        assert not isinstance(raised.value, StudentMisbehaviorError)

    def test_a_non_root_grader_is_not_misbehavior(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _as_euid(monkeypatch, 501)
        _ = _cohort_kill_returning(monkeypatch, [False])
        monkeypatch.setattr(
            "karotte.process_utils.psutil.process_iter", _fake_process_iter([])
        )

        with pytest.raises(RuntimeError) as raised:
            kill_processes(1000, pass_interval=0, deadline=0.2)

        assert not isinstance(raised.value, StudentMisbehaviorError)

    def test_being_the_uid_already_is_not_enough(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Running as the uid buys no way to issue the in-cohort kill: dropping
        into it needs CAP_SETGID for the helper's ``setgroups``, and a helper
        forked from a grader that is already the uid would SIGKILL the grader
        along with the cohort."""
        _as_euid(monkeypatch, 1000)
        _ = _cohort_kill_returning(monkeypatch, [False])
        monkeypatch.setattr(
            "karotte.process_utils.psutil.process_iter", _fake_process_iter([])
        )

        with pytest.raises(RuntimeError) as raised:
            kill_processes(1000, pass_interval=0, deadline=0.2)

        assert not isinstance(raised.value, StudentMisbehaviorError)

    def test_it_gives_up_on_the_first_pass(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No retry fixes it, so it must not sit out the deadline first."""
        _as_euid(monkeypatch, 501)
        calls = _cohort_kill_returning(monkeypatch, [False])
        monkeypatch.setattr(
            "karotte.process_utils.psutil.process_iter", _fake_process_iter([])
        )

        with pytest.raises(RuntimeError):
            kill_processes(1000, pass_interval=0, deadline=30)

        assert len(calls) == 1

    def test_a_pass_that_ran_the_kill_settles_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The euid check predicts whether the in-cohort kill can run; a pass
        that ran it knows. A later pass that loses the fork is then the pid
        exhaustion the student caused, not a grader that never had one."""
        _as_euid(monkeypatch, 501)
        _ = _cohort_kill_returning(monkeypatch, [True, False])
        stuck = _FakeProc(10, uid=1000, killable=False)
        monkeypatch.setattr(
            "karotte.process_utils.psutil.process_iter", _fake_process_iter([stuck])
        )

        with pytest.raises(StudentMisbehaviorError):
            kill_processes(1000, pass_interval=0, deadline=0.2)


def test_kill_processes_retries_until_the_in_cohort_kill_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Losing the helper fork is usually transient -- the kill sweep frees the
    cohort's own pids, so a later pass gets one. It must keep going rather than
    fail on the first unverifiable pass."""
    _as_euid(monkeypatch, 0)
    calls = _cohort_kill_returning(monkeypatch, [False, False, True])
    monkeypatch.setattr(
        "karotte.process_utils.psutil.process_iter", _fake_process_iter([])
    )

    kill_processes(1000, pass_interval=0, deadline=5)

    assert len(calls) == 3


def test_kill_cohort_from_within_refuses_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``kill(-1)`` as root would signal every process in the container, this
    one included, so UID 0 must never reach the helper."""
    forks: list[None] = []
    monkeypatch.setattr(os, "fork", lambda: forks.append(None))

    assert _kill_cohort_from_within(0) is False
    assert forks == []


def test_kill_cohort_from_within_refuses_a_non_root_grader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The helper's ``setgroups`` needs CAP_SETGID, so a non-root fork can only
    fail -- and if it did somehow drop into the cohort, its ``kill(-1)`` would
    signal the grader that forked it."""
    _as_euid(monkeypatch, 1000)
    forks: list[None] = []
    monkeypatch.setattr(os, "fork", lambda: forks.append(None))

    assert _kill_cohort_from_within(1000) is False
    assert forks == []


def _fake_ns_links(monkeypatch: pytest.MonkeyPatch, links: dict[int, str]) -> None:
    """Point ``/proc/<pid>/ns/pid`` reads at ``links``; our own ns is ``pid:[1]``."""

    def fake_readlink(path: str) -> str:
        if path == "/proc/self/ns/pid":
            return "pid:[1]"
        pid = int(path.split("/")[2])
        if pid not in links:
            raise FileNotFoundError(path)
        return links[pid]

    monkeypatch.setattr("karotte.process_utils.os.readlink", fake_readlink)


def test_kill_pidns_cohort_kills_only_processes_in_a_foreign_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SIGKILLing a PID namespace's init drops every process inside it at once,
    which is the only reap a fork-and-die chain cannot evade. A process sharing
    our own namespace has no such guarantee behind it and is left to the sweep."""
    inside = _FakeProc(10, uid=1000)
    alongside = _FakeProc(11, uid=1000)
    monkeypatch.setattr(
        "karotte.process_utils.psutil.process_iter",
        _fake_process_iter([inside, alongside]),
    )
    _fake_ns_links(monkeypatch, {inside.pid: "pid:[42]", alongside.pid: "pid:[1]"})

    killed = _kill_pidns_cohort(1000)

    assert killed == [inside.pid]
    assert inside.kill_calls == 1
    assert alongside.kill_calls == 0


def test_kill_pidns_cohort_tolerates_a_vanished_proc_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The namespace link is read one process later than the walk that listed
    it, so the entry can be gone by then. That is a dead process, not a reason
    to abandon the rest of the cohort."""
    gone = _FakeProc(10, uid=1000)
    present = _FakeProc(11, uid=1000)
    monkeypatch.setattr(
        "karotte.process_utils.psutil.process_iter",
        _fake_process_iter([gone, present]),
    )
    _fake_ns_links(monkeypatch, {present.pid: "pid:[42]"})

    killed = _kill_pidns_cohort(1000)

    assert killed == [present.pid]


def test_kill_processes_reaps_namespaces_before_verifying(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The namespace kill has to run inside the reap loop, not merely be
    available: it is the step that removes fork chains, and the sweep that
    follows is only a check on it."""
    inside = _FakeProc(10, uid=1000)
    monkeypatch.setattr(
        "karotte.process_utils.psutil.process_iter", _fake_process_iter([inside])
    )
    _fake_ns_links(monkeypatch, {inside.pid: "pid:[42]"})

    kill_processes(1000, pass_interval=0, deadline=5)

    assert inside.kill_calls >= 1
