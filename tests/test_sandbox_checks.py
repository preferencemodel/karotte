"""Sandbox checks: the student's limits and isolation, measured against the real
kernel. Each check is bounded by size, count and time, and skips where this
sandbox has no mechanism for it.

Run as root inside the sandbox under test (``just test-root``). The student is
an unused uid, so nothing here touches a real user's processes.
"""

import errno
import os
import select
import shutil
import signal
import stat
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import psutil
import pytest

from karotte import subprocess as student_subprocess
from karotte.cgroups import (
    HARNESS_LEAF,
    own_cgroup_dir,
    read_mounts,
    unregister_student_cgroup,
)
from karotte.confinement import (
    CgroupConfinement,
    Contract,
    build_confinement,
    get_confinement,
)
from karotte.confinement_check import (
    _unused_uid,  # pyright: ignore[reportPrivateUsage]
    gather,
)
from karotte.container import mount_points
from karotte.file_quota import mount_file_quota, unmount_file_quota
from karotte.process_utils import kill_processes
from karotte.subprocess import student_session_command
from karotte.trusted_bin import trusted_binary

pytestmark = pytest.mark.requires_root

MIB = 1 << 20
_TIMEOUT_SECONDS = 30


@pytest.fixture
def uid() -> int:
    return _unused_uid(None)


@pytest.fixture
def cgroup(uid: int) -> Iterator[CgroupConfinement]:
    confinement = build_confinement(uid=uid)
    if not isinstance(confinement, CgroupConfinement):
        pytest.skip("no writable cgroup hierarchy")
    try:
        yield confinement
    finally:
        _ = confinement.group.kill_all()
        _ = confinement.limit_memory(None)
        _ = confinement.limit_processes(None)
        confinement.group.destroy()
        unregister_student_cgroup(uid)


@pytest.fixture
def world_readable_scratch() -> Iterator[Path]:
    """A directory the student can reach. Not pytest's tmp_path (under a 0700
    root directory) and not $TMPDIR, which a dev shell may point at a private
    directory such as /tmp/nix-shell.XXXX."""
    scratch = Path(tempfile.mkdtemp(prefix="karotte-sandbox-check-", dir="/tmp"))
    scratch.chmod(0o755)
    closed = [p for p in scratch.parents if not p.stat().st_mode & stat.S_IXOTH]
    if closed:
        shutil.rmtree(scratch, ignore_errors=True)
        pytest.skip(f"{closed[0]} is not traversable by the student")
    try:
        yield scratch
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _as_student(
    uid: int,
    work: Callable[[], str],
    *,
    join: Callable[[], None] | None = None,
) -> tuple[int, str]:
    """Run ``work`` in a forked child that joined the group (as root) and then
    dropped to ``uid``. Returns the child's wait status and whatever it
    reported before it exited or was killed; a child still running at the
    timeout is killed."""
    read_fd, write_fd = os.pipe()
    child = os.fork()
    if child == 0:
        os.close(read_fd)
        try:
            if join is not None:
                join()
            os.setgroups([])
            os.setgid(uid)
            os.setuid(uid)

            def report(text: str) -> None:
                _ = os.write(write_fd, text.encode())

            report(work())
        except BaseException:
            os._exit(1)
        os._exit(0)
    os.close(write_fd)
    chunks: list[bytes] = []
    deadline = time.monotonic() + _TIMEOUT_SECONDS
    with os.fdopen(read_fd, "rb", buffering=0) as pipe:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                os.kill(child, signal.SIGKILL)
                break
            ready, _, _ = select.select([pipe], [], [], remaining)
            if not ready:
                continue
            chunk = pipe.read(4096)
            if not chunk:
                break
            chunks.append(chunk)
    _, status = os.waitpid(child, 0)
    return status, b"".join(chunks).decode()


def test_memory_past_the_limit_is_refused(uid: int, cgroup: CgroupConfinement) -> None:
    limit = 32 * MIB
    if cgroup.limit_memory(limit) is not Contract.PREVENTED:
        pytest.skip("the cgroup does not take a memory limit")

    def allocate_twice_the_limit() -> str:
        held: list[bytearray] = []
        chunk = 4 * MIB
        while len(held) * chunk < 2 * limit:
            # bytearray zero-fills, so every page is touched and charged.
            held.append(bytearray(chunk))
        return str(len(held) * chunk)

    status, reported = _as_student(
        uid, allocate_twice_the_limit, join=cgroup.group.join_self
    )

    assert os.WIFSIGNALED(status) and os.WTERMSIG(status) == signal.SIGKILL, (
        f"the student allocated {reported or '?'} bytes against a {limit} byte limit"
    )
    assert reported == ""


def test_processes_past_the_limit_are_refused(
    uid: int, cgroup: CgroupConfinement
) -> None:
    limit = 8
    if cgroup.limit_processes(limit) is not Contract.PREVENTED:
        pytest.skip("the cgroup does not take a process limit")

    def fork_past_the_limit() -> str:
        children: list[int] = []
        try:
            for _ in range(4 * limit):
                try:
                    pid = os.fork()
                except OSError:
                    break
                if pid == 0:
                    time.sleep(_TIMEOUT_SECONDS)
                    os._exit(0)
                children.append(pid)
            return str(len(children))
        finally:
            for pid in children:
                os.kill(pid, signal.SIGKILL)
                _ = os.waitpid(pid, 0)

    status, reported = _as_student(
        uid, fork_past_the_limit, join=cgroup.group.join_self
    )

    assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0
    # The forking process itself counts against the limit.
    assert int(reported) < limit


@pytest.fixture
def quota(world_readable_scratch: Path) -> Iterator[Path]:
    """A small kernel quota over a student-writable directory, as a run's
    first file limit mounts it."""
    try:
        _ = trusted_binary("mkfs.ext4")
    except FileNotFoundError:
        pytest.skip("mkfs.ext4 is not installed")
    target = world_readable_scratch / "workdir"
    target.mkdir(mode=0o777)
    target.chmod(0o777)
    quota_dir = world_readable_scratch / "quota"
    mounted = mount_file_quota((target,), 8 * MIB, 64, quota_dir=quota_dir)
    if mounted is None:
        pytest.skip("this sandbox cannot loop-mount a quota")
    try:
        yield target
    finally:
        assert unmount_file_quota(mounted, quota_dir)


def test_disk_bytes_past_the_quota_are_refused(uid: int, quota: Path) -> None:
    def write_twice_the_quota() -> str:
        written = 0
        try:
            with open(quota / "fill", "wb") as f:
                while written < 16 * MIB:
                    written += f.write(b"\0" * MIB)
                    os.fsync(f.fileno())
        except OSError as exc:
            return f"{written} {exc.errno}"
        return f"{written} 0"

    status, reported = _as_student(uid, write_twice_the_quota)

    assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0
    written, error = (int(part) for part in reported.split())
    assert written < 8 * MIB
    assert error in (errno.ENOSPC, errno.EDQUOT)


def test_files_past_the_inode_quota_are_refused(uid: int, quota: Path) -> None:
    def create_many_files() -> str:
        created = 0
        try:
            for index in range(1000):
                (quota / f"f{index}").touch()
                created += 1
        except OSError as exc:
            return f"{created} {exc.errno}"
        return f"{created} 0"

    status, reported = _as_student(uid, create_many_files)

    assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0
    created, error = (int(part) for part in reported.split())
    assert created < 1000
    assert error in (errno.ENOSPC, errno.EDQUOT)


_CHAIN = """\
n=0
beat() { n=$((n + 1)); echo "$n" > "$1/beat"; }
link() { beat "$1"; sleep 0.05; ( link "$1" & ); exit 0; }
( link "$1" & )
sleep {timeout}
""".replace("{timeout}", str(_TIMEOUT_SECONDS))


def _beat(marker: Path) -> str | None:
    try:
        return marker.read_text()
    except OSError:
        return None


def test_a_fork_and_die_chain_in_a_session_is_reaped(
    uid: int, world_readable_scratch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every link lives about 50 ms before handing over to a fresh process.
    Started the way a tool starts a student session, so it lands in whatever
    group and PID namespace this sandbox gives sessions."""
    monkeypatch.setenv("KAROTTE_CONTAINERIZED", "1")
    monkeypatch.setenv("KAROTTE_DEMOTE_ID", str(uid))
    student_subprocess.reset_pid_namespace_probe()
    marker_dir = world_readable_scratch / "chain"
    marker_dir.mkdir(mode=0o777)
    marker_dir.chmod(0o777)
    marker = marker_dir / "beat"

    argv, preexec = student_session_command(
        [trusted_binary("sh"), "-c", _CHAIN, "chain", str(marker_dir)],
        disable_networking=False,
    )
    session = subprocess.Popen(
        argv, preexec_fn=preexec, cwd="/", stdin=subprocess.DEVNULL
    )
    try:
        time.sleep(1)
        first = _beat(marker)
        time.sleep(0.5)
        assert first is not None and _beat(marker) != first, (
            "precondition failed: the chain is not running"
        )

        kill_processes(uid, deadline=_TIMEOUT_SECONDS)

        _ = session.wait(timeout=_TIMEOUT_SECONDS)
        stopped = _beat(marker)
        time.sleep(0.5)
        assert _beat(marker) == stopped, "the chain kept running after the reap"
        assert [p.pid for p in psutil.process_iter() if _owned_by(p, uid)] == []
    finally:
        if session.poll() is None:
            session.kill()
            _ = session.wait()
        confinement = get_confinement(uid)
        if isinstance(confinement, CgroupConfinement):
            _ = confinement.group.kill_all()
            confinement.group.destroy()
            unregister_student_cgroup(uid)


def _owned_by(process: psutil.Process, uid: int) -> bool:
    """A live process of ``uid``. Zombies don't count, as in karotte's own
    sweep: they are dead, and where nothing reaps orphans (a container whose
    pid 1 is a shell) they stay listed."""
    try:
        return process.uids().real == uid and process.status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


def test_the_student_cannot_read_or_signal_the_harness(uid: int) -> None:
    harness = os.getpid()

    def probe() -> str:
        reached: list[str] = []
        for path in ("environ", "mem", "root/"):
            try:
                if path.endswith("/"):
                    _ = os.listdir(f"/proc/{harness}/{path}")
                else:
                    with open(f"/proc/{harness}/{path}", "rb") as f:
                        _ = f.read(1)
                reached.append(path)
            except OSError:
                pass
        try:
            os.kill(harness, 0)
            reached.append("signal")
        except OSError:
            pass
        return ",".join(reached)

    status, reported = _as_student(uid, probe)

    assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0
    assert reported == ""


def test_a_transcript_under_a_root_only_parent_is_out_of_reach(uid: int) -> None:
    out = Path(tempfile.mkdtemp(prefix="karotte-out-"))
    try:
        out.chmod(0o700)
        transcript = out / "transcript.json"
        transcript.write_text('{"run_id": "run"}')
        transcript.chmod(0o644)
        (out / "artifacts").mkdir(mode=0o777)

        def probe() -> str:
            reached: list[str] = []
            attempts: list[tuple[str, Callable[[], object]]] = [
                ("list", lambda: os.listdir(out)),
                ("stat", lambda: transcript.stat()),
                ("read", lambda: transcript.read_text()),
                ("write", lambda: transcript.write_text("{}")),
                ("create", lambda: (out / "artifacts" / "x").write_text("x")),
            ]
            for name, attempt in attempts:
                try:
                    _ = attempt()
                    reached.append(name)
                except OSError:
                    pass
            return ",".join(reached)

        status, reported = _as_student(uid, probe)

        assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0
        assert reported == ""
        assert transcript.read_text() == '{"run_id": "run"}'
    finally:
        shutil.rmtree(out, ignore_errors=True)


def _firewall_state() -> list[str]:
    listings: list[str] = []
    for command in ("iptables", "ip6tables"):
        try:
            binary = trusted_binary(command)
        except FileNotFoundError:
            continue
        listings.append(
            subprocess.run([binary, "-S"], capture_output=True, text=True).stdout
        )
    return listings


def _cgroup_state() -> set[Path]:
    return {
        path
        for mount in read_mounts()
        if mount.is_cgroup2
        for path in own_cgroup_dir(mount.path).iterdir()
        if path.is_dir() and path.name != HARNESS_LEAF
    }


def test_the_confinement_check_leaves_nothing_behind(
    uid: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What ``karotte check confinement`` does must not change a later run
    in the same sandbox."""
    monkeypatch.setenv("KAROTTE_CONTAINERIZED", "1")
    monkeypatch.setenv("KAROTTE_DEMOTE_ID", str(uid))
    student_subprocess.reset_pid_namespace_probe()
    before = (_firewall_state(), _cgroup_state(), read_mounts(), mount_points())

    observations = gather(None)

    assert observations.session.error is None
    # Only the harness leaf may stay, which every run creates itself.
    assert all("karotte_harness" in item for item in observations.left_behind)
    assert (_firewall_state(), _cgroup_state(), read_mounts(), mount_points()) == before


def test_the_confinement_check_sees_the_session_group_and_the_firewall_hold(
    uid: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real student session lands in the student's group, and with the
    rules in place the probe uid reaches none of the canaries."""
    monkeypatch.setenv("KAROTTE_CONTAINERIZED", "1")
    monkeypatch.setenv("KAROTTE_DEMOTE_ID", str(uid))
    student_subprocess.reset_pid_namespace_probe()

    observations = gather(None)

    assert observations.session.error is None
    if observations.student_cgroup is not None:
        assert observations.session.cgroup == observations.student_cgroup
    if not observations.firewall_took:
        pytest.skip(f"the firewall did not take here: {observations.firewall_error}")
    assert observations.firewall_check_error is None
    assert observations.reachable_with_rules == []
