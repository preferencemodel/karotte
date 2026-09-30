"""Shared subprocess helpers for tool implementations.

Provides a privilege-demotion preexec_fn so that tool subprocesses
run as the student user when inside a container, plus helpers for
wrapping a command in namespaces via `unshare(1)`: a network namespace to deny
it network access, and a PID namespace so the whole session can be reaped
race-free. Demoted children also get their own IPC namespace, so SysV objects
never outlive a launch.
"""

import ctypes
import os
import pwd
import subprocess
from collections.abc import Callable, Mapping, Sequence
from importlib.metadata import entry_points

from loguru import logger

from karotte.container import demoted_uid_gid as demoted_uid_gid
from karotte.trusted_bin import trusted_binary as trusted_binary


def chdir_to_workdir() -> None:
    """Change to the student working directory when inside a container.

    Reads ``KAROTTE_WORKDIR`` from the environment.  Does nothing when the
    variable is not set (e.g. outside a container).
    """
    workdir = os.environ.get("KAROTTE_WORKDIR")
    if workdir is not None:
        os.chdir(workdir)


def _reenter_cwd() -> None:
    """Re-resolve the working directory by name in a forked child, before exec."""
    try:
        os.chdir(os.getcwd())
    except OSError:
        pass


def student_identity_env() -> dict[str, str]:
    """Env overrides that tell a demoted child who it is.

    ``setuid(2)`` changes the uid and nothing else, so a child demoted by
    :func:`make_demote_fn` (or by ``unshare --setuid``) keeps the environment of
    whoever launched it. That launcher is root, so ``HOME`` stays ``/root``,
    which the student cannot write: anything that caches under ``$HOME``
    fails until the student works around it.

    ``USER``/``LOGNAME`` matter for the same reason, and more so under gVisor,
    where the child sees the overflow uid (see
    :func:`wrap_to_disable_networking`) and so cannot look its own name up.

    Returns an empty dict when there is no demotion to compensate for. Falls
    back to ``KAROTTE_WORKDIR`` for ``HOME`` when the uid has no passwd entry, since
    that is where the student's files live either way.
    """
    uid_gid = demoted_uid_gid()
    if uid_gid is None:
        return {}

    try:
        pw = pwd.getpwuid(uid_gid)
    except KeyError:
        workdir = os.environ.get("KAROTTE_WORKDIR")
        if workdir is None:
            logger.warning(
                f"Demote uid {uid_gid} has no passwd entry and KAROTTE_WORKDIR is unset; leaving HOME pointed at the launcher's home."
            )
            return {}
        return {"HOME": workdir}

    return {"HOME": pw.pw_dir, "USER": pw.pw_name, "LOGNAME": pw.pw_name}


HARNESS_SECRET_ENTRY_POINT_GROUP = "karotte.harness_secret_env"


def harness_secret_env() -> frozenset[str]:
    """Env var names of harness credentials, as registered by installed packages.

    An explicit list, not a pattern over credential-shaped names: environments
    hand the student secrets on purpose (a dataset token, a scoped API key).
    """
    names: set[str] = set()
    for ep in entry_points(group=HARNESS_SECRET_ENTRY_POINT_GROUP):
        try:
            names.update(ep.load())
        except Exception as e:  # noqa: BLE001 - a broken plugin must not break karotte
            logger.warning("Ignoring harness secrets from {!r}: {}", ep.name, e)
    return frozenset(names)


def is_harness_secret_env(name: str) -> bool:
    """Whether ``name`` is an env var the student must not inherit."""
    return name in harness_secret_env()


def scrub_harness_secrets(env: Mapping[str, str]) -> dict[str, str]:
    """``env`` without the variables :func:`is_harness_secret_env` names."""
    secrets = harness_secret_env()
    dropped = sorted(k for k in env if k in secrets)
    if dropped:
        logger.info("Withholding {} from the student's environment", ", ".join(dropped))
    return {k: v for k, v in env.items() if k not in dropped}


def student_env() -> dict[str, str]:
    """The environment a process running as the student starts with.

    This process's environment minus the harness's secrets, with the identity
    overrides of :func:`student_identity_env` on top.
    """
    return scrub_harness_secrets(os.environ) | student_identity_env()


_ipcns_probed: bool = False
_ipcns_available: bool = False


def reset_ipc_namespace_probe() -> None:
    """Forget the cached probe result. For tests."""
    global _ipcns_probed, _ipcns_available
    _ipcns_probed = False
    _ipcns_available = False


def _try_unshare_ipc() -> None:
    """Enter a fresh IPC namespace, swallowing a refusal: this runs in a
    preexec_fn, where raising kills the launch."""
    try:
        os.unshare(os.CLONE_NEWIPC)
    except OSError:
        pass


def ipc_namespace_available() -> bool:
    """Whether demoted children can start in their own IPC namespace, so each
    launch begins with no SysV objects and they die with its last process.

    ``unshare(CLONE_NEWIPC)`` needs CAP_SYS_ADMIN, so it is probed once — in a
    throwaway child, because unsharing here would move this process itself.
    """
    global _ipcns_probed, _ipcns_available
    if _ipcns_probed:
        return _ipcns_available
    _ipcns_probed = True
    if not hasattr(os, "unshare"):
        return False
    try:
        result = subprocess.run(
            [trusted_binary("true")],
            preexec_fn=lambda: os.unshare(os.CLONE_NEWIPC),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
            check=False,
        )
        _ipcns_available = result.returncode == 0
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug(
            f"Could not unshare an IPC namespace for student processes ({exc}) — is CAP_SYS_ADMIN missing? SysV segments, queues and semaphores will be shared with the harness and survive across launches."
        )
        return False
    return _ipcns_available


CLONE_NEWNS = 0x00020000

MS_RDONLY = 1
MS_NOSUID = 2
MS_NODEV = 4
MS_NOEXEC = 8
MS_REMOUNT = 32
MS_BIND = 4096
MS_REC = 16384
MS_PRIVATE = 1 << 18

_libc: ctypes.CDLL | None = None


def _ensure_libc() -> None:
    """Load libc while still in the parent. `CDLL(None)` takes the symbols
    already in the process rather than opening a library, so there is nothing
    to dlopen in the forked child."""
    global _libc
    if _libc is None:
        _libc = ctypes.CDLL(None, use_errno=True)


def _mount(
    source: str, target: str, fstype: str | None, flags: int, data: str | None = None
) -> None:
    assert _libc is not None, "_ensure_libc must run in the parent first"
    encoded = [
        arg.encode() if arg is not None else None
        for arg in (source, target, fstype, data)
    ]
    if _libc.mount(encoded[0], encoded[1], encoded[2], flags, encoded[3]) != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, f"mount {source} on {target}: {os.strerror(errno)}")


def _isolate_mounts(
    ephemeral_dirs: Sequence[str],
    read_only_dirs: Sequence[str],
    covered_dirs: Sequence[str],
) -> None:
    """Enter a mount namespace where `ephemeral_dirs` are empty and throwaway,
    `read_only_dirs` cannot be written or executed from, and `covered_dirs` are
    gone: covered by an empty tmpfs that is read-only from the moment it is
    mounted.

    Runs in the forked child while it is still root: `CLONE_NEWNS` and
    `mount(2)` both want CAP_SYS_ADMIN, which the drop about to follow takes
    away. A failure raises, which kills the launch — carrying on would run
    with the directories this was meant to take away.

    The root is made private first, or every mount below propagates back out.
    A read-only directory is bound rather than covered, so root-owned data
    under it stays readable; `noexec` denies the other half of the same
    surface, a program left there for the caller to be talked into running.
    The bind is recursive so mounts underneath it survive.

    A cover is for a directory whose contents are the danger, so nothing under
    it survives — including mounts, which it shadows. Read-only at mount time
    rather than remounted afterwards, because an empty writable tmpfs is worse
    than what it replaced: a child that can map executable pages writes its own
    library into the cover and loads that. Read-only also makes the cover
    independent of everything else a preexec does, so it holds wherever in one
    it runs.
    """
    try:
        os.unshare(CLONE_NEWNS)
    except OSError as exc:
        raise OSError(
            exc.errno,
            f"could not unshare a mount namespace ({os.strerror(exc.errno or 0)}) — is CAP_SYS_ADMIN missing? Refusing to continue with the shared directories still in place.",
        ) from exc

    _mount("none", "/", None, MS_REC | MS_PRIVATE)

    for target in ephemeral_dirs:
        _mount("tmpfs", target, "tmpfs", MS_NOSUID | MS_NODEV)

    for target in read_only_dirs:
        _mount(target, target, None, MS_BIND | MS_REC)
        _mount(
            "none",
            target,
            None,
            MS_REMOUNT | MS_BIND | MS_RDONLY | MS_NOSUID | MS_NODEV | MS_NOEXEC,
        )

    for target in covered_dirs:
        _mount("tmpfs", target, "tmpfs", MS_RDONLY | MS_NOSUID | MS_NODEV | MS_NOEXEC)


PLATFORM_TOOLING_ENTRY_POINT_GROUP = "karotte.platform_tooling_dirs"


def platform_tooling_dirs() -> tuple[str, ...]:
    """Dirs where a platform mounts its own tooling into every container, as registered by installed packages.

    Only those that exist here, since covering a missing path fails the launch.
    """
    paths: set[str] = set()
    for ep in entry_points(group=PLATFORM_TOOLING_ENTRY_POINT_GROUP):
        try:
            paths.update(ep.load())
        except Exception as e:  # noqa: BLE001 - a broken plugin must not break karotte
            logger.warning("Ignoring platform tooling dirs from {!r}: {}", ep.name, e)
    return tuple(sorted(path for path in paths if os.path.isdir(path)))


def make_demote_fn(
    *chown_fds: int,
    uid_gid: int | None = None,
    ephemeral_dirs: Sequence[str] = (),
    read_only_dirs: Sequence[str] = (),
    covered_dirs: Sequence[str] = (),
) -> Callable[[], None] | None:
    """Return a callable that drops privileges, or None.

    The returned callable calls ``os.setgid()`` and ``os.setuid()`` in
    the forked child before exec. Returns ``None`` when there is nothing to
    demote to (see :func:`demoted_uid_gid`), which is safe to pass directly to
    ``asyncio.create_subprocess_exec``.

    ``uid_gid`` overrides who to drop to; the default is the student uid from
    ``KAROTTE_DEMOTE_ID``. It says who to become, not whether to demote: off the
    container the return value is ``None`` either way.

    ``chown_fds`` are child fds to hand to the demoted uid first. A pipe
    created by ``subprocess.PIPE`` belongs to this process, which is root in a
    container, and its inode is permission-checked like any other on open. So a
    demoted child can read and write the descriptors it inherited but cannot
    reopen them by path — ``/dev/stdin``, ``/dev/stdout``, ``/proc/self/fd/N`` —
    which is how programs reach their own
    standard streams. ``fchown`` runs in the forked child while it is still
    root, and settles it. By then the child's fds 0, 1 and 2 are already the
    ends this process gave it.

    Name only fds opened for this child. An fd merely inherited from whoever
    started us is not ours to give away, and neither is a shared device such as
    the ``/dev/null`` behind ``subprocess.DEVNULL``.

    Where the kernel allows it, the child also enters a fresh IPC namespace
    before the drop, so each launch starts with no SysV objects and its
    segments, queues and semaphores die with its last process.

    Naming any of ``ephemeral_dirs``, ``read_only_dirs`` or ``covered_dirs``
    additionally gives the child a private mount namespace where those hold —
    see :func:`_isolate_mounts`. Which directories those are is the caller's to
    say, since it is the caller who knows the image; for the ones a platform
    rather than the image puts there, see :func:`platform_tooling_dirs`.
    """
    default = demoted_uid_gid()
    if default is None:
        return None
    uid_gid = default if uid_gid is None else uid_gid
    unshare_ipc = ipc_namespace_available()
    isolate_mounts = any((ephemeral_dirs, read_only_dirs, covered_dirs))
    if isolate_mounts:
        _ensure_libc()

    def _demote() -> None:
        _reenter_cwd()
        for fd in chown_fds:
            os.fchown(fd, uid_gid, uid_gid)
        if isolate_mounts:
            _isolate_mounts(ephemeral_dirs, read_only_dirs, covered_dirs)
        if unshare_ipc:
            _try_unshare_ipc()
        os.setgroups([])
        os.setgid(uid_gid)
        os.setuid(uid_gid)

    return _demote


def make_preexec(
    *chown_fds: int,
    uid_gid: int | None = None,
    ephemeral_dirs: Sequence[str] = (),
    read_only_dirs: Sequence[str] = (),
    covered_dirs: Sequence[str] = (),
) -> Callable[[], None] | None:
    """Return a preexec_fn that starts a new session and drops privileges, or None.

    Composes ``os.setsid()`` with :func:`make_demote_fn`, and takes the same
    ``chown_fds``, ``uid_gid`` and mount-isolation arguments. Use this for
    subprocesses that need session isolation (e.g. bash, where we ``killpg``
    the whole process group). For simple subprocesses that only need privilege
    demotion, use :func:`make_demote_fn` directly.

    Returns ``None`` when ``KAROTTE_DEMOTE_ID`` is not set (e.g. outside a
    container), which is safe to pass directly to
    ``asyncio.create_subprocess_exec``.
    """
    demote_fn = make_demote_fn(
        *chown_fds,
        uid_gid=uid_gid,
        ephemeral_dirs=ephemeral_dirs,
        read_only_dirs=read_only_dirs,
        covered_dirs=covered_dirs,
    )
    if demote_fn is None:
        return None

    def _preexec() -> None:
        os.setsid()
        demote_fn()

    return _preexec


def wrap_to_disable_networking(argv: list[str]) -> list[str]:
    """Wrap ``argv`` so the spawned process runs in fresh user + network
    namespaces, with no network interfaces attached.

    We can't do the unshare in a Python ``preexec_fn`` because the
    forked-from-multithreaded-Python child fails the kernel's
    ``thread_group_empty`` check on ``unshare(CLONE_NEWUSER)``. Routing
    through ``unshare(1)`` sidesteps that — by the time it runs the
    syscall, it's a fresh execve'd, single-threaded process.

    Outside gVisor, ``--map-current-user`` writes a uid_map / gid_map
    so ``getuid()`` inside the new userns matches the outer uid. Under
    gVisor (``KAROTTE_GVISOR`` env var set), gVisor's procfs does not
    support those map writes, so we omit the flag and accept that the
    child will see the overflow uid (typically 65534) inside the new
    userns. Callers who care about a real uid view inside the sandbox
    must compensate (e.g. set ``HOME``/``USER`` explicitly).
    """
    unshare = trusted_binary("unshare")
    if os.environ.get("KAROTTE_GVISOR") is not None:
        return [unshare, "--user", "--net", "--", *argv]
    return [unshare, "--user", "--net", "--map-current-user", "--", *argv]


_PIDNS_FLAGS = ["--pid", "--fork", "--mount-proc", "--kill-child=SIGKILL"]
"""``unshare(1)`` flags that put a session in its own PID namespace.

``--mount-proc`` gives it a private ``/proc`` (which also implies ``--mount``),
so ``ps`` and ``pkill`` see the namespace's own pids and nothing else.
``--kill-child`` tears the namespace down if the launcher dies, so an orphaned
session cannot outlive us.

Deliberately no ``--user``. Unsharing a user namespace is the other way to get
the CAP_SYS_ADMIN this needs, but gVisor's procfs rejects the uid_map / gid_map
writes that would make the student's uid meaningful inside it, leaving them as
the overflow uid: ``ps`` shows ``nobody``, root-owned files look unowned, and
setuid stops working. We are root already, so we take the capability we have
and let ``unshare`` drop to the student itself via ``--setuid`` / ``--setgid``.
"""

_pidns_probed: bool = False
_pidns_available: bool = False


def reset_pid_namespace_probe() -> None:
    """Forget the cached probe result. For tests."""
    global _pidns_probed, _pidns_available
    _pidns_probed = False
    _pidns_available = False


def _pid_namespace_available(uid_gid: int) -> bool:
    """Whether student sessions can be put in their own PID namespace.

    Only on gVisor. A PID namespace is the one reap that a fork-and-die chain
    cannot evade — SIGKILLing its init drops every process inside at once,
    where a ``/proc`` sweep never even observes a chain whose every generation
    is shorter-lived than a walk. But on runc ``/proc`` is masked, so
    ``--mount-proc`` cannot replace it and the namespace is left reading an
    outer ``/proc`` it cannot address: ``ps`` and ``pkill`` report pids that
    mean nothing inside. Students lean on both, so runc keeps the sweep.

    Probed once by actually building a namespace, because the capability
    depends on how the pod was launched (gVisor eval pods add CAP_SYS_ADMIN;
    a local ``docker run`` may not).
    """
    global _pidns_probed, _pidns_available
    if _pidns_probed:
        return _pidns_available
    _pidns_probed = True
    if os.environ.get("KAROTTE_GVISOR") is None:
        return False
    try:
        result = subprocess.run(
            [
                trusted_binary("unshare"),
                *_PIDNS_FLAGS,
                *_setuid_flags(uid_gid),
                "--",
                "true",
            ],
            preexec_fn=_make_root_preexec(uid_gid),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning(f"PID namespace probe failed to run ({exc}); reaping by sweep")
        return False
    _pidns_available = result.returncode == 0
    if not _pidns_available:
        logger.warning(
            f"Could not create a PID namespace for student processes (unshare exited {result.returncode}) — is CAP_SYS_ADMIN missing? Falling back to best-effort sweep reaping, which cannot see a fork-and-die chain."
        )
    return _pidns_available


def _confinement_preexec() -> Callable[[], None] | None:
    from karotte.confinement import get_confinement

    return get_confinement().student_preexec()


def _setuid_flags(uid_gid: int) -> list[str]:
    return ["--setuid", str(uid_gid), "--setgid", str(uid_gid)]


def _make_root_preexec(uid_gid: int, *chown_fds: int) -> Callable[[], None]:
    """A preexec that prepares the child but leaves it root, for when
    ``unshare`` does the demotion (it needs CAP_SYS_ADMIN to build the
    namespace, so it cannot already have dropped to the student).

    Supplementary groups still go here: ``unshare`` only sets uid and gid, and
    they have to be cleared while we are still privileged enough to do it. The
    IPC namespace too, for the same reason it lives in :func:`make_demote_fn`.
    """
    unshare_ipc = ipc_namespace_available()

    def _preexec() -> None:
        os.setsid()
        _reenter_cwd()
        for fd in chown_fds:
            os.fchown(fd, uid_gid, uid_gid)
        if unshare_ipc:
            _try_unshare_ipc()
        os.setgroups([])

    return _preexec


def student_session_command(
    argv: list[str],
    *chown_fds: int,
    disable_networking: bool,
) -> tuple[list[str], Callable[[], None] | None]:
    """The argv to exec for a student session, and the preexec_fn that goes
    with it.

    The two are returned together because they divide one job between them: if
    the session gets a PID namespace, ``unshare`` performs the privilege drop
    and the preexec must leave the child root long enough for it to; otherwise
    the preexec drops privileges itself. Pairing the wrong two runs the student
    as root or the namespace not at all.

    ``chown_fds`` are child fds to hand to the student, as in
    :func:`make_preexec`.

    The session also joins the student cgroup where the sandbox has one.
    Joining here, before anything else runs, is what makes the uid-keyed
    limits true: every descendant inherits membership, so a fork-and-die
    chain cannot be born outside it.
    """
    uid_gid = demoted_uid_gid()
    if uid_gid is None or not _pid_namespace_available(uid_gid):
        wrapped = wrap_to_disable_networking(argv) if disable_networking else argv
        demote_fn = make_demote_fn(*chown_fds)
        if demote_fn is None:
            return wrapped, None
        join = _confinement_preexec()

        def _preexec() -> None:
            os.setsid()
            # Join while still root: the cgroup's files are root-owned.
            if join is not None:
                join()
            demote_fn()

        return wrapped, _preexec

    # The network namespace is unshared by the already-demoted student inside
    # the PID namespace, exactly as it is without one.
    inner = wrap_to_disable_networking(argv) if disable_networking else argv
    return (
        [
            trusted_binary("unshare"),
            *_PIDNS_FLAGS,
            *_setuid_flags(uid_gid),
            "--",
            *inner,
        ],
        _make_root_preexec(uid_gid, *chown_fds),
    )
