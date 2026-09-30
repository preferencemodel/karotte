"""Minimal PID-1 init shim (the core of what tini does).

When karotte is a container's entrypoint it runs as PID 1, and every orphaned
process in the container gets reparented to it. A regular process never
``wait()``s for children it didn't spawn, so those orphans linger as zombies.
The shim forks at startup: the child continues as the real karotte process,
while the parent stays behind as a minimal init that forwards signals to the
child, reaps anything reparented to PID 1, and exits with the child's exit
code.

Reaping from a separate process cannot race with subprocess exit-status
collection: the init parent can only ever wait on its single fork child plus
reparented orphans, while the real karotte process keeps exclusive ownership
of its own children.
"""

import os
import signal
import sys

_FORWARDED_SIGNALS = (
    signal.SIGHUP,
    signal.SIGINT,
    signal.SIGQUIT,
    signal.SIGTERM,
    signal.SIGUSR1,
    signal.SIGUSR2,
)


def maybe_become_init() -> None:
    """Fork if running as PID 1, with the parent acting as a minimal init.

    Returns in the fork child (and immediately when not PID 1), which then
    continues as the real karotte process. The parent never returns: it runs
    the init loop and ``_exit``s with the child's exit code. Must be called
    before any threads or event loops start, so the fork child inherits a
    single-threaded interpreter.
    """
    if os.getpid() != 1:
        return
    # Block the forwarded signals across the fork so none can be lost (or kill
    # the parent) between forking and installing the forwarding handlers; the
    # parent unblocks them once its handlers are in place.
    signal.pthread_sigmask(signal.SIG_BLOCK, _FORWARDED_SIGNALS)
    child_pid = os.fork()
    if child_pid == 0:
        signal.pthread_sigmask(signal.SIG_UNBLOCK, _FORWARDED_SIGNALS)
        return
    os._exit(_run_init_loop(child_pid))


def _run_init_loop(child_pid: int) -> int:
    """Forward signals to ``child_pid`` and reap children until it exits.

    Returns the child's exit code, mapping signal deaths to 128+signum.
    """

    def forward(signum: int, _frame: object) -> None:
        try:
            os.kill(child_pid, signum)
        except ProcessLookupError:
            pass

    # PID 1 gets no default signal dispositions, so without explicit handlers
    # a SIGTERM from the container runtime would be silently dropped.
    for sig in _FORWARDED_SIGNALS:
        signal.signal(sig, forward)
    # Deliver anything that arrived while the signals were blocked around the
    # fork; a no-op if the caller never blocked them.
    signal.pthread_sigmask(signal.SIG_UNBLOCK, _FORWARDED_SIGNALS)

    while True:
        try:
            pid, status = os.waitpid(-1, 0)
        except ChildProcessError:
            sys.stderr.write(
                "karotte init shim: lost track of the main child process\n"
            )
            return 1
        if pid != child_pid:
            continue
        exit_code = os.waitstatus_to_exitcode(status)
        return 128 - exit_code if exit_code < 0 else exit_code
