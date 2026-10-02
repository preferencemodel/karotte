"""Tests for :mod:`karotte.init_shim`.

The shim replaces tini: when karotte is a container's PID 1, it forks and the
parent stays behind as a minimal init that forwards signals to the child,
reaps orphans reparented to PID 1, and exits with the child's exit code.

``_run_init_loop`` only relies on ``waitpid(-1)`` semantics that hold for any
parent process, so tests exercise it with real fork children instead of
needing to actually run as PID 1. Every test that runs the loop for real does
so in a subprocess: ``waitpid(-1)`` in the pytest process would steal exit
statuses of children spawned by unrelated tests in the same session.
Reparenting of orphans is the kernel's job; the one test that exercises it
end-to-end makes the shim process a child subreaper via ``prctl`` and is
Linux-only.
"""

import os
import signal
import subprocess
import sys
import textwrap
import time
from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

import pytest

from karotte.init_shim import (
    _WAITED_SIGNALS,  # pyright: ignore[reportPrivateUsage]
    _run_init_loop,  # pyright: ignore[reportPrivateUsage]
    maybe_become_init,
)

pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="fork/signals are POSIX-only"
)


@pytest.fixture(autouse=True)
def _restore_signal_state() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    """The shim installs signal handlers and manipulates the signal mask
    in the calling process; restore pytest's state afterwards so leakage can't
    poison other tests (a blocked SIGTERM is inherited by forked children)."""
    original_handlers = {sig: signal.getsignal(sig) for sig in _WAITED_SIGNALS}
    original_mask = signal.pthread_sigmask(signal.SIG_BLOCK, [])
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, original_mask)
        for sig, handler in original_handlers.items():
            signal.signal(sig, handler)


# --- maybe_become_init: fork/no-fork decision (no real forking) ---


def test_noop_when_not_pid_1(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_fork() -> int:
        raise AssertionError("must not fork when not PID 1")

    monkeypatch.setattr(os, "fork", fail_fork)
    assert maybe_become_init() is None


def test_child_returns_to_caller(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "getpid", lambda: 1)
    monkeypatch.setattr(os, "fork", lambda: 0)

    def fail_exit(code: int) -> None:
        raise AssertionError(f"child must not _exit (got {code})")

    monkeypatch.setattr(os, "_exit", fail_exit)
    assert maybe_become_init() is None


def test_parent_exits_with_loop_result(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "getpid", lambda: 1)
    monkeypatch.setattr(os, "fork", lambda: 4242)

    seen_child_pids: list[int] = []

    def fake_loop(child_pid: int) -> int:
        seen_child_pids.append(child_pid)
        return 7

    monkeypatch.setattr("karotte.init_shim._run_init_loop", fake_loop)

    def fake_exit(code: int) -> None:
        raise SystemExit(code)

    monkeypatch.setattr(os, "_exit", fake_exit)

    with pytest.raises(SystemExit) as excinfo:
        maybe_become_init()
    assert excinfo.value.code == 7
    assert seen_child_pids == [4242]


# --- _run_init_loop: exit-code propagation with real children ---


EXIT_CODE_SCRIPT = textwrap.dedent(
    """
    import os, signal, sys, time
    from karotte.init_shim import _run_init_loop

    mode, value = sys.argv[1], int(sys.argv[2])
    child = os.fork()
    if child == 0:
        if mode == "exit":
            os._exit(value)
        os.kill(os.getpid(), value)
        time.sleep(30)
        os._exit(250)

    code = _run_init_loop(child)
    # The loop reaped everything before returning.
    try:
        os.waitpid(-1, os.WNOHANG)
    except ChildProcessError:
        print("all-reaped", flush=True)
    sys.exit(code)
    """
)


def run_loop_subprocess(script: str, *argv: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", script, *argv],
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.parametrize("exit_code", [0, 1, 7, 200])
def test_returns_child_exit_code(exit_code: int) -> None:
    result = run_loop_subprocess(EXIT_CODE_SCRIPT, "exit", str(exit_code))
    assert result.returncode == exit_code, result.stderr
    assert "all-reaped" in result.stdout


@pytest.mark.parametrize("signum", [signal.SIGKILL, signal.SIGTERM])
def test_child_signal_death_maps_to_128_plus_signum(signum: signal.Signals) -> None:
    result = run_loop_subprocess(EXIT_CODE_SCRIPT, "signal", str(int(signum)))
    assert result.returncode == 128 + signum, result.stderr
    assert "all-reaped" in result.stdout


# --- _run_init_loop: reaping of children other than the main child ---


REAP_OTHERS_SCRIPT = textwrap.dedent(
    """
    import os, sys, time
    from karotte.init_shim import _run_init_loop

    # Children that exit at once (stand-ins for reparented orphans), and a main
    # child that exits only once the loop has reaped every one of them: a
    # reaped pid is gone, while a zombie still answers kill(pid, 0). Waiting on
    # a pipe instead would race: a child closes its end before it becomes a
    # zombie the loop can reap.
    others = []
    for _ in range(5):
        pid = os.fork()
        if pid == 0:
            os._exit(0)
        others.append(pid)

    main = os.fork()
    if main == 0:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            alive = 0
            for pid in others:
                try:
                    os.kill(pid, 0)
                    alive += 1
                except ProcessLookupError:
                    pass
            if not alive:
                os._exit(5)
            time.sleep(0.01)
        os._exit(6)

    code = _run_init_loop(main)
    try:
        os.waitpid(-1, os.WNOHANG)
    except ChildProcessError:
        print("all-reaped", flush=True)
    sys.exit(code)
    """
)


def test_reaps_other_children_while_waiting() -> None:
    """Children that exit before the main child are reaped by the loop instead
    of lingering as zombies."""
    result = run_loop_subprocess(REAP_OTHERS_SCRIPT)
    assert result.returncode == 5, result.stderr
    assert "all-reaped" in result.stdout


def test_returns_one_when_all_children_vanish(monkeypatch: pytest.MonkeyPatch) -> None:
    """ECHILD without having seen the main child exit means the shim lost
    track of it; the loop reports failure instead of raising."""

    def raise_echild(_pid: int, _options: int) -> tuple[int, int]:
        raise ChildProcessError

    monkeypatch.setattr(os, "waitpid", raise_echild)
    assert _run_init_loop(999999) == 1


# --- signal forwarding through a real, separate shim process ---


SHIM_SCRIPT = textwrap.dedent(
    """
    import os, signal, sys, time
    from karotte.init_shim import _FORWARDED_SIGNALS, _run_init_loop

    child_behavior = sys.argv[1]
    # Same sequence as maybe_become_init: block across the fork so a signal
    # arriving before the init loop starts cannot kill it.
    signal.pthread_sigmask(signal.SIG_BLOCK, _FORWARDED_SIGNALS)
    child = os.fork()
    if child == 0:
        signal.pthread_sigmask(signal.SIG_UNBLOCK, _FORWARDED_SIGNALS)
        if child_behavior == "exit-on-sigterm":
            signal.signal(signal.SIGTERM, lambda *a: os._exit(43))
        elif child_behavior == "exit-on-sigusr1":
            signal.signal(signal.SIGUSR1, lambda *a: os._exit(44))
        elif child_behavior == "ignore-sigterm-exit-on-sigusr1":
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGUSR1, lambda *a: os._exit(44))
        # "default-dispositions": leave handlers untouched.
        print("child-ready", flush=True)
        # Short sleeps, not one long one: a signal that lands just before a
        # sleep starts has its Python handler deferred to the next bytecode
        # check, which a single time.sleep(30) would put off for 30 seconds.
        for _ in range(300):
            time.sleep(0.1)
        os._exit(250)
    sys.exit(_run_init_loop(child))
    """
)


def spawn_shim(child_behavior: str) -> subprocess.Popen[str]:
    proc = subprocess.Popen(
        [sys.executable, "-c", SHIM_SCRIPT, child_behavior],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout is not None
    assert proc.stdout.readline().strip() == "child-ready"
    return proc


def test_forwards_sigterm_to_child() -> None:
    proc = spawn_shim("exit-on-sigterm")
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=10) == 43


def test_forwards_sigusr1_to_child() -> None:
    proc = spawn_shim("exit-on-sigusr1")
    proc.send_signal(signal.SIGUSR1)
    assert proc.wait(timeout=10) == 44


def test_sigterm_terminates_child_with_default_disposition() -> None:
    """A child that doesn't catch SIGTERM dies from the forwarded signal, and
    the shim reports 128+15."""
    proc = spawn_shim("default-dispositions")
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=10) == 128 + signal.SIGTERM


PENDING_SIGNAL_SCRIPT = textwrap.dedent(
    """
    import os, signal, sys, time
    from karotte.init_shim import _FORWARDED_SIGNALS, _run_init_loop

    signal.pthread_sigmask(signal.SIG_BLOCK, _FORWARDED_SIGNALS)
    child = os.fork()
    if child == 0:
        signal.signal(signal.SIGTERM, lambda *a: os._exit(43))
        signal.pthread_sigmask(signal.SIG_UNBLOCK, _FORWARDED_SIGNALS)
        for _ in range(300):  # short sleeps, as in SHIM_SCRIPT
            time.sleep(0.1)
        os._exit(250)

    # Stays pending until the init loop waits for it: the shim must forward
    # it instead of dying from it.
    os.kill(os.getpid(), signal.SIGTERM)
    sys.exit(_run_init_loop(child))
    """
)


def test_signal_arriving_before_init_loop_is_forwarded() -> None:
    result = subprocess.run(
        [sys.executable, "-c", PENDING_SIGNAL_SCRIPT],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 43, result.stderr


def test_shim_keeps_running_while_child_ignores_signal() -> None:
    """Forwarding a signal the child ignores must not kill the shim itself."""
    proc = spawn_shim("ignore-sigterm-exit-on-sigusr1")
    proc.send_signal(signal.SIGTERM)
    time.sleep(0.3)
    assert proc.poll() is None
    proc.send_signal(signal.SIGUSR1)
    assert proc.wait(timeout=10) == 44


# --- end-to-end orphan reparenting (Linux-only, needs PR_SET_CHILD_SUBREAPER) ---


REPARENT_SCRIPT = textwrap.dedent(
    """
    import ctypes, os, sys, time
    from karotte.init_shim import _run_init_loop

    PR_SET_CHILD_SUBREAPER = 36
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        sys.exit(90)

    # Double fork: the middle process dies at once, so its child becomes an
    # orphan that reparents to us (the subreaper) and exits while the main
    # child is still running.
    middle = os.fork()
    if middle == 0:
        orphan = os.fork()
        if orphan == 0:
            time.sleep(0.2)  # outlive the middle process to become an orphan
            os._exit(0)
        os._exit(7)

    main = os.fork()
    if main == 0:
        time.sleep(1.0)
        os._exit(5)

    code = _run_init_loop(main)
    # By the time the main child exits, the loop must have reaped both the
    # middle process and the reparented orphan; a leftover zombie would still
    # be waitable here.
    try:
        os.waitpid(-1, os.WNOHANG)
    except ChildProcessError:
        print("all-reaped", flush=True)
    sys.exit(code)
    """
)


@pytest.mark.skipif(sys.platform != "linux", reason="prctl subreaper is Linux-only")
def test_reaps_reparented_orphans_end_to_end() -> None:
    result = subprocess.run(
        [sys.executable, "-c", REPARENT_SCRIPT],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 5, result.stderr
    assert "all-reaped" in result.stdout


# --- console-script wiring ---


def test_entry_forks_before_running_app(monkeypatch: pytest.MonkeyPatch) -> None:
    import karotte.cli

    calls: list[str] = []
    monkeypatch.setattr(karotte.cli, "configure_logging", lambda: calls.append("log"))
    monkeypatch.setattr(karotte.cli, "maybe_become_init", lambda: calls.append("init"))
    monkeypatch.setattr(
        karotte.cli, "hide_run_config_from_procfs", lambda: calls.append("hide")
    )

    def fake_app(*_args: Any, **_kwargs: Any) -> None:
        calls.append("app")

    monkeypatch.setattr(karotte.cli, "app", fake_app)
    karotte.cli.entry()
    assert calls == ["log", "hide", "init", "app"]


def test_entry_hides_run_config_before_the_fork(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The init parent never execs, so anything left on argv at fork time stays
    in its world-readable /proc/<pid>/cmdline for the container's lifetime."""
    import karotte.cli

    argv = ["karotte", "run", "--config", '{"secret": "key"}', "--no-containerized"]
    monkeypatch.setenv("KAROTTE_CONTAINERIZED", "1")
    monkeypatch.setattr(sys, "argv", argv)

    def noop_app(*_args: Any, **_kwargs: Any) -> None:
        pass

    monkeypatch.setattr(karotte.cli, "app", noop_app)
    monkeypatch.setattr(karotte.cli, "configure_logging", lambda: None)

    forked_argv: list[list[str]] = []
    monkeypatch.setattr(
        karotte.cli, "maybe_become_init", lambda: forked_argv.append(list(sys.argv))
    )

    def fake_execv(_path: str, new_argv: list[str]) -> None:
        monkeypatch.setattr(sys, "argv", new_argv[1:])

    with (
        patch("karotte.hide_run_config.Path.write_text"),
        patch("karotte.hide_run_config.os.chmod"),
        patch("karotte.hide_run_config.os.execv", side_effect=fake_execv),
    ):
        karotte.cli.entry()

    assert forked_argv, "the init shim never ran"
    assert '{"secret": "key"}' not in forked_argv[0]
