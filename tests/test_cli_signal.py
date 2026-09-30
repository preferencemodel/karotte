import os
import signal
import time

import pytest

from karotte.cli import (
    _install_sigterm_handler,  # pyright: ignore[reportPrivateUsage]
    _sigterm_handler,  # pyright: ignore[reportPrivateUsage]
)


@pytest.fixture
def _restore_sigterm():  # pyright: ignore[reportUnusedFunction]
    original = signal.getsignal(signal.SIGTERM)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, original)


def test_sigterm_handler_raises_keyboard_interrupt():
    with pytest.raises(KeyboardInterrupt):
        _sigterm_handler(signal.SIGTERM, None)


def test_install_sets_sigterm_disposition(_restore_sigterm: None):
    _install_sigterm_handler()
    assert signal.getsignal(signal.SIGTERM) is _sigterm_handler


def test_sigterm_is_translated_to_keyboard_interrupt(_restore_sigterm: None):
    """End-to-end: once installed, an actual SIGTERM delivered to this process
    raises KeyboardInterrupt in the main thread, exactly like SIGINT/Ctrl-C."""
    _install_sigterm_handler()
    with pytest.raises(KeyboardInterrupt):
        os.kill(os.getpid(), signal.SIGTERM)
        # Signals are delivered between bytecode instructions on the main
        # thread; sleep gives the pending handler a chance to run.
        time.sleep(1.0)
