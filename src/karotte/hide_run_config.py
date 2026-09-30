"""Keep the inline run config JSON out of the world-readable /proc/[pid]/cmdline."""

import os
import stat
import sys
from pathlib import Path

from karotte.container import is_containerized

RUN_CONFIG_PATH = "/tmp/.karotte_run_config.json"


def hide_run_config_from_procfs() -> None:
    """Re-exec with the run config in a root-only file, so its secrets (API keys)
    leave the 0444 /proc/[pid]/cmdline the student can read.

    Must run before the init shim forks: the init parent never execs, so it would
    keep the original argv for the container's whole lifetime.
    """
    if not is_containerized():
        return

    config_index = _run_config_argv_index()
    if config_index is None:
        return

    config = sys.argv[config_index]
    try:
        if Path(config).is_file():
            return
    except OSError:
        pass  # Config is not a file path (e.g. too long for the filesystem)

    Path(RUN_CONFIG_PATH).write_text(config)
    os.chmod(RUN_CONFIG_PATH, stat.S_IRUSR | stat.S_IWUSR)

    new_argv = list(sys.argv)
    new_argv[config_index - 1 : config_index + 1] = ["--config", RUN_CONFIG_PATH]

    os.execv(sys.executable, [sys.executable, *new_argv])


def _run_config_argv_index() -> int | None:
    """Index in ``sys.argv`` of the value given to ``run``'s ``--config``/``-c``."""
    for i, arg in enumerate(sys.argv):
        if arg in ("--config", "-c") and i + 1 < len(sys.argv):
            return i + 1 if "run" in sys.argv[:i] else None
    return None
