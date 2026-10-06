"""Keep the inline run config JSON and API key out of the world-readable /proc/[pid]/cmdline."""

import os
import stat
import sys
from pathlib import Path

from karotte.container import is_containerized

RUN_CONFIG_PATH = "/tmp/.karotte_run_config.json"
MODEL_API_KEY_PATH = "/tmp/.karotte_model_api_key"


def hide_run_config_from_procfs() -> None:
    """Re-exec with ``run``'s secrets in root-only files, so they leave the 0444
    /proc/[pid]/cmdline the student can read.

    Must run before the init shim forks: the init parent never execs, so it would
    keep the original argv for the container's whole lifetime.
    """
    if not is_containerized():
        return

    new_argv = _argv_without_secrets(sys.argv)
    if new_argv != sys.argv:
        os.execv(sys.executable, [sys.executable, *new_argv])


def _argv_without_secrets(argv: list[str]) -> list[str]:
    new_argv: list[str] = []
    seen_run = False
    i = 0
    while i < len(argv):
        arg = argv[i]
        name, equals, value = (
            arg.partition("=") if arg.startswith("--") else (arg, "", "")
        )
        if not seen_run or not (equals or i + 1 < len(argv)):
            new_argv.append(arg)
            seen_run |= arg == "run"
            i += 1
            continue
        if not equals:
            value = argv[i + 1]
        hidden = _hidden(name, value)
        if hidden is None:
            new_argv.append(arg)
            i += 1
            continue
        flag, path = hidden
        Path(path).write_text(value)
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
        new_argv += [flag, path]
        i += 1 if equals else 2
    return new_argv


def _hidden(name: str, value: str) -> tuple[str, str] | None:
    """The flag and file that replace a secret ``name value`` pair, or None to keep it."""
    if name in ("--config", "-c"):
        try:
            if Path(value).is_file():
                return None
        except OSError:
            pass  # Config is not a file path (e.g. too long for the filesystem)
        return "--config", RUN_CONFIG_PATH
    if name == "--model-api-key" and not value.startswith("$"):
        return "--model-api-key-file", MODEL_API_KEY_PATH
    return None
