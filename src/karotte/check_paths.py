"""Reject loader paths with an empty or relative component, which make the dynamic loader search the cwd."""

import os
import re
from collections.abc import Mapping

_LOADER_VARS = ("PATH", "LD_LIBRARY_PATH", "LD_PRELOAD", "LD_AUDIT")


class UnsafeLoaderPath(Exception):
    pass


def split_loader_path(var: str, value: str) -> list[str]:
    """glibc splits LD_PRELOAD on spaces as well as colons; the other vars on colons only."""
    if var == "LD_PRELOAD":
        return re.split(r"[: ]", value)
    return value.split(os.pathsep)


def check_paths(env: Mapping[str, str] | None = None) -> None:
    if env is None:
        env = os.environ
    for var in _LOADER_VARS:
        value = env.get(var)
        if not value:
            continue
        for entry in split_loader_path(var, value):
            if not entry.startswith("/"):
                raise UnsafeLoaderPath(
                    f"{var} has a non-absolute component {entry!r} in {value!r}: an empty or relative entry makes the loader search the cwd."
                )
