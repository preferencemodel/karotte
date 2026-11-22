"""Folders of plugins Karotte loads in addition to the installed packages."""

import importlib
import os
import stat
import sys
from collections.abc import Iterable
from importlib.metadata import EntryPoint, distributions, version

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

from karotte.container import demoted_uid_gid

PLUGIN_PATH_ENV = "KAROTTE_PLUGIN_PATH"
"""``:``-separated absolute folders holding plugin packages and their ``.dist-info``."""

PLUGIN_MOUNT_ROOT = "/root/.karotte-plugins"
"""Where launchers mount plugin folders: under ``/root``, which only root can enter."""


class PluginPathError(RuntimeError):
    pass


def plugin_folders() -> list[str]:
    folders: list[str] = []
    for entry in os.environ.get(PLUGIN_PATH_ENV, "").split(":"):
        if not entry:
            continue
        if not os.path.isabs(entry):
            raise PluginPathError(
                f"{PLUGIN_PATH_ENV} entry {entry} isn't an absolute path"
            )
        folder = os.path.normpath(entry)
        if folder not in folders:
            folders.append(folder)
    return folders


def from_plugin_path(entry_point: EntryPoint) -> bool:
    dist = entry_point.dist
    return (
        dist is not None
        and os.path.normpath(str(dist.locate_file(""))) in plugin_folders()
    )


def apply_plugin_path() -> None:
    """Put the folders :data:`PLUGIN_PATH_ENV` names first on ``sys.path``, in order."""
    folders = plugin_folders()
    if not folders:
        return
    student = demoted_uid_gid()
    for folder in folders:
        if not os.path.isdir(folder):
            raise PluginPathError(
                f"{PLUGIN_PATH_ENV} names {folder}, which isn't a folder"
            )
        if student is not None and _can_reach(student, folder):
            raise PluginPathError(
                f"{PLUGIN_PATH_ENV} names {folder}, which the student (uid {student}) can reach"
            )
        _check_karotte_requirement(folder)
    rest = [p for p in sys.path if os.path.normpath(p) not in folders]
    sys.path[:] = folders + rest
    importlib.invalidate_caches()


def _can_reach(uid_gid: int, folder: str) -> bool:
    """Whether every directory from ``/`` down to ``folder`` lets ``uid_gid`` search it."""
    path = os.path.realpath(folder)
    parts = [path]
    while (parent := os.path.dirname(path)) != path:
        parts.append(parent)
        path = parent
    for part in parts:
        st = os.stat(part)
        if st.st_uid == uid_gid:
            search = stat.S_IXUSR
        elif st.st_gid == uid_gid:
            search = stat.S_IXGRP
        else:
            search = stat.S_IXOTH
        if not st.st_mode & search:
            return False
    return True


def _check_karotte_requirement(folder: str) -> None:
    installed = Version(version("karotte"))
    for dist in distributions(path=[folder]):
        for line in dist.requires or ():
            try:
                requirement = Requirement(line)
            except InvalidRequirement as e:
                msg = f"{dist.name} in {folder} has an invalid Requires-Dist {line!r}: {e}"
                raise PluginPathError(msg) from e
            if canonicalize_name(requirement.name) != "karotte":
                continue
            if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
                continue
            if not requirement.specifier.contains(installed, prereleases=True):
                msg = (
                    f"{dist.name} in {folder} needs {requirement}, "
                    + f"but karotte {installed} is installed"
                )
                raise PluginPathError(msg)


def plugin_path_env(mount_targets: Iterable[str]) -> dict[str, str]:
    """The :data:`PLUGIN_PATH_ENV` setting for a sandbox with these mounts."""
    plugins = [t for t in mount_targets if t.startswith(f"{PLUGIN_MOUNT_ROOT}/")]
    return {PLUGIN_PATH_ENV: ":".join(plugins)} if plugins else {}
