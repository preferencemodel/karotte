"""A kernel-enforced cap on new files: overlays whose upper layer lives on a
loop-mounted filesystem of exactly the budgeted size."""

from __future__ import annotations

import json
import os
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from karotte.container import mount_points
from karotte.trusted_bin import trusted_binary

QUOTA_DIR = Path("/var/karotte_quota")

_BACKING = "quota.img"
_MOUNT = "mnt"
_MANIFEST = "manifest.json"


@dataclass(frozen=True)
class FileQuota:
    """A mounted quota: new writes under ``paths`` share ``size_bytes`` of space."""

    size_bytes: int
    paths: tuple[Path, ...]


def mount_file_quota(
    paths: tuple[Path, ...],
    max_bytes: int,
    max_count: int | None,
    quota_dir: Path | None = None,
) -> FileQuota | None:
    """Cap what may be written under ``paths`` at ``max_bytes`` (and
    ``max_count`` files), enforced by the kernel; ``None`` where this sandbox
    cannot mount, so the caller can fall back to detection.

    ``quota_dir`` holds the backing image and its mount, ``QUOTA_DIR`` by
    default. A probe passes its own so it never touches the run's."""
    if os.geteuid() != 0:
        return None
    quota_dir = QUOTA_DIR if quota_dir is None else quota_dir
    backing = quota_dir / _BACKING
    mount_point = quota_dir / _MOUNT

    # A privileged helper outside this container may have prepared the quota
    # filesystem already (on k8s, loop devices are unreachable in-container
    # even with CAP_SYS_ADMIN: the device cgroup denies them, so a sidecar
    # loop-mounts the sized image onto a shared volume). Its size *is* the
    # budget; the overlays below are the only mounts made here.
    premounted = os.path.ismount(mount_point)
    if premounted:
        stats = os.statvfs(mount_point)
        max_bytes = stats.f_frsize * stats.f_blocks
    elif backing.exists():
        logger.warning("A file quota is already mounted; not mounting another")
        return None
    else:
        try:
            quota_dir.mkdir(mode=0o700, exist_ok=True)
            with open(backing, "wb") as f:
                # Preallocate rather than truncate: a sparse image that runs out
                # of backing space mid-write surfaces as EIO / remount-ro inside
                # the quota filesystem, not the clean ENOSPC this exists to give.
                # macOS (dev machines) has no posix_fallocate; sandboxes are Linux.
                fallocate = getattr(os, "posix_fallocate", None)
                if fallocate is not None:
                    fallocate(f.fileno(), 0, max_bytes)
                else:
                    f.truncate(max_bytes)
            mkfs = [trusted_binary("mkfs.ext4"), "-q", "-F", "-m", "0"]
            if max_count is not None:
                mkfs += ["-N", str(max_count)]
            _run([*mkfs, str(backing)])
            mount_point.mkdir(exist_ok=True)
            _run(
                [trusted_binary("mount"), "-o", "loop", str(backing), str(mount_point)]
            )
        except (OSError, RuntimeError) as exc:
            logger.debug(f"No kernel file quota on this sandbox: {exc}")
            _remove_quota_dir(quota_dir)
            return None

    overlaid: list[Path] = []
    try:
        for index, path in enumerate(paths):
            _mount_overlay(mount_point, index, path)
            overlaid.append(path)
        _ = (quota_dir / _MANIFEST).write_text(
            json.dumps({"paths": [str(p) for p in paths]})
        )
    except (OSError, RuntimeError) as exc:
        logger.warning(f"Could not overlay every path; unmounting the quota: {exc}")
        if premounted:
            # The pre-mounted filesystem is the helper's to tear down; undo
            # only the overlays this process made.
            _unmount_all(overlaid)
        else:
            _unmount_all([*overlaid, mount_point])
            _remove_quota_dir(quota_dir)
        return None

    _reenter_cwd()
    over = list(map(str, paths))
    if premounted:
        logger.info(
            f"Adopted pre-mounted file quota of {max_bytes} bytes at {mount_point} over {over}"
        )
    else:
        logger.info(f"Created loop file quota of {max_bytes} bytes over {over}")
    return FileQuota(max_bytes, tuple(paths))


def ensure_file_quota() -> None:
    """Remount a quota whose mounts are gone, e.g. after a restore that kept
    the disk but not the kernel's mount state.

    A no-op without a manifest or with everything mounted. Raises
    ``RuntimeError`` when a recorded mount cannot be restored: the student's
    writes live in the quota, so grading without it would see none of them.
    """
    try:
        manifest = json.loads((QUOTA_DIR / _MANIFEST).read_text())
    except (OSError, ValueError):
        return
    mount_point = QUOTA_DIR / _MOUNT
    try:
        if not os.path.ismount(mount_point):
            _run(
                [
                    trusted_binary("mount"),
                    "-o",
                    "loop",
                    str(QUOTA_DIR / _BACKING),
                    str(mount_point),
                ]
            )
        for index, path in enumerate(manifest["paths"]):
            if not os.path.ismount(path):
                _mount_overlay(mount_point, index, Path(path))
    except (OSError, RuntimeError, KeyError) as exc:
        raise RuntimeError(f"Could not remount the file quota: {exc}") from exc
    _reenter_cwd()


def _reenter_cwd() -> None:
    """A mount over it would otherwise leave the process in the hidden lower directory."""
    try:
        cwd = os.getcwd()
        os.chdir(cwd)
        here, there = os.stat("."), os.stat(cwd)
    except OSError:
        return
    if (here.st_dev, here.st_ino) != (there.st_dev, there.st_ino):
        raise RuntimeError(
            f"The working directory still resolves to a directory hidden by the file quota: {cwd}"
        )


def _mount_overlay(mount_point: Path, index: int, path: Path) -> None:
    """Overlay ``path`` so new writes land on the quota filesystem, mirroring
    the lower root's owner and mode. Mounts below ``path`` (which the overlay
    would hide) are rebound onto the merged view."""
    lower_stat = path.stat()
    upper = mount_point / f"upper{index}"
    work = mount_point / f"work{index}"
    for directory in (upper, work):
        directory.mkdir(exist_ok=True)
    os.chmod(upper, stat.S_IMODE(lower_stat.st_mode))
    os.chown(upper, lower_stat.st_uid, lower_stat.st_gid)

    kept: list[tuple[Path, Path]] = []
    try:
        for sub_index, submount in enumerate(_submounts_under(path)):
            aside = mount_point / f"keep{index}-{sub_index}"
            aside.mkdir(exist_ok=True)
            _run([trusted_binary("mount"), "--rbind", str(submount), str(aside)])
            kept.append((aside, submount))
        _run(
            [
                trusted_binary("mount"),
                "-t",
                "overlay",
                "overlay",
                "-o",
                f"lowerdir={path},upperdir={upper},workdir={work}",
                str(path),
            ]
        )
        for aside, submount in kept:
            _run([trusted_binary("mount"), "--rbind", str(aside), str(submount)])
    finally:
        for aside, _submount in kept:
            try:
                _run([trusted_binary("umount"), "-l", str(aside)])
                aside.rmdir()
            except (OSError, RuntimeError):
                pass


def _submounts_under(path: Path) -> list[Path]:
    """Roots of the mounts strictly below ``path``, which an overlay on
    ``path`` would otherwise hide. Nested mounts travel with their root's
    rbind, so only the topmost of each subtree is returned."""
    prefix = str(path).rstrip("/") + "/"
    points = sorted(
        str(point) for point in mount_points() if str(point).startswith(prefix)
    )
    tops: list[Path] = []
    for point in points:
        if not any(point.startswith(f"{top}/") for top in tops):
            tops.append(Path(point))
    return tops


def _unmount_all(mounted: list[Path]) -> None:
    # -R: an overlaid path may carry rebound submounts on top, and a plain
    # umount of it would fail busy.
    for path in mounted:
        try:
            _run([trusted_binary("umount"), "-R", str(path)])
        except (OSError, RuntimeError):
            continue


def unmount_file_quota(quota: FileQuota, quota_dir: Path) -> bool:
    """Undo :func:`mount_file_quota` for a quota mounted under ``quota_dir``.
    Returns whether everything came off; what was written under the quota is
    gone with it."""
    mounted = [*quota.paths, quota_dir / _MOUNT]
    _unmount_all(mounted)
    left = [path for path in mounted if os.path.ismount(path)]
    if left:
        logger.warning(f"Could not unmount the file quota at {list(map(str, left))}")
        return False
    _remove_quota_dir(quota_dir)
    return True


def _remove_quota_dir(quota_dir: Path) -> None:
    for path in (quota_dir / _MANIFEST, quota_dir / _BACKING):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
    for directory in (quota_dir / _MOUNT, quota_dir):
        try:
            directory.rmdir()
        except OSError:
            pass


def _run(argv: list[str]) -> None:
    result = subprocess.run(argv, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"{argv[0]} failed ({result.returncode}): {result.stderr.strip()}"
        )
