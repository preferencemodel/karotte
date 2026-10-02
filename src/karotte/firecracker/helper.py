"""A small pinned Alpine image with e2fsprogs, libarchive and zstd.

``mkfs.ext4 -d -`` builds a filesystem from a tar stream only with e2fsprogs
1.47.1 or newer built with libarchive, which many hosts lack, so the base
drive is made in this container instead.

It stays on Alpine 3.22 (e2fsprogs 1.47.2): 3.23 and later build e2fsprogs
without libarchive. The fix in 1.47.4 for files over 2 GiB is to the
directory input (``-d <dir>``), which 1.47.3 broke; the tar input reads such
files correctly in 1.47.2.
"""

import hashlib
import re
import subprocess

from karotte.firecracker import FirecrackerError

HELPER_BASE_IMAGE = "docker.io/library/alpine:3.22@sha256:5291449c3df73caf6ed85e649dec1b9e818b39a5d8c871e97afc13e9cd5e8fa8"
HELPER_PACKAGES = ("e2fsprogs", "e2fsprogs-extra", "libarchive", "tar", "zstd")

MIN_MKE2FS_VERSION = (1, 47, 1)
"""The first e2fsprogs whose ``mkfs.ext4 -d`` reads a tar stream."""

# Prints mke2fs's version line, then tar-ok if it made a filesystem from a tar.
_PROBE_SCRIPT = """\
mke2fs -V 2>&1 | head -n 1
echo x > /tmp/f && truncate -s 8388608 /tmp/probe.ext4 &&
  tar -cf - -C /tmp f | mkfs.ext4 -q -F -d - /tmp/probe.ext4 >/dev/null 2>&1 &&
  echo tar-ok
"""


class HelperError(FirecrackerError):
    pass


def helper_containerfile() -> str:
    return (
        f"FROM {HELPER_BASE_IMAGE}\n"
        + f"RUN apk add --no-cache {' '.join(HELPER_PACKAGES)}\n"
    )


def helper_image(engine: str = "docker") -> str:
    """The helper image's tag, built on first use and checked to read a tar."""
    containerfile = helper_containerfile()
    tag = (
        "karotte-firecracker-helper:"
        + hashlib.sha256(containerfile.encode()).hexdigest()[:12]
    )
    exists = subprocess.run(
        [engine, "image", "inspect", tag], capture_output=True, check=False
    )
    if exists.returncode != 0:
        subprocess.run(
            [engine, "build", "--tag", tag, "-"],
            input=containerfile.encode(),
            check=True,
        )
    probe = subprocess.run(
        [engine, "run", "--rm", "--network", "none", tag, "sh", "-c", _PROBE_SCRIPT],
        capture_output=True,
        text=True,
        check=False,
    )
    check_mke2fs(probe.stdout, tag)
    return tag


def check_mke2fs(probe_output: str, image: str) -> None:
    """Refuse a helper whose mke2fs can't build the base drive from a tar."""
    found = re.search(r"mke2fs (\d+)\.(\d+)(?:\.(\d+))?", probe_output)
    if found is None:
        raise HelperError(
            f"Could not read the mke2fs version in {image}: {probe_output.strip()!r}"
        )
    version = tuple(int(part or 0) for part in found.groups())
    if version < MIN_MKE2FS_VERSION:
        raise HelperError(
            f"{image} has mke2fs {'.'.join(map(str, version))}; building the base"
            + " drive from a tar stream needs e2fsprogs"
            + f" {'.'.join(map(str, MIN_MKE2FS_VERSION))} or newer"
        )
    if "tar-ok" not in probe_output.splitlines():
        raise HelperError(
            f"mke2fs in {image} can't build a filesystem from a tar stream: it needs"
            + " e2fsprogs built with libarchive, and libarchive installed"
        )
