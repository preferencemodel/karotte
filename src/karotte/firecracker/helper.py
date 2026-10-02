"""A small pinned Alpine image with e2fsprogs, libarchive and zstd.

``mkfs.ext4 -d -`` builds a filesystem from a tar stream only with e2fsprogs
1.47.1 or newer built with libarchive, which many hosts lack, so the base
drive is made in this container instead.
"""

import hashlib
import subprocess

HELPER_BASE_IMAGE = "docker.io/library/alpine:3.22@sha256:5291449c3df73caf6ed85e649dec1b9e818b39a5d8c871e97afc13e9cd5e8fa8"
HELPER_PACKAGES = ("e2fsprogs", "e2fsprogs-extra", "libarchive", "tar", "zstd")


def helper_containerfile() -> str:
    return (
        f"FROM {HELPER_BASE_IMAGE}\n"
        + f"RUN apk add --no-cache {' '.join(HELPER_PACKAGES)}\n"
    )


def helper_image(engine: str = "docker") -> str:
    """The helper image's tag, built on first use."""
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
    return tag
