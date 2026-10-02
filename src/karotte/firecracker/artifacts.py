"""The pinned Firecracker release and guest kernel, downloaded into the karotte
cache and checked against their SHA-256 before every use.

The kernel is Kata Containers' guest vmlinux: it has everything karotte needs
in the guest (xt_owner, nftables, loop, ext4, overlayfs, vsock, all
namespaces) built in. Kata publishes it only inside its full static release
tarball, so that tarball is downloaded once per pinned version, checked, and
only the kernel is kept.
"""

import hashlib
import os
import platform
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from loguru import logger

from karotte.firecracker import FirecrackerError

FIRECRACKER_VERSION = "v1.17.0"
KATA_VERSION = "4.0.0"
KERNEL_NAME = "vmlinux-6.18.35-200"

Arch = Literal["x86_64", "aarch64"]
SUPPORTED_ARCHES: tuple[Arch, ...] = ("x86_64", "aarch64")

_KATA_ARCH = {"x86_64": "amd64", "aarch64": "arm64"}

# Download checksums are the ones the GitHub releases publish; member
# checksums were taken from those downloads.
_FIRECRACKER_TGZ_SHA256 = {
    "x86_64": "06094a1108ae9e82aa4c23a775aa92758f53f1175d422270d9d6162cb9ade558",
    "aarch64": "e351ebe4f7a16b5873bbd51005d2e6767103cff4d5ebc829df2d3f95a93e2256",
}
_FIRECRACKER_SHA256 = {
    "x86_64": "99ad0f5cd0514a88aad0e9ae8cfdb3cc3b4ab9d190e1194602406c786b5de7a5",
    "aarch64": "fe726e0b43c04363ac07e358be4dee982c3947c65ed3ae10c770fef5e1cd756c",
}
_JAILER_SHA256 = {
    "x86_64": "65ef226e96f0ceda55ba643f445801ef2cc0ea667ef67cad8ac4f406c9c8434f",
    "aarch64": "4d8d2dd4dfc1d47932b2bd261479181dd0c54f0a2f32714421a7e7f9f07ddec8",
}
_KATA_TARBALL_SHA256 = {
    "x86_64": "2c3b9dfeba355582b40aee462b12916c9740654d0230f696adf719d67b063a8c",
    "aarch64": "730c789efa2a1e0a762875f5241126c8558fb238e87562ae52e1f1f2a748385c",
}
_KERNEL_SHA256 = {
    "x86_64": "6abc48fa83c58e3db314037105d04ec00c1bc80d85eb2dfb2e2854f414573bca",
    "aarch64": "4a8998a2e7ac12d6ad1f15b5e7d00571e4518ea9f33db1a1a568310373ca428d",
}


class ArtifactError(FirecrackerError):
    pass


@dataclass(frozen=True)
class ArchiveFile:
    member: str
    """Path inside the archive."""
    name: str
    """File name in the cache."""
    sha256: str
    executable: bool = False


@dataclass(frozen=True)
class Archive:
    url: str
    sha256: str
    compression: Literal["gz", "zst"]
    files: tuple[ArchiveFile, ...]


@dataclass(frozen=True)
class Artifacts:
    firecracker: Path
    jailer: Path
    kernel: Path


def cache_dir() -> Path:
    """Where the Firecracker runtime keeps artifacts, base drives and runs."""
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "karotte" / "firecracker"


def host_arch() -> str:
    machine = platform.machine()
    return {"amd64": "x86_64", "arm64": "aarch64"}.get(machine, machine)


def firecracker_archive(arch: Arch) -> Archive:
    v = FIRECRACKER_VERSION
    return Archive(
        url=f"https://github.com/firecracker-microvm/firecracker/releases/download/{v}/firecracker-{v}-{arch}.tgz",
        sha256=_FIRECRACKER_TGZ_SHA256[arch],
        compression="gz",
        files=(
            ArchiveFile(
                member=f"release-{v}-{arch}/firecracker-{v}-{arch}",
                name=f"firecracker-{v}-{arch}",
                sha256=_FIRECRACKER_SHA256[arch],
                executable=True,
            ),
            ArchiveFile(
                member=f"release-{v}-{arch}/jailer-{v}-{arch}",
                name=f"jailer-{v}-{arch}",
                sha256=_JAILER_SHA256[arch],
                executable=True,
            ),
        ),
    )


def kernel_archive(arch: Arch) -> Archive:
    v = KATA_VERSION
    return Archive(
        url=f"https://github.com/kata-containers/kata-containers/releases/download/{v}/kata-static-{v}-{_KATA_ARCH[arch]}.tar.zst",
        sha256=_KATA_TARBALL_SHA256[arch],
        compression="zst",
        files=(
            ArchiveFile(
                member=f"./opt/kata/share/kata-containers/{KERNEL_NAME}",
                name=f"{KERNEL_NAME}-kata-{v}-{arch}",
                sha256=_KERNEL_SHA256[arch],
            ),
        ),
    )


Downloader = Callable[[str, Path], str]
"""Writes the URL's content to the path and returns its SHA-256 hex digest."""

Extractor = Callable[[Path, Archive, ArchiveFile, Path], None]
"""Writes one member of the downloaded archive to the path."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path) -> str:
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=60) as response, dest.open("wb") as f:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        next_report = 0.1
        while chunk := response.read(1 << 20):
            f.write(chunk)
            digest.update(chunk)
            done += len(chunk)
            if total and done / total >= next_report:
                logger.info(f"Downloaded {done >> 20} of {total >> 20} MiB")
                next_report += 0.1
    return digest.hexdigest()


def extract(
    archive_path: Path, archive: Archive, file: ArchiveFile, dest: Path
) -> None:
    if archive.compression == "gz":
        with tarfile.open(archive_path, "r:gz") as tar:
            src = tar.extractfile(file.member)
            if src is None:
                raise ArtifactError(f"{file.member} is not a file in {archive.url}")
            with src, dest.open("wb") as out:
                shutil.copyfileobj(src, out)
        return
    # Python before 3.14 has no zstd, and the host may not either; the helper
    # container has both zstd and tar.
    from karotte.firecracker.helper import helper_image

    with dest.open("wb") as out:
        subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--volume",
                f"{archive_path.parent}:/in:ro",
                helper_image(),
                "tar",
                "--zstd",
                "-xOf",
                f"/in/{archive_path.name}",
                file.member,
            ],
            stdout=out,
            check=True,
        )


def _verified(path: Path, sha256: str) -> bool:
    return path.is_file() and sha256_file(path) == sha256


def ensure_archive(
    archive: Archive,
    directory: Path,
    *,
    download: Downloader = download,
    extract: Extractor = extract,
) -> list[Path]:
    """The archive's files in ``directory``, downloading and extracting them
    if any is missing or doesn't match its checksum."""
    paths = [directory / f.name for f in archive.files]
    if all(_verified(p, f.sha256) for p, f in zip(paths, archive.files)):
        return paths

    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory, prefix=".download-") as tmp:
        archive_path = Path(tmp) / Path(archive.url).name
        logger.info(f"Downloading {archive.url}")
        digest = download(archive.url, archive_path)
        if digest != archive.sha256:
            raise ArtifactError(
                f"{archive.url} has SHA-256 {digest}, expected {archive.sha256}"
            )
        for path, file in zip(paths, archive.files):
            staged = Path(tmp) / file.name
            extract(archive_path, archive, file, staged)
            digest = sha256_file(staged)
            if digest != file.sha256:
                raise ArtifactError(
                    f"{file.member} from {archive.url} has SHA-256 {digest}, expected {file.sha256}"
                )
            staged.chmod(0o755 if file.executable else 0o644)
            os.replace(staged, path)
    return paths


def ensure_artifacts(
    directory: Path | None = None,
    arch: str | None = None,
    *,
    download: Downloader = download,
    extract: Extractor = extract,
) -> Artifacts:
    """The Firecracker binary, jailer and guest kernel, downloaded on first use."""
    arch = arch or host_arch()
    if arch not in SUPPORTED_ARCHES:
        raise ArtifactError(f"Firecracker runs on x86_64 and aarch64, not {arch}")
    directory = directory or cache_dir() / "artifacts"
    firecracker, jailer = ensure_archive(
        firecracker_archive(arch),
        directory,
        download=download,
        extract=extract,
    )
    (kernel,) = ensure_archive(
        kernel_archive(arch),
        directory,
        download=download,
        extract=extract,
    )
    return Artifacts(firecracker=firecracker, jailer=jailer, kernel=kernel)
