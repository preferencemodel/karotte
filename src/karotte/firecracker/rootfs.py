"""The read-only base drive: the image's filesystem as ext4 with the guest init
written into it, cached by image ID.

``docker export | mkfs.ext4 -d -`` keeps owners, modes, hardlinks and file
capabilities with no root and no loop mount. The filesystem is made in a
large sparse file and then shrunk to fit, since an export's size isn't known
until it has streamed.
"""

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from karotte.firecracker import FirecrackerError
from karotte.firecracker.helper import helper_image

GUEST_DIR = Path(__file__).parent / "guest"
GUEST_FILES = {"init": "0100755", "agent.py": "0100644"}
"""Files written into the base drive's /.karotte, with their inode modes."""

REQUIRED_IMAGE_ENV = ("KAROTTE_CONTAINERIZED", "KAROTTE_DEMOTE_ID")
"""Without these karotte runs unconfined in the guest, so the image must set them."""

_SPARSE_BYTES = 1 << 40
"""Size of the file the base filesystem is made in before it's shrunk."""

_KEEP_BASE_DRIVES = 4

_IMAGE_ARCH = {"amd64": "x86_64", "arm64": "aarch64"}


class RootfsError(FirecrackerError):
    pass


@dataclass(frozen=True)
class ImageConfig:
    id: str
    env: tuple[str, ...]
    working_dir: str
    architecture: str
    """Normalized to the kernel's name (``x86_64``, ``aarch64``)."""


def inspect_image(image: str, engine: str = "docker") -> ImageConfig:
    result = subprocess.run(
        [engine, "image", "inspect", "--format", "{{json .}}", image],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RootfsError(
            f"Cannot inspect image {image!r}: {result.stderr.strip() or 'unknown error'}"
        )
    return parse_image_config(json.loads(result.stdout))


def parse_image_config(data: dict[str, object]) -> ImageConfig:
    config = data.get("Config")
    config = config if isinstance(config, dict) else {}
    arch = str(data.get("Architecture") or "")
    return ImageConfig(
        id=str(data["Id"]),
        env=tuple(str(e) for e in config.get("Env") or ()),
        working_dir=str(config.get("WorkingDir") or "/"),
        architecture=_IMAGE_ARCH.get(arch, arch),
    )


def check_image_env(config: ImageConfig) -> None:
    names = {e.partition("=")[0] for e in config.env}
    missing = [name for name in REQUIRED_IMAGE_ENV if name not in names]
    if missing:
        raise RootfsError(
            f"The image doesn't set {', '.join(missing)}; karotte would run unconfined in the VM. Set them in the Containerfile."
        )


def guest_files_digest() -> str:
    digest = hashlib.sha256()
    for name in sorted(GUEST_FILES):
        digest.update(name.encode() + b"\0")
        digest.update((GUEST_DIR / name).read_bytes())
    return digest.hexdigest()


def base_drive_path(config: ImageConfig, directory: Path) -> Path:
    image_id = config.id.removeprefix("sha256:")[:20]
    return directory / f"{image_id}-{guest_files_digest()[:10]}.ext4"


def debugfs_script() -> str:
    """Writes the guest files into /.karotte, owned by root."""
    lines = ["mkdir /.karotte", "cd /.karotte"]
    for name, mode in GUEST_FILES.items():
        lines += [
            f"write /guest/{name} {name}",
            f"sif {name} mode {mode}",
            f"sif {name} uid 0",
            f"sif {name} gid 0",
        ]
    lines += ["sif /.karotte mode 040755", "sif /.karotte uid 0", "sif /.karotte gid 0"]
    return "\n".join(lines) + "\n"


# Runs in the helper container with the export on stdin. Arguments: output
# file name, then the uid and gid to hand it to.
_BUILD_SCRIPT = f"""\
set -e
out="/out/$1"
truncate -s {_SPARSE_BYTES} "$out"
mkfs.ext4 -q -F -L karotte-base -O ^has_journal -E lazy_itable_init=1,root_owner=0:0 -d - "$out"
debugfs -w -f /guest/debugfs.cmds "$out" >/dev/null
for f in {" ".join(GUEST_FILES)}; do
  debugfs -R "cat /.karotte/$f" "$out" 2>/dev/null | cmp -s - "/guest/$f"
done
e2fsck -fy "$out" >/dev/null || [ $? -eq 1 ]
resize2fs -M "$out" >/dev/null 2>&1
chown "$2:$3" "$out"
"""


def build_base_drive(
    image: str, directory: Path, engine: str = "docker", link_to: Path | None = None
) -> tuple[Path, ImageConfig]:
    """The base drive for the image, built on first use.

    With ``link_to``, the drive is hard-linked there and that path returned:
    another run's eviction of old drives then can't take it away before this
    run's VM opens it. A drive evicted between the check and the link is
    built again."""
    config = inspect_image(image, engine)
    check_image_env(config)
    path = base_drive_path(config, directory)
    for _ in range(3):
        if path.exists():
            try:
                os.utime(path)
            except FileNotFoundError:
                continue
        else:
            _build(engine, image, config, directory, path)
        if link_to is None:
            return path, config
        try:
            os.link(path, link_to)
        except FileNotFoundError:
            continue
        return link_to, config
    raise RootfsError(f"The base drive for {image} kept being removed while in use")


def _build(
    engine: str, image: str, config: ImageConfig, directory: Path, path: Path
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    logger.info(f"Building the Firecracker base drive for {image} ({config.id[:19]})")
    helper = helper_image(engine)
    with tempfile.TemporaryDirectory(dir=directory, prefix=".build-") as tmp:
        guest = Path(tmp) / "guest"
        guest.mkdir()
        for name in GUEST_FILES:
            shutil.copyfile(GUEST_DIR / name, guest / name)
        (guest / "debugfs.cmds").write_text(debugfs_script())
        _export_to_ext4(engine, image, helper, Path(tmp), guest, "base.ext4")
        os.replace(Path(tmp) / "base.ext4", path)
    _evict_old(directory, keep=path)


def _export_to_ext4(
    engine: str, image: str, helper: str, out_dir: Path, guest: Path, name: str
) -> None:
    created = subprocess.run(
        [engine, "create", image, "true"], capture_output=True, text=True, check=True
    )
    container = created.stdout.strip()
    try:
        export = subprocess.Popen([engine, "export", container], stdout=subprocess.PIPE)
        assert export.stdout is not None
        build = subprocess.run(
            [
                engine,
                "run",
                "--interactive",
                "--rm",
                "--network",
                "none",
                "--volume",
                f"{out_dir}:/out",
                "--volume",
                f"{guest}:/guest:ro",
                helper,
                "sh",
                "-c",
                _BUILD_SCRIPT,
                "sh",
                name,
                str(os.getuid()),
                str(os.getgid()),
            ],
            stdin=export.stdout,
            check=False,
        )
        export.stdout.close()
        export_rc = export.wait()
        if export_rc != 0 or build.returncode != 0:
            raise RootfsError(
                f"Building the base drive for {image} failed (export exit {export_rc}, mkfs exit {build.returncode})"
            )
    finally:
        subprocess.run(
            [engine, "rm", "--force", container], capture_output=True, check=False
        )


def _evict_old(directory: Path, keep: Path) -> None:
    drives = sorted(
        directory.glob("*.ext4"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    for old in drives[_KEEP_BASE_DRIVES:]:
        if old != keep:
            logger.info(f"Removing old base drive {old}")
            old.unlink(missing_ok=True)
