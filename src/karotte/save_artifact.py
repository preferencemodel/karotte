import gzip
import io
import os
import stat
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import httpx
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_delay,
    wait_exponential,
)

from karotte.backend_client import read_token
from karotte.container import is_containerized
from karotte.schemas.evaluation_run_config import EvaluationRunConfig
from karotte.untrusted_paths import (
    DIR_FLAGS,
    FILE_FLAGS,
    lstat_at,
    open_at,
    walk_to_parent,
)

_MAX_FILE_BYTES = 1024**3
_MAX_TOTAL_BYTES = 4 * 1024**3
_MAX_ENTRIES = 10_000


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code >= 500:
        return True
    return False


def save_artifact(config: EvaluationRunConfig, path: Path) -> None:
    """Saves artifact from the specified path. Artifacts can be files or directories.

    If `config.backend_uri` is set, uploads the artifact directly to object storage via presigned URLs.
    Otherwise, saves locally to:
    - `$KAROTTE_ARTIFACT_DIR/<run_id>_artifacts/` if `KAROTTE_ARTIFACT_DIR` is set
    - `<run_id>_artifacts/` next to `config.transcript_file` if it is set
    - `/out/<run_id>_artifacts/` if containerized
    - `out/<run_id>_artifacts/` if not containerized

    No-ops if `config.save_artifacts` is False.

    The path may be written by the student, so symlinks, special files and files
    over the size limits are skipped, and any failure is logged instead of raised.
    """
    if not config.save_artifacts:
        logger.debug("Artifact saving disabled, skipping {}", path)
        return

    if not os.path.lexists(path):
        logger.warning("Artifact path {} does not exist, skipping save.", path)
        return

    try:
        if config.backend_uri:
            _upload_artifact_via_presigned_url(config, path)
        else:
            _save_artifact_locally(config, path)
    except (_Refused, OSError) as e:
        logger.warning("Not saving artifact {}: {}", path, e)


class _Refused(Exception):
    pass


class _Listing(NamedTuple):
    is_dir: bool
    dirs: list[str]
    files: list[str]


@dataclass
class _Budget:
    total_bytes: int = 0

    def take(self, n: int) -> None:
        self.total_bytes += n
        if self.total_bytes > _MAX_TOTAL_BYTES:
            raise _Refused(f"artifact is larger than {_MAX_TOTAL_BYTES} bytes")


def _list_artifact(path: Path) -> _Listing:
    """Directories and regular files under `path`, relative to it, without following symlinks."""
    parent_fd, final = walk_to_parent(Path(os.path.abspath(path)))
    try:
        st = lstat_at(parent_fd, final)
    finally:
        os.close(parent_fd)
    if stat.S_ISREG(st.st_mode):
        if st.st_size > _MAX_FILE_BYTES:
            raise _Refused(f"larger than {_MAX_FILE_BYTES} bytes")
        return _Listing(False, [], [""])
    if not stat.S_ISDIR(st.st_mode):
        raise _Refused("not a regular file or directory")

    dirs: list[str] = []
    files: list[str] = []
    entries = 0
    pending = [""]
    while pending:
        rel = pending.pop()
        fd = _open_under(path, rel, DIR_FLAGS)
        try:
            with os.scandir(fd) as it:
                for entry in it:
                    entries += 1
                    if entries > _MAX_ENTRIES:
                        raise _Refused(f"more than {_MAX_ENTRIES} entries")
                    child = f"{rel}/{entry.name}" if rel else entry.name
                    if entry.is_dir(follow_symlinks=False):
                        dirs.append(child)
                        pending.append(child)
                    elif not entry.is_file(follow_symlinks=False):
                        logger.warning(
                            "Skipping {}/{}: not a regular file", path, child
                        )
                    elif entry.stat(follow_symlinks=False).st_size > _MAX_FILE_BYTES:
                        logger.warning("Skipping {}/{}: too large", path, child)
                    else:
                        files.append(child)
        finally:
            os.close(fd)
    return _Listing(True, sorted(dirs), sorted(files))


def _open_under(path: Path, rel: str, flags: int) -> int:
    """Open `path/rel` without following a symlink anywhere in either."""
    parts = rel.split("/") if rel else []
    parent_fd, final = walk_to_parent(Path(os.path.abspath(path)))
    try:
        fd = open_at(parent_fd, final, DIR_FLAGS if parts else flags)
    finally:
        os.close(parent_fd)
    for i, name in enumerate(parts):
        try:
            next_fd = os.open(
                name, flags if i == len(parts) - 1 else DIR_FLAGS, dir_fd=fd
            )
        finally:
            os.close(fd)
        fd = next_fd
    return fd


def _open_file(path: Path, rel: str) -> int:
    fd = _open_under(path, rel, FILE_FLAGS)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise _Refused(f"{rel or path} is not a regular file")
    return fd


def _chunks(fd: int, budget: _Budget) -> Iterator[bytes]:
    read = 0
    while chunk := os.read(fd, _UPLOAD_CHUNK_SIZE):
        read += len(chunk)
        if read > _MAX_FILE_BYTES:
            raise _Refused(f"a file grew past {_MAX_FILE_BYTES} bytes")
        budget.take(len(chunk))
        yield chunk


def _save_artifact_locally(config: EvaluationRunConfig, path: Path) -> None:
    listing = _list_artifact(path)
    dest_root = _maybe_create_target_dir(config) / path.name
    logger.info("Saving artifact {} to {}", path, dest_root)

    if listing.is_dir:
        dest_root.mkdir(exist_ok=True)
        for rel in listing.dirs:
            (dest_root / rel).mkdir(exist_ok=True)

    budget = _Budget()
    for rel in listing.files:
        _copy_file(path, rel, dest_root / rel if rel else dest_root, budget)


def _copy_file(path: Path, rel: str, dest: Path, budget: _Budget) -> None:
    src = _open_file(path, rel)
    try:
        st = os.fstat(src)
        out = os.open(
            dest, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600
        )
        try:
            for chunk in _chunks(src, budget):
                view = memoryview(chunk)
                while view:
                    view = view[os.write(out, view) :]
            os.fchmod(out, stat.S_IMODE(st.st_mode) & 0o777)
            os.utime(out, ns=(st.st_atime_ns, st.st_mtime_ns))
        except BaseException:
            dest.unlink(missing_ok=True)
            raise
        finally:
            os.close(out)
    finally:
        os.close(src)


_created_target_dir: Path | None = None

ARTIFACT_DIR_ENV_VAR = "KAROTTE_ARTIFACT_DIR"
"""Overrides where local artifacts land. Set by an image whose external
harness collects artifacts from a directory of its choosing."""


def local_artifact_dir() -> Path | None:
    """The directory this process saved artifacts to locally, if any."""
    return _created_target_dir


def _artifact_base_dir(config: EvaluationRunConfig) -> Path:
    if override := os.environ.get(ARTIFACT_DIR_ENV_VAR):
        return Path(override)
    if config.transcript_file:
        return Path(config.transcript_file).parent
    return Path("/out") if is_containerized() else Path("out")


def _maybe_create_target_dir(config: EvaluationRunConfig) -> Path:
    global _created_target_dir

    if not _created_target_dir:
        target_dir = _artifact_base_dir(config)
        target_dir = target_dir / f"{config.run_id}_artifacts"
        base_name = target_dir.name

        suffix = 2

        while target_dir.exists():
            # Add a numeric suffix to avoid overwriting existing artifacts
            target_dir = target_dir.with_name(f"{base_name}_{suffix}")
            suffix += 1

        target_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Created artifact directory at {}", target_dir)

        _created_target_dir = target_dir

    return _created_target_dir


def _upload_artifact_via_presigned_url(config: EvaluationRunConfig, path: Path) -> None:
    """Upload each file of the artifact to a presigned URL from the backend.

    Upload failures are logged, not raised.
    """
    artifact_name = path.name
    listing = _list_artifact(path)
    files_to_upload = [
        (rel, f"{artifact_name}/{rel}" if rel else artifact_name)
        for rel in listing.files
    ]
    if not files_to_upload:
        logger.info("No files to upload for artifact {}", artifact_name)
        return

    try:
        presigned_urls = _get_presigned_urls(
            config, [artifact_path for _, artifact_path in files_to_upload]
        )
        if not presigned_urls:
            return  # Error already logged

        budget = _Budget()
        uploaded_count = 0
        for rel, artifact_path in files_to_upload:
            presigned_url = presigned_urls.get(artifact_path)
            if not presigned_url:
                logger.error("No presigned URL for {}", artifact_path)
                continue

            fd = _open_file(path, rel)
            try:
                if _upload_file(_chunks(fd, budget), artifact_path, presigned_url):
                    uploaded_count += 1
            finally:
                os.close(fd)

        logger.info(
            "Uploaded {}/{} files for artifact {}",
            uploaded_count,
            len(files_to_upload),
            artifact_name,
        )

    except _Refused:
        raise
    except Exception as e:
        logger.error("Failed to upload artifact {}: {}", artifact_name, e)


@retry(
    retry=retry_if_exception(_is_retryable),
    stop=stop_after_delay(300),
    wait=wait_exponential(multiplier=1, max=10),
    reraise=True,
)
def _post_presigned_urls(
    url: str, config: EvaluationRunConfig, artifact_paths: list[str]
) -> dict[str, str]:
    token = read_token()
    response = httpx.post(
        url,
        json={
            "run_id": config.run_id,
            "artifact_paths": artifact_paths,
        },
        headers={"Authorization": f"Bearer {token}"} if token else {},
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()["presigned_urls"]


def _get_presigned_urls(
    config: EvaluationRunConfig, artifact_paths: list[str]
) -> dict[str, str]:
    """Request presigned URLs from the backend."""
    url = f"{config.backend_uri}/api/artifacts/presign"

    try:
        return _post_presigned_urls(url, config, artifact_paths)
    except httpx.HTTPStatusError as e:
        logger.error(
            "Failed to get presigned URLs: HTTP {} - {}",
            e.response.status_code,
            e.response.text,
        )
        return {}
    except httpx.RequestError as e:
        logger.error("Failed to get presigned URLs: {}", e)
        return {}


_UPLOAD_CHUNK_SIZE = 1 << 20


def _gzip_stream(chunks: Iterator[bytes]) -> Iterator[bytes]:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=6) as gz:
        for chunk in chunks:
            gz.write(chunk)
            if buf.tell():
                yield buf.getvalue()
                buf.seek(0)
                buf.truncate()
    yield buf.getvalue()


def _upload_file(chunks: Iterator[bytes], name: str, presigned_url: str) -> bool:
    """Stream one file, gzipped, to a presigned URL. Returns whether it succeeded."""
    try:
        response = httpx.put(
            presigned_url,
            content=_gzip_stream(chunks),
            headers={
                "Content-Encoding": "gzip",
            },
            timeout=httpx.Timeout(
                connect=30.0,
                read=300.0,
                write=300.0,
                pool=30.0,
            ),
        )
        response.raise_for_status()
        return True

    except httpx.HTTPStatusError as e:
        logger.error("Failed to upload {}: HTTP {}", name, e.response.status_code)
        return False
    except httpx.RequestError as e:
        logger.error("Failed to upload {}: {}", name, e)
        return False
