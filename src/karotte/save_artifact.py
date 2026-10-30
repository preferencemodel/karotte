import gzip
import io
import os
import shutil
from collections.abc import Iterator
from pathlib import Path

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

    Logs a warning and no-ops if the specified path does not exist.
    """
    if not config.save_artifacts:
        logger.debug("Artifact saving disabled, skipping {}", path)
        return

    if not path.exists():
        logger.warning("Artifact path {} does not exist, skipping save.", path)
        return

    if config.backend_uri:
        _upload_artifact_via_presigned_url(config, path)
    else:
        _save_artifact_locally(config, path)


def _save_artifact_locally(config: EvaluationRunConfig, path: Path) -> None:
    """Save artifact to local filesystem."""
    target_dir = _maybe_create_target_dir(config)

    if path.is_dir():
        logger.info("Saving artifact from {} to {}", path, target_dir / path.name)
        shutil.copytree(path, target_dir / path.name, dirs_exist_ok=True)
    else:
        logger.info("Saving artifact {} to {}", path, target_dir / path.name)
        shutil.copy2(path, target_dir / path.name)


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
    """Upload artifact to object storage via presigned URLs from the backend.

    1. Collect all file paths to upload
    2. Request presigned URLs from backend in batch
    3. Upload each file directly to storage

    Logs errors but does not raise - artifact upload failures should not fail the run.
    """
    artifact_name = path.name

    try:
        # Collect all files to upload
        if path.is_dir():
            files_to_upload = [
                (f, f"{artifact_name}/{f.relative_to(path)}")
                for f in path.rglob("*")
                if f.is_file()
            ]
        else:
            files_to_upload = [(path, artifact_name)]

        if not files_to_upload:
            logger.info("No files to upload for artifact {}", artifact_name)
            return

        # Request presigned URLs from backend
        artifact_paths = [artifact_path for _, artifact_path in files_to_upload]
        presigned_urls = _get_presigned_urls(config, artifact_paths)

        if not presigned_urls:
            return  # Error already logged

        # Upload each file to storage
        uploaded_count = 0
        for file_path, artifact_path in files_to_upload:
            presigned_url = presigned_urls.get(artifact_path)
            if not presigned_url:
                logger.error("No presigned URL for {}", artifact_path)
                continue

            if _upload_file(file_path, presigned_url):
                uploaded_count += 1

        logger.info(
            "Uploaded {}/{} files for artifact {}",
            uploaded_count,
            len(files_to_upload),
            artifact_name,
        )

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


def _gzip_stream(file_path: Path) -> Iterator[bytes]:
    buf = io.BytesIO()
    with (
        file_path.open("rb") as src,
        gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=6) as gz,
    ):
        while chunk := src.read(_UPLOAD_CHUNK_SIZE):
            gz.write(chunk)
            if buf.tell():
                yield buf.getvalue()
                buf.seek(0)
                buf.truncate()
    yield buf.getvalue()


def _upload_file(file_path: Path, presigned_url: str) -> bool:
    """Stream a single file, gzipped, to a presigned URL.

    Returns True if successful, False otherwise.
    """
    try:
        response = httpx.put(
            presigned_url,
            content=_gzip_stream(file_path),
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
        logger.error(
            "Failed to upload {}: HTTP {}",
            file_path.name,
            e.response.status_code,
        )
        return False
    except httpx.RequestError as e:
        logger.error("Failed to upload {}: {}", file_path.name, e)
        return False
