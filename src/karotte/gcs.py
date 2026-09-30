import os
from pathlib import Path


def _get_storage_client():
    try:
        from google.cloud import storage
    except ImportError:
        raise ImportError(
            "google-cloud-storage is required for GCS operations. Install it with: uv add 'karotte[gcs]'"
        )
    return storage.Client()


def download_gcs_dir(bucket_name: str, gcs_prefix: str, local_dir: str):
    """
    Download an entire directory from Google Cloud Storage to the local filesystem.

    Args:
        bucket_name: Name of the GCS bucket.
        gcs_prefix: Path prefix in GCS (e.g. 'shared/models/my-model/').
        local_dir: Local directory path where files will be downloaded.

    Example:
        from karotte.gcs import download_gcs_dir # doctest: +SKIP
        download_gcs_dir(
            bucket_name='my-bucket',
            gcs_prefix='shared/models/Qwen2-1.5B-Instruct/',
            local_dir='./Qwen2-1.5B-Instruct'
        )
    """
    client = _get_storage_client()
    bucket = client.bucket(bucket_name)

    for blob in bucket.list_blobs(prefix=gcs_prefix):
        relative = blob.name[len(gcs_prefix) :]
        if not relative:
            continue
        local_path = os.path.join(local_dir, relative)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)


def upload_gcs_dir(local_dir: str, bucket_name: str, gcs_prefix: str):
    """
    Upload an entire local directory to Google Cloud Storage.

    Args:
        local_dir: Local directory path to upload.
        bucket_name: Name of the GCS bucket.
        gcs_prefix: Path prefix in GCS where files will be uploaded
                     (e.g. 'shared/models/my-model/').

    Example:
        from karotte.gcs import upload_gcs_dir
        upload_gcs_dir(
            local_dir='./Qwen2-1.5B-Instruct',
            bucket_name='my-bucket',
            gcs_prefix='shared/models/Qwen2-1.5B-Instruct/'
        )
    """
    client = _get_storage_client()
    bucket = client.bucket(bucket_name)
    local_root = Path(local_dir)

    for path in local_root.rglob("*"):
        if not path.is_file():
            continue
        blob_name = gcs_prefix + str(path.relative_to(local_root))
        bucket.blob(blob_name).upload_from_filename(str(path))
