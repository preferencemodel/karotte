from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestGetStorageClient:
    def test_raises_import_error_with_install_hint_when_missing(self):
        with patch.dict("sys.modules", {"google.cloud": None, "google": None}):
            from importlib import reload

            import karotte.gcs as gcs_module

            reload(gcs_module)

            with pytest.raises(ImportError, match=r"uv add 'karotte\[gcs\]'"):
                gcs_module._get_storage_client()  # pyright: ignore[reportPrivateUsage]


class TestDownloadGcsDir:
    def test_downloads_files_to_local_dir(self, tmp_path: Path):
        blob_a = MagicMock()
        blob_a.name = "prefix/a.txt"

        blob_b = MagicMock()
        blob_b.name = "prefix/sub/b.txt"

        bucket = MagicMock()
        bucket.list_blobs.return_value = [blob_a, blob_b]

        client = MagicMock()
        client.bucket.return_value = bucket

        with patch("karotte.gcs._get_storage_client", return_value=client):
            from karotte.gcs import download_gcs_dir

            local = tmp_path / "out"
            download_gcs_dir("my-bucket", "prefix/", str(local))

        client.bucket.assert_called_once_with("my-bucket")
        bucket.list_blobs.assert_called_once_with(prefix="prefix/")

        blob_a.download_to_filename.assert_called_once_with(str(local / "a.txt"))
        blob_b.download_to_filename.assert_called_once_with(
            str(local / "sub" / "b.txt")
        )

    def test_creates_intermediate_directories(self, tmp_path: Path):
        blob = MagicMock()
        blob.name = "pfx/deep/nested/file.bin"

        bucket = MagicMock()
        bucket.list_blobs.return_value = [blob]

        client = MagicMock()
        client.bucket.return_value = bucket

        with patch("karotte.gcs._get_storage_client", return_value=client):
            from karotte.gcs import download_gcs_dir

            local = tmp_path / "dl"
            download_gcs_dir("b", "pfx/", str(local))

        # The parent directories should have been created
        assert (local / "deep" / "nested").is_dir()

    def test_skips_prefix_only_blob(self, tmp_path: Path):
        """A blob whose name equals the prefix exactly (empty relative path) is skipped."""
        blob = MagicMock()
        blob.name = "prefix/"

        bucket = MagicMock()
        bucket.list_blobs.return_value = [blob]

        client = MagicMock()
        client.bucket.return_value = bucket

        with patch("karotte.gcs._get_storage_client", return_value=client):
            from karotte.gcs import download_gcs_dir

            download_gcs_dir("b", "prefix/", str(tmp_path / "out"))

        blob.download_to_filename.assert_not_called()

    def test_handles_empty_listing(self, tmp_path: Path):
        bucket = MagicMock()
        bucket.list_blobs.return_value = []

        client = MagicMock()
        client.bucket.return_value = bucket

        with patch("karotte.gcs._get_storage_client", return_value=client):
            from karotte.gcs import download_gcs_dir

            download_gcs_dir("b", "empty/", str(tmp_path / "out"))


class TestUploadGcsDir:
    def test_uploads_all_files_with_correct_blob_names(self, tmp_path: Path):
        local_dir = tmp_path / "src"
        local_dir.mkdir()
        (local_dir / "a.txt").write_text("a")
        sub = local_dir / "sub"
        sub.mkdir()
        (sub / "b.txt").write_text("b")

        blob_mock = MagicMock()
        bucket = MagicMock()
        bucket.blob.return_value = blob_mock

        client = MagicMock()
        client.bucket.return_value = bucket

        with patch("karotte.gcs._get_storage_client", return_value=client):
            from karotte.gcs import upload_gcs_dir

            upload_gcs_dir(str(local_dir), "my-bucket", "dest/")

        client.bucket.assert_called_once_with("my-bucket")

        blob_names = sorted(c.args[0] for c in bucket.blob.call_args_list)
        assert blob_names == ["dest/a.txt", "dest/sub/b.txt"]

        assert blob_mock.upload_from_filename.call_count == 2

    def test_skips_directories(self, tmp_path: Path):
        local_dir = tmp_path / "src"
        local_dir.mkdir()
        (local_dir / "subdir").mkdir()
        (local_dir / "file.txt").write_text("x")

        blob_mock = MagicMock()
        bucket = MagicMock()
        bucket.blob.return_value = blob_mock

        client = MagicMock()
        client.bucket.return_value = bucket

        with patch("karotte.gcs._get_storage_client", return_value=client):
            from karotte.gcs import upload_gcs_dir

            upload_gcs_dir(str(local_dir), "b", "p/")

        # Only the file, not the directory
        assert blob_mock.upload_from_filename.call_count == 1
        bucket.blob.assert_called_once_with("p/file.txt")

    def test_handles_empty_directory(self, tmp_path: Path):
        local_dir = tmp_path / "empty"
        local_dir.mkdir()

        bucket = MagicMock()
        client = MagicMock()
        client.bucket.return_value = bucket

        with patch("karotte.gcs._get_storage_client", return_value=client):
            from karotte.gcs import upload_gcs_dir

            upload_gcs_dir(str(local_dir), "b", "p/")

        bucket.blob.assert_not_called()
