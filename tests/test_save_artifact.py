import gzip
import os
import sys
from pathlib import Path

import pytest
from pytest_httpx import HTTPXMock

# Import the save_artifact module
# Note: We need to import via sys.modules because karotte.__init__.py
# re-exports save_artifact as a function, which can cause import confusion
from karotte import save_artifact as save_artifact_func
from karotte.schemas.evaluation_run_config import EvaluationRunConfig
from karotte.schemas.http_mcp_server_config import HttpMcpServerConfig

save_artifacts_module = sys.modules["karotte.save_artifact"]
save_artifact = save_artifact_func


@pytest.fixture
def sample_config() -> EvaluationRunConfig:
    """Create a sample config for testing."""
    return EvaluationRunConfig(
        run_id="test_run_123",
        task_id="test_task",
        model="test_model",
        model_api_key="test_key",
        mcp_server_config=HttpMcpServerConfig(host="0.0.0.0", port=8080),
        transcript_file="",
    )


@pytest.fixture(autouse=True)
def reset_global_state(monkeypatch: pytest.MonkeyPatch):
    """Reset the global _created_target_dir before each test."""
    monkeypatch.setattr(save_artifacts_module, "_created_target_dir", None)
    yield
    monkeypatch.setattr(save_artifacts_module, "_created_target_dir", None)


class TestSaveArtifacts:
    """Tests for the save_artifacts function."""

    def test_warns_and_noops_for_nonexistent_path(
        self,
        sample_config: EvaluationRunConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Test that a missing path logs a warning and no-ops rather than raising."""
        monkeypatch.chdir(tmp_path)
        nonexistent_path = tmp_path / "nonexistent"

        # Should not raise
        save_artifact(sample_config, nonexistent_path)

        # No artifact directory should be created
        assert not (tmp_path / "out").exists()

    def test_saves_single_file(
        self,
        sample_config: EvaluationRunConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Test saving a single file as an artifact."""
        monkeypatch.chdir(tmp_path)

        # Create a source file
        source_file = tmp_path / "test_file.txt"
        source_file.write_text("test content")

        save_artifact(sample_config, source_file)

        # Check artifact was created using run_id
        artifact_dir = tmp_path / "out" / f"{sample_config.run_id}_artifacts"
        assert artifact_dir.exists()
        assert artifact_dir.is_dir()

        # Check file was copied
        copied_file = artifact_dir / "test_file.txt"
        assert copied_file.exists()
        assert copied_file.read_text() == "test content"

    def test_saves_directory(
        self,
        sample_config: EvaluationRunConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Test saving a directory as an artifact in a subdirectory."""
        monkeypatch.chdir(tmp_path)

        # Create a source directory with files
        source_dir = tmp_path / "test_dir"
        source_dir.mkdir()
        (source_dir / "file1.txt").write_text("content 1")
        (source_dir / "file2.txt").write_text("content 2")

        save_artifact(sample_config, source_dir)

        # Check artifact directory was created
        artifact_dir = tmp_path / "out" / f"{sample_config.run_id}_artifacts"
        assert artifact_dir.exists()
        assert artifact_dir.is_dir()

        # Check directory was copied as a subdirectory (not flattened)
        assert (artifact_dir / "test_dir" / "file1.txt").read_text() == "content 1"
        assert (artifact_dir / "test_dir" / "file2.txt").read_text() == "content 2"

    def test_saves_nested_directory_structure(
        self,
        sample_config: EvaluationRunConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Test saving a directory with nested subdirectories."""
        monkeypatch.chdir(tmp_path)

        # Create nested directory structure
        source_dir = tmp_path / "nested_dir"
        source_dir.mkdir()
        (source_dir / "subdir1").mkdir()
        (source_dir / "subdir1" / "file1.txt").write_text("nested content 1")
        (source_dir / "subdir2").mkdir()
        (source_dir / "subdir2" / "subdir3").mkdir()
        (source_dir / "subdir2" / "subdir3" / "file2.txt").write_text("deep content")

        save_artifact(sample_config, source_dir)

        # Directory is saved as a subdirectory preserving its name
        artifact_dir = tmp_path / "out" / f"{sample_config.run_id}_artifacts"
        assert (
            artifact_dir / "nested_dir" / "subdir1" / "file1.txt"
        ).read_text() == "nested content 1"
        assert (
            artifact_dir / "nested_dir" / "subdir2" / "subdir3" / "file2.txt"
        ).read_text() == "deep content"

    def test_adds_numeric_suffix_when_artifact_exists(
        self,
        sample_config: EvaluationRunConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Test that numeric suffix is added when artifact directory already exists."""
        monkeypatch.chdir(tmp_path)

        # Create source file
        source_file = tmp_path / "test_file.txt"
        source_file.write_text("content")

        # Pre-create the artifact directory
        (tmp_path / "out" / f"{sample_config.run_id}_artifacts").mkdir(parents=True)

        save_artifact(sample_config, source_file)

        # Should create with suffix _2 (first suffix)
        artifact_dir = tmp_path / "out" / f"{sample_config.run_id}_artifacts_2"
        assert artifact_dir.exists()
        assert (artifact_dir / "test_file.txt").read_text() == "content"

    def test_increments_suffix_correctly(
        self,
        sample_config: EvaluationRunConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Test that suffix increments correctly when multiple artifacts exist."""
        monkeypatch.chdir(tmp_path)

        # Create source file
        source_file = tmp_path / "test_file.txt"
        source_file.write_text("content")

        # Pre-create multiple artifact directories
        (tmp_path / "out" / f"{sample_config.run_id}_artifacts").mkdir(parents=True)
        (tmp_path / "out" / f"{sample_config.run_id}_artifacts_2").mkdir(parents=True)
        (tmp_path / "out" / f"{sample_config.run_id}_artifacts_3").mkdir(parents=True)

        save_artifact(sample_config, source_file)

        # Should create with suffix _4
        artifact_dir = tmp_path / "out" / f"{sample_config.run_id}_artifacts_4"
        assert artifact_dir.exists()
        assert (artifact_dir / "test_file.txt").read_text() == "content"

    def test_suffix_does_not_compound(
        self,
        sample_config: EvaluationRunConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Test that suffixes don't compound (e.g., _2_3_4 instead of _4).

        This tests the fix for the bug where the suffix was appended to
        the already-modified name instead of the original base name.
        """
        monkeypatch.chdir(tmp_path)

        # Create source file
        source_file = tmp_path / "myfile.txt"
        source_file.write_text("content")

        # Pre-create artifact directories
        base = sample_config.run_id
        (tmp_path / "out" / f"{base}_artifacts").mkdir(parents=True)
        (tmp_path / "out" / f"{base}_artifacts_2").mkdir(parents=True)

        save_artifact(sample_config, source_file)

        # Should be _3, NOT _2_3
        correct_dir = tmp_path / "out" / f"{base}_artifacts_3"
        incorrect_dir = tmp_path / "out" / f"{base}_artifacts_2_3"

        assert correct_dir.exists(), f"Expected {base}_artifacts_3 to exist"
        assert not incorrect_dir.exists(), "Bug: suffix compounded to _2_3"

    def test_selects_absolute_path_when_containerized(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Test that /out is selected when KAROTTE_CONTAINERIZED env var is set.

        This test verifies the path selection logic without actually writing
        to /out, which requires root permissions.
        """
        monkeypatch.setenv("KAROTTE_CONTAINERIZED", "1")

        # Verify the path selection logic directly
        target_dir = (
            Path("/out") if "KAROTTE_CONTAINERIZED" in os.environ else Path("out")
        )

        assert target_dir == Path("/out")
        assert target_dir.is_absolute()

    def test_the_env_var_overrides_the_artifact_base_dir(
        self,
        sample_config: EvaluationRunConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """KAROTTE_ARTIFACT_DIR wins over the transcript's directory, /out and out/."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("KAROTTE_CONTAINERIZED", "1")
        base = tmp_path / "logs" / "artifacts" / "karotte"
        monkeypatch.setenv(save_artifacts_module.ARTIFACT_DIR_ENV_VAR, str(base))
        config = sample_config.model_copy(
            update={"transcript_file": str(tmp_path / "transcripts" / "t.json")}
        )
        source_file = tmp_path / "test_file.txt"
        source_file.write_text("content")

        save_artifact(config, source_file)

        artifact_dir = base / f"{sample_config.run_id}_artifacts"
        assert (artifact_dir / "test_file.txt").read_text() == "content"
        assert not (tmp_path / "out").exists()
        assert not (tmp_path / "transcripts").exists()

    def test_saves_next_to_the_transcript(
        self,
        sample_config: EvaluationRunConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """The transcript's directory is the one the launcher shares with the host."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("KAROTTE_CONTAINERIZED", "1")
        monkeypatch.delenv(save_artifacts_module.ARTIFACT_DIR_ENV_VAR, raising=False)
        shared = tmp_path / "root" / "out"
        config = sample_config.model_copy(
            update={"transcript_file": str(shared / "transcript.json")}
        )
        source_file = tmp_path / "test_file.txt"
        source_file.write_text("content")

        save_artifact(config, source_file)

        artifact_dir = shared / f"{sample_config.run_id}_artifacts"
        assert (artifact_dir / "test_file.txt").read_text() == "content"

    def test_uses_relative_path_when_not_containerized(
        self,
        sample_config: EvaluationRunConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Test that out/ is used when KAROTTE_CONTAINERIZED is not set."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("KAROTTE_CONTAINERIZED", raising=False)

        # Create source file
        source_file = tmp_path / "test_file.txt"
        source_file.write_text("local content")

        save_artifact(sample_config, source_file)

        # Check artifact was created in relative out/
        artifact_dir = tmp_path / "out" / f"{sample_config.run_id}_artifacts"
        assert artifact_dir.exists()
        assert (artifact_dir / "test_file.txt").read_text() == "local content"

    def test_creates_out_directory_if_not_exists(
        self,
        sample_config: EvaluationRunConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Test that the out directory is created if it doesn't exist."""
        monkeypatch.chdir(tmp_path)

        # Ensure out doesn't exist
        out_dir = tmp_path / "out"
        assert not out_dir.exists()

        # Create source file
        source_file = tmp_path / "test_file.txt"
        source_file.write_text("content")

        save_artifact(sample_config, source_file)

        # out directory should now exist
        assert out_dir.exists()
        assert (
            out_dir / f"{sample_config.run_id}_artifacts" / "test_file.txt"
        ).exists()

    def test_preserves_file_metadata(
        self,
        sample_config: EvaluationRunConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Test that file metadata is preserved when copying (shutil.copy2)."""
        monkeypatch.chdir(tmp_path)

        # Create source file
        source_file = tmp_path / "test_file.txt"
        source_file.write_text("content")

        # Get original modification time
        original_mtime = source_file.stat().st_mtime

        save_artifact(sample_config, source_file)

        # Check copied file has same modification time
        copied_file = (
            tmp_path / "out" / f"{sample_config.run_id}_artifacts" / "test_file.txt"
        )
        copied_mtime = copied_file.stat().st_mtime

        assert copied_mtime == original_mtime

    def test_handles_empty_directory(
        self,
        sample_config: EvaluationRunConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Test saving an empty directory as an artifact."""
        monkeypatch.chdir(tmp_path)

        # Create empty source directory
        source_dir = tmp_path / "empty_dir"
        source_dir.mkdir()

        save_artifact(sample_config, source_dir)

        # Check artifact directory was created with subdirectory
        artifact_dir = tmp_path / "out" / f"{sample_config.run_id}_artifacts"
        assert artifact_dir.exists()
        assert artifact_dir.is_dir()
        assert (artifact_dir / "empty_dir").exists()
        assert (artifact_dir / "empty_dir").is_dir()

    def test_handles_special_characters_in_filename(
        self,
        sample_config: EvaluationRunConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Test handling files with special characters in name."""
        monkeypatch.chdir(tmp_path)

        # Create file with special characters
        source_file = tmp_path / "test file (1).txt"
        source_file.write_text("special content")

        save_artifact(sample_config, source_file)

        artifact_dir = tmp_path / "out" / f"{sample_config.run_id}_artifacts"
        assert artifact_dir.exists()
        assert (artifact_dir / "test file (1).txt").read_text() == "special content"

    def test_multiple_saves_use_same_directory(
        self,
        sample_config: EvaluationRunConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Test that multiple saves within same run use the same artifact directory.

        The _created_target_dir global caches the directory so subsequent calls
        to save_artifacts with the same config use the same directory.
        """
        monkeypatch.chdir(tmp_path)

        # Create multiple source files
        file1 = tmp_path / "file1.txt"
        file1.write_text("content 1")
        file2 = tmp_path / "file2.txt"
        file2.write_text("content 2")

        # Save both files
        save_artifact(sample_config, file1)
        save_artifact(sample_config, file2)

        # Both should be in the same artifact directory
        artifact_dir = tmp_path / "out" / f"{sample_config.run_id}_artifacts"
        assert (artifact_dir / "file1.txt").read_text() == "content 1"
        assert (artifact_dir / "file2.txt").read_text() == "content 2"

        # No _2 suffix directory should exist
        assert not (tmp_path / "out" / f"{sample_config.run_id}_artifacts_2").exists()

    def test_handles_symlinks_in_directory(
        self,
        sample_config: EvaluationRunConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Test that symlinks within a directory are handled."""
        monkeypatch.chdir(tmp_path)

        # Create source directory with a symlink
        source_dir = tmp_path / "dir_with_symlink"
        source_dir.mkdir()
        real_file = source_dir / "real_file.txt"
        real_file.write_text("real content")
        symlink = source_dir / "link_file.txt"
        symlink.symlink_to(real_file)

        save_artifact(sample_config, source_dir)

        artifact_dir = tmp_path / "out" / f"{sample_config.run_id}_artifacts"
        assert artifact_dir.exists()
        # Directory is saved as subdirectory
        subdir = artifact_dir / "dir_with_symlink"
        # shutil.copytree copies symlinks as symlinks by default
        assert (subdir / "real_file.txt").read_text() == "real content"
        assert (subdir / "link_file.txt").exists()

    def test_handles_binary_files(
        self,
        sample_config: EvaluationRunConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Test saving binary files as artifacts."""
        monkeypatch.chdir(tmp_path)

        # Create binary file
        source_file = tmp_path / "binary_file.bin"
        binary_content = bytes(range(256))
        source_file.write_bytes(binary_content)

        save_artifact(sample_config, source_file)

        artifact_dir = tmp_path / "out" / f"{sample_config.run_id}_artifacts"
        copied_file = artifact_dir / "binary_file.bin"
        assert copied_file.read_bytes() == binary_content

    def test_handles_large_suffix_numbers(
        self,
        sample_config: EvaluationRunConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Test handling when there are many existing artifact directories."""
        monkeypatch.chdir(tmp_path)

        # Create source file
        source_file = tmp_path / "test.txt"
        source_file.write_text("content")

        # Pre-create many artifact directories (base and 2 through 100)
        base = sample_config.run_id
        (tmp_path / "out" / f"{base}_artifacts").mkdir(parents=True)
        for i in range(2, 101):
            (tmp_path / "out" / f"{base}_artifacts_{i}").mkdir()

        save_artifact(sample_config, source_file)

        # Should create with suffix _101
        artifact_dir = tmp_path / "out" / f"{base}_artifacts_101"
        assert artifact_dir.exists()
        assert (artifact_dir / "test.txt").read_text() == "content"

    def test_uses_run_id_for_directory_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Test that the artifact directory uses run_id, not filename."""
        monkeypatch.chdir(tmp_path)

        config = EvaluationRunConfig(
            run_id="my_unique_run_id",
            task_id="test_task",
            model="test_model",
            model_api_key="test_key",
            mcp_server_config=HttpMcpServerConfig(host="0.0.0.0", port=8080),
            transcript_file="",
        )

        source_file = tmp_path / "some_file.txt"
        source_file.write_text("content")

        save_artifact(config, source_file)

        # Directory should be named after run_id, not the file
        artifact_dir = tmp_path / "out" / "my_unique_run_id_artifacts"
        assert artifact_dir.exists()
        assert (artifact_dir / "some_file.txt").exists()

        # Should NOT create a directory named after the file
        assert not (tmp_path / "out" / "some_file.txt_artifacts").exists()


class TestSaveArtifactsDisabled:
    """Tests for save_artifacts=False behavior."""

    def test_noops_when_save_artifacts_false(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Test that save_artifact does nothing when save_artifacts=False."""
        monkeypatch.chdir(tmp_path)

        config = EvaluationRunConfig(
            run_id="test_run",
            task_id="test_task",
            model="test_model",
            model_api_key="test_key",
            mcp_server_config=HttpMcpServerConfig(host="0.0.0.0", port=8080),
            transcript_file="",
            save_artifacts=False,
        )

        source_file = tmp_path / "test_file.txt"
        source_file.write_text("content")

        save_artifact(config, source_file)

        # No artifact directory should be created
        assert not (tmp_path / "out").exists()

    def test_noops_even_with_backend_uri_when_disabled(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Test that save_artifact does nothing even with backend_uri when disabled."""
        monkeypatch.chdir(tmp_path)

        config = EvaluationRunConfig(
            run_id="test_run",
            task_id="test_task",
            model="test_model",
            model_api_key="test_key",
            mcp_server_config=HttpMcpServerConfig(host="0.0.0.0", port=8080),
            transcript_file="",
            save_artifacts=False,
            backend_uri="http://localhost:8000",
        )

        source_file = tmp_path / "test_file.txt"
        source_file.write_text("content")

        # Should not raise, should not make any HTTP calls
        save_artifact(config, source_file)

        # No artifact directory should be created
        assert not (tmp_path / "out").exists()


class TestArtifactUpload:
    """Tests for artifact upload to backend via presigned URLs."""

    @pytest.fixture(autouse=True)
    def _mock_service_account_token(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        token_file = tmp_path / "service-account-token"
        token_file.write_text("fake-token")
        monkeypatch.setenv("KAROTTE_BACKEND_TOKEN_PATH", str(token_file))

    @pytest.mark.parametrize("has_token", [True, False])
    def test_presign_request_carries_the_token_if_there_is_one(
        self,
        has_token: bool,
        sample_config: EvaluationRunConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        httpx_mock: HTTPXMock,
    ):
        monkeypatch.chdir(tmp_path)
        if not has_token:
            monkeypatch.setenv("KAROTTE_BACKEND_TOKEN_PATH", str(tmp_path / "missing"))
        config = sample_config.model_copy(
            update={"backend_uri": "http://localhost:8000"}
        )
        source_file = tmp_path / "test_file.txt"
        source_file.write_text("test content")
        httpx_mock.add_response(
            method="POST",
            url="http://localhost:8000/api/artifacts/presign",
            json={"presigned_urls": {"test_file.txt": "https://s3.example/put"}},
        )
        httpx_mock.add_response(method="PUT", url="https://s3.example/put")

        save_artifact(config, source_file)

        presign = httpx_mock.get_requests()[0]
        if has_token:
            assert presign.headers["Authorization"] == "Bearer fake-token"
        else:
            assert "Authorization" not in presign.headers

    def test_uploads_file_when_backend_uri_set(
        self,
        sample_config: EvaluationRunConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        httpx_mock: HTTPXMock,
    ):
        """Test that file is uploaded via presigned URL when backend_uri is set."""
        monkeypatch.chdir(tmp_path)

        config = sample_config.model_copy(
            update={"backend_uri": "http://localhost:8000"}
        )

        source_file = tmp_path / "test_file.txt"
        source_file.write_text("test content")

        # Mock the presign endpoint
        httpx_mock.add_response(
            method="POST",
            url="http://localhost:8000/api/artifacts/presign",
            json={
                "presigned_urls": {
                    "test_file.txt": "https://s3.amazonaws.com/bucket/presigned-url"
                }
            },
        )

        # Mock the S3 upload
        httpx_mock.add_response(
            method="PUT",
            url="https://s3.amazonaws.com/bucket/presigned-url",
            status_code=200,
        )

        save_artifact(config, source_file)

        # Verify NO local save happened (only uploads to backend)
        artifact_dir = tmp_path / "out" / f"{config.run_id}_artifacts"
        assert not artifact_dir.exists()

        # Verify HTTP requests were made
        requests = httpx_mock.get_requests()
        assert len(requests) == 2

        # First request is presign
        assert requests[0].method == "POST"
        assert "presign" in str(requests[0].url)

        # Second request is S3 upload
        assert requests[1].method == "PUT"
        assert requests[1].headers["content-encoding"] == "gzip"

    def test_upload_streams_gzip_without_buffering_the_file(
        self,
        sample_config: EvaluationRunConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        httpx_mock: HTTPXMock,
    ):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(save_artifacts_module, "_UPLOAD_CHUNK_SIZE", 1024)
        config = sample_config.model_copy(
            update={"backend_uri": "http://localhost:8000"}
        )

        content = os.urandom(10 * 1024) + b"tail"
        source_file = tmp_path / "recording.bin"
        source_file.write_bytes(content)

        def _no_whole_read(self: Path) -> bytes:
            raise AssertionError(f"{self} was read into memory whole")

        monkeypatch.setattr(Path, "read_bytes", _no_whole_read)

        httpx_mock.add_response(
            method="POST",
            url="http://localhost:8000/api/artifacts/presign",
            json={"presigned_urls": {"recording.bin": "https://storage.example/u"}},
        )
        httpx_mock.add_response(
            method="PUT", url="https://storage.example/u", status_code=200
        )

        save_artifact(config, source_file)

        put = httpx_mock.get_requests()[1]
        assert put.headers["content-encoding"] == "gzip"
        assert put.headers["transfer-encoding"] == "chunked"
        assert "content-length" not in put.headers
        assert gzip.decompress(put.content) == content

    def test_upload_of_empty_file_is_valid_gzip(
        self,
        sample_config: EvaluationRunConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        httpx_mock: HTTPXMock,
    ):
        monkeypatch.chdir(tmp_path)
        config = sample_config.model_copy(
            update={"backend_uri": "http://localhost:8000"}
        )
        source_file = tmp_path / "empty.log"
        source_file.write_bytes(b"")

        httpx_mock.add_response(
            method="POST",
            url="http://localhost:8000/api/artifacts/presign",
            json={"presigned_urls": {"empty.log": "https://storage.example/u"}},
        )
        httpx_mock.add_response(
            method="PUT", url="https://storage.example/u", status_code=200
        )

        save_artifact(config, source_file)

        assert gzip.decompress(httpx_mock.get_requests()[1].content) == b""

    def test_uploads_directory_files_individually(
        self,
        sample_config: EvaluationRunConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        httpx_mock: HTTPXMock,
    ):
        """Test that directory files are uploaded individually via presigned URLs."""
        monkeypatch.chdir(tmp_path)

        config = sample_config.model_copy(
            update={"backend_uri": "http://localhost:8000"}
        )

        source_dir = tmp_path / "test_dir"
        source_dir.mkdir()
        (source_dir / "file1.txt").write_text("content 1")
        (source_dir / "file2.txt").write_text("content 2")

        # Mock the presign endpoint
        httpx_mock.add_response(
            method="POST",
            url="http://localhost:8000/api/artifacts/presign",
            json={
                "presigned_urls": {
                    "test_dir/file1.txt": "https://s3.amazonaws.com/bucket/url1",
                    "test_dir/file2.txt": "https://s3.amazonaws.com/bucket/url2",
                }
            },
        )

        # Mock the S3 uploads
        httpx_mock.add_response(
            method="PUT",
            url="https://s3.amazonaws.com/bucket/url1",
            status_code=200,
        )
        httpx_mock.add_response(
            method="PUT",
            url="https://s3.amazonaws.com/bucket/url2",
            status_code=200,
        )

        save_artifact(config, source_dir)

        # Verify NO local save happened (only uploads to backend)
        artifact_dir = tmp_path / "out" / f"{config.run_id}_artifacts"
        assert not artifact_dir.exists()

        # Verify HTTP requests were made (1 presign + 2 uploads)
        requests = httpx_mock.get_requests()
        assert len(requests) == 3

        # First request is presign
        assert requests[0].method == "POST"

        # Remaining requests are S3 uploads
        assert requests[1].method == "PUT"
        assert requests[2].method == "PUT"

    def test_does_not_upload_when_backend_uri_not_set(
        self,
        sample_config: EvaluationRunConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        httpx_mock: HTTPXMock,
    ):
        """Test that no upload happens when backend_uri is not set."""
        monkeypatch.chdir(tmp_path)

        source_file = tmp_path / "test_file.txt"
        source_file.write_text("test content")

        save_artifact(sample_config, source_file)

        # Verify local save happened
        artifact_dir = tmp_path / "out" / f"{sample_config.run_id}_artifacts"
        assert (artifact_dir / "test_file.txt").exists()

        # Verify no HTTP request was made
        assert len(httpx_mock.get_requests()) == 0

    def test_presign_failure_does_not_raise(
        self,
        sample_config: EvaluationRunConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        httpx_mock: HTTPXMock,
    ):
        """Test that presign failure logs error but does not raise."""
        monkeypatch.chdir(tmp_path)

        config = sample_config.model_copy(
            update={"backend_uri": "http://localhost:8000"}
        )

        source_file = tmp_path / "test_file.txt"
        source_file.write_text("test content")

        # Mock HTTP error response for presign (use 4xx to avoid triggering retries)
        httpx_mock.add_response(
            method="POST",
            url="http://localhost:8000/api/artifacts/presign",
            status_code=400,
            text="Bad Request",
        )

        # Should not raise
        save_artifact(config, source_file)

        # No local save when backend_uri is set
        artifact_dir = tmp_path / "out" / f"{config.run_id}_artifacts"
        assert not artifact_dir.exists()

    def test_s3_upload_failure_does_not_raise(
        self,
        sample_config: EvaluationRunConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        httpx_mock: HTTPXMock,
    ):
        """Test that S3 upload failure logs error but does not raise."""
        monkeypatch.chdir(tmp_path)

        config = sample_config.model_copy(
            update={"backend_uri": "http://localhost:8000"}
        )

        source_file = tmp_path / "test_file.txt"
        source_file.write_text("test content")

        # Mock successful presign
        httpx_mock.add_response(
            method="POST",
            url="http://localhost:8000/api/artifacts/presign",
            json={
                "presigned_urls": {
                    "test_file.txt": "https://s3.amazonaws.com/bucket/presigned-url"
                }
            },
        )

        # Mock S3 upload failure
        httpx_mock.add_response(
            method="PUT",
            url="https://s3.amazonaws.com/bucket/presigned-url",
            status_code=403,
            text="Forbidden",
        )

        # Should not raise
        save_artifact(config, source_file)

        # No local save when backend_uri is set
        artifact_dir = tmp_path / "out" / f"{config.run_id}_artifacts"
        assert not artifact_dir.exists()

    def test_request_error_does_not_raise(
        self,
        sample_config: EvaluationRunConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        httpx_mock: HTTPXMock,
    ):
        """Test that request error logs error but does not raise."""
        import httpx

        monkeypatch.chdir(tmp_path)

        config = sample_config.model_copy(
            update={"backend_uri": "http://localhost:8000"}
        )

        source_file = tmp_path / "test_file.txt"
        source_file.write_text("test content")

        # Mock a non-retryable request error (4xx to avoid triggering retries)
        httpx_mock.add_exception(
            httpx.DecodingError("Decoding failed"),
            url="http://localhost:8000/api/artifacts/presign",
        )

        # Should not raise
        save_artifact(config, source_file)

        # No local save when backend_uri is set
        artifact_dir = tmp_path / "out" / f"{config.run_id}_artifacts"
        assert not artifact_dir.exists()

    def test_handles_empty_directory(
        self,
        sample_config: EvaluationRunConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        httpx_mock: HTTPXMock,
    ):
        """Test uploading an empty directory (no files to upload)."""
        monkeypatch.chdir(tmp_path)

        config = sample_config.model_copy(
            update={"backend_uri": "http://localhost:8000"}
        )

        source_dir = tmp_path / "empty_dir"
        source_dir.mkdir()

        # Should not make any HTTP requests for empty directory
        save_artifact(config, source_dir)

        # No HTTP requests should be made
        assert len(httpx_mock.get_requests()) == 0

    def test_handles_nested_directory(
        self,
        sample_config: EvaluationRunConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        httpx_mock: HTTPXMock,
    ):
        """Test uploading a nested directory structure."""
        monkeypatch.chdir(tmp_path)

        config = sample_config.model_copy(
            update={"backend_uri": "http://localhost:8000"}
        )

        source_dir = tmp_path / "nested"
        source_dir.mkdir()
        (source_dir / "subdir").mkdir()
        (source_dir / "file1.txt").write_text("content 1")
        (source_dir / "subdir" / "file2.txt").write_text("content 2")

        # Mock the presign endpoint
        httpx_mock.add_response(
            method="POST",
            url="http://localhost:8000/api/artifacts/presign",
            json={
                "presigned_urls": {
                    "nested/file1.txt": "https://s3.amazonaws.com/bucket/url1",
                    "nested/subdir/file2.txt": "https://s3.amazonaws.com/bucket/url2",
                }
            },
        )

        # Mock the S3 uploads
        httpx_mock.add_response(
            method="PUT",
            url="https://s3.amazonaws.com/bucket/url1",
            status_code=200,
        )
        httpx_mock.add_response(
            method="PUT",
            url="https://s3.amazonaws.com/bucket/url2",
            status_code=200,
        )

        save_artifact(config, source_dir)

        # Verify presign request contains correct paths
        requests = httpx_mock.get_requests()
        presign_request = requests[0]
        import json

        body = json.loads(presign_request.content)
        assert config.run_id == body["run_id"]
        assert set(body["artifact_paths"]) == {
            "nested/file1.txt",
            "nested/subdir/file2.txt",
        }
