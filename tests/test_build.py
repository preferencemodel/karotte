from subprocess import CompletedProcess
from unittest.mock import MagicMock, patch

import pytest
import typer

from karotte.build import (
    _buildx_available,  # pyright: ignore[reportPrivateUsage]
    build_container,
    get_container_build_command,
)
from karotte.runtime import get_engine


class TestGetContainerBuildCommand:
    def test_basic_podman_command(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("CI", raising=False)

        command = get_container_build_command("podman", "/build/context", "karotte")

        assert command == [
            "podman",
            "build",
            "--tag",
            "karotte",
            "/build/context",
        ]

    def test_basic_docker_command(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("CI", raising=False)

        command = get_container_build_command("docker", "/build/context", "karotte")

        # Docker needs explicit Containerfile specification
        assert command == [
            "docker",
            "build",
            "--file",
            "Containerfile",
            "--tag",
            "karotte",
            "/build/context",
        ]

    def test_ci_environment_adds_sudo(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CI", "true")

        command = get_container_build_command("podman", ".", "karotte")

        assert command[0] == "sudo"

    def test_non_ci_environment_no_sudo(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("CI", raising=False)

        command = get_container_build_command("podman", ".", "karotte")

        assert "sudo" not in command

    def test_custom_tag(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("CI", raising=False)

        command = get_container_build_command("podman", "/build/context", "custom_tag")

        tag_index = command.index("--tag")
        assert command[tag_index + 1] == "custom_tag"

    def test_custom_tag_with_docker(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("CI", raising=False)

        command = get_container_build_command("docker", ".", "my-image:v1")

        tag_index = command.index("--tag")
        assert command[tag_index + 1] == "my-image:v1"

    def test_cache_from(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("CI", raising=False)

        command = get_container_build_command(
            "podman", ".", "karotte", cache_from=["type=registry,ref=repo:cache"]
        )

        idx = command.index("--cache-from")
        assert command[idx + 1] == "type=registry,ref=repo:cache"

    def test_cache_to(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("CI", raising=False)

        command = get_container_build_command(
            "podman", ".", "karotte", cache_to=["type=local,dest=/tmp/cache"]
        )

        idx = command.index("--cache-to")
        assert command[idx + 1] == "type=local,dest=/tmp/cache"

    def test_multiple_cache_from(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("CI", raising=False)

        command = get_container_build_command(
            "podman", ".", "karotte", cache_from=["src1", "src2"]
        )

        indices = [i for i, v in enumerate(command) if v == "--cache-from"]
        assert len(indices) == 2
        assert command[indices[0] + 1] == "src1"
        assert command[indices[1] + 1] == "src2"

    def test_no_cache_flags_by_default(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("CI", raising=False)

        command = get_container_build_command("podman", ".", "karotte")

        assert "--cache-from" not in command
        assert "--cache-to" not in command


class TestBuildSecrets:
    def test_a_build_secret_is_mounted_from_its_file(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.delenv("CI", raising=False)

        command = get_container_build_command(
            "podman", ".", "karotte", build_secrets=["uv_env=/tmp/uv_env"]
        )

        assert command == [
            "podman",
            "build",
            "--secret=id=uv_env,src=/tmp/uv_env",
            "--tag",
            "karotte",
            ".",
        ]

    def test_several_build_secrets(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("CI", raising=False)

        command = get_container_build_command(
            "podman", ".", "karotte", build_secrets=["a=/x", "b=/y=z"]
        )

        assert "--secret=id=a,src=/x" in command
        assert "--secret=id=b,src=/y=z" in command

    def test_a_secret_without_a_path_is_rejected(self):
        with pytest.raises(typer.BadParameter, match="name=path"):
            get_container_build_command("podman", ".", build_secrets=["uv_env"])

    def test_no_secret_is_passed_unless_asked(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CI", "true")
        monkeypatch.delenv("PYTEST_CURRENT_TEST")

        command = get_container_build_command("podman", ".", "karotte")

        assert command[:3] == ["sudo", "podman", "build"]
        assert not [a for a in command if a.startswith("--secret")]


class TestBuildContainerWithTag:
    def test_build_container_uses_default_tag(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("CI", raising=False)

        mock_run = MagicMock()
        mock_run.return_value.returncode = 0

        with patch("karotte.build.subprocess.run", mock_run):
            build_container("podman", ".")

        call_args = mock_run.call_args[0][0]
        tag_index = call_args.index("--tag")
        assert call_args[tag_index + 1] == "karotte"

    def test_build_container_uses_custom_tag(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("CI", raising=False)

        mock_run = MagicMock()
        mock_run.return_value.returncode = 0

        with patch("karotte.build.subprocess.run", mock_run):
            build_container("podman", ".", "custom-tag")

        call_args = mock_run.call_args[0][0]
        tag_index = call_args.index("--tag")
        assert call_args[tag_index + 1] == "custom-tag"


class TestPodmanThroughARemoteService:
    def test_builds_with_secrets_without_asking_podman(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.delenv("CI", raising=False)
        remote = CompletedProcess(args=[], returncode=0, stdout="true\n", stderr="")

        with patch("karotte.build.subprocess.run", return_value=remote) as mock_run:
            command = get_container_build_command(
                "podman", ".", "karotte", build_secrets=["uv_env=/tmp/uv_env"]
            )

        mock_run.assert_not_called()
        assert command[:3] == ["podman", "build", "--secret=id=uv_env,src=/tmp/uv_env"]


class TestGetEngine:
    def test_podman(self):
        assert get_engine("podman") == "podman"

    def test_docker(self):
        assert get_engine("docker") == "docker"

    def test_docker_gvisor(self):
        assert get_engine("docker:gvisor") == "docker"


class TestDockerGvisorBuildCommand:
    """Tests that docker:gvisor runtime builds using docker engine."""

    def test_uses_docker_engine(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("CI", raising=False)

        command = get_container_build_command(
            "docker:gvisor", "/build/context", "karotte"
        )

        assert command[0] == "docker"
        assert "podman" not in command

    def test_includes_containerfile_flag(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("CI", raising=False)

        command = get_container_build_command("docker:gvisor", ".", "karotte")

        assert "--file" in command
        assert "Containerfile" in command

    def test_does_not_include_runtime_flag(self, monkeypatch: pytest.MonkeyPatch):
        """Build commands should not include --runtime=runsc (only run commands do)."""
        monkeypatch.delenv("CI", raising=False)

        command = get_container_build_command("docker:gvisor", ".", "karotte")

        assert "--runtime=runsc" not in command


class TestBuildxAvailable:
    def test_true_when_buildx_answers(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("CI", raising=False)
        result = CompletedProcess(args=[], returncode=0)
        with patch("karotte.build.subprocess.run", return_value=result) as run:
            assert _buildx_available() is True
        assert run.call_args[0][0] == ["docker", "buildx", "version"]

    def test_false_when_buildx_is_missing(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("CI", raising=False)
        result = CompletedProcess(args=[], returncode=1)
        with patch("karotte.build.subprocess.run", return_value=result):
            assert _buildx_available() is False

    def test_asks_the_docker_the_build_uses(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CI", "true")
        result = CompletedProcess(args=[], returncode=0)
        with patch("karotte.build.subprocess.run", return_value=result) as run:
            _buildx_available()
        assert run.call_args[0][0] == ["sudo", "docker", "buildx", "version"]
