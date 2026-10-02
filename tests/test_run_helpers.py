import json
import os
import socket
import sys
from pathlib import Path
from typing import Any, final
from unittest.mock import MagicMock, patch

import pytest
import typer

from karotte import run_helpers
from karotte.cli.run import (
    _expand_mount_file_references,  # pyright: ignore[reportPrivateUsage]
    _validate_mount_specs,  # pyright: ignore[reportPrivateUsage]
)
from karotte.forwarded_env import EXIT_ON_RUN_ERROR_ENV_VAR
from karotte.providers import SERVICE_TIER_ENV
from karotte.run_helpers import (
    _maybe_block_internet,  # pyright: ignore[reportPrivateUsage]
    _set_up_runner,  # pyright: ignore[reportPrivateUsage]
    build_configs,
    chown_outputs,
    clean_up_old_containers,
    common_run_id_prefix,
    get_container_run_command,
    parse_config,
    run_containerized,
    run_non_containerized,
    validate_gvisor_runtime,
)
from karotte.runtime import Runtime
from karotte.schemas import RunStatus
from karotte.schemas.evaluation_run_config import EvaluationRunConfig
from karotte.schemas.transcript import ErrorEvent, TaskCompletedEvent
from karotte.task import Task
from tests.conftest import register_hardware_plugins


class TestParseConfig:
    def test_parses_json_string(self):
        config_json = json.dumps(
            {
                "run_id": "test-run",
                "task_id": "test-task",
                "model": "test-model",
                "model_api_key": "test-key",
            }
        )

        config = parse_config(config_json)

        assert config.run_id == "test-run"
        assert config.task_id == "test-task"
        assert config.model == "test-model"

    def test_parses_config_from_file(self, tmp_path: Path):
        config_file = tmp_path / "config.json"
        config_file.write_text(
            json.dumps(
                {
                    "run_id": "file-run",
                    "task_id": "file-task",
                    "model": "file-model",
                    "model_api_key": "file-key",
                }
            )
        )

        config = parse_config(str(config_file))

        assert config.run_id == "file-run"
        assert config.task_id == "file-task"

    def test_raises_error_for_missing_api_key(self, capsys: pytest.CaptureFixture[str]):
        config_json = json.dumps(
            {
                "run_id": "test-run",
                "task_id": "test-task",
                "model": "test-model",
                # no model_api_key, non-vertex, non-training
            }
        )

        with pytest.raises(typer.Abort):
            parse_config(config_json)

        assert "model_api_key is required" in capsys.readouterr().err

    def test_prepare_only_allows_missing_api_key(self):
        config_json = json.dumps(
            {
                "run_id": "test-run",
                "task_id": "test-task",
                "model": "test-model",
                # no model_api_key: allowed because prepare-only never calls the model
            }
        )

        config = parse_config(config_json, prepare_only=True)

        assert config.model_api_key is None

    def _rubric_config(self, rubric_key: str) -> str:
        return json.dumps(
            {
                "run_id": "test-run",
                "task_id": "test-task",
                "model": "test-model",
                "model_api_key": "test-key",
                "rubric_judge_api_key": rubric_key,
            }
        )

    def test_resolves_a_rubric_judge_key_reference(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("MY_RUBRIC_KEY", "sk-rubric")

        config = parse_config(self._rubric_config("$MY_RUBRIC_KEY"))

        assert config.rubric_judge_api_key == "sk-rubric"

    def test_keeps_a_literal_rubric_judge_key(self):
        config = parse_config(self._rubric_config("sk-literal"))

        assert config.rubric_judge_api_key == "sk-literal"

    def test_an_unset_rubric_judge_key_variable_does_not_abort(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.delenv("MY_RUBRIC_KEY", raising=False)

        config = parse_config(self._rubric_config("$MY_RUBRIC_KEY"))

        assert config.rubric_judge_api_key is None

    def test_raises_error_for_nonexistent_file(self):
        with pytest.raises(typer.Abort):
            parse_config("/nonexistent/path/config.json")

    def test_raises_error_for_invalid_json_string(self):
        with pytest.raises(typer.Abort):
            parse_config("not valid json and not a file path")

    def test_raises_error_for_invalid_json_in_file(self, tmp_path: Path):
        config_file = tmp_path / "invalid.json"
        config_file.write_text("not valid json")

        with pytest.raises(typer.Abort):
            parse_config(str(config_file))

    def test_shows_validation_errors_for_missing_required_field(
        self, capsys: pytest.CaptureFixture[str]
    ):
        config_json = json.dumps(
            {
                "run_id": "test-run",
                # missing task_id, model, model_api_key
            }
        )

        with pytest.raises(typer.Abort):
            parse_config(config_json)

        err = capsys.readouterr().err
        assert "Validation errors:" in err
        assert "task_id" in err
        assert "model" in err

    def test_shows_validation_errors_for_invalid_field_type(
        self, capsys: pytest.CaptureFixture[str]
    ):
        config_json = json.dumps(
            {
                "run_id": "test-run",
                "task_id": "test-task",
                "model": "test-model",
                "model_api_key": "test-key",
                "turn_limit": "not-an-int",  # should be int
            }
        )

        with pytest.raises(typer.Abort):
            parse_config(config_json)

        err = capsys.readouterr().err
        assert "Validation errors:" in err
        assert "turn_limit" in err


class TestParseConfigEnvVarResolution:
    def test_resolves_env_var_in_json_string(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("MY_API_KEY", "sk-ant-actual-key")
        config_json = json.dumps(
            {
                "run_id": "test-run",
                "task_id": "test-task",
                "model": "test-model",
                "model_api_key": "$MY_API_KEY",
            }
        )

        config = parse_config(config_json)

        assert config.model_api_key == "sk-ant-actual-key"

    def test_resolves_env_var_in_config_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        monkeypatch.setenv("MY_API_KEY", "sk-ant-actual-key")
        config_file = tmp_path / "config.json"
        config_file.write_text(
            json.dumps(
                {
                    "run_id": "test-run",
                    "task_id": "test-task",
                    "model": "test-model",
                    "model_api_key": "$MY_API_KEY",
                }
            )
        )

        config = parse_config(str(config_file))

        assert config.model_api_key == "sk-ant-actual-key"

    def test_aborts_when_env_var_not_set(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("MISSING_KEY", raising=False)
        config_json = json.dumps(
            {
                "run_id": "test-run",
                "task_id": "test-task",
                "model": "test-model",
                "model_api_key": "$MISSING_KEY",
            }
        )

        with pytest.raises(typer.Abort):
            parse_config(config_json)

    def test_fake_model_does_not_need_env_var(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("MISSING_KEY", raising=False)
        config_json = json.dumps(
            {
                "run_id": "test-run",
                "task_id": "test-task",
                "model": "test-model",
                "model_api_key": "$MISSING_KEY",
                "use_fake_model": True,
            }
        )

        config = parse_config(config_json)

        assert config.use_fake_model

    def test_fake_model_does_not_need_api_key(self):
        config_json = json.dumps(
            {
                "run_id": "test-run",
                "task_id": "test-task",
                "model": "test-model",
                "use_fake_model": True,
            }
        )

        config = parse_config(config_json)

        assert config.model_api_key is None

    def test_prepare_only_does_not_need_env_var(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("MISSING_KEY", raising=False)
        config_json = json.dumps(
            {
                "run_id": "test-run",
                "task_id": "test-task",
                "model": "test-model",
                "model_api_key": "$MISSING_KEY",
            }
        )

        config = parse_config(config_json, prepare_only=True)

        assert config.model_api_key == "$MISSING_KEY"

    def test_fake_model_still_resolves_set_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("MY_API_KEY", "sk-ant-actual-key")
        config_json = json.dumps(
            {
                "run_id": "test-run",
                "task_id": "test-task",
                "model": "test-model",
                "model_api_key": "$MY_API_KEY",
                "use_fake_model": True,
            }
        )

        config = parse_config(config_json)

        assert config.model_api_key == "sk-ant-actual-key"

    def test_aborts_with_helpful_message_when_env_var_not_set(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ):
        monkeypatch.delenv("MISSING_KEY", raising=False)
        config_json = json.dumps(
            {
                "run_id": "test-run",
                "task_id": "test-task",
                "model": "test-model",
                "model_api_key": "$MISSING_KEY",
            }
        )

        with pytest.raises(typer.Abort):
            parse_config(config_json)

        err = capsys.readouterr().err
        assert "MISSING_KEY" in err
        assert "not set as an environment variable" in err

    def test_does_not_resolve_plain_api_key(self):
        config_json = json.dumps(
            {
                "run_id": "test-run",
                "task_id": "test-task",
                "model": "test-model",
                "model_api_key": "sk-ant-plain-key",
            }
        )

        config = parse_config(config_json)

        assert config.model_api_key == "sk-ant-plain-key"

    def test_does_not_affect_vertex_ai_config_with_no_key(self):
        config_json = json.dumps(
            {
                "run_id": "test-run",
                "task_id": "test-task",
                "model": "vertex_ai/gemini-3-pro-preview",
            }
        )

        config = parse_config(config_json)

        assert config.model_api_key is None

    def test_resolves_to_empty_string_when_env_var_is_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("EMPTY_KEY", "")
        config_json = json.dumps(
            {
                "run_id": "test-run",
                "task_id": "test-task",
                "model": "test-model",
                "model_api_key": "$EMPTY_KEY",
            }
        )

        config = parse_config(config_json)

        assert config.model_api_key == ""

    def test_aborts_when_env_var_name_is_empty(self):
        """$-only value (no env var name) should abort."""
        config_json = json.dumps(
            {
                "run_id": "test-run",
                "task_id": "test-task",
                "model": "test-model",
                "model_api_key": "$",
            }
        )

        with pytest.raises(typer.Abort):
            parse_config(config_json)

    def test_env_var_reference_survives_model_copy(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Resolved key should be preserved after model_copy (e.g. in build_configs)."""
        monkeypatch.setenv("MY_API_KEY", "sk-ant-resolved-key")
        config_json = json.dumps(
            {
                "run_id": "test-run",
                "task_id": "test-task",
                "model": "test-model",
                "model_api_key": "$MY_API_KEY",
            }
        )

        config = parse_config(config_json)
        copied = config.model_copy(update={"run_id": "other-run"})

        assert copied.model_api_key == "sk-ant-resolved-key"


class TestBuildConfigs:
    def test_single_config_returns_unchanged(self, sample_config: EvaluationRunConfig):
        configs = build_configs(sample_config, n=1)

        assert len(configs) == 1
        assert configs[0] is sample_config

    def test_multiple_configs_have_unique_run_ids(
        self, sample_config: EvaluationRunConfig
    ):
        configs = build_configs(sample_config, n=3)

        assert len(configs) == 3
        assert configs[0].run_id == "test_run-0"
        assert configs[1].run_id == "test_run-1"
        assert configs[2].run_id == "test_run-2"

    def test_multiple_configs_have_unique_websocket_ports(
        self, sample_config: EvaluationRunConfig
    ):
        base_port = sample_config.websocket_config.port
        configs = build_configs(sample_config, n=3)

        assert configs[0].websocket_config.port == base_port
        assert configs[1].websocket_config.port == base_port + 1
        assert configs[2].websocket_config.port == base_port + 2

    def test_multiple_configs_have_unique_transcript_files(
        self, sample_config: EvaluationRunConfig
    ):
        configs = build_configs(sample_config, n=3)

        assert configs[0].transcript_file == "out/transcript_0.json"
        assert configs[1].transcript_file == "out/transcript_1.json"
        assert configs[2].transcript_file == "out/transcript_2.json"

    @pytest.mark.parametrize(
        ("transcript_file", "expected"),
        [
            ("out/run.json", "out/run_0.json"),
            ("out/session.json", "out/session_0.json"),
            ("out/transcript", "out/transcript_0.json"),
            ("out/nosuffix", "out/nosuffix_0.json"),
        ],
    )
    def test_transcript_file_suffix_is_removed_not_stripped(
        self, sample_config: EvaluationRunConfig, transcript_file: str, expected: str
    ):
        sample_config = sample_config.model_copy(
            update={"transcript_file": transcript_file}
        )
        configs = build_configs(sample_config, n=2)

        assert configs[0].transcript_file == expected

    def test_handles_none_transcript_file(self, sample_config: EvaluationRunConfig):
        sample_config = sample_config.model_copy(update={"transcript_file": None})
        configs = build_configs(sample_config, n=2)

        assert configs[0].transcript_file is None
        assert configs[1].transcript_file is None


class TestGetContainerRunCommand:
    def test_basic_command_structure(self, sample_config: EvaluationRunConfig):
        command, _ = get_container_run_command(
            sample_config, "podman", dev=False, keep_container=False
        )

        assert "podman" in command
        assert "run" in command
        assert "--rm" in command
        assert "--cap-add=NET_ADMIN" in command

    def test_includes_port_publish(self, sample_config: EvaluationRunConfig):
        command, _ = get_container_run_command(
            sample_config, "podman", dev=False, keep_container=False
        )

        port = sample_config.websocket_config.port
        assert "--publish" in command
        assert f"{port}:{port}" in command

    def test_dev_mode_mounts_environment_src(
        self, sample_config: EvaluationRunConfig, tmp_path: Path
    ):
        build_context = str(tmp_path)
        command, _ = get_container_run_command(
            sample_config,
            "podman",
            dev=True,
            keep_container=False,
            build_context=build_context,
        )

        mount_args = [arg for arg in command if "site-packages/environment" in arg]
        assert len(mount_args) == 1
        assert mount_args[0].startswith("type=bind,source=")
        assert f"source={tmp_path / 'src' / 'environment'}" in mount_args[0]

    def test_non_dev_mode_no_dev_bind_mount(self, sample_config: EvaluationRunConfig):
        command, _ = get_container_run_command(
            sample_config, "podman", dev=False, keep_container=False
        )

        assert not any("site-packages/environment" in arg for arg in command)

    def test_mount_adds_bind_mount(self, sample_config: EvaluationRunConfig):
        command, _ = get_container_run_command(
            sample_config,
            "podman",
            dev=False,
            keep_container=False,
            mounts=["/data/models:/models:ro"],
        )

        mount_args = [
            arg for arg in command if arg.startswith("type=bind,source=/data/models")
        ]
        assert len(mount_args) == 1
        assert mount_args[0] == "type=bind,source=/data/models,target=/models,readonly"

    def test_multiple_mounts(self, sample_config: EvaluationRunConfig):
        mounts = ["/data/models:/models:ro", "/tmp/results:/results"]
        command, _ = get_container_run_command(
            sample_config,
            "podman",
            dev=False,
            keep_container=False,
            mounts=mounts,
        )

        bind_mount_args = [
            arg
            for arg in command
            if arg.startswith("type=bind,") and "target=/out" not in arg
        ]
        assert len(bind_mount_args) == 2
        assert (
            bind_mount_args[0]
            == "type=bind,source=/data/models,target=/models,readonly"
        )
        assert bind_mount_args[1] == "type=bind,source=/tmp/results,target=/results"

    def test_no_user_bind_mounts_by_default(self, sample_config: EvaluationRunConfig):
        command, _ = get_container_run_command(
            sample_config, "podman", dev=False, keep_container=False
        )

        # No --volume flags should ever appear (we use --mount type=bind)
        assert "--volume" not in command
        # Only the transcript file mount should exist, no user bind mounts
        bind_mount_args = [
            arg
            for arg in command
            if arg.startswith("type=bind,") and "target=/out" not in arg
        ]
        assert len(bind_mount_args) == 0

    def test_transcript_file_creates_bind_mount(
        self, sample_config: EvaluationRunConfig, tmp_path: Path
    ):
        transcript_path = tmp_path / "out" / "transcript.json"
        config = sample_config.model_copy(
            update={"transcript_file": str(transcript_path)}
        )

        command, updated_config = get_container_run_command(
            config, "podman", dev=False, keep_container=False
        )

        assert "--mount" in command
        # Config should be updated with container path
        assert updated_config.transcript_file == "/out/transcript.json"

    def test_returns_updated_config_with_container_transcript_path(
        self, sample_config: EvaluationRunConfig, tmp_path: Path
    ):
        transcript_path = tmp_path / "transcripts" / "run.json"
        config = sample_config.model_copy(
            update={"transcript_file": str(transcript_path)}
        )

        _, updated_config = get_container_run_command(
            config, "podman", dev=False, keep_container=False
        )

        # Original config unchanged
        assert config.transcript_file == str(transcript_path)
        # Updated config has container path
        assert updated_config.transcript_file == "/out/run.json"

    def test_no_transcript_file_returns_unchanged_config(
        self, sample_config: EvaluationRunConfig
    ):
        sample_config.transcript_file = ""

        _, updated_config = get_container_run_command(
            sample_config, "podman", dev=False, keep_container=False
        )

        assert updated_config.transcript_file == sample_config.transcript_file

    def test_ci_environment_adds_sudo(
        self,
        sample_config: EvaluationRunConfig,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setenv("CI", "true")

        command, _ = get_container_run_command(
            sample_config, "podman", dev=False, keep_container=False
        )

        assert command[0] == "sudo"

    def test_podman_uses_localhost_image(self, sample_config: EvaluationRunConfig):
        command, _ = get_container_run_command(
            sample_config, "podman", dev=False, keep_container=False
        )

        assert "localhost/karotte" in command

    def test_docker_uses_plain_image_name(self, sample_config: EvaluationRunConfig):
        command, _ = get_container_run_command(
            sample_config, "docker", dev=False, keep_container=False
        )

        assert "karotte" in command
        assert "localhost/karotte" not in command

    def test_includes_karotte_run_command(self, sample_config: EvaluationRunConfig):
        command, _ = get_container_run_command(
            sample_config, "podman", dev=False, keep_container=False
        )

        assert "/root/.venv/bin/karotte" in command
        assert "run" in command
        assert "--no-containerized" in command
        assert "--config" in command

    def test_keep_container_false_includes_rm_flag(
        self, sample_config: EvaluationRunConfig
    ):
        """When keep_container=False, --rm flag should be present."""
        command, _ = get_container_run_command(
            sample_config, "podman", dev=False, keep_container=False
        )

        assert "--rm" in command

    def test_keep_container_true_excludes_rm_flag(
        self, sample_config: EvaluationRunConfig
    ):
        """When keep_container=True, --rm flag should NOT be present."""
        command, _ = get_container_run_command(
            sample_config, "podman", dev=False, keep_container=True
        )

        assert "--rm" not in command

    def test_container_has_name_based_on_run_id(
        self, sample_config: EvaluationRunConfig
    ):
        """Container should be named karotte_run_<run_id>."""
        command, _ = get_container_run_command(
            sample_config, "podman", dev=False, keep_container=False
        )

        name_index = command.index("--name")
        assert command[name_index + 1] == f"karotte_run_{sample_config.run_id}"

    def test_asks_the_inner_run_to_exit_non_zero_on_an_error(
        self, sample_config: EvaluationRunConfig
    ):
        command, _ = get_container_run_command(
            sample_config, "podman", dev=False, keep_container=False
        )

        envs = [command[i + 1] for i, arg in enumerate(command) if arg == "--env"]
        assert f"{EXIT_ON_RUN_ERROR_ENV_VAR}=1" in envs

    def test_proxy_url_adds_anthropic_base_url(
        self, sample_config: EvaluationRunConfig
    ):
        """--proxy should set ANTHROPIC_BASE_URL to the given URL."""
        command, _ = get_container_run_command(
            sample_config,
            "podman",
            dev=False,
            keep_container=False,
            proxy_url="https://proxy.example",
        )

        envs = [command[i + 1] for i, arg in enumerate(command) if arg == "--env"]
        assert "ANTHROPIC_BASE_URL=https://proxy.example" in envs
        assert "KAROTTE_PROXY_URL=https://proxy.example" in envs

        # Env vars should appear before the image name
        env_indices = [i for i, arg in enumerate(command) if arg == "--env"]
        image_index = command.index("localhost/karotte")
        assert max(env_indices) < image_index

    def test_loguru_level_is_forwarded(
        self, sample_config: EvaluationRunConfig, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("LOGURU_LEVEL", "DEBUG")
        command, _ = get_container_run_command(
            sample_config, "podman", dev=False, keep_container=False
        )

        envs = [command[i + 1] for i, arg in enumerate(command) if arg == "--env"]
        assert "LOGURU_LEVEL=DEBUG" in envs

    def test_service_tier_is_forwarded(
        self, sample_config: EvaluationRunConfig, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv(SERVICE_TIER_ENV, "auto")
        command, _ = get_container_run_command(
            sample_config, "podman", dev=False, keep_container=False
        )

        envs = [command[i + 1] for i, arg in enumerate(command) if arg == "--env"]
        assert f"{SERVICE_TIER_ENV}=auto" in envs

    def test_the_student_network_setting_is_forwarded(
        self, sample_config: EvaluationRunConfig, monkeypatch: pytest.MonkeyPatch
    ):
        """The firewall reads it inside the container."""
        monkeypatch.setenv("KAROTTE_STUDENT_NETWORK", "internal")
        command, _ = get_container_run_command(
            sample_config, "podman", dev=False, keep_container=False
        )

        envs = [command[i + 1] for i, arg in enumerate(command) if arg == "--env"]
        assert "KAROTTE_STUDENT_NETWORK=internal" in envs

    def test_custom_proxy_url(self, sample_config: EvaluationRunConfig):
        """--proxy with a custom URL should use that URL."""
        command, _ = get_container_run_command(
            sample_config,
            "podman",
            dev=False,
            keep_container=False,
            proxy_url="https://custom-proxy.example.com",
        )

        envs = [command[i + 1] for i, arg in enumerate(command) if arg == "--env"]
        assert "ANTHROPIC_BASE_URL=https://custom-proxy.example.com" in envs
        assert "KAROTTE_PROXY_URL=https://custom-proxy.example.com" in envs

    def test_no_proxy_skips_anthropic_base_url(
        self, sample_config: EvaluationRunConfig
    ):
        """--no-proxy should not add any proxy --env flags."""
        command, _ = get_container_run_command(
            sample_config, "podman", dev=False, keep_container=False, proxy_url=None
        )

        envs = [command[i + 1] for i, arg in enumerate(command) if arg == "--env"]
        assert not any(
            env.startswith(("ANTHROPIC_BASE_URL=", "KAROTTE_PROXY_URL="))
            for env in envs
        )


class TestRunContainerizedPluginArgs:
    """Plugins add container engine `run` arguments, e.g. to pass devices through."""

    def _command(
        self, sample_config: EvaluationRunConfig, runtime: Runtime
    ) -> list[str]:
        captured: list[str] = []

        def capture_run(cmd: list[str], **_: Any) -> None:
            captured.extend(cmd)

        with patch("karotte.run_helpers.subprocess.run", side_effect=capture_run):
            run_containerized(sample_config, runtime=runtime, dev=False)
        return captured

    def test_no_devices_without_a_plugin(
        self, sample_config: EvaluationRunConfig, monkeypatch: pytest.MonkeyPatch
    ):
        register_hardware_plugins(monkeypatch)
        assert "--device" not in self._command(sample_config, "podman")

    def test_plugin_args_come_before_the_image(
        self, sample_config: EvaluationRunConfig, monkeypatch: pytest.MonkeyPatch
    ):
        seen: list[tuple[str, Runtime]] = []

        def gpu(task: Task, runtime: Runtime) -> list[str]:
            seen.append((task.id, runtime))
            return ["--device", "nvidia.com/gpu=all"]

        register_hardware_plugins(monkeypatch, container_run_args={"a": gpu})
        command = self._command(sample_config, "docker")

        at = command.index("--device")
        assert command[at + 1] == "nvidia.com/gpu=all"
        assert at < command.index("karotte")
        assert seen == [(sample_config.task_id, "docker")]


class TestCommonRunIdPrefix:
    def test_empty_list(self):
        assert common_run_id_prefix([]) == ""

    def test_single_run_id(self):
        assert common_run_id_prefix(["my-run-0"]) == "my-run-0"

    def test_parallel_suffixes(self):
        assert common_run_id_prefix(["my-run-0", "my-run-1"]) == "my-run-"

    def test_no_shared_prefix(self):
        assert common_run_id_prefix(["alpha", "beta"]) == ""


class TestCleanUpOldContainers:
    """Tests for clean_up_old_containers function."""

    def test_lists_containers_with_shared_run_id_prefix(self):
        """Should query for containers matching this invocation's run-id prefix."""
        with patch("karotte.run_helpers.subprocess.run") as mock_run:
            mock_run.return_value.stdout = ""
            mock_run.return_value.returncode = 0

            clean_up_old_containers("podman", ["my-run-0", "my-run-1"])

        # First call should be the ps command
        list_call = mock_run.call_args_list[0]
        cmd = list_call[0][0]
        assert cmd[0] == "podman"
        assert "ps" in cmd
        assert "-a" in cmd
        assert "--filter" in cmd
        assert "name=karotte_run_my-run-" in cmd
        assert "-q" in cmd

    def test_stops_and_removes_found_containers(self):
        """Should remove all found containers in a single rm call."""
        with patch("karotte.run_helpers.subprocess.run") as mock_run:
            # First call (ps) returns container IDs
            mock_run.return_value.stdout = "abc123\ndef456"
            mock_run.return_value.returncode = 0

            clean_up_old_containers("podman", ["run-a"])

        # Should have: 1 ps + 1 rm = 2 calls
        assert mock_run.call_count == 2

        # Second call should be rm with both container IDs
        rm_call = mock_run.call_args_list[1]
        cmd = rm_call[0][0]
        assert cmd[0] == "podman"
        assert "rm" in cmd
        assert "--force" in cmd
        assert "abc123" in cmd
        assert "def456" in cmd

    def test_handles_empty_container_list(self):
        """Should not call rm when no containers exist."""
        with patch("karotte.run_helpers.subprocess.run") as mock_run:
            mock_run.return_value.stdout = ""
            mock_run.return_value.returncode = 0

            clean_up_old_containers("podman", ["run-a"])

        # Only the ps command should be called
        assert mock_run.call_count == 1

    def test_works_with_docker_runtime(self):
        """Should use docker command when runtime is docker."""
        with patch("karotte.run_helpers.subprocess.run") as mock_run:
            mock_run.return_value.stdout = "container1"
            mock_run.return_value.returncode = 0

            clean_up_old_containers("docker", ["run-a"])

        # Both calls should use docker
        ps_call = mock_run.call_args_list[0]
        rm_call = mock_run.call_args_list[1]
        assert ps_call[0][0][0] == "docker"
        assert rm_call[0][0][0] == "docker"

    def test_logs_warning_when_ps_command_fails(self):
        """Should log warning and return early when ps command fails."""
        with (
            patch("karotte.run_helpers.subprocess.run") as mock_run,
            patch("loguru.logger.warning") as mock_warning,
        ):
            mock_run.return_value.stdout = ""
            mock_run.return_value.stderr = "Cannot connect to daemon"
            mock_run.return_value.returncode = 1

            clean_up_old_containers("podman", ["run-a"])

        # Only ps should be called, not rm
        assert mock_run.call_count == 1
        mock_warning.assert_called_once()
        assert "Cannot connect to daemon" in mock_warning.call_args[0][0]

    def test_logs_warning_when_rm_command_fails(self):
        """Should log warning when rm command fails."""
        with (
            patch("karotte.run_helpers.subprocess.run") as mock_run,
            patch("loguru.logger.warning") as mock_warning,
        ):
            # First call (ps) succeeds, second call (rm) fails
            mock_run.side_effect = [
                MagicMock(stdout="container1", stderr="", returncode=0),
                MagicMock(stdout="", stderr="Permission denied", returncode=1),
            ]

            clean_up_old_containers("podman", ["run-a"])

        assert mock_run.call_count == 2
        mock_warning.assert_called_once()
        assert "Permission denied" in mock_warning.call_args[0][0]

    def test_handles_ps_failure_with_empty_stderr(self):
        """Should show 'unknown error' when ps fails with empty stderr."""
        with (
            patch("karotte.run_helpers.subprocess.run") as mock_run,
            patch("loguru.logger.warning") as mock_warning,
        ):
            mock_run.return_value.stdout = ""
            mock_run.return_value.stderr = ""
            mock_run.return_value.returncode = 1

            clean_up_old_containers("podman", ["run-a"])

        mock_warning.assert_called_once()
        assert "unknown error" in mock_warning.call_args[0][0]


class TestRunContainerizedKeepContainer:
    """Tests for run_containerized with keep_container parameter."""

    def test_keep_container_true_passes_to_command(
        self, sample_config: EvaluationRunConfig
    ):
        """run_containerized should pass keep_container=True to get_container_run_command."""
        captured_command: list[str] = []

        def capture_run(cmd: list[str], **_: Any) -> None:
            captured_command.extend(cmd)

        with (
            patch("karotte.run_helpers.subprocess.run", side_effect=capture_run),
        ):
            run_containerized(
                sample_config, runtime="podman", dev=False, keep_container=True
            )

        # --rm should NOT be in the command when keep_container=True
        assert "--rm" not in captured_command

    def test_keep_container_false_passes_to_command(
        self, sample_config: EvaluationRunConfig
    ):
        """run_containerized should pass keep_container=False to get_container_run_command."""
        captured_command: list[str] = []

        def capture_run(cmd: list[str], **_: Any) -> None:
            captured_command.extend(cmd)

        with (
            patch("karotte.run_helpers.subprocess.run", side_effect=capture_run),
        ):
            run_containerized(
                sample_config, runtime="podman", dev=False, keep_container=False
            )

        # --rm should be in the command when keep_container=False
        assert "--rm" in captured_command


class TestContainerNamingWithBuildConfigs:
    """Tests for container naming when using build_configs for parallel runs."""

    def test_parallel_configs_have_unique_container_names(
        self, sample_config: EvaluationRunConfig
    ):
        """Each config from build_configs should produce a unique container name."""
        configs = build_configs(sample_config, n=3)

        commands = [
            get_container_run_command(config, "podman", dev=False, keep_container=False)
            for config in configs
        ]

        # Extract container names from each command
        container_names = []
        for command, _ in commands:
            name_index = command.index("--name")
            container_names.append(command[name_index + 1])

        # All names should be unique
        assert len(container_names) == len(set(container_names))

        # Names should follow expected pattern
        assert container_names[0] == "karotte_run_test_run-0"
        assert container_names[1] == "karotte_run_test_run-1"
        assert container_names[2] == "karotte_run_test_run-2"


class TestBuildContextPassThrough:
    """Tests that build_context is properly forwarded through the call chain.

    These tests would have failed before the fix that added build_context
    forwarding in _run_without_ui and _run_containerized_worker.
    """

    def test_run_without_ui_forwards_build_context(
        self, sample_config: EvaluationRunConfig, tmp_path: Path
    ):
        """_run_without_ui should forward build_context to run_containerized."""
        from karotte.cli.run import (
            _run_without_ui,  # pyright: ignore[reportPrivateUsage]
        )

        build_context = str(tmp_path)
        captured_commands: list[list[str]] = []

        def capture_run(cmd: list[str], **_: Any) -> None:
            captured_commands.append(cmd)

        with (
            patch("karotte.run_helpers.subprocess.run", side_effect=capture_run),
            patch("karotte.cli.run.clean_up_old_containers"),
        ):
            _run_without_ui(
                [sample_config],
                runtime="podman",
                dev=True,
                build_context=build_context,
                keep_containers=False,
            )

        assert len(captured_commands) == 1
        mount_args = [
            a for a in captured_commands[0] if "site-packages/environment" in a
        ]
        assert len(mount_args) == 1
        assert f"source={tmp_path / 'src' / 'environment'}" in mount_args[0]

    def test_run_without_ui_forwards_mounts(self, sample_config: EvaluationRunConfig):
        """_run_without_ui should forward mounts to run_containerized."""
        from karotte.cli.run import (
            _run_without_ui,  # pyright: ignore[reportPrivateUsage]
        )

        captured_commands: list[list[str]] = []

        def capture_run(cmd: list[str], **_: Any) -> None:
            captured_commands.append(cmd)

        with (
            patch("karotte.run_helpers.subprocess.run", side_effect=capture_run),
            patch("karotte.cli.run.clean_up_old_containers"),
            patch("karotte.cli.run.build_container"),
        ):
            _run_without_ui(
                [sample_config],
                runtime="podman",
                dev=False,
                build_context=".",
                keep_containers=False,
                mounts=["/host/path:/container/path:ro"],
            )

        assert len(captured_commands) == 1
        assert (
            "type=bind,source=/host/path,target=/container/path,readonly"
            in captured_commands[0]
        )

    def test_run_containerized_worker_forwards_build_context(
        self, sample_config: EvaluationRunConfig, tmp_path: Path
    ):
        """_run_containerized_worker should forward build_context to get_container_run_command."""
        from karotte.terminal.app import (
            _run_containerized_worker,  # pyright: ignore[reportPrivateUsage]
        )

        build_context = str(tmp_path)
        captured_commands: list[list[str]] = []

        def capture_run(cmd: list[str], **_: Any) -> None:
            captured_commands.append(cmd)

        with patch("karotte.terminal.app.subprocess.run", side_effect=capture_run):
            _run_containerized_worker(
                sample_config,
                runtime="podman",
                dev=True,
                keep_container=False,
                build_context=build_context,
            )

        assert len(captured_commands) == 1
        mount_args = [
            a for a in captured_commands[0] if "site-packages/environment" in a
        ]
        assert len(mount_args) == 1
        assert f"source={tmp_path / 'src' / 'environment'}" in mount_args[0]

    def test_run_containerized_worker_forwards_mounts(
        self, sample_config: EvaluationRunConfig
    ):
        """_run_containerized_worker should forward mounts to get_container_run_command."""
        from karotte.terminal.app import (
            _run_containerized_worker,  # pyright: ignore[reportPrivateUsage]
        )

        captured_commands: list[list[str]] = []

        def capture_run(cmd: list[str], **_: Any) -> None:
            captured_commands.append(cmd)

        with patch("karotte.terminal.app.subprocess.run", side_effect=capture_run):
            _run_containerized_worker(
                sample_config,
                runtime="podman",
                dev=False,
                keep_container=False,
                mounts=["/data:/data:ro"],
            )

        assert len(captured_commands) == 1
        assert "type=bind,source=/data,target=/data,readonly" in captured_commands[0]


class TestValidateMountSpecs:
    """Tests for _validate_mount_specs."""

    def test_valid_source_target(self, tmp_path: Path):
        _validate_mount_specs([f"{tmp_path}:/container"])

    def test_valid_source_target_ro(self, tmp_path: Path):
        _validate_mount_specs([f"{tmp_path}:/container:ro"])

    def test_multiple_valid_specs(self, tmp_path: Path):
        a = tmp_path / "a"
        c = tmp_path / "c"
        a.mkdir()
        c.mkdir()
        _validate_mount_specs([f"{a}:/b", f"{c}:/d:ro"])

    def test_rejects_missing_target(self):
        with pytest.raises(typer.Abort):
            _validate_mount_specs(["/host-only"])

    def test_rejects_too_many_colons(self):
        with pytest.raises(typer.Abort):
            _validate_mount_specs(["a:b:c:d"])

    def test_rejects_rw_option(self):
        with pytest.raises(typer.Abort):
            _validate_mount_specs(["/host:/container:rw"])

    def test_rejects_unknown_option(self):
        with pytest.raises(typer.Abort):
            _validate_mount_specs(["/host:/container:z"])

    def test_rejects_bad_spec_among_valid_ones(self):
        with pytest.raises(typer.Abort):
            _validate_mount_specs(["/a:/b", "/c:/d:Z"])

    def test_rejects_empty_target(self):
        with pytest.raises(typer.Abort):
            _validate_mount_specs(["/host:"])

    def test_rejects_relative_target(self):
        with pytest.raises(typer.Abort):
            _validate_mount_specs(["/host:relative/path"])

    def test_rejects_nonexistent_source_path(self):
        with pytest.raises(typer.Abort):
            _validate_mount_specs(["/nonexistent/path/xyz:/container"])

    def test_accepts_existing_source_path(self, tmp_path: Path):
        _validate_mount_specs([f"{tmp_path}:/container"])


class TestExpandMountFileReferences:
    """Tests for _expand_mount_file_references."""

    def test_returns_plain_specs_unchanged(self):
        result = _expand_mount_file_references(["/a:/b", "/c:/d:ro"])
        assert result == ["/a:/b", "/c:/d:ro"]

    def test_expands_file_reference(self, tmp_path: Path):
        mounts_file = tmp_path / "mounts.txt"
        mounts_file.write_text("/host1:/container1\n/host2:/container2:ro\n")

        result = _expand_mount_file_references([f"@{mounts_file}"])
        assert result == ["/host1:/container1", "/host2:/container2:ro"]

    def test_skips_blank_lines(self, tmp_path: Path):
        mounts_file = tmp_path / "mounts.txt"
        mounts_file.write_text("/a:/b\n\n  \n/c:/d\n")

        result = _expand_mount_file_references([f"@{mounts_file}"])
        assert result == ["/a:/b", "/c:/d"]

    def test_skips_comment_lines(self, tmp_path: Path):
        mounts_file = tmp_path / "mounts.txt"
        mounts_file.write_text(
            "# Models directory\n/models:/models:ro\n# Data\n/data:/data\n"
        )

        result = _expand_mount_file_references([f"@{mounts_file}"])
        assert result == ["/models:/models:ro", "/data:/data"]

    def test_strips_whitespace_from_lines(self, tmp_path: Path):
        mounts_file = tmp_path / "mounts.txt"
        mounts_file.write_text("  /a:/b  \n  /c:/d:ro  \n")

        result = _expand_mount_file_references([f"@{mounts_file}"])
        assert result == ["/a:/b", "/c:/d:ro"]

    def test_mixes_plain_specs_and_file_references(self, tmp_path: Path):
        mounts_file = tmp_path / "mounts.txt"
        mounts_file.write_text("/from_file:/target\n")

        result = _expand_mount_file_references(
            ["/plain:/spec", f"@{mounts_file}", "/another:/one:ro"]
        )
        assert result == ["/plain:/spec", "/from_file:/target", "/another:/one:ro"]

    def test_aborts_on_nonexistent_file(self, capsys: pytest.CaptureFixture[str]):
        with pytest.raises(typer.Abort):
            _expand_mount_file_references(["@/nonexistent/mounts.txt"])

        _, stderr = capsys.readouterr()
        assert "/nonexistent/mounts.txt" in stderr

    def test_empty_file_produces_no_entries(self, tmp_path: Path):
        mounts_file = tmp_path / "empty.txt"
        mounts_file.write_text("")

        result = _expand_mount_file_references([f"@{mounts_file}"])
        assert result == []

    def test_file_with_only_comments_and_blanks(self, tmp_path: Path):
        mounts_file = tmp_path / "mounts.txt"
        mounts_file.write_text("# just a comment\n\n  # another\n  \n")

        result = _expand_mount_file_references([f"@{mounts_file}"])
        assert result == []

    def test_inline_comments_are_not_stripped(self, tmp_path: Path):
        """A # in the middle of a line is NOT treated as a comment."""
        mounts_file = tmp_path / "mounts.txt"
        mounts_file.write_text("/path#1:/target\n")

        result = _expand_mount_file_references([f"@{mounts_file}"])
        assert result == ["/path#1:/target"]


class TestWritableMountWarning:
    """Tests for the writable mount warning when n_parallel > 1."""

    def test_warns_on_writable_mounts_with_parallel(
        self, sample_config: EvaluationRunConfig, capsys: pytest.CaptureFixture[str]
    ):
        """Writable mounts shared across parallel containers should emit a warning."""
        from unittest.mock import patch

        from karotte.cli.run import run

        with (
            patch("karotte.cli.run.parse_config", return_value=sample_config),
            patch("karotte.cli.run._validate_mount_specs"),
            patch("karotte.cli.run._run_without_ui"),
            patch("karotte.build.which", return_value="/usr/bin/podman"),
        ):
            run(
                config="{}",
                containerized=True,
                runtime="podman",
                build_context=".",
                n_parallel=2,
                dev=False,
                mount=["/data:/data"],
                no_ui=True,
                keep_containers=False,
                cache_from=None,
                cache_to=None,
            )

        captured = capsys.readouterr()
        assert "writable bind mount" in captured.err
        assert "2 parallel containers" in captured.err

    def test_no_warning_when_all_mounts_readonly(
        self, sample_config: EvaluationRunConfig, capsys: pytest.CaptureFixture[str]
    ):
        """Read-only mounts should not trigger a warning."""
        from unittest.mock import patch

        from karotte.cli.run import run

        with (
            patch("karotte.cli.run.parse_config", return_value=sample_config),
            patch("karotte.cli.run._validate_mount_specs"),
            patch("karotte.cli.run._run_without_ui"),
            patch("karotte.build.which", return_value="/usr/bin/podman"),
        ):
            run(
                config="{}",
                containerized=True,
                runtime="podman",
                build_context=".",
                n_parallel=2,
                dev=False,
                mount=["/data:/data:ro"],
                no_ui=True,
                keep_containers=False,
                cache_from=None,
                cache_to=None,
            )

        captured = capsys.readouterr()
        assert "writable bind mount" not in captured.err

    def test_no_warning_when_single_container(
        self, sample_config: EvaluationRunConfig, capsys: pytest.CaptureFixture[str]
    ):
        """Single container should not trigger a warning even with writable mounts."""
        from unittest.mock import patch

        from karotte.cli.run import run

        with (
            patch("karotte.cli.run.parse_config", return_value=sample_config),
            patch("karotte.cli.run._validate_mount_specs"),
            patch("karotte.cli.run._run_without_ui"),
            patch("karotte.build.which", return_value="/usr/bin/podman"),
        ):
            run(
                config="{}",
                containerized=True,
                runtime="podman",
                build_context=".",
                n_parallel=1,
                dev=False,
                mount=["/data:/data"],
                no_ui=True,
                keep_containers=False,
                cache_from=None,
                cache_to=None,
            )

        captured = capsys.readouterr()
        assert "writable bind mount" not in captured.err


class TestDockerGvisorRunCommand:
    """Tests for get_container_run_command with docker:gvisor runtime."""

    def test_uses_docker_engine(self, sample_config: EvaluationRunConfig):
        command, _ = get_container_run_command(
            sample_config, "docker:gvisor", dev=False, keep_container=False
        )

        assert "docker" in command
        assert "podman" not in command

    def test_includes_runtime_runsc(self, sample_config: EvaluationRunConfig):
        command, _ = get_container_run_command(
            sample_config, "docker:gvisor", dev=False, keep_container=False
        )

        assert "--runtime=runsc" in command

    def test_runsc_before_cap_add(self, sample_config: EvaluationRunConfig):
        """--runtime=runsc should appear after 'run' but before container flags."""
        command, _ = get_container_run_command(
            sample_config, "docker:gvisor", dev=False, keep_container=False
        )

        run_index = command.index("run")
        runsc_index = command.index("--runtime=runsc")
        cap_index = command.index("--cap-add=NET_ADMIN")
        assert run_index < runsc_index < cap_index

    @pytest.mark.parametrize("runtime", ["docker:gvisor", "docker", "podman"])
    def test_only_gvisor_gets_sys_ptrace(
        self, sample_config: EvaluationRunConfig, runtime: Runtime
    ):
        """Under gVisor, root needs it to read the student's /proc/<pid>/smaps, which the memory watchdog weighs."""
        command, _ = get_container_run_command(
            sample_config, runtime, dev=False, keep_container=False
        )

        assert ("--cap-add=SYS_PTRACE" in command) == (runtime == "docker:gvisor")

    def test_declares_the_gvisor_sandbox(self, sample_config: EvaluationRunConfig):
        command, _ = get_container_run_command(
            sample_config, "docker:gvisor", dev=False, keep_container=False
        )

        env_indices = [i for i, arg in enumerate(command) if arg == "--env"]
        env_values = [command[i + 1] for i in env_indices]
        assert "KAROTTE_SANDBOX=gvisor" in env_values
        assert not [v for v in env_values if v.startswith("KAROTTE_GVISOR")]

    def test_uses_plain_image_name(self, sample_config: EvaluationRunConfig):
        """docker:gvisor should use 'karotte', not 'localhost/karotte'."""
        command, _ = get_container_run_command(
            sample_config, "docker:gvisor", dev=False, keep_container=False
        )

        assert "karotte" in command
        assert "localhost/karotte" not in command

    def test_includes_standard_flags(self, sample_config: EvaluationRunConfig):
        command, _ = get_container_run_command(
            sample_config, "docker:gvisor", dev=False, keep_container=False
        )

        assert "--cap-add=NET_ADMIN" in command
        assert "--rm" in command
        assert f"karotte_run_{sample_config.run_id}" in command

    def test_keep_container_excludes_rm(self, sample_config: EvaluationRunConfig):
        command, _ = get_container_run_command(
            sample_config, "docker:gvisor", dev=False, keep_container=True
        )

        assert "--rm" not in command

    def test_plain_docker_does_not_include_runsc(
        self, sample_config: EvaluationRunConfig
    ):
        """Sanity: plain docker runtime should not get --runtime=runsc."""
        command, _ = get_container_run_command(
            sample_config, "docker", dev=False, keep_container=False
        )

        assert "--runtime=runsc" not in command

    def test_plain_docker_declares_runc(self, sample_config: EvaluationRunConfig):
        command, _ = get_container_run_command(
            sample_config, "docker", dev=False, keep_container=False
        )

        env_indices = [i for i, arg in enumerate(command) if arg == "--env"]
        env_values = [command[i + 1] for i in env_indices]
        assert "KAROTTE_SANDBOX=runc" in env_values

    def test_podman_does_not_include_runsc(self, sample_config: EvaluationRunConfig):
        """Sanity: podman runtime should not get --runtime=runsc."""
        command, _ = get_container_run_command(
            sample_config, "podman", dev=False, keep_container=False
        )

        assert "--runtime=runsc" not in command


class TestDockerGvisorCleanUp:
    """Tests for clean_up_old_containers with docker:gvisor runtime."""

    def test_uses_docker_engine_for_ps(self):
        with patch("karotte.run_helpers.subprocess.run") as mock_run:
            mock_run.return_value.stdout = ""
            mock_run.return_value.returncode = 0

            clean_up_old_containers("docker:gvisor", ["run-a"])

        cmd = mock_run.call_args_list[0][0][0]
        assert cmd[0] == "docker"

    def test_uses_docker_engine_for_rm(self):
        with patch("karotte.run_helpers.subprocess.run") as mock_run:
            mock_run.return_value.stdout = "container1"
            mock_run.return_value.returncode = 0

            clean_up_old_containers("docker:gvisor", ["run-a"])

        assert mock_run.call_count == 2
        rm_cmd = mock_run.call_args_list[1][0][0]
        assert rm_cmd[0] == "docker"
        assert "rm" in rm_cmd


@pytest.fixture(autouse=True)
def _fixed_network_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):  # pyright: ignore[reportUnusedFunction]
    """The strict firewall allows the sandbox's own addresses; pin them so rule
    lists don't depend on the machine running the tests. Proxy pinning writes
    to a temp hosts file, never the real /etc/hosts."""
    monkeypatch.setattr("karotte.run_helpers.HOSTS_FILE", tmp_path / "etc-hosts")
    monkeypatch.setattr("karotte.run_helpers.reachable_as", lambda _uid, _targets: [])  # pyright: ignore[reportUnknownLambdaType]
    monkeypatch.delenv("KAROTTE_STUDENT_NETWORK", raising=False)
    # A karotte container always names its student; the firewall check needs it.
    monkeypatch.setenv("KAROTTE_DEMOTE_ID", "1000")
    own: tuple[list[str], list[str]] = (["10.1.2.3"], [])
    monkeypatch.setattr("karotte.confinement._own_addresses", lambda: own)


class TestMaybeBlockInternetGvisor:
    """Tests for _maybe_block_internet firewall rule generation under gvisor."""

    def test_uses_iptables_nft_by_default(self, monkeypatch: pytest.MonkeyPatch):
        """Outside gVisor, should use standard iptables (nft)."""
        monkeypatch.setenv("KAROTTE_CONTAINERIZED", "1")
        monkeypatch.setenv("KAROTTE_SANDBOX", "runc")

        rules: list[str] = []
        with patch(
            "karotte.confinement.subprocess.run",
            return_value=MagicMock(stderr="", returncode=0),
        ) as mock_run:
            _maybe_block_internet()
            rules = [call[0][0] for call in mock_run.call_args_list]

        assert all(
            r.startswith("iptables ") or r.startswith("ip6tables ") for r in rules
        )
        assert not any("iptables-legacy" in r for r in rules)

    def test_uses_iptables_legacy_under_gvisor(self, monkeypatch: pytest.MonkeyPatch):
        """Under gVisor, should use iptables-legacy."""
        monkeypatch.setenv("KAROTTE_CONTAINERIZED", "1")
        monkeypatch.setenv("KAROTTE_SANDBOX", "gvisor")

        rules: list[str] = []
        with patch(
            "karotte.confinement.subprocess.run",
            return_value=MagicMock(stderr=""),
        ) as mock_run:
            _maybe_block_internet()
            rules = [call[0][0] for call in mock_run.call_args_list]

        ipv4_rules = [r for r in rules if not r.startswith("ip6")]
        ipv6_rules = [r for r in rules if r.startswith("ip6")]
        assert all(r.startswith("iptables-legacy ") for r in ipv4_rules)
        assert all(r.startswith("ip6tables-legacy ") for r in ipv6_rules)

    def test_uses_drop_instead_of_reject_under_gvisor(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Under gVisor, catch-all rules should use DROP instead of REJECT."""
        monkeypatch.setenv("KAROTTE_CONTAINERIZED", "1")
        monkeypatch.setenv("KAROTTE_SANDBOX", "gvisor")

        rules: list[str] = []
        with patch(
            "karotte.confinement.subprocess.run",
            return_value=MagicMock(stderr="", returncode=0),
        ) as mock_run:
            _maybe_block_internet()
            rules = [call[0][0] for call in mock_run.call_args_list]

        reject_rules = [r for r in rules if "REJECT" in r]
        drop_rules = [r for r in rules if "-j DROP" in r]
        assert len(reject_rules) == 0
        assert len(drop_rules) == 2  # one IPv4, one IPv6

    def test_uses_reject_without_gvisor(self, monkeypatch: pytest.MonkeyPatch):
        """Outside gVisor, catch-all rules should use REJECT."""
        monkeypatch.setenv("KAROTTE_CONTAINERIZED", "1")
        monkeypatch.setenv("KAROTTE_SANDBOX", "runc")

        rules: list[str] = []
        with patch(
            "karotte.confinement.subprocess.run",
            return_value=MagicMock(stderr="", returncode=0),
        ) as mock_run:
            _maybe_block_internet()
            rules = [call[0][0] for call in mock_run.call_args_list]

        reject_rules = [r for r in rules if "-j REJECT" in r]
        drop_rules = [r for r in rules if "-j DROP" in r]
        assert len(reject_rules) == 2  # one IPv4, one IPv6
        assert len(drop_rules) == 0

    def test_does_nothing_outside_container(self, monkeypatch: pytest.MonkeyPatch):
        """Should not apply any rules when KAROTTE_CONTAINERIZED is not set."""
        monkeypatch.delenv("KAROTTE_CONTAINERIZED", raising=False)
        monkeypatch.setenv("KAROTTE_SANDBOX", "runc")

        with patch("karotte.confinement.subprocess.run") as mock_run:
            _maybe_block_internet()

        mock_run.assert_not_called()

    def test_applies_all_expected_rules_under_gvisor(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """IPv4: localhost, the own address, reject. IPv6: localhost, reject."""
        monkeypatch.setenv("KAROTTE_CONTAINERIZED", "1")
        monkeypatch.setenv("KAROTTE_SANDBOX", "gvisor")

        with patch(
            "karotte.confinement.subprocess.run",
            return_value=MagicMock(stderr="", returncode=0),
        ) as mock_run:
            _maybe_block_internet()

        assert mock_run.call_count == 5


class TestMaybeBlockInternetBlockedPorts:
    """Tests for _maybe_block_internet blocked_ports parameter."""

    def _get_rules(
        self, monkeypatch: pytest.MonkeyPatch, blocked_ports: list[int] | None = None
    ) -> list[str]:
        monkeypatch.setenv("KAROTTE_CONTAINERIZED", "1")
        monkeypatch.setenv("KAROTTE_SANDBOX", "runc")

        with patch(
            "karotte.confinement.subprocess.run",
            return_value=MagicMock(returncode=0, stderr=""),
        ) as mock_run:
            _maybe_block_internet(blocked_ports=blocked_ports)
            return [call[0][0] for call in mock_run.call_args_list]

    def test_blocked_ports_generate_drop_rules(self, monkeypatch: pytest.MonkeyPatch):
        """Each blocked port should produce a DROP rule for IPv4 and IPv6."""
        rules = self._get_rules(monkeypatch, blocked_ports=[8001, 8080])

        port_rules = [r for r in rules if "--dport" in r]
        assert len(port_rules) == 4  # 2 ports x 2 (IPv4 + IPv6)
        assert all("-j DROP" in r for r in port_rules)

    def test_blocked_port_rules_always_use_drop_not_reject(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Port-block rules should use DROP even when gvisor is off (catch-all uses REJECT)."""
        rules = self._get_rules(monkeypatch, blocked_ports=[8001])

        port_rules = [r for r in rules if "--dport" in r]
        assert all("-j DROP" in r for r in port_rules)
        # The catch-all rules should still use REJECT (non-gvisor)
        catchall_reject = [r for r in rules if "-j REJECT" in r]
        assert len(catchall_reject) == 2  # one IPv4, one IPv6

    def test_blocked_port_rules_come_before_localhost_accept(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Port-block rules must precede the localhost ACCEPT rule in iptables order."""
        rules = self._get_rules(monkeypatch, blocked_ports=[8001])

        ipv4_rules = [r for r in rules if r.startswith("iptables ")]
        ipv4_port_idx = next(i for i, r in enumerate(ipv4_rules) if "--dport 8001" in r)
        ipv4_accept_idx = next(
            i for i, r in enumerate(ipv4_rules) if "127.0.0.0/8" in r and "ACCEPT" in r
        )
        assert ipv4_port_idx < ipv4_accept_idx

        ipv6_rules = [r for r in rules if r.startswith("ip6tables ")]
        ipv6_port_idx = next(i for i, r in enumerate(ipv6_rules) if "--dport 8001" in r)
        ipv6_accept_idx = next(
            i for i, r in enumerate(ipv6_rules) if "::1" in r and "ACCEPT" in r
        )
        assert ipv6_port_idx < ipv6_accept_idx

    def test_no_blocked_ports_produces_same_rules_as_before(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Passing no blocked_ports should not add any port-block rules."""
        rules = self._get_rules(monkeypatch, blocked_ports=None)

        port_rules = [r for r in rules if "--dport" in r]
        assert len(port_rules) == 0
        # 3 IPv4 (localhost, own address, reject) + 2 IPv6 (localhost, reject)
        assert len(rules) == 5

    def test_rule_count_with_blocked_ports(self, monkeypatch: pytest.MonkeyPatch):
        """Each blocked port adds one IPv4 and one IPv6 rule."""
        rules = self._get_rules(monkeypatch, blocked_ports=[8001, 8080])

        # 5 base rules + 2 ports * 2 (IPv4 + IPv6) = 9
        assert len(rules) == 9

    def test_allowed_ips_generate_accept_before_reject(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Allowed egress IPs get an ACCEPT rule ahead of the catch-all reject."""
        monkeypatch.setenv("KAROTTE_CONTAINERIZED", "1")
        monkeypatch.setenv("KAROTTE_SANDBOX", "runc")
        with patch(
            "karotte.confinement.subprocess.run",
            return_value=MagicMock(returncode=0, stderr=""),
        ) as mock_run:
            _maybe_block_internet(allowed_ips=["203.0.113.7"])
            rules = [call[0][0] for call in mock_run.call_args_list]
        ipv4 = [r for r in rules if r.startswith("iptables ")]
        accept_idx = next(
            i for i, r in enumerate(ipv4) if "203.0.113.7" in r and "ACCEPT" in r
        )
        reject_idx = next(i for i, r in enumerate(ipv4) if "-j REJECT" in r)
        assert accept_idx < reject_idx

    def test_blocked_ports_under_gvisor(self, monkeypatch: pytest.MonkeyPatch):
        """Port-block rules should use iptables-legacy under gvisor."""
        monkeypatch.setenv("KAROTTE_CONTAINERIZED", "1")
        monkeypatch.setenv("KAROTTE_SANDBOX", "gvisor")

        with patch(
            "karotte.confinement.subprocess.run",
            return_value=MagicMock(returncode=0, stderr=""),
        ) as mock_run:
            _maybe_block_internet(blocked_ports=[8001])
            rules = [call[0][0] for call in mock_run.call_args_list]

        port_rules = [r for r in rules if "--dport" in r]
        assert len(port_rules) == 2
        assert any(r.startswith("iptables-legacy ") for r in port_rules)
        assert any(r.startswith("ip6tables-legacy ") for r in port_rules)
        assert all("-j DROP" in r for r in port_rules)

    def test_nonzero_return_code_without_gvisor_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """A failing iptables call should raise RuntimeError when gvisor is off."""
        monkeypatch.setenv("KAROTTE_CONTAINERIZED", "1")
        monkeypatch.setenv("KAROTTE_SANDBOX", "runc")

        with patch(
            "karotte.confinement.subprocess.run",
            return_value=MagicMock(returncode=1, stderr="iptables: command failed"),
        ):
            with pytest.raises(RuntimeError, match="failed to apply firewall rule"):
                _maybe_block_internet(blocked_ports=[8001])

    def test_nonzero_return_code_with_gvisor_logs_and_breaks(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Under gvisor, a failing iptables call should warn and stop applying
        further rules instead of raising."""
        monkeypatch.setenv("KAROTTE_CONTAINERIZED", "1")
        monkeypatch.setenv("KAROTTE_SANDBOX", "gvisor")

        with patch(
            "karotte.confinement.subprocess.run",
            return_value=MagicMock(returncode=1, stderr="iptables: command failed"),
        ) as mock_run:
            _maybe_block_internet(blocked_ports=[8001])

        # First rule fails → loop breaks, no further rules attempted.
        assert mock_run.call_count == 1


@final
class _EmptyTask(Task):
    @property
    def system_prompt(self) -> str | None:
        return None

    id = "test-task"

    @property
    def steps(self):
        return []

    @property
    def tools(self):
        return []


class TestSetUpRunnerFirewall:
    """Which ports `_set_up_runner` blocks depends on the selected agent."""

    def _blocked_ports(self, config: EvaluationRunConfig) -> list[int]:
        with patch("karotte.run_helpers._maybe_block_internet") as mock_block:
            _set_up_runner(config, _EmptyTask(config))
        return mock_block.call_args.kwargs["blocked_ports"]

    def test_builtin_agent_blocks_both_ports(self, sample_config: EvaluationRunConfig):
        blocked = self._blocked_ports(sample_config)
        assert set(blocked) == {
            sample_config.websocket_config.port,
            sample_config.mcp_server_config.port,
        }

    def test_external_agent_blocks_both_ports(self, sample_config: EvaluationRunConfig):
        config = sample_config.model_copy(
            update={"model": "pt/foo", "backend_uri": "http://backend"}
        )
        blocked = self._blocked_ports(config)
        assert set(blocked) == {
            config.websocket_config.port,
            config.mcp_server_config.port,
        }

    def test_student_mcp_agent_leaves_mcp_port_open(
        self, sample_config: EvaluationRunConfig, monkeypatch: pytest.MonkeyPatch
    ):
        """An agent that runs as the student keeps only the websocket blocked."""
        monkeypatch.delenv("KAROTTE_PROXY_URL", raising=False)
        runner = MagicMock()
        runner.allows_student_mcp_access = True
        with patch("karotte.evaluation_runner.EvaluationRunner", return_value=runner):
            with patch("karotte.run_helpers._maybe_block_internet") as mock_block:
                _set_up_runner(sample_config, _EmptyTask(sample_config))
        blocked = mock_block.call_args.kwargs["blocked_ports"]
        assert blocked == [sample_config.websocket_config.port]
        assert sample_config.mcp_server_config.port not in blocked

    def test_student_agent_allows_proxy_egress(
        self, sample_config: EvaluationRunConfig, monkeypatch: pytest.MonkeyPatch
    ):
        """In agent mode the proxy host is resolved and allowed for the student."""
        monkeypatch.setenv("KAROTTE_PROXY_URL", "https://proxy.example")
        runner = MagicMock()
        runner.allows_student_mcp_access = True
        with patch("karotte.evaluation_runner.EvaluationRunner", return_value=runner):
            with patch(
                "karotte.run_helpers._resolve_host_ips", return_value=["203.0.113.7"]
            ):
                with patch("karotte.run_helpers._maybe_block_internet") as mock_block:
                    _set_up_runner(sample_config, _EmptyTask(sample_config))
        assert mock_block.call_args.kwargs["allowed_ips"] == ["203.0.113.7"]

    def test_student_agent_pins_the_proxy_host(
        self,
        sample_config: EvaluationRunConfig,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ):
        """The firewall lets the student reach no DNS resolver, so the proxy's
        name has to resolve from /etc/hosts."""
        hosts = tmp_path / "hosts"
        _ = hosts.write_text("127.0.0.1 localhost")
        monkeypatch.setattr("karotte.run_helpers.HOSTS_FILE", hosts)
        monkeypatch.setenv("KAROTTE_CONTAINERIZED", "1")
        monkeypatch.setenv("KAROTTE_PROXY_URL", "https://proxy.example:8443/v1")
        runner = MagicMock()
        runner.allows_student_mcp_access = True
        with patch("karotte.evaluation_runner.EvaluationRunner", return_value=runner):
            with patch(
                "karotte.run_helpers._resolve_host_ips",
                return_value=["203.0.113.7", "203.0.113.8"],
            ):
                with patch("karotte.run_helpers._maybe_block_internet"):
                    _set_up_runner(sample_config, _EmptyTask(sample_config))
                    _set_up_runner(sample_config, _EmptyTask(sample_config))

        assert hosts.read_text() == (
            "127.0.0.1 localhost\n203.0.113.7 proxy.example\n203.0.113.8 proxy.example\n"
        )

    def test_outside_a_container_the_hosts_file_is_left_alone(
        self,
        sample_config: EvaluationRunConfig,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ):
        hosts = tmp_path / "hosts"
        _ = hosts.write_text("127.0.0.1 localhost\n")
        monkeypatch.setattr("karotte.run_helpers.HOSTS_FILE", hosts)
        monkeypatch.delenv("KAROTTE_CONTAINERIZED", raising=False)
        monkeypatch.setenv("KAROTTE_PROXY_URL", "https://proxy.example")
        runner = MagicMock()
        runner.allows_student_mcp_access = True
        with patch("karotte.evaluation_runner.EvaluationRunner", return_value=runner):
            with patch(
                "karotte.run_helpers._resolve_host_ips", return_value=["203.0.113.7"]
            ):
                with patch("karotte.run_helpers._maybe_block_internet"):
                    _set_up_runner(sample_config, _EmptyTask(sample_config))

        assert hosts.read_text() == "127.0.0.1 localhost\n"

    def test_builtin_agent_allows_no_egress(self, sample_config: EvaluationRunConfig):
        with patch("karotte.run_helpers._maybe_block_internet") as mock_block:
            _set_up_runner(sample_config, _EmptyTask(sample_config))
        assert mock_block.call_args.kwargs["allowed_ips"] == []

    @pytest.mark.parametrize("took", [True, False])
    def test_runner_records_whether_the_firewall_took(
        self, sample_config: EvaluationRunConfig, took: bool
    ):
        with patch("karotte.run_helpers._maybe_block_internet", return_value=took):
            runner = _set_up_runner(sample_config, _EmptyTask(sample_config))
        assert runner.network_firewall is took


class TestValidateGvisorRuntime:
    """Tests for validate_gvisor_runtime daemon.json checking."""

    _BASE_ARGS: list[str] = [
        "-net-raw",
        "--systrap-disable-syscall-patching",
        "-overlay2=none",
        "-file-access=shared",
        "-network=sandbox",
        "--net-disconnect-ok",
    ]

    @pytest.fixture(autouse=True)
    def _mock_runsc_installed(self) -> Any:
        with patch("shutil.which", return_value="/usr/bin/runsc"):
            yield

    def _write_daemon_json(self, tmp_path: Path, content: dict[str, Any]) -> Path:
        path = tmp_path / "daemon.json"
        path.write_text(json.dumps(content))
        return path

    def _runsc_config(self, args: list[str]) -> dict[str, Any]:
        return {
            "runtimes": {
                "runsc": {
                    "path": "/usr/bin/runsc",
                    "runtimeArgs": args,
                }
            }
        }

    # --- runsc binary checks ---

    def test_aborts_when_runsc_not_installed(self, tmp_path: Path):
        path = self._write_daemon_json(tmp_path, self._runsc_config(self._BASE_ARGS))

        with (
            patch("shutil.which", return_value=None),
            pytest.raises(typer.Abort),
        ):
            validate_gvisor_runtime(daemon_json_path=path)

    def test_runsc_not_installed_shows_install_url(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ):
        path = self._write_daemon_json(tmp_path, {})

        with (
            patch("shutil.which", return_value=None),
            pytest.raises(typer.Abort),
        ):
            validate_gvisor_runtime(daemon_json_path=path)

        err = capsys.readouterr().err
        assert "not installed" in err
        assert "gvisor.dev" in err

    # --- valid configs ---

    def test_passes_with_valid_config(self, tmp_path: Path):
        path = self._write_daemon_json(tmp_path, self._runsc_config(self._BASE_ARGS))
        validate_gvisor_runtime(daemon_json_path=path)

    def test_passes_with_extra_args(self, tmp_path: Path):
        """Extra runtimeArgs beyond the required ones should be fine."""
        path = self._write_daemon_json(
            tmp_path,
            self._runsc_config(self._BASE_ARGS + ["-some-extra-flag"]),
        )
        validate_gvisor_runtime(daemon_json_path=path)

    def test_passes_with_extra_runtimes(self, tmp_path: Path):
        """Other runtimes alongside runsc should be fine."""
        config = self._runsc_config(self._BASE_ARGS)
        config["runtimes"]["runc"] = {"path": "/usr/bin/runc"}
        path = self._write_daemon_json(tmp_path, config)
        validate_gvisor_runtime(daemon_json_path=path)

    # --- daemon.json structure errors ---

    def test_aborts_when_file_missing(self, tmp_path: Path):
        path = tmp_path / "nonexistent.json"

        with pytest.raises(typer.Abort):
            validate_gvisor_runtime(daemon_json_path=path)

    def test_aborts_when_file_is_invalid_json(self, tmp_path: Path):
        path = tmp_path / "daemon.json"
        path.write_text("not valid json{{{")

        with pytest.raises(typer.Abort):
            validate_gvisor_runtime(daemon_json_path=path)

    def test_aborts_when_runsc_not_registered(self, tmp_path: Path):
        path = self._write_daemon_json(tmp_path, {"runtimes": {}})

        with pytest.raises(typer.Abort):
            validate_gvisor_runtime(daemon_json_path=path)

    def test_aborts_when_runtimes_key_missing(self, tmp_path: Path):
        path = self._write_daemon_json(tmp_path, {"storage-driver": "overlay2"})

        with pytest.raises(typer.Abort):
            validate_gvisor_runtime(daemon_json_path=path)

    # --- missing required args ---

    def test_aborts_when_net_raw_missing(self, tmp_path: Path):
        path = self._write_daemon_json(
            tmp_path, self._runsc_config(["-network=sandbox"])
        )

        with pytest.raises(typer.Abort):
            validate_gvisor_runtime(daemon_json_path=path)

    def test_aborts_when_network_sandbox_missing(self, tmp_path: Path):
        path = self._write_daemon_json(tmp_path, self._runsc_config(["-net-raw"]))

        with pytest.raises(typer.Abort):
            validate_gvisor_runtime(daemon_json_path=path)

    def test_aborts_when_runtime_args_empty(self, tmp_path: Path):
        path = self._write_daemon_json(tmp_path, self._runsc_config([]))

        with pytest.raises(typer.Abort):
            validate_gvisor_runtime(daemon_json_path=path)

    def test_aborts_when_runtime_args_key_missing(self, tmp_path: Path):
        path = self._write_daemon_json(
            tmp_path,
            {"runtimes": {"runsc": {"path": "/usr/bin/runsc"}}},
        )

        with pytest.raises(typer.Abort):
            validate_gvisor_runtime(daemon_json_path=path)

    # --- error messages ---

    def test_error_message_mentions_missing_args(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ):
        path = self._write_daemon_json(
            tmp_path, self._runsc_config(["-network=sandbox"])
        )

        with pytest.raises(typer.Abort):
            validate_gvisor_runtime(daemon_json_path=path)

        err = capsys.readouterr().err
        assert "-net-raw" in err

    def test_instructions_show_the_daemon_config(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ):
        path = tmp_path / "nonexistent.json"

        with pytest.raises(typer.Abort):
            validate_gvisor_runtime(daemon_json_path=path)

        err = capsys.readouterr().err
        assert "sudo tee" in err
        assert "sudo systemctl reload docker" in err
        assert "-net-raw" in err
        assert "-network=sandbox" in err

    def test_instructions_name_the_installed_runsc(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ):
        """gVisor's install puts runsc in /usr/local/bin with its gvisor-bin/ sidecars next to it."""
        with (
            patch("shutil.which", return_value="/usr/local/bin/runsc"),
            pytest.raises(typer.Abort),
        ):
            validate_gvisor_runtime(daemon_json_path=tmp_path / "nonexistent.json")

        err = capsys.readouterr().err
        assert '"path": "/usr/local/bin/runsc"' in err
        assert "gvisor-bin" in err


class TestRunNonContainerizedBackendStreaming:
    async def _backend_events(self, config: EvaluationRunConfig) -> list[object] | None:
        from karotte.run_helpers import run_non_containerized

        event = object()
        received: list[list[object]] = []

        async def fake_backend(events: Any, _run_config: EvaluationRunConfig) -> None:
            received.append([e async for e in events])

        async def drain(*args: Any) -> None:
            async for _ in args[-1]:
                pass

        async def run_events():
            yield event

        runner = MagicMock()
        runner.run = run_events
        with (
            patch("karotte.run_helpers._set_up_runner", return_value=runner),
            patch("karotte.run_helpers.stream_transcript_to_websocket", drain),
            patch("karotte.run_helpers.stream_transcript_to_stdout", drain),
            patch(
                "karotte.transcript_streaming.stream_transcript_to_backend.stream_transcript_to_backend",
                fake_backend,
            ),
        ):
            await run_non_containerized(config, MagicMock())
        if not received:
            return None
        assert received == [[event]]
        return received[0]

    @pytest.mark.asyncio
    async def test_streams_to_the_backend_when_backend_uri_is_set(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("KAROTTE_BACKEND_TOKEN_PATH", str(tmp_path / "missing"))
        config = EvaluationRunConfig(
            run_id="r",
            task_id="t",
            model="m",
            model_api_key="k",
            backend_uri="http://backend",
        )
        assert await self._backend_events(config) is not None

    @pytest.mark.asyncio
    async def test_does_not_stream_without_backend_uri(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        token = tmp_path / "token"
        token.write_text("tok")
        monkeypatch.setenv("KAROTTE_BACKEND_TOKEN_PATH", str(token))
        config = EvaluationRunConfig(
            run_id="r", task_id="t", model="m", model_api_key="k"
        )
        assert await self._backend_events(config) is None


class TestChownOutputs:
    """Outputs written in the container go to the owner of the mounted output dir, without following symlinks."""

    @pytest.fixture
    def layout(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        out = tmp_path / "out"
        artifacts = out / "r_artifacts"
        (artifacts / "sub").mkdir(parents=True)
        (out / "transcript.json").write_text("{}")
        (artifacts / "a.txt").write_text("a")
        (artifacts / "sub" / "b.txt").write_text("b")
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret").write_text("s")
        (artifacts / "link").symlink_to(outside / "secret")
        (artifacts / "linkdir").symlink_to(outside)
        monkeypatch.setattr(
            sys.modules["karotte.save_artifact"], "_created_target_dir", artifacts
        )
        monkeypatch.setenv("KAROTTE_CONTAINERIZED", "1")
        return out

    def _chown_calls(
        self, sample_config: EvaluationRunConfig, out: Path
    ) -> list[tuple[str, int, int, dict[str, Any]]]:
        calls: list[tuple[str, int, int, dict[str, Any]]] = []

        def record(path: Any, uid: int, gid: int, **kwargs: Any) -> None:
            calls.append((os.fsdecode(path), uid, gid, kwargs))

        config = sample_config.model_copy(
            update={"transcript_file": str(out / "transcript.json")}
        )
        with patch("karotte.run_helpers.os.chown", record):
            chown_outputs(config)
        return calls

    def test_chowns_transcript_and_artifacts_to_the_output_dir_owner(
        self, sample_config: EvaluationRunConfig, layout: Path
    ):
        calls = self._chown_calls(sample_config, layout)

        owner = layout.stat()
        names = {Path(path).name for path, _, _, _ in calls}
        assert names == {
            "transcript.json",
            "r_artifacts",
            "a.txt",
            "sub",
            "b.txt",
            "link",
            "linkdir",
        }
        assert all(
            (uid, gid) == (owner.st_uid, owner.st_gid) for _, uid, gid, _ in calls
        )

    def test_never_follows_symlinks(
        self, sample_config: EvaluationRunConfig, layout: Path
    ):
        calls = self._chown_calls(sample_config, layout)

        assert all(kwargs.get("follow_symlinks") is False for *_, kwargs in calls)
        assert "secret" not in {Path(path).name for path, _, _, _ in calls}

    def test_does_nothing_outside_a_container(
        self,
        sample_config: EvaluationRunConfig,
        layout: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.delenv("KAROTTE_CONTAINERIZED")

        assert self._chown_calls(sample_config, layout) == []


class TestOutputDirOwnerOnHost:
    """Under sudo, the host hands the output dir to the invoking user so the container chowns outputs to them."""

    def _command(
        self, sample_config: EvaluationRunConfig, tmp_path: Path
    ) -> list[tuple[Any, ...]]:
        calls: list[tuple[Any, ...]] = []

        def record(*args: Any, **kwargs: Any) -> None:
            calls.append((*args, kwargs))

        config = sample_config.model_copy(
            update={"transcript_file": str(tmp_path / "out" / "transcript.json")}
        )
        with (
            patch("karotte.run_helpers.load_task", return_value=MagicMock()),
            patch("karotte.run_helpers.os.chown", record),
        ):
            get_container_run_command(config, "docker", dev=False, keep_container=False)
        return calls

    def test_chowns_the_output_dir_to_the_sudo_user(
        self,
        sample_config: EvaluationRunConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setattr("karotte.run_helpers.os.geteuid", lambda: 0)
        monkeypatch.setenv("SUDO_UID", "1234")
        monkeypatch.setenv("SUDO_GID", "5678")

        calls = self._command(sample_config, tmp_path)

        assert calls == [(tmp_path / "out", 1234, 5678, {"follow_symlinks": False})]

    def test_leaves_the_output_dir_alone_without_sudo(
        self,
        sample_config: EvaluationRunConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.delenv("SUDO_UID", raising=False)

        assert self._command(sample_config, tmp_path) == []


class TestRunNonContainerizedResult:
    """run_non_containerized reports the error a run ended in, and nothing for a pass or fail."""

    async def _run(
        self, sample_config: EvaluationRunConfig, events: list[Any]
    ) -> ErrorEvent | None:
        async def fake_run():
            for event in events:
                yield event

        async def drain(*args: Any) -> None:
            async for _ in args[-1]:
                pass

        runner = MagicMock()
        runner.run = fake_run
        with (
            patch("karotte.run_helpers._set_up_runner", return_value=runner),
            patch("karotte.run_helpers.stream_transcript_to_websocket", drain),
            patch("karotte.run_helpers.stream_transcript_to_stdout", drain),
        ):
            return await run_non_containerized(sample_config, MagicMock())

    @pytest.mark.asyncio
    async def test_returns_the_error_event_of_an_errored_run(
        self, sample_config: EvaluationRunConfig
    ):
        error = ErrorEvent(exception_type="TurnLimitReachedError", message="x")
        result = await self._run(
            sample_config, [error, TaskCompletedEvent(status="error")]
        )

        assert result == error

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", ["passed", "failed"])
    async def test_returns_none_for_a_run_that_did_not_error(
        self, sample_config: EvaluationRunConfig, status: RunStatus
    ):
        result = await self._run(sample_config, [TaskCompletedEvent(status=status)])

        assert result is None


class TestResolveProxy:
    def test_a_lookup_that_keeps_failing_stops_the_run(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Without the address the firewall cuts the agent off from its model."""
        calls: list[str] = []

        def fail(host: str, *_args: object, **_kwargs: object) -> object:
            calls.append(host)
            raise socket.gaierror("Temporary failure in name resolution")

        monkeypatch.setattr("karotte.run_helpers.socket.getaddrinfo", fail)
        monkeypatch.setattr("karotte.run_helpers.time.sleep", lambda _s: None)  # pyright: ignore[reportUnknownLambdaType]

        with pytest.raises(RuntimeError, match="proxy.example"):
            _ = run_helpers._resolve_host_ips("https://proxy.example/v1")  # pyright: ignore[reportPrivateUsage]
        assert len(calls) == 3

    def test_a_brief_failure_is_retried(self, monkeypatch: pytest.MonkeyPatch):
        answers: list[object] = [
            socket.gaierror("Temporary failure"),
            [(socket.AF_INET, 0, 0, "", ("203.0.113.7", 0))],
        ]

        def lookup(*_args: object, **_kwargs: object) -> object:
            answer = answers.pop(0)
            if isinstance(answer, Exception):
                raise answer
            return answer

        monkeypatch.setattr("karotte.run_helpers.socket.getaddrinfo", lookup)
        monkeypatch.setattr("karotte.run_helpers.time.sleep", lambda _s: None)  # pyright: ignore[reportUnknownLambdaType]

        assert run_helpers._resolve_host_ips("https://proxy.example") == ["203.0.113.7"]  # pyright: ignore[reportPrivateUsage]


class TestFirewallSelfTest:
    """After the rules take, the student must fail to reach every canary."""

    def _block(self, monkeypatch: pytest.MonkeyPatch, reached: list[tuple[str, int]]):
        monkeypatch.setenv("KAROTTE_CONTAINERIZED", "1")
        monkeypatch.setenv("KAROTTE_DEMOTE_ID", "1000")
        monkeypatch.setenv("KAROTTE_SANDBOX", "vm")
        seen: list[tuple[int, list[tuple[str, int]]]] = []

        def fake_reachable(uid: int, targets: list[tuple[str, int]]):
            seen.append((uid, targets))
            return reached

        monkeypatch.setattr("karotte.run_helpers.reachable_as", fake_reachable)
        with patch(
            "karotte.confinement.subprocess.run",
            return_value=MagicMock(returncode=0, stderr=""),
        ):
            _maybe_block_internet()
        return seen

    def test_a_refused_student_passes(self, monkeypatch: pytest.MonkeyPatch):
        seen = self._block(monkeypatch, [])
        assert [uid for uid, _ in seen] == [1000]

    def test_a_student_that_gets_through_stops_the_run(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        with pytest.raises(RuntimeError, match="169.254.169.254"):
            _ = self._block(monkeypatch, [("169.254.169.254", 80)])

    def test_allowed_ips_are_left_out_of_the_check(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(
            "karotte.confinement._default_gateway", lambda: "192.168.64.1"
        )
        monkeypatch.setenv("KAROTTE_CONTAINERIZED", "1")
        monkeypatch.setenv("KAROTTE_DEMOTE_ID", "1000")
        monkeypatch.setenv("KAROTTE_SANDBOX", "vm")
        seen: list[list[tuple[str, int]]] = []

        def fake_reachable(_uid: int, targets: list[tuple[str, int]]):
            seen.append(targets)
            return []

        monkeypatch.setattr("karotte.run_helpers.reachable_as", fake_reachable)
        with patch(
            "karotte.confinement.subprocess.run",
            return_value=MagicMock(returncode=0, stderr=""),
        ):
            _maybe_block_internet(allowed_ips=["192.168.64.1"])

        assert all(host != "192.168.64.1" for host, _ in seen[0])

    def test_rules_that_did_not_take_skip_the_check(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """gVisor accepts rules without enforcing them; a check there would
        only restate that."""
        monkeypatch.setenv("KAROTTE_CONTAINERIZED", "1")
        monkeypatch.setenv("KAROTTE_SANDBOX", "gvisor")
        called: list[object] = []
        monkeypatch.setattr(
            "karotte.run_helpers.reachable_as",
            lambda *args: called.append(args) or [],  # pyright: ignore[reportUnknownLambdaType]
        )
        with patch(
            "karotte.confinement.subprocess.run",
            return_value=MagicMock(returncode=0, stderr=""),
        ):
            _maybe_block_internet()
        assert called == []
