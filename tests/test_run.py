import functools
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import typer

from karotte.cli.run import (
    PROXY_ENTRY_POINT_GROUP,
    _run_without_ui,  # pyright: ignore[reportPrivateUsage]
    default_proxy_url,
    run,
)
from karotte.forwarded_env import EXIT_ON_RUN_ERROR_ENV_VAR
from karotte.run_helpers import sanitize_paths
from karotte.schemas.evaluation_run_config import EvaluationRunConfig
from karotte.schemas.http_mcp_server_config import HttpMcpServerConfig
from karotte.schemas.transcript import ErrorEvent
from tests.conftest import register_hardware_plugins


@pytest.fixture(autouse=True)
def _stub_process_hardening():  # pyright: ignore[reportUnusedFunction]
    """run() calls sanitize_paths_and_reexec(), which sanitizes process-global env
    vars in place and re-execs. pytest can't revert either, so a real call would
    strip non-root PATH entries (e.g. nix-store paths holding `uv`) or replace the
    test process. These tests exercise run()'s orchestration, so stub it out.

    harden_filesystem() goes with it: a test that sets KAROTTE_CONTAINERIZED would
    otherwise chmod the machine's own /dev/mqueue when this suite runs on Linux.
    """
    with (
        patch("karotte.cli.run.sanitize_paths_and_reexec"),
        patch("karotte.cli.run.harden_filesystem"),
        patch("karotte.build.which", return_value="/usr/bin/engine"),
        patch("karotte.build._buildx_available", return_value=True),
    ):
        yield


@pytest.fixture(autouse=True)
def _docker_by_default(monkeypatch: pytest.MonkeyPatch):  # pyright: ignore[reportUnusedFunction]
    """The real default depends on the OS running the tests and its VM setup."""
    # The package re-exports the `run` function under the module's name.
    monkeypatch.setattr(
        sys.modules["karotte.cli.run"],
        "default_runtime",
        lambda _hardware=None: "docker",  # pyright: ignore[reportUnknownLambdaType]
    )


@pytest.fixture
def in_karotte_image(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("KAROTTE_CONTAINERIZED", "1")


def test_raises_error_if_trying_to_run_multiple_runs_without_containerization(
    capsys: pytest.CaptureFixture[str],
):
    with pytest.raises(typer.Abort):
        run(config="", containerized=False, n_parallel=2)

    _, stderr = capsys.readouterr()
    assert "Cannot run multiple runs without containerization." in stderr


def test_raises_error_if_trying_to_mount_dev_files_without_containerization(
    capsys: pytest.CaptureFixture[str],
):
    with pytest.raises(typer.Abort):
        run(config="", containerized=False, dev=True)

    _, stderr = capsys.readouterr()
    assert "Cannot use the `--dev` option without containerization." in stderr


def test_raises_error_if_trying_to_keep_containers_without_containerization(
    capsys: pytest.CaptureFixture[str],
):
    with pytest.raises(typer.Abort):
        run(config="", containerized=False, keep_containers=True)

    _, stderr = capsys.readouterr()
    assert (
        "Cannot use the `--keep-containers` option without containerization." in stderr
    )


@pytest.mark.parametrize("prepare_only", [False, True])
def test_no_containerized_aborts_outside_a_karotte_container(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    prepare_only: bool,
):
    monkeypatch.delenv("KAROTTE_CONTAINERIZED", raising=False)
    with (
        patch("karotte.cli.run.parse_config") as parse,
        pytest.raises(typer.Abort),
    ):
        run(config="{}", containerized=False, prepare_only=prepare_only)

    assert capsys.readouterr().err.strip() == (
        "--no-containerized only runs inside a karotte container image; "
        + "drop the flag to run in a container."
    )
    parse.assert_not_called()


def test_raises_error_if_trying_to_prepare_only_with_parallel_runs(
    capsys: pytest.CaptureFixture[str],
):
    with pytest.raises(typer.Abort):
        run(config="", n_parallel=2, prepare_only=True)

    _, stderr = capsys.readouterr()
    assert "Cannot use --prepare-only with --n-parallel > 1." in stderr


@pytest.fixture(autouse=True)
def _restore_proxy_env(monkeypatch: pytest.MonkeyPatch):  # pyright: ignore[reportUnusedFunction]
    """Uncontainerized runs with a proxy export it into os.environ."""
    for var in ("KAROTTE_PROXY_URL", "ANTHROPIC_BASE_URL"):
        monkeypatch.setenv(var, "")
        monkeypatch.delenv(var)


class TestFirecrackerLaunchErrors:
    @patch("karotte.cli.run._print_output_paths")
    @patch("karotte.cli.run.clean_up_old_containers")
    @patch("karotte.cli.run.build_container")
    def test_a_vm_that_cant_start_fails_only_its_run(
        self, _build: MagicMock, _cleanup: MagicMock, _paths: MagicMock
    ):
        """E.g. pasta can't start for one run: that run fails, the rest still run,
        and nothing escapes as a traceback."""
        from karotte.firecracker.network import NetworkError

        ran: list[str] = []

        def run_containerized(config: EvaluationRunConfig, **_kwargs: object) -> None:
            if config.run_id == "b":
                raise NetworkError("pasta can't set up a network namespace here")
            ran.append(config.run_id)

        configs = [
            EvaluationRunConfig(run_id=r, task_id="t", model="m", model_api_key="k")
            for r in ("a", "b", "c")
        ]
        with patch("karotte.cli.run.run_containerized", run_containerized):
            with pytest.raises(typer.Exit) as exc_info:
                _run_without_ui(
                    configs, "firecracker", True, ".", keep_containers=False
                )

        assert exc_info.value.exit_code == 1
        assert sorted(ran) == ["a", "c"]


class TestRunWithoutUiCache:
    """Test that _run_without_ui forwards cache args to build_container."""

    def _make_config(self) -> MagicMock:
        config = MagicMock()
        config.run_id = "test-run"
        config.transcript_file = None
        return config

    @patch("karotte.cli.run.run_containerized")
    @patch("karotte.cli.run.clean_up_old_containers")
    @patch("karotte.cli.run.build_container")
    def test_forwards_cache_from(
        self, mock_build: MagicMock, _mock_cleanup: MagicMock, _mock_run: MagicMock
    ):
        _run_without_ui(
            [self._make_config()],
            "podman",
            False,
            ".",
            keep_containers=False,
            cache_from=["type=registry,ref=repo:cache"],
        )

        mock_build.assert_called_once_with(
            "podman",
            ".",
            cache_from=["type=registry,ref=repo:cache"],
            cache_to=None,
            build_secrets=(),
        )

    @patch("karotte.cli.run.run_containerized")
    @patch("karotte.cli.run.clean_up_old_containers")
    @patch("karotte.cli.run.build_container")
    def test_forwards_cache_to(
        self, mock_build: MagicMock, _mock_cleanup: MagicMock, _mock_run: MagicMock
    ):
        _run_without_ui(
            [self._make_config()],
            "podman",
            False,
            ".",
            keep_containers=False,
            cache_to=["type=local,dest=/tmp/cache"],
            build_secrets=(),
        )

        mock_build.assert_called_once_with(
            "podman",
            ".",
            cache_from=None,
            cache_to=["type=local,dest=/tmp/cache"],
            build_secrets=(),
        )

    @patch("karotte.cli.run.run_containerized")
    @patch("karotte.cli.run.clean_up_old_containers")
    @patch("karotte.cli.run.build_container")
    def test_forwards_both_cache_args(
        self, mock_build: MagicMock, _mock_cleanup: MagicMock, _mock_run: MagicMock
    ):
        _run_without_ui(
            [self._make_config()],
            "podman",
            False,
            ".",
            keep_containers=False,
            cache_from=["type=registry,ref=a"],
            cache_to=["type=registry,ref=b"],
            build_secrets=(),
        )

        mock_build.assert_called_once_with(
            "podman",
            ".",
            cache_from=["type=registry,ref=a"],
            cache_to=["type=registry,ref=b"],
            build_secrets=(),
        )

    @patch("karotte.cli.run.run_containerized")
    @patch("karotte.cli.run.clean_up_old_containers")
    @patch("karotte.cli.run.build_container")
    def test_no_cache_args_by_default(
        self, mock_build: MagicMock, _mock_cleanup: MagicMock, _mock_run: MagicMock
    ):
        _run_without_ui(
            [self._make_config()], "podman", False, ".", keep_containers=False
        )

        mock_build.assert_called_once_with(
            "podman", ".", cache_from=None, cache_to=None, build_secrets=()
        )

    @patch("karotte.cli.run.run_containerized")
    @patch("karotte.cli.run.clean_up_old_containers")
    @patch("karotte.cli.run.build_container")
    def test_forwards_build_secrets(
        self, mock_build: MagicMock, _mock_cleanup: MagicMock, _mock_run: MagicMock
    ):
        _run_without_ui(
            [self._make_config()],
            "podman",
            False,
            ".",
            keep_containers=False,
            build_secrets=["uv_env=/tmp/uv_env"],
        )

        mock_build.assert_called_once_with(
            "podman",
            ".",
            cache_from=None,
            cache_to=None,
            build_secrets=["uv_env=/tmp/uv_env"],
        )

    @patch("karotte.cli.run.run_containerized")
    @patch("karotte.cli.run.clean_up_old_containers")
    @patch("karotte.cli.run.build_container")
    def test_skips_build_in_dev_mode(
        self, mock_build: MagicMock, _mock_cleanup: MagicMock, _mock_run: MagicMock
    ):
        _run_without_ui(
            [self._make_config()],
            "podman",
            True,
            ".",
            keep_containers=False,
            cache_from=["repo:cache"],
        )

        mock_build.assert_not_called()


def _register_proxies(monkeypatch: pytest.MonkeyPatch, **urls: object) -> None:
    def fake_entry_points(*, group: str) -> list[SimpleNamespace]:
        assert group == PROXY_ENTRY_POINT_GROUP

        def loader(url: object) -> object:
            if isinstance(url, Exception):
                raise url
            return url

        return [
            SimpleNamespace(name=name, load=functools.partial(loader, url))
            for name, url in urls.items()
        ]

    monkeypatch.setattr(
        sys.modules["karotte.cli.run"], "entry_points", fake_entry_points
    )


class TestDefaultProxyUrl:
    def test_none_without_a_plugin(self, monkeypatch: pytest.MonkeyPatch):
        _register_proxies(monkeypatch)
        assert default_proxy_url() is None

    def test_an_installed_plugin_provides_it(self, monkeypatch: pytest.MonkeyPatch):
        _register_proxies(monkeypatch, internal="https://proxy.example")
        assert default_proxy_url() == "https://proxy.example"

    def test_the_first_plugin_by_name_wins(self, monkeypatch: pytest.MonkeyPatch):
        _register_proxies(monkeypatch, b="https://b.example", a="https://a.example")
        assert default_proxy_url() == "https://a.example"

    def test_a_broken_plugin_is_skipped(self, monkeypatch: pytest.MonkeyPatch):
        _register_proxies(monkeypatch, a=ImportError("gone"), b="https://proxy.example")
        assert default_proxy_url() == "https://proxy.example"


@pytest.mark.usefixtures("in_karotte_image")
class TestProxyApiKeyFallback:
    """When a proxy is in use and ANTHROPIC_API_KEY is missing, a dummy key should be injected."""

    def _run_with_proxy(
        self, *, proxy: str | None = "https://proxy.example", no_proxy: bool = False
    ) -> None:
        """Call run() with enough mocked out to test only the proxy/api-key logic."""
        run_config = EvaluationRunConfig(
            run_id="test",
            task_id="example-task",
            model="vertex_ai/gemini",
            model_api_key=None,
            mcp_server_config=HttpMcpServerConfig(),
        )
        mock_task = MagicMock()
        with (
            patch("karotte.cli.run.parse_config", return_value=run_config),
            patch("karotte.cli.run.load_task", return_value=mock_task),
            patch(
                "karotte.mcp_servers.http_mcp_server.run_server",
                return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()),
            ),
            patch("karotte.cli.run.anyio", run=MagicMock(return_value=None)),
        ):
            run(config="{}", containerized=False, proxy=proxy, no_proxy=no_proxy)

    def test_proxy_sets_dummy_env_var_when_not_set(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        self._run_with_proxy()

        assert os.environ["ANTHROPIC_API_KEY"] == "model_api_key"

    def test_proxy_preserves_existing_env_var(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-real-key")

        self._run_with_proxy()

        assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-real-key"

    def test_no_proxy_does_not_set_env_var(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        _register_proxies(monkeypatch, internal="https://proxy.example")

        self._run_with_proxy(no_proxy=True)

        assert "ANTHROPIC_API_KEY" not in os.environ

    def test_no_proxy_by_default(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        _register_proxies(monkeypatch)

        self._run_with_proxy(proxy=None)

        assert "ANTHROPIC_API_KEY" not in os.environ

    def test_a_plugin_proxy_is_the_default(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        _register_proxies(monkeypatch, internal="https://proxy.example")

        self._run_with_proxy(proxy=None)

        assert os.environ["ANTHROPIC_API_KEY"] == "model_api_key"


@pytest.mark.usefixtures("in_karotte_image")
class TestProxyPlaceholderForReferencedKey:
    """A keyless proxy works whichever variable the config's key refers to."""

    def _run(self, config: str, proxy: str | None) -> MagicMock:
        with (
            patch("karotte.cli.run.load_task", return_value=MagicMock()),
            patch(
                "karotte.mcp_servers.http_mcp_server.run_server",
                return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()),
            ),
            patch("karotte.cli.run.anyio") as mock_anyio,
        ):
            run(config=config, containerized=False, proxy=proxy, no_proxy=not proxy)
        return mock_anyio

    def _config(self, key: str) -> str:
        return EvaluationRunConfig(
            run_id="test",
            task_id="example-task",
            model="openai/gpt-5.5",
            model_api_key=key,
        ).model_dump_json()

    @pytest.fixture(autouse=True)
    def _restore_openai_key(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("OPENAI_API_KEY", "")

    def test_fills_in_the_referenced_variable(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        mock_anyio = self._run(self._config("$OPENAI_API_KEY"), "https://proxy.example")

        run_config = mock_anyio.run.call_args.args[1]
        assert run_config.model_api_key == "model_api_key"

    def test_keeps_a_set_variable(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-real")

        mock_anyio = self._run(self._config("$OPENAI_API_KEY"), "https://proxy.example")

        assert mock_anyio.run.call_args.args[1].model_api_key == "sk-real"

    def test_reads_the_reference_from_a_config_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        config_file = tmp_path / "config.json"
        config_file.write_text(self._config("$OPENAI_API_KEY"))

        mock_anyio = self._run(str(config_file), "https://proxy.example")

        assert mock_anyio.run.call_args.args[1].model_api_key == "model_api_key"

    def test_without_a_proxy_an_unset_variable_still_aborts(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with pytest.raises(typer.Abort):
            self._run(self._config("$OPENAI_API_KEY"), None)


class TestUncontainerizedProxyEnv:
    """With --no-containerized, the proxy is applied to this process's environment."""

    PROXY_VARS: tuple[str, ...] = ("KAROTTE_PROXY_URL", "ANTHROPIC_BASE_URL")

    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("KAROTTE_CONTAINERIZED", "1")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")

    def _run(
        self, *, proxy: str | None = None, no_proxy: bool = False
    ) -> dict[str, str | None]:
        run_config = EvaluationRunConfig(
            run_id="test", task_id="example-task", model="m", model_api_key="k"
        )
        seen: dict[str, str | None] = {}

        def record(*_args: object) -> None:
            seen.update({var: os.environ.get(var) for var in self.PROXY_VARS})

        with (
            patch("karotte.cli.run.parse_config", return_value=run_config),
            patch("karotte.cli.run.load_task", return_value=MagicMock()),
            patch(
                "karotte.mcp_servers.http_mcp_server.run_server",
                return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()),
            ),
            patch("karotte.cli.run.anyio.run", side_effect=record),
        ):
            run(config="{}", containerized=False, proxy=proxy, no_proxy=no_proxy)
        return seen

    def test_an_explicit_proxy_is_applied(self, monkeypatch: pytest.MonkeyPatch):
        _register_proxies(monkeypatch)

        seen = self._run(proxy="https://proxy.example")

        assert seen == {
            "KAROTTE_PROXY_URL": "https://proxy.example",
            "ANTHROPIC_BASE_URL": "https://proxy.example",
        }

    def test_nothing_is_set_without_a_proxy(self, monkeypatch: pytest.MonkeyPatch):
        _register_proxies(monkeypatch)

        assert self._run() == {"KAROTTE_PROXY_URL": None, "ANTHROPIC_BASE_URL": None}

    def test_no_proxy_sets_nothing(self, monkeypatch: pytest.MonkeyPatch):
        _register_proxies(monkeypatch, internal="https://plugin.example")

        seen = self._run(no_proxy=True)

        assert seen == {"KAROTTE_PROXY_URL": None, "ANTHROPIC_BASE_URL": None}

    @pytest.mark.parametrize("preset", PROXY_VARS)
    def test_an_environment_that_already_routes_is_left_alone(
        self, monkeypatch: pytest.MonkeyPatch, preset: str
    ):
        _register_proxies(monkeypatch, internal="https://plugin.example")
        monkeypatch.setenv(preset, "https://backend.example")

        seen = self._run(proxy="https://proxy.example")

        assert seen == {
            var: "https://backend.example" if var == preset else None
            for var in self.PROXY_VARS
        }

    def test_a_plugin_proxy_is_not_applied(self, monkeypatch: pytest.MonkeyPatch):
        _register_proxies(monkeypatch, internal="https://plugin.example")

        seen = self._run()

        assert seen == {"KAROTTE_PROXY_URL": None, "ANTHROPIC_BASE_URL": None}


def test_tui_receives_the_cache_options(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("KAROTTE_CONTAINERIZED", "")
    monkeypatch.delenv("KAROTTE_CONTAINERIZED")
    run_config = EvaluationRunConfig(
        run_id="r", task_id="t", model="m", model_api_key="k"
    )
    with (
        patch("karotte.cli.run.parse_config", return_value=run_config),
        patch("karotte.cli.run.load_task", return_value=MagicMock()),
        patch("karotte.terminal.app.KarotteApp") as mock_app_cls,
    ):
        mock_app_cls.return_value.run_failed = False
        run(
            config="{}",
            cache_from=["type=registry,ref=repo:cache"],
            cache_to=["type=registry,ref=repo:cache,mode=max"],
        )

    app = mock_app_cls.return_value
    assert app.cache_from == ["type=registry,ref=repo:cache"]
    assert app.cache_to == ["type=registry,ref=repo:cache,mode=max"]
    app.run.assert_called_once()


@pytest.mark.usefixtures("in_karotte_image")
def test_run_uses_the_preprocessed_config(monkeypatch: pytest.MonkeyPatch):
    from karotte.judges.rubric_judge import RubricJudge

    monkeypatch.setattr(RubricJudge, "default_api_key", None)
    parsed = EvaluationRunConfig(run_id="r", task_id="t", model="m", model_api_key="k")
    preprocessed = parsed.model_copy(
        update={"rubric_judge_api_key": "from-preprocessor"}
    )
    with (
        patch("karotte.cli.run.parse_config", return_value=parsed),
        patch(
            "karotte.cli.run.apply_run_config_preprocessors",
            return_value=preprocessed,
        ) as preprocess,
        patch("karotte.cli.run.load_task", return_value=MagicMock()),
        patch(
            "karotte.mcp_servers.http_mcp_server.run_server",
            return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()),
        ),
        patch("karotte.cli.run.anyio", run=MagicMock(return_value=None)),
        patch("karotte.cli.run.build_configs", return_value=[preprocessed]),
        patch("karotte.cli.run._run_without_ui"),
    ):
        run(config="{}", containerized=False, no_ui=True)

    preprocess.assert_called_once_with(parsed)
    assert RubricJudge.default_api_key == "from-preprocessor"


class TestSanitizePathsTiming:
    """sanitize_paths_and_reexec() must only run in the process that hosts the
    untrusted agent, not in the host process that merely orchestrates a
    containerized run (where stripping non-root PATH entries can hide the
    container runtime)."""

    def _run(self, *, containerized: bool) -> MagicMock:
        return self._run_recording(
            "sanitize_paths_and_reexec", containerized=containerized
        )

    def _run_recording(self, target: str, *, containerized: bool) -> MagicMock:
        run_config = EvaluationRunConfig(
            run_id="test",
            task_id="example-task",
            model="vertex_ai/gemini",
            model_api_key=None,
            mcp_server_config=HttpMcpServerConfig(),
        )
        with (
            patch("karotte.cli.run.sanitize_paths_and_reexec") as mock_sanitize,
            patch("karotte.cli.run.harden_filesystem") as mock_harden,
            patch("karotte.cli.run.parse_config", return_value=run_config),
            patch(
                "karotte.cli.run.apply_run_config_preprocessors",
                return_value=run_config,
            ),
            patch("karotte.cli.run.load_task", return_value=MagicMock()),
            patch(
                "karotte.mcp_servers.http_mcp_server.run_server",
                return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()),
            ),
            patch("karotte.cli.run.anyio", run=MagicMock(return_value=None)),
            patch("karotte.cli.run.build_configs", return_value=[run_config]),
            patch("karotte.cli.run._run_without_ui"),
        ):
            run(config="{}", containerized=containerized, no_ui=True, runtime="podman")
        return {
            "sanitize_paths_and_reexec": mock_sanitize,
            "harden_filesystem": mock_harden,
        }[target]

    def test_not_called_on_host_orchestration_path(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.delenv("KAROTTE_CONTAINERIZED", raising=False)
        mock_sanitize = self._run(containerized=True)
        mock_sanitize.assert_not_called()

    def test_called_on_uncontainerized_path(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("KAROTTE_CONTAINERIZED", "1")
        mock_sanitize = self._run(containerized=False)
        mock_sanitize.assert_called_once()

    def test_called_inside_container(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("KAROTTE_CONTAINERIZED", "1")
        mock_sanitize = self._run(containerized=True)
        mock_sanitize.assert_called_once()

    def test_the_filesystem_is_hardened_on_the_same_path(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """The mounts a runtime leaves world-writable are the container's, so
        they are closed where the untrusted agent is hosted and nowhere else."""
        monkeypatch.setenv("KAROTTE_CONTAINERIZED", "1")
        assert self._run_recording("harden_filesystem", containerized=True).called

    def test_the_filesystem_is_not_hardened_on_the_host(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.delenv("KAROTTE_CONTAINERIZED", raising=False)
        mock_harden = self._run_recording("harden_filesystem", containerized=True)
        mock_harden.assert_not_called()


class TestSanitizeLdPreloadSpaceSeparation:
    """glibc splits LD_PRELOAD on spaces as well as colons; the sanitizer must too."""

    def _sanitize(self, monkeypatch: pytest.MonkeyPatch, preload: str) -> str:
        monkeypatch.setenv("KAROTTE_CONTAINERIZED", "1")
        monkeypatch.setenv("PATH", "/usr/bin")
        monkeypatch.setenv("LD_LIBRARY_PATH", "")
        monkeypatch.setenv("LD_AUDIT", "")
        monkeypatch.setenv("LD_PRELOAD", preload)
        sanitize_paths()
        return os.environ["LD_PRELOAD"]

    def test_strips_space_separated_unsafe_entry(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        assert self._sanitize(monkeypatch, f"/usr/bin {tmp_path}") == "/usr/bin"

    def test_keeps_space_separated_safe_entries(self, monkeypatch: pytest.MonkeyPatch):
        result = self._sanitize(monkeypatch, "/usr/bin /bin")
        assert result.split(os.pathsep) == ["/usr/bin", "/bin"]


class TestSanitizePathsCiSkip:
    """The ``CI`` skip protects a local test runner's PATH, but must never apply
    inside a container: an eval run that happens to set CI must still have the
    student-writable venv stripped from the (root) server's PATH."""

    def _path_with_student_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> str:
        """Set PATH to a non-root (student-writable) dir followed by a root dir,
        and return the student dir. monkeypatch restores PATH on teardown, so the
        in-place mutation sanitize_paths() makes does not leak to other tests."""
        student_dir = tmp_path / "venv" / "bin"
        student_dir.mkdir(parents=True)
        monkeypatch.setenv("PATH", f"{student_dir}{os.pathsep}/usr/bin")
        return str(student_dir)

    def test_skips_in_ci_outside_a_container(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        monkeypatch.setenv("CI", "1")
        monkeypatch.delenv("KAROTTE_CONTAINERIZED", raising=False)
        student_dir = self._path_with_student_dir(monkeypatch, tmp_path)

        sanitize_paths()

        assert student_dir in os.environ["PATH"], "skip should leave PATH untouched"

    def test_runs_in_ci_inside_a_container(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        monkeypatch.setenv("CI", "1")
        monkeypatch.setenv("KAROTTE_CONTAINERIZED", "1")
        student_dir = self._path_with_student_dir(monkeypatch, tmp_path)

        sanitize_paths()

        assert student_dir not in os.environ["PATH"], (
            "the student-writable entry must be stripped even under CI in a container"
        )
        assert "/usr/bin" in os.environ["PATH"]

    def test_logs_removals_at_debug(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        from loguru import logger

        monkeypatch.setenv("KAROTTE_CONTAINERIZED", "1")
        self._path_with_student_dir(monkeypatch, tmp_path)
        levels: list[str] = []
        sink = logger.add(
            lambda m: levels.append(m.record["level"].name),
            filter=lambda r: "Sanitized" in r["message"],
        )
        try:
            sanitize_paths()
        finally:
            logger.remove(sink)

        assert levels and set(levels) == {"DEBUG"}


@patch("karotte.cli.run.run_containerized")
@patch("karotte.cli.run.clean_up_old_containers")
@patch("karotte.cli.run.build_container")
def test_keep_containers_message_uses_the_actual_runtime(
    _mock_build: MagicMock,
    _mock_cleanup: MagicMock,
    _mock_run: MagicMock,
    capsys: pytest.CaptureFixture[str],
):
    _run_without_ui(
        [MagicMock(run_id="r", transcript_file=None)],
        "docker:gvisor",
        False,
        ".",
        keep_containers=True,
    )

    assert "docker cp karotte_run_r:/workdir/ ./out/" in capsys.readouterr().out


@pytest.mark.usefixtures("in_karotte_image")
class TestChownOutputsAfterRun:
    def _run_uncontainerized(self, anyio_mock: MagicMock, chown: MagicMock) -> None:
        run_config = EvaluationRunConfig(
            run_id="r", task_id="t", model="m", model_api_key="k"
        )
        with (
            patch("karotte.cli.run.chown_outputs", chown),
            patch("karotte.cli.run.parse_config", return_value=run_config),
            patch(
                "karotte.cli.run.apply_run_config_preprocessors",
                return_value=run_config,
            ),
            patch("karotte.cli.run.load_task", return_value=MagicMock()),
            patch(
                "karotte.mcp_servers.http_mcp_server.run_server",
                return_value=MagicMock(
                    __enter__=MagicMock(), __exit__=MagicMock(return_value=False)
                ),
            ),
            patch("karotte.cli.run.anyio", anyio_mock),
        ):
            run(config="{}", containerized=False)

    def test_outputs_are_chowned_after_the_run(self):
        chown = MagicMock()

        self._run_uncontainerized(MagicMock(run=MagicMock(return_value=None)), chown)

        chown.assert_called_once()

    def test_outputs_are_chowned_when_the_run_raises(self):
        anyio_mock = MagicMock()
        anyio_mock.run.side_effect = RuntimeError("boom")

        chown = MagicMock()

        with pytest.raises(RuntimeError):
            self._run_uncontainerized(anyio_mock, chown)

        chown.assert_called_once()


def test_the_runtime_defaults_to_the_os_vm():
    """No default in the signature: it is resolved from the platform and the
    task's hardware (see `default_runtime`)."""
    import inspect

    assert inspect.signature(run).parameters["runtime"].default is None


class TestMissingRuntime:
    @pytest.mark.parametrize(
        ("runtime", "message"),
        [
            ("docker", "docker not found. Install it or pass --runtime podman."),
            ("podman", "podman not found. Install it or pass --runtime docker."),
            (
                "docker:gvisor",
                "docker not found. Install it or pass --runtime podman.",
            ),
        ],
    )
    def test_exits_with_one_line_before_doing_anything(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        runtime: str,
        message: str,
    ):
        monkeypatch.delenv("KAROTTE_CONTAINERIZED", raising=False)
        with (
            patch("karotte.build.which", return_value=None),
            patch("karotte.cli.run.parse_config") as parse,
            pytest.raises(typer.Exit) as exc_info,
        ):
            run(config="{}", runtime=runtime)  # pyright: ignore[reportArgumentType]

        assert exc_info.value.exit_code == 1
        assert capsys.readouterr().err.strip() == message
        parse.assert_not_called()

    def test_not_checked_without_containerization(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("KAROTTE_CONTAINERIZED", "1")
        with (
            patch("karotte.build.which", return_value=None) as which,
            patch("karotte.cli.run.parse_config", side_effect=RuntimeError("parsed")),
            pytest.raises(RuntimeError, match="parsed"),
        ):
            run(config="{}", containerized=False)

        which.assert_not_called()

    def test_not_checked_inside_the_container(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("KAROTTE_CONTAINERIZED", "1")
        with (
            patch("karotte.build.which", return_value=None) as which,
            patch("karotte.cli.run.parse_config", side_effect=RuntimeError("parsed")),
            pytest.raises(RuntimeError, match="parsed"),
        ):
            run(config="{}")

        which.assert_not_called()


class TestMissingBuildx:
    def test_exits_with_one_line_before_doing_anything(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ):
        monkeypatch.delenv("KAROTTE_CONTAINERIZED", raising=False)
        with (
            patch("karotte.build._buildx_available", return_value=False),
            patch("karotte.cli.run.parse_config") as parse,
            pytest.raises(typer.Exit) as exc_info,
        ):
            run(config="{}", runtime="docker")

        assert exc_info.value.exit_code == 1
        assert "docker buildx not found" in capsys.readouterr().err
        parse.assert_not_called()

    def test_not_needed_in_dev_mode(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("KAROTTE_CONTAINERIZED", raising=False)
        with (
            patch("karotte.build._buildx_available", return_value=False) as buildx,
            patch("karotte.cli.run.parse_config", side_effect=RuntimeError("parsed")),
            pytest.raises(RuntimeError, match="parsed"),
        ):
            run(config="{}", runtime="docker", dev=True)

        buildx.assert_not_called()


class TestExitCode:
    """A host-side run exits non-zero when a run ends in an error event; the backend's in-pod run and failed tasks exit 0."""

    def _run_uncontainerized(
        self, anyio_result: object, *, exit_on_error: bool = True
    ) -> None:
        run_config = EvaluationRunConfig(
            run_id="r", task_id="t", model="m", model_api_key="k"
        )
        anyio_mock = MagicMock()
        anyio_mock.run.return_value = anyio_result
        env = {"KAROTTE_CONTAINERIZED": "1"}
        if exit_on_error:
            env[EXIT_ON_RUN_ERROR_ENV_VAR] = "1"
        with (
            patch.dict(os.environ, env),
            patch("karotte.cli.run.parse_config", return_value=run_config),
            patch(
                "karotte.cli.run.apply_run_config_preprocessors",
                return_value=run_config,
            ),
            patch("karotte.cli.run.load_task", return_value=MagicMock()),
            patch(
                "karotte.mcp_servers.http_mcp_server.run_server",
                return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()),
            ),
            patch("karotte.cli.run.anyio", anyio_mock),
        ):
            run(config="{}", containerized=False)

    def test_exits_1_when_the_run_ended_in_an_error(self):
        error = ErrorEvent(exception_type="TurnLimitReachedError", message="x")

        with pytest.raises(typer.Exit) as exc_info:
            self._run_uncontainerized(error)

        assert exc_info.value.exit_code == 1

    def test_exits_130_when_the_run_was_interrupted(self):
        error = ErrorEvent(exception_type="KeyboardInterrupt", message="")

        with pytest.raises(typer.Exit) as exc_info:
            self._run_uncontainerized(error)

        assert exc_info.value.exit_code == 130

    def test_exits_0_when_the_run_did_not_error(self):
        self._run_uncontainerized(None)

    def test_backend_invocation_exits_0_on_an_error(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.delenv(EXIT_ON_RUN_ERROR_ENV_VAR, raising=False)
        error = ErrorEvent(exception_type="TurnLimitReachedError", message="x")

        self._run_uncontainerized(error, exit_on_error=False)

    @patch("karotte.cli.run.clean_up_old_containers")
    @patch("karotte.cli.run.build_container")
    def test_containerized_exits_non_zero_when_any_run_fails(
        self,
        _mock_build: MagicMock,
        _mock_cleanup: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ):
        configs: list[EvaluationRunConfig] = [
            MagicMock(run_id="r-0", transcript_file=None),
            MagicMock(run_id="r-1", transcript_file=None),
        ]
        ran: list[str] = []

        def fake_run_containerized(config: MagicMock, **_: object) -> None:
            ran.append(config.run_id)
            if config.run_id == "r-0":
                raise subprocess.CalledProcessError(1, ["podman", "run"])

        with (
            patch("karotte.cli.run.run_containerized", fake_run_containerized),
            pytest.raises(typer.Exit) as exc_info,
        ):
            _run_without_ui(configs, "podman", False, ".", keep_containers=False)

        assert exc_info.value.exit_code == 1
        assert sorted(ran) == ["r-0", "r-1"]
        assert "karotte_run_r-0" in capsys.readouterr().err

    @patch("karotte.cli.run.run_containerized")
    @patch("karotte.cli.run.clean_up_old_containers")
    @patch("karotte.cli.run.build_container")
    def test_containerized_exits_0_when_all_runs_succeed(
        self, _mock_build: MagicMock, _mock_cleanup: MagicMock, _mock_run: MagicMock
    ):
        _run_without_ui(
            [
                MagicMock(run_id="r-0", transcript_file=None),
                MagicMock(run_id="r-1", transcript_file=None),
            ],
            "podman",
            False,
            ".",
            keep_containers=False,
        )

    def test_tui_exits_1_when_a_run_failed(self):
        run_config = EvaluationRunConfig(
            run_id="r", task_id="t", model="m", model_api_key="k"
        )
        app = MagicMock(run_failed=True)
        with (
            patch("karotte.cli.run.parse_config", return_value=run_config),
            patch("karotte.cli.run.load_task", return_value=MagicMock()),
            patch(
                "karotte.cli.run.apply_run_config_preprocessors",
                return_value=run_config,
            ),
            patch("karotte.terminal.app.KarotteApp", return_value=app),
            pytest.raises(typer.Exit) as exc_info,
        ):
            run(config="{}")

        assert exc_info.value.exit_code == 1


class TestHostOutputPaths:
    """Containerized runs print /out paths from inside the container; the host prints its own."""

    def _config(self, run_id: str = "r") -> EvaluationRunConfig:
        return EvaluationRunConfig(
            run_id=run_id,
            task_id="t",
            model="m",
            model_api_key="k",
            transcript_file="out/transcript.json",
        )

    @staticmethod
    def _write_outputs(config: EvaluationRunConfig, **_: object) -> None:
        out = Path("out")
        (out / "transcript.json").write_text("{}")
        (out / f"{config.run_id}_artifacts").mkdir()

    @patch("karotte.cli.run.clean_up_old_containers")
    @patch("karotte.cli.run.build_container")
    def test_prints_host_paths_after_a_run(
        self,
        _mock_build: MagicMock,
        _mock_cleanup: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "out").mkdir()
        with patch("karotte.cli.run.run_containerized", self._write_outputs):
            _run_without_ui([self._config()], "docker", False, ".", False)

        out = capsys.readouterr().out
        assert f"Transcript: {tmp_path / 'out' / 'transcript.json'}" in out
        assert f"Artifacts:  {tmp_path / 'out' / 'r_artifacts'}" in out

    @patch("karotte.cli.run.clean_up_old_containers")
    @patch("karotte.cli.run.build_container")
    def test_prints_host_paths_when_the_run_fails(
        self,
        _mock_build: MagicMock,
        _mock_cleanup: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "out").mkdir()

        def fail(config: EvaluationRunConfig, **_: object) -> None:
            self._write_outputs(config)
            raise subprocess.CalledProcessError(1, ["docker", "run"])

        with (
            patch("karotte.cli.run.run_containerized", fail),
            pytest.raises(typer.Exit),
        ):
            _run_without_ui([self._config()], "docker", False, ".", False)

        assert "Transcript: " in capsys.readouterr().out

    @patch("karotte.cli.run.run_containerized")
    @patch("karotte.cli.run.clean_up_old_containers")
    @patch("karotte.cli.run.build_container")
    def test_prints_nothing_for_missing_outputs(
        self,
        _mock_build: MagicMock,
        _mock_cleanup: MagicMock,
        _mock_run: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ):
        monkeypatch.chdir(tmp_path)
        _run_without_ui([self._config()], "docker", False, ".", False)

        out = capsys.readouterr().out
        assert "Transcript: " not in out
        assert "Artifacts: " not in out

    def test_prints_host_paths_after_the_tui_exits(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ):
        monkeypatch.delenv("KAROTTE_CONTAINERIZED", raising=False)
        monkeypatch.chdir(tmp_path)
        (tmp_path / "out").mkdir()
        run_config = self._config()
        app = MagicMock(run_failed=False)
        app.run.side_effect = lambda: self._write_outputs(run_config)
        with (
            patch("karotte.cli.run.parse_config", return_value=run_config),
            patch("karotte.cli.run.load_task", return_value=MagicMock()),
            patch(
                "karotte.cli.run.apply_run_config_preprocessors",
                return_value=run_config,
            ),
            patch("karotte.cli.run.require_environment"),
            patch("karotte.terminal.app.KarotteApp", return_value=app),
        ):
            run(config="{}")

        assert (
            f"Transcript: {tmp_path / 'out' / 'transcript.json'}"
            in capsys.readouterr().out
        )


class TestPluginRefusesTheLaunch:
    def _launch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("KAROTTE_CONTAINERIZED", raising=False)
        run_config = EvaluationRunConfig(
            run_id="r", task_id="t", model="m", model_api_key="k"
        )
        with (
            patch("karotte.cli.run.require_environment"),
            patch("karotte.cli.run.require_runtime"),
            patch("karotte.cli.run.require_buildx"),
            patch("karotte.cli.run.parse_config", return_value=run_config),
            patch("karotte.cli.run.load_task", return_value=MagicMock()),
            patch("karotte.cli.run._run_without_ui", side_effect=RuntimeError("ran")),
        ):
            run(config="{}", runtime="docker", no_ui=True)

    def test_the_plugin_message_is_shown_and_nothing_runs(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ):
        def refuse(task: object, runtime: str) -> list[str]:
            del task, runtime
            raise RuntimeError("runsc lacks -tpuproxy")

        register_hardware_plugins(monkeypatch, container_run_args={"a": refuse})

        with pytest.raises(typer.Abort):
            self._launch(monkeypatch)

        assert "runsc lacks -tpuproxy" in capsys.readouterr().err

    def test_without_a_refusal_the_run_goes_ahead(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        register_hardware_plugins(monkeypatch)

        with pytest.raises(RuntimeError, match="ran"):
            self._launch(monkeypatch)
