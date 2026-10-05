import sys
from collections.abc import Iterator
from unittest.mock import patch

import pytest
import typer

from karotte.cli.build import build


@pytest.fixture(autouse=True)
def _engine_on_path() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    with (
        patch("karotte.build.which", return_value="/usr/bin/engine"),
        patch("karotte.build._buildx_available", return_value=True),
    ):
        yield


@pytest.mark.parametrize(
    ("runtime", "message"),
    [
        ("docker", "docker not found. Install it or pass --runtime podman."),
        ("podman", "podman not found. Install it or pass --runtime docker."),
    ],
)
def test_missing_runtime_exits_with_one_line(
    capsys: pytest.CaptureFixture[str], runtime: str, message: str
):
    with (
        patch("karotte.build.which", return_value=None),
        patch("karotte.cli.build.build_container") as mock_build,
        pytest.raises(typer.Exit) as exc_info,
    ):
        build(runtime=runtime)  # pyright: ignore[reportArgumentType]

    assert exc_info.value.exit_code == 1
    assert capsys.readouterr().err.strip() == message
    mock_build.assert_not_called()


BUILDX_MISSING = (
    "docker buildx not found, and building the image needs it. "
    + "Install the docker-buildx package or Docker's docker-buildx-plugin."
)


@pytest.mark.parametrize("runtime", ["docker", "docker:gvisor"])
def test_missing_buildx_exits_with_one_line(
    capsys: pytest.CaptureFixture[str], runtime: str
):
    with (
        patch("karotte.build._buildx_available", return_value=False),
        patch("karotte.cli.build.build_container") as mock_build,
        pytest.raises(typer.Exit) as exc_info,
    ):
        build(runtime=runtime)  # pyright: ignore[reportArgumentType]

    assert exc_info.value.exit_code == 1
    assert capsys.readouterr().err.strip() == BUILDX_MISSING
    mock_build.assert_not_called()


def test_buildx_is_only_checked_for_docker():
    with (
        patch("karotte.build._buildx_available") as buildx,
        patch("karotte.cli.build.build_container"),
    ):
        build(runtime="podman")

    buildx.assert_not_called()


@pytest.fixture(autouse=True)
def _docker_by_default(monkeypatch: pytest.MonkeyPatch):  # pyright: ignore[reportUnusedFunction]
    """The real default depends on the OS running the tests."""
    # The package re-exports the `build` function under the module's name.
    monkeypatch.setattr(
        sys.modules["karotte.cli.build"],
        "default_runtime",
        lambda _hardware=None: "docker",  # pyright: ignore[reportUnknownLambdaType]
    )


class TestBuildCommand:
    def _tasks(
        self, monkeypatch: pytest.MonkeyPatch, hardware: dict[str, str | None]
    ) -> None:
        def fake_task(task_id: str, hw: str | None) -> type:
            class FakeTask:
                id: str = task_id
                required_hardware: str | None = hw

                def __init__(self, config: object) -> None:
                    del config

            return FakeTask

        classes = [fake_task(task_id, hw) for task_id, hw in hardware.items()]
        monkeypatch.setattr("karotte.load_tasks.load_all_task_classes", lambda: classes)
        monkeypatch.setattr(
            sys.modules["karotte.cli.build"],
            "default_runtime",
            lambda hw=None: "docker" if hw == "gpu-1" else "apple-container",  # pyright: ignore[reportUnknownLambdaType]
        )

    def test_the_default_runtime_is_the_one_run_picks_for_the_tasks(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """`run` picks by each task's hardware; `build` must fill that store."""
        self._tasks(monkeypatch, {"a": "gpu-1", "b": "gpu-1"})
        with patch("karotte.cli.build.build_container") as mock_build:
            build()
        assert mock_build.call_args.args[0] == "docker"

    def test_tasks_that_would_pick_different_runtimes_need_one_named(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        self._tasks(monkeypatch, {"a": "gpu-1", "b": "cpu-4"})
        with patch("karotte.cli.build.build_container") as mock_build:
            with pytest.raises(typer.Exit):
                build()
        mock_build.assert_not_called()

    def test_outside_an_environment_the_default_hardware_decides(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        def no_environment() -> list[object]:
            raise ModuleNotFoundError("environment")

        monkeypatch.setattr("karotte.load_tasks.load_all_task_classes", no_environment)
        monkeypatch.setattr(
            sys.modules["karotte.cli.build"], "default_hardware", lambda: "gpu-1"
        )
        monkeypatch.setattr(
            sys.modules["karotte.cli.build"],
            "default_runtime",
            lambda hw=None: "docker" if hw == "gpu-1" else "apple-container",  # pyright: ignore[reportUnknownLambdaType]
        )
        with patch("karotte.cli.build.build_container") as mock_build:
            build()
        assert mock_build.call_args.args[0] == "docker"

    def test_calls_build_container_with_defaults(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            sys.modules["karotte.cli.build"],
            "default_runtime",
            lambda _hardware=None: "podman",  # pyright: ignore[reportUnknownLambdaType]
        )
        with patch("karotte.cli.build.build_container") as mock_build:
            build()

        mock_build.assert_called_once_with(
            "podman",
            ".",
            "karotte",
            cache_from=None,
            cache_to=None,
            build_secrets=(),
        )

    def test_calls_build_container_with_custom_runtime(self):
        with patch("karotte.cli.build.build_container") as mock_build:
            build(runtime="podman")

        mock_build.assert_called_once_with(
            "podman",
            ".",
            "karotte",
            cache_from=None,
            cache_to=None,
            build_secrets=(),
        )

    def test_calls_build_container_with_custom_tag(self):
        with patch("karotte.cli.build.build_container") as mock_build:
            build(tag="my-custom-image")

        mock_build.assert_called_once_with(
            "docker",
            ".",
            "my-custom-image",
            cache_from=None,
            cache_to=None,
            build_secrets=(),
        )

    def test_calls_build_container_with_custom_build_context(self):
        with patch("karotte.cli.build.build_container") as mock_build:
            build(build_context="/path/to/context")

        mock_build.assert_called_once_with(
            "docker",
            "/path/to/context",
            "karotte",
            cache_from=None,
            cache_to=None,
            build_secrets=(),
        )

    def test_calls_build_container_with_all_custom_options(self):
        with patch("karotte.cli.build.build_container") as mock_build:
            build(runtime="podman", tag="my-image:v2", build_context="/custom/path")

        mock_build.assert_called_once_with(
            "podman",
            "/custom/path",
            "my-image:v2",
            cache_from=None,
            cache_to=None,
            build_secrets=(),
        )

    def test_calls_build_container_with_cache_from(self):
        with patch("karotte.cli.build.build_container") as mock_build:
            build(cache_from=["type=registry,ref=myrepo:cache"])

        mock_build.assert_called_once_with(
            "docker",
            ".",
            "karotte",
            cache_from=["type=registry,ref=myrepo:cache"],
            cache_to=None,
            build_secrets=(),
        )

    def test_calls_build_container_with_cache_to(self):
        with patch("karotte.cli.build.build_container") as mock_build:
            build(cache_to=["type=local,dest=/tmp/cache"])

        mock_build.assert_called_once_with(
            "docker",
            ".",
            "karotte",
            cache_from=None,
            cache_to=["type=local,dest=/tmp/cache"],
            build_secrets=(),
        )

    def test_calls_build_container_with_multiple_cache_flags(self):
        with patch("karotte.cli.build.build_container") as mock_build:
            build(
                cache_from=["type=registry,ref=a", "type=local,src=/tmp"],
                cache_to=["type=registry,ref=b"],
            )

        mock_build.assert_called_once_with(
            "docker",
            ".",
            "karotte",
            cache_from=["type=registry,ref=a", "type=local,src=/tmp"],
            cache_to=["type=registry,ref=b"],
            build_secrets=(),
        )

    def test_forwards_build_secrets(self):
        with patch("karotte.cli.build.build_container") as mock_build:
            build(build_secret=["uv_env=/tmp/uv_env"])

        mock_build.assert_called_once_with(
            "docker",
            ".",
            "karotte",
            cache_from=None,
            cache_to=None,
            build_secrets=["uv_env=/tmp/uv_env"],
        )
