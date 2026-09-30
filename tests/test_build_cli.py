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
        ("nerdctl", "nerdctl not found. Install it or pass --runtime docker."),
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


@pytest.mark.parametrize("runtime", ["podman", "nerdctl"])
def test_buildx_is_only_checked_for_docker(runtime: str):
    with (
        patch("karotte.build._buildx_available") as buildx,
        patch("karotte.cli.build.build_container"),
    ):
        build(runtime=runtime)  # pyright: ignore[reportArgumentType]

    buildx.assert_not_called()


class TestBuildCommand:
    def test_calls_build_container_with_defaults(self):
        with patch("karotte.cli.build.build_container") as mock_build:
            build()

        mock_build.assert_called_once_with(
            "docker",
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
