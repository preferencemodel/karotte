"""The age-delay applied where uv resolves outside a project.

`uv tool run` and `uv run --script` read no `[tool.uv]`, so these constants are
a hand-copy of the one the templates write. Drift is silent in the dangerous
direction: the flags keep parsing and the resolve keeps succeeding, only now
against a release published minutes ago.
"""

import functools
import subprocess
from types import SimpleNamespace
from typing import Any

import pytest

from karotte.templates import TEMPLATES_DIR
from karotte.uv_supply_chain import (
    AGE_DELAY,
    ENTRY_POINT_GROUP,
    age_delay_exemptions,
    env_with_age_delay,
    exclude_newer_flags,
    run_uv,
)

BASE_TEMPLATE = (TEMPLATES_DIR / "pyproject.base.toml.jinja").read_text()


def _register(monkeypatch: pytest.MonkeyPatch, **packages: tuple[str, ...]) -> None:
    def fake_entry_points(*, group: str) -> list[SimpleNamespace]:
        assert group == ENTRY_POINT_GROUP
        return [
            SimpleNamespace(name=name, load=functools.partial(tuple, pkgs))
            for name, pkgs in packages.items()
        ]

    monkeypatch.setattr("karotte.uv_supply_chain.entry_points", fake_entry_points)


@pytest.fixture
def exempt_karotte(monkeypatch: pytest.MonkeyPatch) -> None:
    _register(monkeypatch, internal=("karotte",))


def test_karotte_is_exempt_without_registered_packages(
    monkeypatch: pytest.MonkeyPatch,
):
    _register(monkeypatch)
    assert age_delay_exemptions() == ("karotte",)
    assert exclude_newer_flags() == ["--exclude-newer-package=karotte=false"]


def test_installed_packages_extend_the_exemptions(monkeypatch: pytest.MonkeyPatch):
    _register(monkeypatch, a=("karotte", "plugin-a"), b=("plugin-a", "plugin-b"))
    assert age_delay_exemptions() == ("karotte", "plugin-a", "plugin-b")
    assert exclude_newer_flags() == [
        "--exclude-newer-package=karotte=false",
        "--exclude-newer-package=plugin-a=false",
        "--exclude-newer-package=plugin-b=false",
    ]


def test_a_broken_package_does_not_stop_the_others(monkeypatch: pytest.MonkeyPatch):
    def broken() -> tuple[str, ...]:
        raise ImportError("missing dependency")

    monkeypatch.setattr(
        "karotte.uv_supply_chain.entry_points",
        lambda *, group: [  # pyright: ignore[reportUnknownLambdaType]
            SimpleNamespace(name="broken", load=broken),
            SimpleNamespace(name="ok", load=lambda: ("karotte",)),
        ],
    )
    assert age_delay_exemptions() == ("karotte",)


def test_the_age_delay_matches_the_one_generated_env_repos_get():
    assert f'exclude-newer = "{AGE_DELAY}"' in BASE_TEMPLATE


def test_generated_env_repos_exempt_karotte():
    assert "\nexclude-newer-package.karotte = false\n" in BASE_TEMPLATE


@pytest.mark.usefixtures("exempt_karotte")
def test_the_flags_parse_as_uv_resolver_options():
    """`--exclude-newer-package=<pkg>=false` is a doubled `=`; uv accepts it.

    `--help` still parses the flags (a bogus value exits 2), so this needs no
    network.
    """
    result = subprocess.run(
        ["uv", "tool", "run", *exclude_newer_flags(), "--help"],
        capture_output=True,
        text=True,
        env=env_with_age_delay(),
    )
    assert result.returncode == 0, result.stderr


def test_the_subprocess_env_carries_the_cutoff():
    assert env_with_age_delay()["UV_EXCLUDE_NEWER"] == AGE_DELAY


@pytest.mark.usefixtures("exempt_karotte")
def test_run_uv_passes_the_cutoff_and_the_exemptions_together(
    monkeypatch: pytest.MonkeyPatch,
):
    """One call site, one thing to remember — the pair is what has to travel."""
    captured: dict[str, Any] = {}

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr("karotte.uv_supply_chain.subprocess.run", fake_run)
    run_uv(("tool", "run"), "karotte", "--version")

    cmd = captured["cmd"]
    assert cmd[:3] == ["uv", "tool", "run"]
    assert cmd[-2:] == ["karotte", "--version"]
    assert set(exclude_newer_flags()) <= set(cmd)
    assert captured["kwargs"]["env"]["UV_EXCLUDE_NEWER"] == AGE_DELAY


@pytest.mark.usefixtures("exempt_karotte")
def test_uv_rejects_the_flags_before_the_subcommand():
    """Why `run_uv` takes the subcommand separately rather than one argv."""
    before = subprocess.run(
        ["uv", *exclude_newer_flags(), "tool", "run", "--help"],
        capture_output=True,
        text=True,
    )
    assert before.returncode == 2
