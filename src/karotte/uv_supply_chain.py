"""The 7-day age-delay for uv runs that resolve outside a project.

`uv tool run` and `uv run --script` ignore the project's `[tool.uv] exclude-newer`,
so `run_uv` passes the cutoff and the exemptions explicitly.
"""

import os
import subprocess
from collections.abc import Sequence
from importlib.metadata import entry_points
from typing import Any

from loguru import logger

# Must match `templates/pyproject.base.toml.jinja`.
AGE_DELAY = "7 days"

ENTRY_POINT_GROUP = "karotte.age_delay_exemptions"


def age_delay_exemptions() -> tuple[str, ...]:
    """Packages exempt from the age-delay: karotte plus those registered by installed packages."""
    packages: set[str] = {"karotte"}
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        try:
            packages.update(ep.load())
        except Exception as e:  # noqa: BLE001 - a broken plugin must not break karotte
            logger.warning("Ignoring age-delay exemptions from {!r}: {}", ep.name, e)
    return tuple(sorted(packages))


def exclude_newer_flags() -> list[str]:
    """uv resolver flags exempting `age_delay_exemptions` from the age-delay.

    Prefer `run_uv`, which passes these and the env var together.
    """
    return [f"--exclude-newer-package={pkg}=false" for pkg in age_delay_exemptions()]


def env_with_age_delay() -> dict[str, str]:
    """`os.environ` plus `UV_EXCLUDE_NEWER`, for a uv subprocess.

    A project's own `exclude-newer-package` still applies on top, so this is
    safe to inherit into a uv command that does read one.
    """
    return os.environ | {"UV_EXCLUDE_NEWER": AGE_DELAY}


def run_uv(
    subcommand: Sequence[str], *args: str, **kwargs: Any
) -> subprocess.CompletedProcess[str]:
    """`subprocess.run` for `uv <subcommand> <args>`, age-delay applied.

    The flags go directly after the subcommand, which is why it is passed
    separately: uv rejects a resolver flag that precedes `tool run`.
    """
    kwargs.setdefault("env", env_with_age_delay())
    return subprocess.run(
        ["uv", *subcommand, *exclude_newer_flags(), *args],
        **kwargs,
    )
