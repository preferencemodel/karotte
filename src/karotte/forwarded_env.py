"""What every launcher puts in the sandbox's environment, besides its own
settings: one place, so a new runtime can't leave any of it out."""

import os

from karotte.confinement import STUDENT_NETWORK_ENV_VAR
from karotte.providers import SERVICE_TIER_ENV

EXIT_ON_RUN_ERROR_ENV_VAR = "KAROTTE_EXIT_ON_RUN_ERROR"
"""Set by the host on its containers so the inner run exits non-zero when the run ends in an error."""

FORWARDED_ENV_VARS = ("LOGURU_LEVEL", SERVICE_TIER_ENV, STUDENT_NETWORK_ENV_VAR)
"""Host settings read inside the sandbox."""


def forwarded_env() -> dict[str, str]:
    """The :data:`FORWARDED_ENV_VARS` set on the host."""
    return {var: value for var in FORWARDED_ENV_VARS if (value := os.environ.get(var))}


def sandbox_env() -> dict[str, str]:
    """The environment every launcher gives the in-sandbox run: exit non-zero
    on a run error, and the forwarded host settings."""
    return {EXIT_ON_RUN_ERROR_ENV_VAR: "1", **forwarded_env()}
