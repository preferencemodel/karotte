import subprocess
import sys

import pytest

_SCRIPT = """
from loguru import logger
from karotte.log import configure_logging
configure_logging()
logger.debug("debug-line")
logger.info("info-line")
"""


def _stderr(env: dict[str, str]) -> str:
    return subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    ).stderr


def test_default_level_is_info(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LOGURU_LEVEL", raising=False)
    import os

    stderr = _stderr(dict(os.environ))
    assert "info-line" in stderr
    assert "debug-line" not in stderr


def test_loguru_level_env_var_overrides_the_default():
    import os

    stderr = _stderr({**os.environ, "LOGURU_LEVEL": "DEBUG"})
    assert "debug-line" in stderr
