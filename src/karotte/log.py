"""Loguru defaults for karotte's own processes."""

import os
import sys

from loguru import logger


def configure_logging() -> None:
    """Log to stderr at INFO, or at ``LOGURU_LEVEL`` when set."""
    logger.remove()
    _ = logger.add(sys.stderr, level=os.environ.get("LOGURU_LEVEL", "INFO"))
