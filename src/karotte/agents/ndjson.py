import json
from typing import Any

from loguru import logger


def parse_json_line(line: str, i: int, tool: str) -> dict[str, Any] | None:
    """Line ``i`` of ``tool``'s output as a JSON object; None, logged, if it isn't one."""
    line = line.strip()
    if not line:
        return None
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        logger.warning("Skipping unparseable {} line {}: {!r}", tool, i, line[:200])
        return None
    if not isinstance(record, dict):
        logger.warning("Skipping non-object {} line {}.", tool, i)
        return None
    return record
