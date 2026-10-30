"""Model forwarders that installed packages register as entry points.

A plugin serves a CLI agent's model calls for providers the agent can't reach on
its own. The entry point's attribute has ``agents`` and ``providers`` (the names
it serves) and ``forwarder(config, files)``, which returns a
:class:`~karotte.model_forwarder.ModelForwarder` for the run, or None. ``files``
is a directory for the forwarder's own files, which the student can't write.
"""

from collections.abc import Iterator
from importlib.metadata import entry_points
from pathlib import Path
from typing import Protocol

from loguru import logger

from karotte.model_forwarder import ModelForwarder
from karotte.schemas.evaluation_run_config import EvaluationRunConfig

ENTRY_POINT_GROUP = "karotte.model_forwarders"


class ForwarderPlugin(Protocol):
    agents: tuple[str, ...]
    providers: tuple[str, ...]

    def forwarder(
        self, config: EvaluationRunConfig, files: Path
    ) -> ModelForwarder | None: ...


def _plugins() -> Iterator[ForwarderPlugin]:
    for ep in sorted(entry_points(group=ENTRY_POINT_GROUP), key=lambda ep: ep.name):
        try:
            yield ep.load()
        except Exception as e:  # noqa: BLE001 - a broken plugin must not break karotte
            logger.warning("Ignoring {} from {!r}: {}", ENTRY_POINT_GROUP, ep.name, e)


def plugin_providers(agent: str) -> tuple[str, ...]:
    """The providers plugins serve for ``agent``."""
    return tuple(
        p for plugin in _plugins() if agent in plugin.agents for p in plugin.providers
    )


def plugin_forwarder(
    agent: str, provider: str, config: EvaluationRunConfig, files: Path
) -> ModelForwarder | None:
    """The first plugin forwarder for ``agent`` on ``provider``, by entry point name."""
    for plugin in _plugins():
        if agent in plugin.agents and provider in plugin.providers:
            if (forwarder := plugin.forwarder(config, files)) is not None:
                return forwarder
    return None
