"""What a task's `required_hardware` means, as installed plugins define it; karotte itself knows no hardware names."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Any, Protocol

from loguru import logger

if TYPE_CHECKING:
    from karotte.runtime import Runtime

DEFAULT_HARDWARE_ENTRY_POINT_GROUP = "karotte.default_hardware"
"""A hardware name, for tasks that name none."""

HARDWARE_LIMITS_ENTRY_POINT_GROUP = "karotte.hardware_limits"
"""``(hardware: str) -> HardwareLimits | None``."""

CONTAINER_RUN_ARGS_ENTRY_POINT_GROUP = "karotte.container_run_args"
"""``(task: Task, runtime: Runtime) -> list[str]``; raises to refuse the launch."""


class _Task(Protocol):
    """The part of ``karotte.Task`` this module reads; importing it would be a cycle."""

    @property
    def required_hardware(self) -> str | None: ...


@dataclass(frozen=True)
class HardwareLimits:
    """What a sandbox on one hardware type holds; ``None`` means unknown."""

    memory_bytes: int | None = None
    disk_bytes: int | None = None


def _plugins(group: str) -> Iterator[Any]:
    for ep in sorted(entry_points(group=group), key=lambda ep: ep.name):
        try:
            yield ep.load()
        except Exception as e:  # noqa: BLE001 - a broken plugin must not break karotte
            logger.warning("Ignoring {} from {!r}: {}", group, ep.name, e)


def default_hardware() -> str | None:
    """The hardware of a task that names none, from the first plugin by entry point name."""
    return next(_plugins(DEFAULT_HARDWARE_ENTRY_POINT_GROUP), None)


def hardware_limits(hardware: str | None) -> HardwareLimits | None:
    """What a sandbox on ``hardware`` holds, from the first plugin by entry point name."""
    if hardware is None:
        return None
    limits = next(_plugins(HARDWARE_LIMITS_ENTRY_POINT_GROUP), None)
    return None if limits is None else limits(hardware)


def container_run_args(task: _Task, runtime: Runtime) -> list[str]:
    """Extra container engine ``run`` arguments for ``task`` from every plugin, in entry point name order."""
    return [
        arg
        for hook in _plugins(CONTAINER_RUN_ARGS_ENTRY_POINT_GROUP)
        for arg in hook(task, runtime)
    ]
