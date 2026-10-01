"""What a task's `required_hardware` means, as installed plugins define it; karotte itself knows no hardware names."""

from __future__ import annotations

import platform
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from importlib.metadata import entry_points
from pathlib import Path
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
    cpus: int | None = None
    """CPUs for a VM runtime to give the sandbox."""
    passthrough: bool = False
    """Needs host devices (a GPU, a TPU) that a VM runtime can't pass through,
    so it runs in a container."""


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


_GIB = 1024**3
DEFAULT_VM_CPUS = 2
DEFAULT_SANDBOX_MEMORY_BYTES = 4 * _GIB
"""The sandbox's RAM in a VM when no plugin knows the hardware: the student
gets it less the harness reserve."""
VM_MEMORY_HEADROOM_BYTES = 1 * _GIB
"""RAM a VM gets above the sandbox's, for the guest kernel and page cache: with
none, a student spread over many processes pushes the guest into a global OOM
that can pick the harness."""


@dataclass(frozen=True)
class VmSize:
    cpus: int
    sandbox_memory_bytes: int
    vm_memory_bytes: int
    disk_bytes: int | None
    """The plugin's disk budget, if it has one."""


def vm_size(hardware: str | None) -> VmSize:
    """How big a VM runtime makes the VM for ``hardware``. Raises ``ValueError``
    for hardware a VM can't run."""
    limits = hardware_limits(hardware) or HardwareLimits()
    if limits.passthrough:
        raise ValueError(
            f"{hardware} needs devices a VM can't pass through; use --runtime docker"
        )
    memory = limits.memory_bytes or DEFAULT_SANDBOX_MEMORY_BYTES
    return VmSize(
        cpus=limits.cpus or DEFAULT_VM_CPUS,
        sandbox_memory_bytes=memory,
        vm_memory_bytes=memory + VM_MEMORY_HEADROOM_BYTES,
        disk_bytes=limits.disk_bytes,
    )


def default_runtime(required_hardware: str | None = None) -> Runtime:
    """The runtime used when none is given: the OS's VM, so a local run gets
    the confinement a guest kernel of its own gives (Apple `container` on
    macOS, Firecracker on Linux), where the machine can run it. Hardware a
    hardware plugin marks ``passthrough`` (a GPU, a TPU) stays on docker, as
    do machines that can't run the VM: an Intel Mac or one before macOS 26,
    Linux without KVM. A VM runtime the machine can run but that isn't set up
    stops the run with what's missing instead."""
    limits = hardware_limits(required_hardware)
    if limits is not None and limits.passthrough:
        return "docker"
    if sys.platform == "darwin" and _mac_runs_apple_container():
        return "apple-container"
    if sys.platform.startswith("linux") and _linux_runs_firecracker():
        return "firecracker"
    return "docker"


_MIN_MACOS_MAJOR = 26
KVM_DEVICE = Path("/dev/kvm")


def _mac_runs_apple_container() -> bool:
    major = platform.mac_ver()[0].split(".")[0]
    return (
        platform.machine() == "arm64"
        and major.isdigit()
        and int(major) >= _MIN_MACOS_MAJOR
    )


def _linux_runs_firecracker() -> bool:
    return platform.machine() in ("x86_64", "aarch64") and KVM_DEVICE.exists()
