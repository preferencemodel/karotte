from types import SimpleNamespace

import pytest

from karotte.hardware import (
    DEFAULT_SANDBOX_MEMORY_BYTES,
    DEFAULT_VM_CPUS,
    VM_MEMORY_HEADROOM_BYTES,
    HardwareLimits,
    VmSize,
    container_run_args,
    default_hardware,
    hardware_limits,
    vm_size,
)
from tests.conftest import register_hardware_plugins


def test_no_default_hardware_without_a_plugin(monkeypatch: pytest.MonkeyPatch):
    register_hardware_plugins(monkeypatch)
    assert default_hardware() is None


def test_the_default_hardware_comes_from_the_first_plugin(
    monkeypatch: pytest.MonkeyPatch,
):
    register_hardware_plugins(monkeypatch, default={"b": "big", "a": "small"})
    assert default_hardware() == "small"


def test_a_broken_default_hardware_plugin_is_skipped(monkeypatch: pytest.MonkeyPatch):
    register_hardware_plugins(
        monkeypatch, default={"a": ImportError("gone"), "b": "small"}
    )
    assert default_hardware() == "small"


def test_no_limits_without_a_plugin(monkeypatch: pytest.MonkeyPatch):
    register_hardware_plugins(monkeypatch)
    assert hardware_limits("small") is None


def test_limits_come_from_the_plugin(monkeypatch: pytest.MonkeyPatch):
    def limits(hardware: str) -> HardwareLimits | None:
        return (
            HardwareLimits(memory_bytes=5, disk_bytes=7)
            if hardware == "small"
            else None
        )

    register_hardware_plugins(monkeypatch, limits={"a": limits})
    assert hardware_limits("small") == HardwareLimits(memory_bytes=5, disk_bytes=7)
    assert hardware_limits("huge") is None


def test_no_hardware_has_no_limits(monkeypatch: pytest.MonkeyPatch):
    def limits(hardware: str) -> HardwareLimits:
        raise AssertionError(f"asked about {hardware!r}")

    register_hardware_plugins(monkeypatch, limits={"a": limits})
    assert hardware_limits(None) is None


def test_no_container_run_args_without_a_plugin(monkeypatch: pytest.MonkeyPatch):
    register_hardware_plugins(monkeypatch)
    assert container_run_args(SimpleNamespace(required_hardware="gpu"), "docker") == []  # pyright: ignore[reportArgumentType]


def test_container_run_args_from_every_plugin_in_name_order(
    monkeypatch: pytest.MonkeyPatch,
):
    seen: list[tuple[object, str]] = []

    def gpu(task: object, runtime: str) -> list[str]:
        seen.append((task, runtime))
        return ["--device", "gpu"]

    def env(task: object, runtime: str) -> list[str]:
        del task, runtime
        return ["--env", "X=1"]

    register_hardware_plugins(
        monkeypatch, container_run_args={"b": env, "a": gpu, "c": ImportError("gone")}
    )
    task = SimpleNamespace(required_hardware="gpu")

    assert container_run_args(task, "docker:gvisor") == [  # pyright: ignore[reportArgumentType]
        "--device",
        "gpu",
        "--env",
        "X=1",
    ]
    assert seen == [(task, "docker:gvisor")]


def test_a_plugin_refuses_the_launch_by_raising(monkeypatch: pytest.MonkeyPatch):
    def refuse(task: object, runtime: str) -> list[str]:
        del task, runtime
        raise RuntimeError("runsc lacks -tpuproxy")

    register_hardware_plugins(monkeypatch, container_run_args={"a": refuse})
    with pytest.raises(RuntimeError, match="tpuproxy"):
        _ = container_run_args(SimpleNamespace(required_hardware="tpu"), "docker")  # pyright: ignore[reportArgumentType]


class TestVmSize:
    """How big a VM runtime makes its VM: the plugin's numbers, else defaults,
    with headroom above the sandbox's RAM for the guest kernel."""

    def test_defaults_without_a_plugin(self, monkeypatch: pytest.MonkeyPatch):
        register_hardware_plugins(monkeypatch)

        size = vm_size(None)

        assert size == VmSize(
            cpus=DEFAULT_VM_CPUS,
            sandbox_memory_bytes=DEFAULT_SANDBOX_MEMORY_BYTES,
            vm_memory_bytes=DEFAULT_SANDBOX_MEMORY_BYTES + VM_MEMORY_HEADROOM_BYTES,
            disk_bytes=None,
        )

    def test_the_plugins_numbers(self, monkeypatch: pytest.MonkeyPatch):
        gib = 1 << 30
        register_hardware_plugins(
            monkeypatch,
            limits={
                "a": lambda _hw: HardwareLimits(  # pyright: ignore[reportUnknownLambdaType]
                    memory_bytes=14 * gib, disk_bytes=80 * gib, cpus=4
                )
            },
        )

        size = vm_size("cpu-4")

        assert (
            size.cpus,
            size.sandbox_memory_bytes,
            size.vm_memory_bytes,
            size.disk_bytes,
        ) == (
            4,
            14 * gib,
            15 * gib,
            80 * gib,
        )

    def test_passthrough_hardware_is_refused(self, monkeypatch: pytest.MonkeyPatch):
        register_hardware_plugins(
            monkeypatch,
            limits={"a": lambda _hw: HardwareLimits(passthrough=True)},  # pyright: ignore[reportUnknownLambdaType]
        )

        with pytest.raises(ValueError, match="gpu-1 needs devices"):
            _ = vm_size("gpu-1")
