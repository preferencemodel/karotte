import sys
from pathlib import Path

import pytest

from karotte import hardware
from karotte.hardware import HardwareLimits, default_runtime
from tests.conftest import register_hardware_plugins


def _limits(name: str) -> HardwareLimits | None:
    return {
        "cpu": HardwareLimits(),
        "gpu": HardwareLimits(passthrough=True),
    }.get(name)


@pytest.mark.parametrize(
    ("platform", "required", "expected"),
    [
        ("darwin", None, "apple-container"),
        ("darwin", "cpu", "apple-container"),
        ("linux", None, "firecracker"),
        ("linux", "cpu", "firecracker"),
        ("linux", "unknown", "firecracker"),
        ("linux", "gpu", "docker"),
        ("darwin", "gpu", "docker"),
        ("win32", None, "docker"),
    ],
)
def test_the_default_runtime_is_the_os_vm(
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
    required: str | None,
    expected: str,
) -> None:
    """Hardware a plugin marks passthrough stays on docker: neither VM passes a
    GPU or TPU through."""
    register_hardware_plugins(monkeypatch, limits={"a": _limits})
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(hardware, "_mac_runs_apple_container", lambda: True)
    monkeypatch.setattr(hardware, "_linux_runs_firecracker", lambda: True)
    assert default_runtime(required) == expected


@pytest.mark.parametrize("platform", ["darwin", "linux"])
def test_a_machine_that_cant_run_the_vm_gets_docker(
    monkeypatch: pytest.MonkeyPatch, platform: str
) -> None:
    """An Intel Mac, a Mac before macOS 26, Linux without KVM (a cloud VM
    without nested virtualization)."""
    register_hardware_plugins(monkeypatch)
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(hardware, "_mac_runs_apple_container", lambda: False)
    monkeypatch.setattr(hardware, "_linux_runs_firecracker", lambda: False)
    assert default_runtime(None) == "docker"


@pytest.mark.parametrize(
    ("machine", "mac_version", "expected"),
    [("arm64", "26.0", True), ("arm64", "15.6", False), ("x86_64", "26.0", False)],
)
def test_apple_container_needs_apple_silicon_and_macos_26(
    monkeypatch: pytest.MonkeyPatch, machine: str, mac_version: str, expected: bool
) -> None:
    monkeypatch.setattr("karotte.hardware.platform.machine", lambda: machine)
    monkeypatch.setattr(
        "karotte.hardware.platform.mac_ver", lambda: (mac_version, ("", "", ""), "")
    )
    assert hardware._mac_runs_apple_container() is expected  # pyright: ignore[reportPrivateUsage]


def test_firecracker_needs_kvm(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("karotte.hardware.platform.machine", lambda: "x86_64")
    monkeypatch.setattr(hardware, "KVM_DEVICE", tmp_path / "kvm")
    assert not hardware._linux_runs_firecracker()  # pyright: ignore[reportPrivateUsage]
    (tmp_path / "kvm").touch()
    assert hardware._linux_runs_firecracker()  # pyright: ignore[reportPrivateUsage]
