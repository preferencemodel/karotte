"""Checks before a ``--runtime firecracker`` run, each failing with what to do."""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Never

import typer
from loguru import logger

from karotte.firecracker import network
from karotte.firecracker.artifacts import (
    SUPPORTED_ARCHES,
    ArtifactError,
    ensure_artifacts,
    host_arch,
)
from karotte.firecracker.drives import DriveError, e2fs_tool
from karotte.firecracker.vm import (
    JAILER_ENV_VAR,
    JailNetns,
    VmError,
    jailer_ids,
    parse_mounts,
)
from karotte.hardware import vm_size

KVM_DEVICE = Path("/dev/kvm")


def firecracker_problems(
    required_hardware: str | None,
    mounts: list[str] | None,
    kvm: Path = KVM_DEVICE,
) -> list[str]:
    """What stops a Firecracker run here; empty when nothing does. Downloads
    the pinned artifacts if they're missing."""
    if not sys.platform.startswith("linux"):
        return ["The firecracker runtime needs Linux with KVM."]
    arch = host_arch()
    if arch not in SUPPORTED_ARCHES:
        return [f"The firecracker runtime needs x86_64 or aarch64, not {arch}."]

    problems: list[str] = []
    if not kvm.exists():
        problems.append(
            f"{kvm} is missing: this host has no KVM (on a cloud VM, enable nested virtualization)."
        )
    elif not os.access(kvm, os.R_OK | os.W_OK):
        problems.append(
            f"{kvm} isn't readable and writable for you: `sudo usermod -aG kvm $USER`, then log in again."
        )

    try:
        _ = vm_size(required_hardware)
    except ValueError as e:
        problems.append(str(e))

    try:
        parse_mounts(mounts, dev=False, build_context=".")
    except VmError as e:
        problems.append(str(e))

    if shutil.which("docker") is None:
        problems.append("docker not found: it builds the image the VM boots.")
    elif not _docker_has_buildkit():
        problems.append(
            "docker has no BuildKit (buildx), which the environment's Containerfile"
            + " needs: install it (Debian and Ubuntu: `apt install docker-buildx`)."
        )
    for tool in ("mkfs.ext4", "debugfs"):
        try:
            e2fs_tool(tool)
        except DriveError as e:
            problems.append(str(e))

    try:
        mode = network.network_mode()
    except network.NetworkError as e:
        problems.append(str(e))
    else:
        problems += _network_problems(mode)

    if not problems:
        try:
            ensure_artifacts()
        except (ArtifactError, OSError) as e:
            problems.append(f"Cannot get the Firecracker artifacts: {e}")
    return problems


def _network_problems(mode: network.NetworkMode) -> list[str]:
    if mode == "none":
        return []
    missing = [
        t
        for t in ("pasta", "ip", "iptables")
        if shutil.which(t) is None and shutil.which(t, path="/usr/sbin:/sbin") is None
    ]
    if missing:
        return [f"{', '.join(missing)} not found; pasta networking needs them."]
    if os.environ.get(JAILER_ENV_VAR) == "1":
        return _jailed_pasta_problems()
    return _pasta_problems()


def _jailed_pasta_problems() -> list[str]:
    """Set up the jailed VMM's network namespace once, the way a run does:
    as root, so the unprivileged probe below says nothing about it."""
    try:
        JailNetns.start([], [], *jailer_ids()).stop()
    except (VmError, OSError) as e:
        return [f"pasta can't give a jailed VM a network here: {e}"]
    return []


def _pasta_problems() -> list[str]:
    """Start pasta once the way a run does. It can be installed and still
    unable to run: Ubuntu 24.04 blocks unprivileged user namespaces with
    AppArmor, and its passt package predates the profile that allows pasta
    one."""
    try:
        result = subprocess.run(
            ["pasta", "--config-net", "--quiet", "--", "true"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as e:
        error = str(e)
    else:
        if result.returncode == 0:
            return []
        # Every line: the first is often a harmless warning (no IPv6).
        error = (
            "; ".join(result.stderr.strip().splitlines()) or f"exit {result.returncode}"
        )
    return [
        f"pasta can't set up a network namespace here ({error}). On Ubuntu"
        + " 24.04, AppArmor keeps it from making the user namespace it needs:"
        + " `sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0`"
        + " lifts that until reboot. Or set KAROTTE_FIRECRACKER_NETWORK=none"
        + " to run without a network."
    ]


def _docker_has_buildkit() -> bool:
    try:
        result = subprocess.run(
            ["docker", "buildx", "version"],
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def validate_firecracker_runtime(
    required_hardware: str | None, mounts: list[str] | None
) -> None:
    """Abort with instructions unless a Firecracker run can start here."""
    problems = firecracker_problems(required_hardware, mounts)
    if problems:
        _print_and_abort(
            "Cannot use the firecracker runtime:\n  - "
            + "\n  - ".join(problems)
            + "\nFix the above, or run in a container instead with --runtime docker (or podman)."
        )
    logger.info("Firecracker runtime checks passed")


def _print_and_abort(message: str) -> Never:
    typer.secho(message, fg=typer.colors.RED, err=True)
    raise typer.Abort()
