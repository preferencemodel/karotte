"""One confinement object per sandbox, each reporting which contract a limit delivered."""

from __future__ import annotations

import errno
import os
import shlex
import shutil
import stat
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from functools import cache
from pathlib import Path

from loguru import logger

import karotte.memory_watch as memory_watch
from karotte.cgroups import StudentCgroup, detect_cgroups, register_student_cgroup
from karotte.container import (
    demoted_uid_gid,
    is_containerized,
    is_gvisor,
    mount_points,
)
from karotte.file_quota import FileQuota, mount_file_quota
from karotte.hardware import hardware_limits

SANDBOX_ENV_VAR = "KAROTTE_SANDBOX"
STUDENT_CGROUP_NAME = "karotte_student"

GIB = 1024**3

HARNESS_RESERVE_BYTES = 1 * GIB
"""RAM held back from the student so the harness next to it keeps working."""

STUDENT_PROCESS_LIMIT = 2048
"""Far above legitimate use, far below a fork bomb."""

STUDENT_FILE_COUNT_LIMIT = 1_000_000
"""Far above legitimate use. Without it, data spread across more files than
the watchdog can walk would be invisible to the byte cap."""


class Sandbox(StrEnum):
    RUNC = "runc"
    GVISOR = "gvisor"
    FIRECRACKER = "firecracker"


@dataclass(frozen=True)
class FileLimit:
    """Cap on the files a uid may hold under ``path``; ``None`` marks no cap."""

    path: str | Path | tuple[str | Path, ...]
    bytes: int | None = None
    count: int | None = None

    @property
    def paths(self) -> tuple[Path, ...]:
        if isinstance(self.path, (str, Path)):
            return (Path(self.path),)
        return tuple(Path(p) for p in self.path)


@dataclass(frozen=True)
class ResourceLimits:
    """The limits in force for one uid; ``None`` marks no limit."""

    memory_bytes: int | None
    process_count: int | None
    file: FileLimit | None = None


class Contract(StrEnum):
    """What a limit actually bought."""

    PREVENTED = "prevented"
    """The kernel refuses at the boundary; the student's allocation fails."""

    REAPED = "detected_and_reaped"
    """Nothing stops the student crossing the line, but we notice and kill."""

    UNSUPPORTED = "not_supported"
    """No mechanism here. Said out loud so it is never mistaken for the above."""


@cache
def current_sandbox() -> Sandbox:
    """Which sandbox we are inside.

    Sniffs for gVisor when ``KAROTTE_SANDBOX`` is absent, for host CLIs predating it.
    """
    declared = os.environ.get(SANDBOX_ENV_VAR)
    if declared:
        try:
            return Sandbox(declared)
        except ValueError:
            logger.warning(f"Unknown {SANDBOX_ENV_VAR}={declared!r}; sniffing instead")
    return Sandbox.GVISOR if is_gvisor() else Sandbox.RUNC


def network_needs_namespace() -> bool:
    """gVisor's iptables silently no-op on GKE, so student sessions get a
    network namespace instead of trusting the firewall."""
    return current_sandbox() is Sandbox.GVISOR


_WRITE_BITS = stat.S_IWGRP | stat.S_IWOTH


FIREWALL_TOLERATE_ENV_VAR = "KAROTTE_FIREWALL_TOLERATE"
"""When set, a refused iptables rule is logged and reported as not taken
instead of aborting the run. Set by images run under an external harness
whose container has no NET_ADMIN and whose network isolation is the harness's
job; callers of ``deny_all_network``/``restrict_network`` already
handle the "rules did not take" result."""

HARDEN_EXEMPT_ENV_VAR = "KAROTTE_HARDEN_EXEMPT"
"""Extra mount points hardening leaves writable, ``os.pathsep``-separated. Set
by an image whose external harness mounts directories the student must write
to, e.g. its own log directories."""


def _promised_writable() -> tuple[Path, ...]:
    """Where an environment hands the student write access on purpose."""
    exempt = tuple(
        Path(os.path.abspath(part))
        for part in os.environ.get(HARDEN_EXEMPT_ENV_VAR, "").split(os.pathsep)
        if part
    )
    workdir = os.environ.get("KAROTTE_WORKDIR")
    if not workdir:
        logger.warning("KAROTTE_WORKDIR is unset; no workdir is exempt from hardening")
        return tuple(memory_watch.TEMP_DIRS) + exempt
    return (Path(workdir),) + tuple(memory_watch.TEMP_DIRS) + exempt


class Confinement:
    """What this sandbox can do to the student.

    The default is honest about having nothing: subclasses add what they have.
    """

    sandbox: Sandbox
    uid: int | None

    def __init__(self, sandbox: Sandbox, uid: int | None = None) -> None:
        self.sandbox = sandbox
        self.uid = uid
        self._file_limit: FileLimit | None = None
        self._file_watch: memory_watch.RunningFileWatch | None = None
        self._quota: FileQuota | None = None

    def limit_memory(self, nbytes: int | None) -> Contract:
        """Cap the student's RAM, or lift the cap when ``nbytes`` is ``None``."""
        del nbytes
        return Contract.UNSUPPORTED

    def limit_processes(self, count: int | None) -> Contract:
        """Cap the student's process count, or lift the cap when ``count`` is ``None``."""
        del count
        return Contract.UNSUPPORTED

    def limit_files(self, limit: FileLimit | None) -> Contract:
        """Cap the files the student may hold, or lift the cap when ``limit`` is ``None``.

        The first byte-capped limit gets the kernel-enforced quota mount where
        the sandbox permits mounting; later adjustments run the watchdog inside
        that backstop, since a mounted filesystem can grow but never shrink.
        The quota is a mount, not a cgroup, so it lives here rather than on
        ``CgroupConfinement`` — a sandbox with no writable cgroup hierarchy
        (an unprivileged pod) still gets it."""
        if (
            self._quota is None
            and limit is not None
            and limit.bytes is not None
            and self.uid is not None
        ):
            self._quota = mount_file_quota(limit.paths, limit.bytes, limit.count)
            if self._quota is not None:
                self._file_limit = limit
                return Contract.PREVENTED
        if self._quota is not None and (limit is None or limit.bytes is None):
            logger.info(
                f"The mounted quota keeps capping writes at {self._quota.size_bytes} bytes"
            )
        if self._file_watch is not None:
            _ = self._file_watch.stop()
            self._file_watch = None
        if self.uid is None:
            self._file_limit = None
            return Contract.UNSUPPORTED
        self._file_limit = limit
        if limit is not None:
            self._file_watch = memory_watch.start_file_watch(
                self.uid, limit.paths, max_bytes=limit.bytes, max_count=limit.count
            )
        return Contract.REAPED

    def current_limits(self) -> ResourceLimits:
        """What is in force right now, read from the enforcing mechanism."""
        return ResourceLimits(None, None, self._file_limit)

    def student_preexec(self) -> Callable[[], None] | None:
        """Places a student session under confinement, or ``None`` if there is none."""
        return None

    def close(self) -> None:
        """Stop whatever the limits started."""
        if self._file_watch is not None:
            _ = self._file_watch.stop()
            self._file_watch = None

    def harden_filesystem(self) -> list[Path]:
        """Take group and world write off every mounted directory carrying it,
        except the ones an environment promises the student. Returns the ones
        that changed."""
        if not is_containerized():
            return []

        promised = _promised_writable()
        closed: list[Path] = []
        for point in mount_points():
            if any(point == p or p in point.parents for p in promised):
                continue
            try:
                info = point.stat()
            except OSError:
                continue
            mode = stat.S_IMODE(info.st_mode)
            if not stat.S_ISDIR(info.st_mode) or not mode & _WRITE_BITS:
                continue
            try:
                os.chmod(point, mode & ~_WRITE_BITS)
            except OSError as e:
                # A read-only mount can't be written whatever its mode says.
                log = logger.debug if e.errno == errno.EROFS else logger.warning
                log(f"Could not close world-writable {point}: {e}")
                continue
            closed.append(point)

        if closed:
            logger.info(
                "Closed world-writable mounts: {}", ", ".join(str(p) for p in closed)
            )
        return closed

    def restrict_to_internal_network(
        self,
        uid: int | str,
        blocked_ports: list[int] | None = None,
        allowed_ips: list[str] | None = None,
    ) -> bool:
        """Keep a uid off the network, except localhost, metadata servers,
        private ranges and ``allowed_ips``. Returns whether the rules took:
        never on gVisor, whose iptables accepts rules without enforcing them.
        """
        if not is_containerized():
            return False

        iptables, ip6tables, reject = self._firewall_commands()
        owner = shlex.quote(str(uid))

        # IPv4 rules: allow localhost, metadata servers, and private networks
        firewall_rules = [
            # Block student from accessing internal service ports (websocket, MCP)
            # on localhost. These rules must come before the localhost ACCEPT rule.
            *[
                f"{iptables} -A OUTPUT -m owner --uid-owner {owner} -p tcp --dport {port} -j DROP"
                for port in (blocked_ports or [])
            ],
            # Allow localhost
            f"{iptables} -A OUTPUT -m owner --uid-owner {owner} -d 127.0.0.0/8 -j ACCEPT",
            # Allow GCP/AWS metadata server (both use 169.254.169.254)
            f"{iptables} -A OUTPUT -m owner --uid-owner {owner} -d 169.254.169.254 -j ACCEPT",
            # Allow link-local addresses (169.254.0.0/16) for metadata and other services
            f"{iptables} -A OUTPUT -m owner --uid-owner {owner} -d 169.254.0.0/16 -j ACCEPT",
            # Allow private networks (RFC 1918)
            f"{iptables} -A OUTPUT -m owner --uid-owner {owner} -d 10.0.0.0/8 -j ACCEPT",
            f"{iptables} -A OUTPUT -m owner --uid-owner {owner} -d 172.16.0.0/12 -j ACCEPT",
            f"{iptables} -A OUTPUT -m owner --uid-owner {owner} -d 192.168.0.0/16 -j ACCEPT",
            # Allow specific external hosts (e.g. the model proxy)
            *[
                f"{iptables} -A OUTPUT -m owner --uid-owner {owner} -d {ip} -j ACCEPT"
                for ip in (allowed_ips or [])
            ],
            # Reject everything else
            f"{iptables} -A OUTPUT -m owner --uid-owner {owner} -j {reject}",
        ]

        # IPv6 rules: allow localhost, link-local, and unique local addresses
        firewall_rules_v6 = [
            # Block student from accessing internal service ports on IPv6 localhost
            *[
                f"{ip6tables} -A OUTPUT -m owner --uid-owner {owner} -p tcp --dport {port} -j DROP"
                for port in (blocked_ports or [])
            ],
            # Allow localhost
            f"{ip6tables} -A OUTPUT -m owner --uid-owner {owner} -d ::1 -j ACCEPT",
            # Allow AWS IPv6 metadata server (IMDSv2)
            f"{ip6tables} -A OUTPUT -m owner --uid-owner {owner} -d fd00:ec2::254 -j ACCEPT",
            # Allow link-local (fe80::/10)
            f"{ip6tables} -A OUTPUT -m owner --uid-owner {owner} -d fe80::/10 -j ACCEPT",
            # Allow unique local (fd00::/8) - includes AWS metadata
            f"{ip6tables} -A OUTPUT -m owner --uid-owner {owner} -d fd00::/8 -j ACCEPT",
            # Reject everything else
            f"{ip6tables} -A OUTPUT -m owner --uid-owner {owner} -j {reject}",
        ]

        took = self._apply_firewall_rules(firewall_rules + firewall_rules_v6)
        return took and self.sandbox is not Sandbox.GVISOR

    def deny_all_network(self, uid: int | str) -> bool:
        """Cut a uid off the network completely. Returns whether the rules
        took: never on gVisor, whose iptables accepts rules without enforcing
        them.
        """
        if not is_containerized():
            return False

        iptables, ip6tables, reject = self._firewall_commands()
        owner = shlex.quote(str(uid))

        took = self._apply_firewall_rules(
            [
                f"{iptables} -A OUTPUT -m owner --uid-owner {owner} -j {reject}",
                f"{ip6tables} -A OUTPUT -m owner --uid-owner {owner} -j {reject}",
            ]
        )
        return took and self.sandbox is not Sandbox.GVISOR

    def _firewall_commands(self) -> tuple[str, str, str]:
        """The iptables/ip6tables commands and refusal target for this sandbox.

        gVisor has no nftables, so it gets iptables-legacy; it has no
        ICMP/TCP-RST either, so refusal is a silent DROP rather than a REJECT.
        Everything else uses the real thing.
        """
        if self.sandbox is Sandbox.GVISOR:
            return "iptables-legacy", "ip6tables-legacy", "DROP"
        return "iptables", "ip6tables", "REJECT"

    def _apply_firewall_rules(self, rules: list[str]) -> bool:
        """Apply rules in order, reporting whether they all took."""
        for rule in rules:
            result = subprocess.run(
                rule,
                capture_output=True,
                shell=True,
                text=True,
            )
            if result.returncode != 0:
                if self.sandbox is Sandbox.GVISOR:
                    logger.warning(
                        "iptables doesn't work when gVisor is enabled, you'll need to use additional controls for "
                        + "disabling networking when spawning child processes, e.g. `disable_networking` on the "
                        + "bash tool."
                    )
                    return False
                elif os.environ.get(FIREWALL_TOLERATE_ENV_VAR):
                    logger.warning(
                        "Firewall rule {!r} was refused (return code {}, stderr: {}); "
                        + "{} is set, so the run continues without it. The harness "
                        + "running this container is expected to isolate the network.",
                        rule,
                        result.returncode,
                        result.stderr.strip(),
                        FIREWALL_TOLERATE_ENV_VAR,
                    )
                    return False
                else:
                    raise RuntimeError(
                        f"failed to apply firewall rule '{rule}': return code: {result.returncode} "
                        + f"| stdout: {result.stdout} | stderr: {result.stderr}"
                    )
        return True


class WatchdogConfinement(Confinement):
    """Polls what the student holds and reaps past a limit; nothing is prevented."""

    def __init__(self, sandbox: Sandbox, uid: int) -> None:
        super().__init__(sandbox, uid)
        self._uid: int = uid
        self._max_bytes: int | None = None
        self._max_processes: int | None = None
        self._running: memory_watch.RunningWatch | None = None

    def limit_memory(self, nbytes: int | None) -> Contract:
        self._max_bytes = nbytes
        self._restart()
        return Contract.REAPED

    def limit_processes(self, count: int | None) -> Contract:
        self._max_processes = count
        self._restart()
        return Contract.REAPED

    def current_limits(self) -> ResourceLimits:
        return ResourceLimits(self._max_bytes, self._max_processes, self._file_limit)

    def close(self) -> None:
        self._stop_watch()
        super().close()

    def _stop_watch(self) -> None:
        if self._running is not None:
            _ = self._running.stop()
            self._running = None

    def _restart(self) -> None:
        """One watch carries the memory and process limits, so a new limit
        replaces the watch; with nothing left to enforce, there is no watch."""
        self._stop_watch()
        if self._max_bytes is None and self._max_processes is None:
            return
        self._running = memory_watch.start_watch(
            self._uid, self._max_bytes, max_processes=self._max_processes
        )


class CgroupConfinement(Confinement):
    """Backed by a real cgroup, on whichever version the sandbox offers."""

    group: StudentCgroup

    def __init__(
        self, sandbox: Sandbox, group: StudentCgroup, uid: int | None = None
    ) -> None:
        super().__init__(sandbox, uid)
        self.group = group

    def limit_memory(self, nbytes: int | None) -> Contract:
        if self.group.set_memory_limit(nbytes):
            return Contract.PREVENTED
        return Contract.UNSUPPORTED

    def limit_processes(self, count: int | None) -> Contract:
        if self.group.set_process_limit(count):
            return Contract.PREVENTED
        return Contract.UNSUPPORTED

    def current_limits(self) -> ResourceLimits:
        return ResourceLimits(
            self.group.memory_limit(), self.group.process_limit(), self._file_limit
        )

    def student_preexec(self) -> Callable[[], None] | None:
        return self.group.join_self


def build_confinement(
    sandbox: Sandbox | None = None, uid: int | None = None
) -> Confinement:
    """The confinement for this sandbox: a cgroup where one works, else the watchdog.

    gVisor is excluded from the cgroup path: it accepts every write and
    enforces none, so success there is not evidence of a limit. ``uid`` is the
    student uid to watch, defaulting to the demotion target.
    """
    sandbox = current_sandbox() if sandbox is None else sandbox
    uid = _student_uid() if uid is None else uid

    if sandbox is Sandbox.GVISOR:
        logger.debug("gVisor: cgroup limits are accepted but not enforced; skipping")
    else:
        confinement = _cgroup_confinement(sandbox, uid)
        if confinement is not None:
            return confinement

    if uid is None:
        logger.debug(f"{sandbox}: no student uid to watch; limits unavailable")
        return Confinement(sandbox)

    logger.debug(f"{sandbox}: confining the student with the uid {uid} watchdog")
    return WatchdogConfinement(sandbox, uid)


def get_confinement(uid: int | None = None) -> Confinement:
    """The confinement for a uid, defaulting to the demotion target.

    Built once per uid, since building one may create a cgroup as a side
    effect; the uid resolves before the cache so the default and its explicit
    spelling share one instance."""
    return _confinement_for(_student_uid() if uid is None else uid)


@cache
def _confinement_for(uid: int | None) -> Confinement:
    return build_confinement(uid=uid)


def _student_uid() -> int | None:
    """The demotion target, or ``None`` where there is none to confine.

    A container without ``KAROTTE_DEMOTE_ID`` makes :func:`demoted_uid_gid` raise;
    the student-spawning path still refuses there, so confinement reports
    honestly instead of doubling the refusal.
    """
    try:
        return demoted_uid_gid()
    except RuntimeError as exc:
        logger.error(f"No student uid to confine: {exc}")
        return None


def _cgroup_confinement(sandbox: Sandbox, uid: int | None) -> CgroupConfinement | None:
    backend = detect_cgroups()
    if backend is None:
        logger.debug(f"{sandbox}: no writable cgroup hierarchy")
        return None

    # Named by uid so every process confining the same uid lands in the same
    # group, and different uids never share one.
    name = STUDENT_CGROUP_NAME if uid is None else f"karotte_uid_{uid}"
    try:
        group = backend.create(name)
    except OSError as exc:
        logger.warning(f"{sandbox}: could not create the student cgroup: {exc}")
        return None

    if uid is not None:
        register_student_cgroup(uid, group)
    logger.debug(f"{sandbox}: confining the student with cgroup v{backend.version}")
    return CgroupConfinement(sandbox, group, uid)


def sandbox_memory_bytes(required_hardware: str | None) -> int | None:
    """The sandbox's RAM: the hardware plugin's number, else the tightest cgroup
    limit above the student, else the machine's RAM; ``None`` without a working cgroup."""
    limits = hardware_limits(required_hardware)
    if limits is not None and limits.memory_bytes is not None:
        return limits.memory_bytes
    if not is_containerized():
        return None
    confinement = get_confinement()
    if not isinstance(confinement, CgroupConfinement):
        return None
    return confinement.group.inherited_memory_limit() or _physical_memory_bytes()


def _physical_memory_bytes() -> int:
    return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")


class _Unset:
    """Tells "not passed" apart from an explicit ``None``."""


_UNSET = _Unset()


def limit_resources(
    *,
    uid: int | None = None,
    memory_bytes: int | None | _Unset = _UNSET,
    process_count: int | None | _Unset = _UNSET,
    file: FileLimit | None | _Unset = _UNSET,
) -> dict[str, Contract]:
    """Set or lift resource limits for a uid (default: the student): a limit
    not passed is kept, one passed as ``None`` is lifted.

    Returns the contract each applied limit delivered, keyed by limit name.
    """
    if not is_containerized():
        return {}

    confinement = get_confinement(uid)
    contracts: dict[str, Contract] = {}

    if not isinstance(memory_bytes, _Unset):
        contract = confinement.limit_memory(memory_bytes)
        if memory_bytes is None:
            logger.info(f"Memory limit for uid {confinement.uid} lifted")
        else:
            contracts["memory"] = contract
            logger.info(
                f"Memory limit for uid {confinement.uid}: {memory_bytes / GIB:.1f}GiB ({contract})"
            )

    if not isinstance(process_count, _Unset):
        contract = confinement.limit_processes(process_count)
        if process_count is None:
            logger.info(f"Process limit for uid {confinement.uid} lifted")
        else:
            contracts["processes"] = contract
            logger.info(
                f"Process limit for uid {confinement.uid}: {process_count} ({contract})"
            )

    if not isinstance(file, _Unset):
        contract = confinement.limit_files(file)
        if file is None:
            logger.info(f"File limit for uid {confinement.uid} lifted")
        else:
            contracts["files"] = contract
            logger.info(f"File limit for uid {confinement.uid}: {file} ({contract})")

    return contracts


_FREE_DISK_FRACTION = 0.8
"""Of the disk space free at task start, what the student may fill."""


def apply_default_limits(required_hardware: str | None) -> dict[str, Contract]:
    """Apply the default student limits: the sandbox's RAM less the harness
    reserve, a process cap, and most of the free disk for files.

    Tasks adjust any of them afterwards with :func:`limit_resources` from
    ``pre_hook``.
    """
    ram = sandbox_memory_bytes(required_hardware)
    return limit_resources(
        memory_bytes=ram - HARNESS_RESERVE_BYTES if ram is not None else None,
        process_count=STUDENT_PROCESS_LIMIT,
        file=_default_file_limit(required_hardware),
    )


_STATE = {
    Contract.PREVENTED: "on",
    Contract.REAPED: "off (watchdog)",
    Contract.UNSUPPORTED: "off (none)",
}


def describe_confinement(
    contracts: Mapping[str, Contract],
    *,
    network_firewall: bool | None,
    ipc_namespace: bool,
    sandbox: Sandbox,
) -> tuple[str, bool]:
    """The confinement in force as one line, and whether any of it is off."""
    memory = contracts.get("memory", Contract.UNSUPPORTED)
    processes = contracts.get("processes", Contract.UNSUPPORTED)
    files = contracts.get("files")
    if memory == processes:
        parts = [f"cgroup limits {_STATE[memory]}"]
    else:
        parts = [f"memory limit {_STATE[memory]}", f"process limit {_STATE[processes]}"]
    parts.append(
        "file quota off (no disk-backed paths)"
        if files is None
        else f"file quota {_STATE[files]}"
    )
    if network_firewall is not None:
        parts.append(f"network firewall {'on' if network_firewall else 'off'}")
    parts.append(f"IPC namespace {'on' if ipc_namespace else 'off'}")
    parts.append(
        "Firecracker VM"
        if sandbox is Sandbox.FIRECRACKER
        else f"gVisor {'on' if sandbox is Sandbox.GVISOR else 'off'}"
    )
    degraded = (
        any(c is not Contract.PREVENTED for c in (memory, processes, files))
        or network_firewall is False
        or not ipc_namespace
    )
    return "Confinement: " + ", ".join(parts), degraded


def apply_default_file_limit(required_hardware: str | None) -> dict[str, Contract]:
    """Apply only the default file limit, leaving memory and processes alone."""
    return limit_resources(file=_default_file_limit(required_hardware))


def ensure_default_limits(required_hardware: str | None) -> dict[str, Contract]:
    """Apply the default memory and process limits, each only where no limit
    is already in force."""
    current = get_resource_limits()
    memory_bytes: int | None | _Unset = _UNSET
    if current.memory_bytes is None:
        ram = sandbox_memory_bytes(required_hardware)
        if ram is not None:
            memory_bytes = ram - HARNESS_RESERVE_BYTES
    process_count: int | _Unset = _UNSET
    if current.process_count is None:
        process_count = STUDENT_PROCESS_LIMIT
    return limit_resources(memory_bytes=memory_bytes, process_count=process_count)


def _default_file_limit(required_hardware: str | None = None) -> FileLimit | None:
    """Everywhere the student can write on real disk, capped at a share of the
    free space; tmpfs locations are the memory limit's to weigh.

    Outside Firecracker the free-space reading may be the host's, not the
    sandbox's, so it is additionally capped at the hardware plugin's disk budget."""
    candidates = [Path(w) for w in (os.environ.get("KAROTTE_WORKDIR"),) if w]
    candidates += list(memory_watch.TEMP_DIRS)
    paths = tuple(
        path for path in candidates if path.is_dir() and memory_watch.disk_backed(path)
    )
    if not paths:
        return None
    budget = int(shutil.disk_usage(paths[0]).free * _FREE_DISK_FRACTION)
    if current_sandbox() is not Sandbox.FIRECRACKER:
        limits = hardware_limits(required_hardware)
        if limits is not None and limits.disk_bytes is not None:
            budget = min(budget, limits.disk_bytes)
    return FileLimit(
        path=paths,
        bytes=budget,
        count=STUDENT_FILE_COUNT_LIMIT,
    )


def get_resource_limits(*, uid: int | None = None) -> ResourceLimits:
    """The limits currently in force for a uid; ``None`` marks no limit.

    Reads through to the enforcing mechanism, so it reflects what the last
    ``limit_resources`` call actually achieved rather than what was asked for.
    """
    if not is_containerized():
        return ResourceLimits(None, None)
    return get_confinement(uid).current_limits()
