"""One confinement object per sandbox, each reporting which contract a limit delivered."""

from __future__ import annotations

import errno
import ipaddress
import os
import pwd
import shlex
import shutil
import socket
import stat
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from functools import cache
from pathlib import Path

import psutil
from loguru import logger

import karotte.memory_watch as memory_watch
from karotte.cgroups import (
    StudentCgroup,
    detect_cgroups,
    read_mounts,
    register_student_cgroup,
)
from karotte.container import (
    demoted_uid_gid,
    is_containerized,
    is_gvisor,
    mount_points,
)
from karotte.file_quota import FileQuota, mount_file_quota
from karotte.hardware import hardware_limits
from karotte.trusted_bin import trusted_binary

SANDBOX_ENV_VAR = "KAROTTE_SANDBOX"
STUDENT_NETWORK_ENV_VAR = "KAROTTE_STUDENT_NETWORK"
"""What the student may reach besides ``allowed_ips``. Unset or ``strict``:
localhost and the sandbox's own addresses. ``internal``: also the link-local
and private ranges (the metadata server, the rest of the private network), for
a task that needs them. Any other value is treated as ``strict``."""
SANDBOX_MEMORY_ENV_VAR = "KAROTTE_SANDBOX_MEMORY_BYTES"
"""The sandbox's RAM in bytes, from the launcher of a VM that holds headroom
above it for the guest kernel. Read when no hardware plugin knows the
hardware, before the cgroup limit and the machine's RAM."""
DISK_BUDGET_ENV_VAR = "KAROTTE_DISK_BUDGET_BYTES"
"""The student's disk budget in bytes, chosen by the host that launched the
sandbox. Set by karotte's VM launchers, which see the host's real free space;
inside the guest, ``df`` reports the VM's own disk, which may be sparse and
larger than the host can back."""
VM_LAUNCHER_ENV_VAR = "KAROTTE_VM_LAUNCHER"
"""Set by karotte's own VM launchers to the runtime's name. They size the VM
with :data:`karotte.hardware.VM_MEMORY_HEADROOM_BYTES` above the sandbox's
RAM; a VM someone else launched may be sized differently."""
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
    VM = "vm"
    """A virtual machine with its own guest kernel (Firecracker, Apple
    `container`, Kata): cgroups and iptables in the guest are real. What
    differs between VM hosts (the disk behind the guest, the network around
    it) is passed in by whoever launched it, not inferred from this value."""
    FIRECRACKER = "vm"
    """The older name for ``VM``; ``KAROTTE_SANDBOX=firecracker`` still selects it."""

    @classmethod
    def _missing_(cls, value: object) -> Sandbox | None:
        if value == "firecracker":
            return cls.VM
        return None


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

FIREWALL_BACKEND_ENV_VAR = "KAROTTE_FIREWALL_BACKEND"
"""``nft`` writes the student firewall as nftables ``meta skuid`` rules
instead of iptables ``-m owner`` rules. Set by a launcher whose guest kernel
has nftables but not the iptables owner match (Modal's VMs)."""

NFT_TABLE = "karotte"
"""The ``inet`` table the nftables firewall keeps one chain per uid in."""

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
        """Keep a uid off the network, except localhost, the sandbox's own
        addresses and ``allowed_ips``. Returns whether the rules took: never on
        gVisor, whose iptables accepts rules without enforcing them.

        ``KAROTTE_STUDENT_NETWORK=internal`` also lets through the link-local
        and private ranges. Off by default: they hold the metadata server and
        its credentials, other workloads, and on a developer machine the
        machine itself and its LAN.
        """
        if not is_containerized():
            return False

        iptables, ip6tables, reject = self._firewall_commands()
        owner = shlex.quote(str(uid))
        internal = _student_network_is_internal()
        own_v4, own_v6 = _own_addresses()

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
            # Allow the sandbox's own addresses (a server the student runs,
            # reached by its interface address)
            *[
                f"{iptables} -A OUTPUT -m owner --uid-owner {owner} -d {ip} -j ACCEPT"
                for ip in own_v4
            ],
            *(
                []
                if not internal
                else [
                    # Allow GCP/AWS metadata server (both use 169.254.169.254)
                    f"{iptables} -A OUTPUT -m owner --uid-owner {owner} -d 169.254.169.254 -j ACCEPT",
                    # Allow link-local addresses (169.254.0.0/16) for metadata and other services
                    f"{iptables} -A OUTPUT -m owner --uid-owner {owner} -d 169.254.0.0/16 -j ACCEPT",
                    # Allow private networks (RFC 1918)
                    f"{iptables} -A OUTPUT -m owner --uid-owner {owner} -d 10.0.0.0/8 -j ACCEPT",
                    f"{iptables} -A OUTPUT -m owner --uid-owner {owner} -d 172.16.0.0/12 -j ACCEPT",
                    f"{iptables} -A OUTPUT -m owner --uid-owner {owner} -d 192.168.0.0/16 -j ACCEPT",
                ]
            ),
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
            *[
                f"{ip6tables} -A OUTPUT -m owner --uid-owner {owner} -d {ip} -j ACCEPT"
                for ip in own_v6
            ],
            *(
                []
                if not internal
                else [
                    # Allow AWS IPv6 metadata server (IMDSv2)
                    f"{ip6tables} -A OUTPUT -m owner --uid-owner {owner} -d fd00:ec2::254 -j ACCEPT",
                    # Allow link-local (fe80::/10)
                    f"{ip6tables} -A OUTPUT -m owner --uid-owner {owner} -d fe80::/10 -j ACCEPT",
                    # Allow unique local (fd00::/8) - includes AWS metadata
                    f"{ip6tables} -A OUTPUT -m owner --uid-owner {owner} -d fd00::/8 -j ACCEPT",
                ]
            ),
            # Reject everything else
            f"{ip6tables} -A OUTPUT -m owner --uid-owner {owner} -j {reject}",
        ]

        took = self._apply_firewall_rules(
            self._for_backend(uid, firewall_rules + firewall_rules_v6)
        )
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
            self._for_backend(
                uid,
                [
                    f"{iptables} -A OUTPUT -m owner --uid-owner {owner} -j {reject}",
                    f"{ip6tables} -A OUTPUT -m owner --uid-owner {owner} -j {reject}",
                ],
            )
        )
        return took and self.sandbox is not Sandbox.GVISOR

    def lift_network_rules(self, uid: int) -> bool:
        """Delete every OUTPUT rule matching ``uid``, as the two methods above
        add them. Returns whether none are left."""
        if _uses_nft():
            return _lift_nft_chain(uid)
        iptables, ip6tables, _ = self._firewall_commands()
        clean = True
        for command in (iptables, ip6tables):
            try:
                binary = trusted_binary(command)
            except FileNotFoundError:
                continue
            listing = subprocess.run(
                [binary, "-S", "OUTPUT"], capture_output=True, text=True
            )
            if listing.returncode != 0:
                # Nothing listable was added either.
                continue
            for rule in rules_owned_by(listing.stdout, uid):
                result = subprocess.run(
                    [binary, "-D", *rule], capture_output=True, text=True
                )
                if result.returncode != 0:
                    logger.warning(
                        f"Could not delete {command} rule {rule}: {result.stderr.strip()}"
                    )
                    clean = False
        return clean

    def _firewall_commands(self) -> tuple[str, str, str]:
        """The iptables/ip6tables commands and refusal target for this sandbox.

        gVisor has no nftables, so it gets iptables-legacy; it has no
        ICMP/TCP-RST either, so refusal is a silent DROP rather than a REJECT.
        Everything else uses the real thing.
        """
        if self.sandbox is Sandbox.GVISOR:
            return "iptables-legacy", "ip6tables-legacy", "DROP"
        return "iptables", "ip6tables", "REJECT"

    def _for_backend(self, uid: int | str, rules: list[str]) -> list[str]:
        """``rules`` as written, or as nftables rules on the nft backend."""
        return _nft_commands(_numeric_uid(uid), rules) if _uses_nft() else rules

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


def _uses_nft() -> bool:
    return os.environ.get(FIREWALL_BACKEND_ENV_VAR) == "nft"


def _numeric_uid(uid: int | str) -> int:
    return int(uid) if str(uid).isdigit() else pwd.getpwnam(str(uid)).pw_uid


def _nft_chain(uid: int) -> str:
    return f"uid_{uid}"


def _nft_commands(uid: int, iptables_rules: Sequence[str]) -> list[str]:
    """The same rules for nftables, in a base chain of the uid's own on the
    OUTPUT hook. A packet passes only if every base chain accepts it, so one
    uid's accepts can't let another uid's packets through."""
    chain = _nft_chain(uid)
    spec = "{ type filter hook output priority 0 ; policy accept ; }"
    commands = [
        f"nft add table inet {NFT_TABLE}",
        f"nft add chain inet {NFT_TABLE} {chain} {shlex.quote(spec)}",
    ]
    for rule in iptables_rules:
        words = shlex.split(rule)
        args = dict(zip(words, words[1:]))
        flags = {w for w in words if w.startswith("-")}
        if (
            flags - {"-A", "-m", "--uid-owner", "-p", "--dport", "-d", "-j"}
            or args.get("-p", "tcp") != "tcp"
        ):
            raise ValueError(f"No nftables translation for {rule!r}")
        family = "ipv6" if words[0].startswith("ip6tables") else "ipv4"
        match = [f"meta skuid {uid}"]
        if "-d" in args:
            match.append(f"{'ip6' if family == 'ipv6' else 'ip'} daddr {args['-d']}")
        else:
            match.append(f"meta nfproto {family}")
        if "--dport" in args:
            match.append(f"tcp dport {args['--dport']}")
        commands.append(
            f"nft add rule inet {NFT_TABLE} {chain} {' '.join(match)} {args['-j'].lower()}"
        )
    return commands


def _lift_nft_chain(uid: int) -> bool:
    """Delete the uid's chain and its rules; whether none are left."""
    chain = _nft_chain(uid)
    try:
        nft = trusted_binary("nft")
    except FileNotFoundError:
        return True
    listing = subprocess.run(
        [nft, "list", "chain", "inet", NFT_TABLE, chain], capture_output=True, text=True
    )
    if listing.returncode != 0:
        # Nothing was added.
        return True
    for command in (["flush", "chain"], ["delete", "chain"]):
        result = subprocess.run(
            [nft, *command, "inet", NFT_TABLE, chain], capture_output=True, text=True
        )
        if result.returncode != 0:
            logger.warning(
                f"Could not {command[0]} nft chain {chain}: {result.stderr.strip()}"
            )
            return False
    return True


def rules_owned_by(listing: str, uid: int) -> list[list[str]]:
    """The OUTPUT rules in an ``iptables -S`` listing that match ``uid``,
    each as the arguments that follow ``-D``."""
    rules: list[list[str]] = []
    for line in listing.splitlines():
        words = shlex.split(line)
        if words[:2] != ["-A", "OUTPUT"]:
            continue
        if any(
            word == "--uid-owner" and value == str(uid)
            for word, value in zip(words, words[1:])
        ):
            rules.append(words[1:])
    return rules


# Fixed device numbers (the kernel's Documentation/admin-guide/devices.txt):
# /dev/loop-control is character device 10:237, and /dev/loopN is block device
# 7:N. Eight nodes, as a stock system creates, cover the run's file quota and a
# confinement check's probe quota at once.
_LOOP_CONTROL = (10, 237)
_LOOP_MAJOR = 7
_LOOP_DEVICES = 8


def prepare_vm_guest(dev: Path | None = None) -> list[str]:
    """Make the guest kernel's cgroups and loop devices usable, in a VM.

    Some VM guests hand the container a read-only cgroupfs and no loop device
    nodes (a Kata guest has both), which leaves karotte on the watchdog and
    without a kernel file quota. Root inside the VM may fix both: the kernel is
    the VM's own. Runs before any confinement is built; each step is logged,
    and a step that fails leaves the weaker fallback in place. Returns what it
    changed.
    """
    if (
        current_sandbox() is not Sandbox.VM
        or os.geteuid() != 0
        or not is_containerized()
    ):
        return []
    changed: list[str] = []
    for mount in read_mounts():
        if not (mount.is_cgroup2 and mount.read_only):
            continue
        result = subprocess.run(
            [trusted_binary("mount"), "-o", "remount,rw", str(mount.path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            changed.append(f"remounted {mount.path} read-write")
        else:
            logger.warning(
                f"Could not remount {mount.path} read-write ({result.stderr.strip()}); student limits fall back to the watchdog"
            )
    dev = Path("/dev") if dev is None else dev
    nodes = [(dev / "loop-control", stat.S_IFCHR, os.makedev(*_LOOP_CONTROL))]
    nodes += [
        (dev / f"loop{n}", stat.S_IFBLK, os.makedev(_LOOP_MAJOR, n))
        for n in range(_LOOP_DEVICES)
    ]
    for path, kind, device in nodes:
        if path.exists():
            continue
        try:
            os.mknod(path, kind | 0o660, device)
        except OSError as exc:
            logger.warning(f"Could not create {path} ({exc}); no kernel file quota")
            break
        changed.append(f"created {path}")
    for step in changed:
        logger.info(f"VM guest setup: {step}")
    return changed


def _student_network_is_internal() -> bool:
    value = os.environ.get(STUDENT_NETWORK_ENV_VAR)
    if value == "internal":
        return True
    if value not in (None, "", "strict"):
        logger.warning(
            f"Unknown {STUDENT_NETWORK_ENV_VAR}={value!r}; using the strict student firewall"
        )
    return False


def _own_addresses() -> tuple[list[str], list[str]]:
    """The sandbox's own IPv4 and IPv6 addresses, read from its interfaces.
    Loopback is covered separately; link-local addresses are left out."""
    v4: set[str] = set()
    v6: set[str] = set()
    for addrs in psutil.net_if_addrs().values():
        for addr in addrs:
            if addr.family not in (socket.AF_INET, socket.AF_INET6):
                continue
            try:
                ip = ipaddress.ip_address(addr.address.split("%", 1)[0])
            except ValueError:
                continue
            if ip.is_loopback or ip.is_link_local:
                continue
            (v4 if ip.version == 4 else v6).add(str(ip))
    return sorted(v4), sorted(v6)


_METADATA_ADDRESS = "169.254.169.254"
_PUBLIC_ADDRESS = "1.1.1.1"
_SELF_TEST_TIMEOUT_SECONDS = 1.0


def firewall_canaries(allowed_ips: Sequence[str] = ()) -> list[tuple[str, int]]:
    """Destinations the student firewall must refuse, for :func:`reachable_as`.

    A connection that succeeds proves the firewall is not doing its job; one
    that fails proves nothing (a REJECT and a closed port look alike), so these
    are addresses that answer when nothing is in the way: a public host, and
    unless the private ranges are deliberately open, the metadata server and
    the default gateway. ``allowed_ips`` are left out: the student may reach
    them on any port, and a model proxy can sit on the gateway (on a Mac
    running Apple `container`, the gateway is the Mac)."""
    canaries = [(_PUBLIC_ADDRESS, 80)]
    if not _student_network_is_internal():
        canaries.append((_METADATA_ADDRESS, 80))
        gateway = _default_gateway()
        if gateway is not None:
            canaries += [(gateway, 80), (gateway, 53)]
    allowed = set(allowed_ips)
    return [(host, port) for host, port in canaries if host not in allowed]


_ROUTE_TABLE = Path("/proc/net/route")
_RTF_GATEWAY = 0x2


def _default_gateway(route_table: Path | None = None) -> str | None:
    try:
        lines = (route_table or _ROUTE_TABLE).read_text().splitlines()[1:]
    except OSError:
        return None
    for line in lines:
        fields = line.split()
        if len(fields) < 4 or fields[1] != "00000000":
            continue
        try:
            gateway, flags = int(fields[2], 16), int(fields[3], 16)
        except ValueError:
            continue
        # A default route with no gateway (an on-link one) has 0.0.0.0 here,
        # which connects to localhost: not a canary.
        if flags & _RTF_GATEWAY and gateway:
            try:
                return socket.inet_ntoa(gateway.to_bytes(4, "little"))
            except OverflowError:
                continue
    return None


def reachable_as(uid: int, targets: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """The ``targets`` a process running as ``uid`` can open a TCP connection to.

    Forks a helper that drops to ``uid`` and tries each; no exec, so nothing on
    disk is trusted. Raises ``RuntimeError`` if the helper can't run."""
    read_fd, write_fd = os.pipe()
    child = os.fork()
    if child == 0:
        os.close(read_fd)
        try:
            os.setgroups([])
            os.setgid(uid)
            os.setuid(uid)
            reached = []
            for index, (host, port) in enumerate(targets):
                try:
                    with socket.create_connection(
                        (host, port), timeout=_SELF_TEST_TIMEOUT_SECONDS
                    ):
                        reached.append(str(index))
                except OSError:
                    pass
            _ = os.write(write_fd, (",".join(reached) + "\n").encode())
        except BaseException:
            os._exit(1)
        os._exit(0)
    os.close(write_fd)
    with os.fdopen(read_fd, "rb") as pipe:
        report = pipe.read().decode()
    _, status = os.waitpid(child, 0)
    if not os.WIFEXITED(status) or os.WEXITSTATUS(status) != 0 or not report:
        raise RuntimeError(f"The firewall self-test as uid {uid} could not run")
    return [targets[int(i)] for i in report.strip().split(",") if i]


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
    """The sandbox's RAM: the hardware plugin's number, else the launcher's
    (``KAROTTE_SANDBOX_MEMORY_BYTES``, when above the harness reserve), else the
    tightest cgroup limit above the student, else the machine's RAM. ``None``
    outside a container, or in one without a working cgroup, when neither the
    plugin nor the launcher says."""
    limits = hardware_limits(required_hardware)
    if limits is not None and limits.memory_bytes is not None:
        return limits.memory_bytes
    launcher = _positive_int_env(SANDBOX_MEMORY_ENV_VAR)
    if launcher is not None and launcher <= HARNESS_RESERVE_BYTES:
        # The student gets the sandbox's RAM less the reserve: zero or less
        # would OOM-kill every student process.
        logger.warning(
            f"Ignoring {SANDBOX_MEMORY_ENV_VAR}={launcher}: not above the harness reserve of {HARNESS_RESERVE_BYTES} bytes"
        )
        launcher = None
    if launcher is not None:
        return launcher
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


def _launcher_disk_budget() -> int | None:
    return _positive_int_env(DISK_BUDGET_ENV_VAR)


def _positive_int_env(name: str) -> int | None:
    value = os.environ.get(name)
    if not value:
        return None
    try:
        number = int(value)
    except ValueError:
        logger.warning(f"Ignoring {name}={value!r}: not an integer")
        return None
    if number <= 0:
        logger.warning(f"Ignoring {name}={value!r}: not positive")
        return None
    return number


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
        "VM"
        if sandbox is Sandbox.VM
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

    Capped at a budget chosen outside the sandbox: the launcher's
    (``KAROTTE_DISK_BUDGET_BYTES``), which saw the host's real free space, or
    else the hardware plugin's disk budget. The reading alone is never trusted:
    in a pod it is the node's disk, and a VM's disk can be a sparse file far
    larger than the host behind it."""
    candidates = [Path(w) for w in (os.environ.get("KAROTTE_WORKDIR"),) if w]
    candidates += list(memory_watch.TEMP_DIRS)
    paths = tuple(
        path for path in candidates if path.is_dir() and memory_watch.disk_backed(path)
    )
    if not paths:
        return None
    budget = int(shutil.disk_usage(paths[0]).free * _FREE_DISK_FRACTION)
    cap = _launcher_disk_budget()
    if cap is None:
        limits = hardware_limits(required_hardware)
        if limits is not None:
            cap = limits.disk_bytes
    if cap is not None:
        budget = min(budget, cap)
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
