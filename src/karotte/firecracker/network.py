"""Guest networking for Firecracker VMs, and the egress filter around it.

pasta gives the guest a network without root: Firecracker runs in a user and
network namespace from pasta, where the launcher is root, so it makes the tap,
NATs it onto pasta's interface and filters it. A jailed VMM (root) runs in a
network namespace root makes and pasta attaches to instead.

The filter runs outside the guest, so it holds for guest root too.
Guest traffic may not reach link-local addresses (the cloud metadata server),
private and shared address space (the host's LAN, a VPN) or the host
itself, except for the allowlisted model proxy. karotte's in-guest firewall
only covers the student.
"""

import ipaddress
import os
import shlex
import shutil
import socket
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import psutil
from loguru import logger

from karotte.firecracker import FirecrackerError

NETWORK_ENV_VAR = "KAROTTE_FIRECRACKER_NETWORK"
"""``pasta``, or ``none`` for a VM without a network; unset means pasta."""

DNS_ENV_VAR = "KAROTTE_FIRECRACKER_DNS"
"""Comma-separated nameservers for the guest, instead of the host's."""

BLOCKED_NETWORKS = (
    "169.254.0.0/16",
    "10.0.0.0/8",
    "172.16.0.0/12",
    "192.168.0.0/16",
    "100.64.0.0/10",
)
"""Link-local, RFC 1918 and shared address space."""

PUBLIC_DNS = ("1.1.1.1", "8.8.8.8")


GUEST_MAC = "06:00:ac:10:00:02"

PASTA_TAP = "fc-tap0"
# TEST-NET-1: never routed, so it can't collide with anything the host uses.
PASTA_HOST_IP = "192.0.2.1"
PASTA_GUEST_IP = "192.0.2.2"

_TAP_PREFIX_LEN = 30

NetworkMode = Literal["pasta", "none"]


class NetworkError(FirecrackerError):
    pass


@dataclass(frozen=True)
class Allow:
    """A destination the guest may reach despite the filter."""

    address: str
    port: int | None = None

    @classmethod
    def parse(cls, spec: str) -> "Allow":
        address, sep, port = spec.rpartition(":")
        if not sep:
            address, port = spec, ""
        try:
            ipaddress.IPv4Address(address)
        except ValueError as e:
            raise NetworkError(f"Expected IPv4[:PORT], got {spec!r}") from e
        return cls(address, int(port) if port else None)

    def __str__(self) -> str:
        return self.address if self.port is None else f"{self.address}:{self.port}"

    def rule(self) -> list[str]:
        port = ["-p", "tcp", "--dport", str(self.port)] if self.port else []
        return ["-d", f"{self.address}/32", *port, "-j", "ACCEPT"]

    def covers(self, other: "Allow") -> bool:
        return self.address == other.address and self.port in (None, other.port)


def is_blocked(address: str) -> bool:
    ip = ipaddress.IPv4Address(address)
    return any(ip in ipaddress.IPv4Network(net) for net in BLOCKED_NETWORKS)


def egress_rules(allow: Sequence[Allow], blocked: Sequence[str]) -> list[list[str]]:
    """iptables rule specs for the chain guest traffic is sent through."""
    rules = [a.rule() for a in allow]
    rules += [["-d", net, "-j", "REJECT"] for net in blocked]
    rules.append(["-j", "ACCEPT"])
    return rules


def host_ipv4_addresses() -> list[str]:
    addresses = {
        a.address
        for addrs in psutil.net_if_addrs().values()
        for a in addrs
        if a.family == socket.AF_INET and not a.address.startswith("127.")
    }
    return sorted(addresses)


def blocked_destinations(host_addresses: Iterable[str]) -> list[str]:
    return [*BLOCKED_NETWORKS, *(f"{a}/32" for a in host_addresses)]


def proxy_allows(proxy_url: str | None) -> list[Allow]:
    """The proxy's addresses as the host resolves them, with its port."""
    if not proxy_url:
        return []
    parsed = urlparse(proxy_url)
    if not parsed.hostname:
        return []
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(parsed.hostname, port, family=socket.AF_INET)
    except OSError as e:
        raise NetworkError(
            f"Cannot resolve the proxy host {parsed.hostname}: {e}"
        ) from e
    return [Allow(a, port) for a in sorted({str(i[4][0]) for i in infos})]


def proxy_hosts_lines(proxy_url: str | None, allows: Sequence[Allow]) -> list[str]:
    """/etc/hosts lines that pin the proxy host in the guest, which resolves
    through public DNS only."""
    host = urlparse(proxy_url).hostname if proxy_url else None
    if not host or not allows:
        return []
    try:
        ipaddress.ip_address(host)
        return []
    except ValueError:
        pass
    return [f"{a.address} {host}" for a in dict.fromkeys(allows)]


HOST_RESOLV_CONFS = (
    Path("/run/systemd/resolve/resolv.conf"),
    Path("/etc/resolv.conf"),
)
"""systemd-resolved's upstream servers first: /etc/resolv.conf then often
names only its 127.0.0.53 stub."""


def guest_nameservers(resolv_paths: Sequence[Path] = HOST_RESOLV_CONFS) -> list[str]:
    """The host's public nameservers, else well-known public ones: the guest
    can't reach a loopback, link-local or private resolver."""
    if override := os.environ.get(DNS_ENV_VAR):
        return [s.strip() for s in override.split(",") if s.strip()]
    for path in resolv_paths:
        try:
            lines = path.read_text().splitlines()
        except OSError:
            continue
        servers: list[str] = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "nameserver":
                try:
                    ip = ipaddress.ip_address(parts[1])
                except ValueError:
                    continue
                if ip.version == 4 and ip.is_global:
                    servers.append(parts[1])
        if servers:
            return servers
    return list(PUBLIC_DNS)


def resolv_conf(nameservers: Sequence[str]) -> str:
    return "".join(f"nameserver {s}\n" for s in nameservers)


# --- choosing and holding a network for one VM ---


@dataclass
class GuestNetwork:
    tap: str
    host_ip: str
    guest_ip: str

    def kernel_ip_arg(self) -> str:
        return f"ip={self.guest_ip}::{self.host_ip}:255.255.255.252::eth0:off"


def network_mode() -> NetworkMode:
    value = os.environ.get(NETWORK_ENV_VAR, "").strip()
    if value in ("pasta", "none"):
        return value
    if value:
        raise NetworkError(f"{NETWORK_ENV_VAR} must be pasta or none, not {value!r}")
    if shutil.which("pasta"):
        return "pasta"
    raise NetworkError(
        "A Firecracker VM needs pasta (install passt) for its network. "
        + f"Set {NETWORK_ENV_VAR}=none to run without a network."
    )


def check_pasta_allows(needed: Sequence[Allow]) -> None:
    """pasta gives its namespace the host's addresses, so a proxy on one of
    them is the namespace itself, not the host: unreachable."""
    host = set(host_ipv4_addresses())
    on_host = [a for a in needed if a.address in host]
    if on_host:
        raise NetworkError(
            f"The proxy at {', '.join(map(str, on_host))} is on this host, which a "
            + "guest can't reach through pasta. Run it on another host, or use"
            + " --runtime docker."
        )


def pasta_network() -> GuestNetwork:
    return GuestNetwork(PASTA_TAP, PASTA_HOST_IP, PASTA_GUEST_IP)


def _pasta_setup_lines(
    allow: Sequence[Allow], blocked: Sequence[str], tap_owner: tuple[int, int] | None
) -> list[str]:
    """Shell lines that make the guest's tap in the current network
    namespace, NAT it out and filter it. ``tap_owner`` (uid, gid) may open the
    tap without CAP_NET_ADMIN there, as a jailed VMM must."""
    subnet = f"{PASTA_HOST_IP}/{_TAP_PREFIX_LEN}"
    owner = "" if tap_owner is None else f" user {tap_owner[0]} group {tap_owner[1]}"
    return [
        "set -e",
        # ip and iptables can live in sbin, which a user's PATH may lack;
        # preflight looks there too.
        'PATH="$PATH:/usr/sbin:/sbin"',
        f"ip tuntap add dev {PASTA_TAP} mode tap{owner}",
        f"ip addr add {subnet} dev {PASTA_TAP}",
        f"ip link set {PASTA_TAP} up",
        "echo 1 > /proc/sys/net/ipv4/ip_forward",
        f"echo 1 > /proc/sys/net/ipv6/conf/{PASTA_TAP}/disable_ipv6",
        f"iptables -w -t nat -A POSTROUTING -s {subnet} ! -o {PASTA_TAP} -j MASQUERADE",
        "iptables -w -N KAROTTE-FC",
        *(
            f"iptables -w -A KAROTTE-FC {shlex.join(spec)}"
            for spec in egress_rules(allow, blocked)
        ),
        f"iptables -w -A FORWARD -i {PASTA_TAP} -j KAROTTE-FC",
        f"iptables -w -A INPUT -i {PASTA_TAP} -j REJECT",
    ]


def pasta_script(
    command: Sequence[str],
    pid_file: Path,
    allow: Sequence[Allow],
    blocked: Sequence[str],
) -> str:
    """Runs inside pasta's namespace: tap, NAT, filter, then ``command``."""
    lines = [
        *_pasta_setup_lines(allow, blocked, None),
        f"{shlex.join(command)} </dev/null &",
        "vmm=$!",
        f"echo $vmm > {shlex.quote(str(pid_file))}",
        # pasta drops the launcher's parent-death signal when it enters its
        # user namespace, so the launcher holds our stdin open instead: EOF
        # means it died, and the VMM goes with it. (A background job's stdin
        # is /dev/null unless redirected.)
        "exec 3<&0",
        "(cat <&3 >/dev/null; kill -9 $vmm 2>/dev/null) &",
        "set +e",
        "wait $vmm",
    ]
    return "\n".join(lines) + "\n"


def pasta_command(
    command: Sequence[str],
    pid_file: Path,
    allow: Sequence[Allow],
    blocked: Sequence[str],
) -> list[str]:
    """``command`` wrapped in pasta. No port forwarding either way, and the
    gateway isn't mapped to the host's loopback."""
    return [
        "pasta",
        "--config-net",
        "--no-map-gw",
        "--quiet",
        "-t",
        "none",
        "-u",
        "none",
        "-T",
        "none",
        "-U",
        "none",
        "--",
        "sh",
        "-c",
        pasta_script(command, pid_file, allow, blocked),
    ]


# --- pasta for a jailed VMM ---
#
# The jailer needs real root (it makes device nodes and chroots), which root in
# pasta's user namespace isn't. So root makes the network namespace instead,
# with a holder process, and pasta attaches to it from outside; the jailer then
# joins it with --netns. No unprivileged user namespace is involved.


def jail_netns_holder_command(
    allow: Sequence[Allow], blocked: Sequence[str], uid: int, gid: int
) -> list[str]:
    """A process that owns a fresh network namespace, set up for the guest
    with a tap ``uid``:``gid`` may open. It prints ``ready <pid>`` and lives
    until its stdin closes.

    It gives up its capabilities once the namespace is set up: pasta drops
    most of its own, and the kernel only lets a process open another's
    namespace when it holds every capability that one does."""
    script = "\n".join(
        [
            *_pasta_setup_lines(allow, blocked, (uid, gid)),
            "echo ready $$",
            "exec setpriv --inh-caps=-all --ambient-caps=-all --bounding-set=-all cat >/dev/null",
        ]
    )
    return ["unshare", "--net", "--", "sh", "-c", script + "\n"]


def pasta_attach_command(netns: str) -> list[str]:
    """pasta, as root, giving the network namespace at ``netns`` (a
    ``/proc/<pid>/ns/net`` path: pasta's AppArmor profile allows those) the
    host's network. It stays in the foreground, so its launcher stops it."""
    return [
        "pasta",
        "--config-net",
        "--no-map-gw",
        "--quiet",
        "--foreground",
        # Started as root, pasta would switch to nobody, which can't open
        # another process's namespace.
        "--runas",
        "0",
        "-t",
        "none",
        "-u",
        "none",
        "-T",
        "none",
        "-U",
        "none",
        "--netns",
        netns,
        "--netns-only",
    ]


def log_filter(mode: str, allow: Sequence[Allow]) -> None:
    allowed = ", ".join(map(str, allow)) or "nothing"
    logger.info(
        f"Firecracker guest network: {mode}; the host drops traffic to link-local, private and host addresses, allowing {allowed}"
    )
