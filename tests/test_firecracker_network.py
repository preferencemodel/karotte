import socket
from pathlib import Path

import pytest

from karotte.firecracker import network
from karotte.firecracker.network import (
    Allow,
    NetworkError,
    blocked_destinations,
    egress_rules,
    guest_nameservers,
    is_blocked,
    jail_netns_holder_command,
    network_mode,
    pasta_attach_command,
    pasta_command,
    proxy_allows,
    proxy_hosts_lines,
)


class TestEgressRules:
    def test_allows_come_first_then_rejects_then_accept(self):
        rules = egress_rules([Allow("10.1.2.3", 443)], ["10.0.0.0/8", "10.9.9.9/32"])

        assert rules == [
            ["-d", "10.1.2.3/32", "-p", "tcp", "--dport", "443", "-j", "ACCEPT"],
            ["-d", "10.0.0.0/8", "-j", "REJECT"],
            ["-d", "10.9.9.9/32", "-j", "REJECT"],
            ["-j", "ACCEPT"],
        ]

    def test_blocks_metadata_private_shared_and_host_addresses(self):
        blocked = blocked_destinations(["203.0.113.7"])

        for net in (
            "169.254.0.0/16",
            "10.0.0.0/8",
            "172.16.0.0/12",
            "192.168.0.0/16",
            "100.64.0.0/10",
            "203.0.113.7/32",
        ):
            assert net in blocked

    @pytest.mark.parametrize(
        ("address", "blocked"),
        [
            ("169.254.169.254", True),
            ("10.138.0.3", True),
            ("172.30.103.1", True),
            ("192.168.1.1", True),
            ("100.74.9.91", True),
            ("1.1.1.1", False),
            ("8.8.8.8", False),
        ],
    )
    def test_is_blocked(self, address: str, blocked: bool):
        assert is_blocked(address) is blocked


class TestAllow:
    def test_parse(self):
        assert Allow.parse("10.0.0.5") == Allow("10.0.0.5", None)
        assert Allow.parse("10.0.0.5:8443") == Allow("10.0.0.5", 8443)
        assert str(Allow("10.0.0.5", 8443)) == "10.0.0.5:8443"

    def test_parse_rejects_hostnames(self):
        with pytest.raises(NetworkError):
            Allow.parse("proxy.internal:443")

    def test_portless_allow_covers_any_port(self):
        assert Allow("10.0.0.5").covers(Allow("10.0.0.5", 443))
        assert not Allow("10.0.0.5", 80).covers(Allow("10.0.0.5", 443))


class TestProxy:
    def test_proxy_allows_resolve_on_the_host(self, monkeypatch: pytest.MonkeyPatch):
        def getaddrinfo(host: str, port: int, family: int):
            assert (host, family) == ("proxy.example", socket.AF_INET)
            return [(family, 0, 0, "", ("10.0.0.5", port))]

        monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)

        assert proxy_allows("https://proxy.example") == [Allow("10.0.0.5", 443)]
        assert proxy_allows("http://proxy.example") == [Allow("10.0.0.5", 80)]
        assert proxy_allows("http://proxy.example:4000/v1") == [Allow("10.0.0.5", 4000)]
        assert proxy_allows(None) == []

    def test_proxy_host_is_pinned_in_the_guest(self):
        allows = [Allow("10.0.0.5", 443), Allow("10.0.0.6", 443)]

        assert proxy_hosts_lines("https://proxy.example", allows) == [
            "10.0.0.5 proxy.example",
            "10.0.0.6 proxy.example",
        ]
        assert proxy_hosts_lines("https://10.0.0.5", allows) == []
        assert proxy_hosts_lines(None, allows) == []

    def test_pasta_refuses_a_proxy_on_the_host(self, monkeypatch: pytest.MonkeyPatch):
        """pasta's namespace has the host's addresses, so the host's proxy
        address is the namespace itself."""
        monkeypatch.setattr(network, "host_ipv4_addresses", lambda: ["203.0.113.9"])

        with pytest.raises(NetworkError, match="on this host"):
            network.check_pasta_allows([Allow("203.0.113.9", 8080)])
        network.check_pasta_allows([Allow("198.51.100.4", 443)])


class TestNameservers:
    def test_only_public_host_resolvers(self, tmp_path: Path):
        resolved = tmp_path / "resolved.conf"
        resolved.write_text("nameserver 169.254.169.254\nnameserver 10.0.0.2\n")
        etc = tmp_path / "resolv.conf"
        etc.write_text("nameserver 127.0.0.53\nnameserver 9.9.9.9\nsearch x\n")

        assert guest_nameservers([resolved, etc]) == ["9.9.9.9"]

    def test_public_fallback(self, tmp_path: Path):
        etc = tmp_path / "resolv.conf"
        etc.write_text("nameserver 127.0.0.53\n")

        assert guest_nameservers([tmp_path / "missing", etc]) == ["1.1.1.1", "8.8.8.8"]

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv(network.DNS_ENV_VAR, "10.0.0.2, 10.0.0.3")

        assert guest_nameservers([]) == ["10.0.0.2", "10.0.0.3"]


class TestNetworkMode:
    def test_env_selects(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv(network.NETWORK_ENV_VAR, "none")
        assert network_mode() == "none"
        monkeypatch.setenv(network.NETWORK_ENV_VAR, "tap")
        with pytest.raises(NetworkError, match="pasta or none"):
            _ = network_mode()

    def test_pasta_when_installed(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv(network.NETWORK_ENV_VAR, raising=False)

        def which(name: str) -> str:
            return f"/usr/bin/{name}"

        monkeypatch.setattr("karotte.firecracker.network.shutil.which", which)
        assert network_mode() == "pasta"

    def test_no_pasta(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv(network.NETWORK_ENV_VAR, raising=False)

        def which(_name: str) -> None:
            return None

        monkeypatch.setattr("karotte.firecracker.network.shutil.which", which)
        with pytest.raises(NetworkError, match="install passt"):
            _ = network_mode()


class TestJailedPasta:
    def test_the_holder_makes_a_tap_the_jailed_user_may_open(self):
        argv = jail_netns_holder_command(
            [Allow("10.0.0.5", 443)], ["10.0.0.0/8"], 1003, 1005
        )
        script = argv[-1].splitlines()

        assert argv[:4] == ["unshare", "--net", "--", "sh"]
        assert "ip tuntap add dev fc-tap0 mode tap user 1003 group 1005" in script
        assert any("iptables -w -A KAROTTE-FC" in line for line in script)
        assert script[-2] == "echo ready $$"
        # Without its capabilities, so pasta may open its namespace.
        assert script[-1].startswith("exec setpriv --inh-caps=-all")

    def test_pasta_attaches_as_root_to_the_namespace_only(self):
        argv = pasta_attach_command("/proc/42/ns/net")

        assert argv[argv.index("--runas") + 1] == "0"
        assert argv[argv.index("--netns") + 1] == "/proc/42/ns/net"
        assert "--netns-only" in argv and "--foreground" in argv
        for flag in ("-t", "-u", "-T", "-U"):
            assert argv[argv.index(flag) + 1] == "none"


class TestPasta:
    def test_the_script_finds_tools_in_sbin(self, tmp_path: Path):
        """Preflight accepts ip and iptables from sbin, which a user's PATH
        may lack."""
        script = pasta_command(["firecracker"], tmp_path / "vmm.pid", [], [])[-1]
        lines = script.splitlines()
        assert lines.index('PATH="$PATH:/usr/sbin:/sbin"') < next(
            i for i, line in enumerate(lines) if line.startswith("ip ")
        )

    def test_no_forwarding_and_no_host_loopback(self, tmp_path: Path):
        argv = pasta_command(["firecracker", "--no-api"], tmp_path / "vmm.pid", [], [])

        assert argv[:2] == ["pasta", "--config-net"]
        assert "--no-map-gw" in argv
        for flag in ("-t", "-u", "-T", "-U"):
            assert argv[argv.index(flag) + 1] == "none"
        assert argv[argv.index("--") + 1 : argv.index("--") + 3] == ["sh", "-c"]

    def test_script_filters_then_runs_the_vmm(self, tmp_path: Path):
        script = pasta_command(
            ["firecracker", "--config-file", "/run dir/fc.json"],
            tmp_path / "vmm.pid",
            [Allow("10.0.0.5", 443)],
            ["169.254.0.0/16", "10.138.0.3/32"],
        )[-1]
        lines = script.splitlines()

        assert lines[0] == "set -e"
        chain = [
            line for line in lines if line.startswith("iptables -w -A KAROTTE-FC ")
        ]
        assert chain == [
            "iptables -w -A KAROTTE-FC -d 10.0.0.5/32 -p tcp --dport 443 -j ACCEPT",
            "iptables -w -A KAROTTE-FC -d 169.254.0.0/16 -j REJECT",
            "iptables -w -A KAROTTE-FC -d 10.138.0.3/32 -j REJECT",
            "iptables -w -A KAROTTE-FC -j ACCEPT",
        ]
        assert "iptables -w -A FORWARD -i fc-tap0 -j KAROTTE-FC" in lines
        assert "iptables -w -A INPUT -i fc-tap0 -j REJECT" in lines
        assert "firecracker --config-file '/run dir/fc.json' </dev/null &" in lines
        # The VMM is killed when the launcher's end of stdin closes.
        assert "exec 3<&0" in lines
        assert lines[-1] == "wait $vmm"
