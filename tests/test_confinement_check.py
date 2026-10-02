import json
import os
import resource
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer
from rich.console import Console

from karotte import confinement_check
from karotte.cgroups import Mount, V1Cgroup
from karotte.cli.check import confinement
from karotte.confinement import GIB, Contract, Sandbox
from karotte.confinement_check import (
    Finding,
    Observations,
    SessionReport,
    evaluate,
    parse_session_output,
    passed,
    render_table,
    report_json,
)
from karotte.hardware import HardwareLimits
from tests.conftest import register_hardware_plugins

STUDENT_GROUP = "/karotte_uid_1000"


def _vm_observations(**changes: object) -> Observations:
    """What a `vm` sandbox that gives the student everything reports."""
    obs = Observations(
        sandbox="vm",
        containerized=True,
        student_uid=1000,
        cgroup_version=2,
        cgroup_writable=True,
        contracts={
            "memory": Contract.PREVENTED.value,
            "processes": Contract.PREVENTED.value,
            "files": Contract.PREVENTED.value,
        },
        readback={
            "memory": confinement_check.PROBE_MEMORY_BYTES,
            "processes": confinement_check.PROBE_PROCESS_COUNT,
        },
        quota_backing="loop ext4",
        firewall_took=True,
        reachable_without_rules=["1.1.1.1:80"],
        pidns_allowed=True,
        pidns_available=True,
        student_cgroup=STUDENT_GROUP,
        session_limits={"memory": 4 * GIB, "processes": 2048},
        session=SessionReport(
            uid=1000,
            pid=1,
            cgroup=STUDENT_GROUP,
            oom_score_adj=1000,
            memory_max=str(4 * GIB),
            pids_max="2048",
            oom_group="1",
        ),
        ram_bytes=6 * GIB,
        student_memory_default=4 * GIB,
    )
    return replace(obs, **changes)


def _weak(obs: Observations) -> set[str]:
    return {f.name for f in evaluate(obs) if f.ok is False}


def test_a_vm_that_gives_everything_passes() -> None:
    findings = evaluate(_vm_observations())
    assert passed(findings)
    assert {f.name for f in findings if f.ok} >= {
        "cgroup",
        "memory limit",
        "files limit",
        "firewall took",
        "reached with rules",
        "PID namespace",
        "session memory.max",
    }


def test_a_session_outside_the_student_group_is_weak() -> None:
    """The bug this check exists for: a session in the unlimited harness leaf."""
    obs = _vm_observations(
        session=SessionReport(
            uid=1000,
            pid=1,
            cgroup="/karotte_harness",
            oom_score_adj=1000,
            memory_max="max",
            pids_max="max",
            oom_group="0",
        )
    )
    assert _weak(obs) == {
        "session cgroup",
        "session memory.max",
        "session pids.max",
        "session memory.oom.group",
    }


@pytest.mark.parametrize(
    ("changes", "weak"),
    [
        ({"cgroup_writable": False}, {"cgroup"}),
        ({"cgroup_version": 1}, {"cgroup"}),
        ({"pidns_available": False}, {"PID namespace"}),
        ({"firewall_took": False}, {"firewall took"}),
        ({"reachable_with_rules": ["10.0.0.1:53"]}, {"reached with rules"}),
        ({"ram_bytes": 4 * GIB}, {"RAM vs student memory.max"}),
        (
            {
                "contracts": {
                    "memory": "prevented",
                    "processes": "prevented",
                    "files": "detected_and_reaped",
                }
            },
            {"files limit"},
        ),
        ({"readback": {"memory": None, "processes": 64}}, {"memory readback"}),
        ({"readback": {"memory": 1, "processes": 64}}, {"memory readback"}),
    ],
)
def test_a_vm_short_of_one_mechanism_is_weak_there(
    changes: dict[str, object], weak: set[str]
) -> None:
    assert _weak(_vm_observations(**changes)) == weak


def _ram_finding(obs: Observations) -> Finding:
    return next(f for f in evaluate(obs) if f.name == "RAM vs student memory.max")


def test_a_karotte_vm_without_headroom_is_weak() -> None:
    """RAM of exactly the student's limit plus the harness reserve is the
    size where a student spread over many processes got the harness killed
    by a global OOM."""
    obs = _vm_observations(ram_bytes=5 * GIB, vm_launcher="firecracker")
    assert _weak(obs) == {"RAM vs student memory.max"}


def test_a_karotte_vm_passes_despite_the_firmware_holes() -> None:
    """A 6 GiB guest's RAM map has a few holes below 1 MiB."""
    obs = _vm_observations(ram_bytes=6 * GIB - (1 << 20), vm_launcher="firecracker")
    assert _ram_finding(obs).ok is True


FIRECRACKER_IOMEM = """\
00000000-00000fff : Reserved
00001000-0009fbff : System RAM
0009fc00-000fffff : Reserved
00100000-bfffffff : System RAM
  01000000-0230ffff : Kernel code
c0001000-c0001fff : virtio-mmio.0
100000000-17fffffff : System RAM
"""


def test_installed_ram_is_read_from_iomem(tmp_path: Path) -> None:
    """A 5 GiB Firecracker guest, where MemTotal says 4.8 GiB."""
    iomem = tmp_path / "iomem"
    _ = iomem.write_text(FIRECRACKER_IOMEM)

    ram = confinement_check.installed_ram_bytes(iomem)

    assert 5 * GIB - (1 << 20) < ram <= 5 * GIB


def test_installed_ram_without_root_falls_back_to_memtotal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    iomem = tmp_path / "iomem"
    _ = iomem.write_text(
        "00000000-00000000 : System RAM\n00000000-00000000 : System RAM\n"
    )
    monkeypatch.setattr(
        "karotte.confinement_check.psutil.virtual_memory",
        lambda: SimpleNamespace(total=1234),
    )

    assert confinement_check.installed_ram_bytes(iomem) == 1234


def test_a_vm_another_launcher_sized_without_headroom_only_warns() -> None:
    """Its RAM is the outer harness's choice; the check must not fail it."""
    obs = _vm_observations(ram_bytes=5 * GIB, vm_launcher=None)
    finding = _ram_finding(obs)
    assert finding.ok is None
    assert "warning" in finding.value
    assert passed(evaluate(obs))


def test_a_vm_short_of_the_harness_reserve_is_weak_whoever_launched_it() -> None:
    obs = _vm_observations(ram_bytes=4 * GIB, vm_launcher=None)
    assert _weak(obs) == {"RAM vs student memory.max"}


def test_a_root_student_is_weak() -> None:
    """KAROTTE_DEMOTE_ID=0 leaves no privilege separation to certify."""
    obs = _vm_observations(
        student_uid=0, session=replace(_vm_observations().session, uid=0)
    )
    assert {"student uid", "session uid"} <= _weak(obs)


def test_memory_max_is_compared_as_the_kernel_stores_it() -> None:
    """The kernel rounds a cap down to a page: 4G as k8s spells it reads back
    lower."""
    applied = 4_000_000_000
    page = resource.getpagesize()
    obs = _vm_observations(
        session_limits={"memory": applied, "processes": 2048},
        session=replace(
            _vm_observations().session, memory_max=str(applied // page * page)
        ),
    )
    assert "session memory.max" not in _weak(obs)


def test_a_firewall_check_that_could_not_run_is_weak() -> None:
    """Reaching nothing because nothing was tried proves nothing."""
    obs = _vm_observations(firewall_check_error="helper died")
    finding = next(f for f in evaluate(obs) if f.name == "reached with rules")
    assert finding.ok is False
    assert "helper died" in finding.value


def test_a_session_that_did_not_run_is_weak() -> None:
    obs = _vm_observations(session=SessionReport(error="exited 1: no unshare"))
    assert _weak(obs) == {"student session"}


def test_the_student_oom_score_is_checked() -> None:
    obs = _vm_observations(session=replace(_vm_observations().session, oom_score_adj=0))
    assert _weak(obs) == {"session oom_score_adj"}


def test_runc_expects_no_pid_namespace_or_cgroup_v2() -> None:
    obs = _vm_observations(
        sandbox=Sandbox.RUNC.value,
        cgroup_version=1,
        pidns_allowed=False,
        pidns_available=False,
        ram_bytes=1 * GIB,
        contracts={
            "memory": Contract.REAPED.value,
            "processes": Contract.REAPED.value,
            "files": Contract.REAPED.value,
        },
        readback={},
        student_cgroup=None,
        session_limits={},
        session=replace(_vm_observations().session, cgroup="/", memory_max="max"),
    )
    findings = evaluate(obs)
    assert passed(findings)
    assert next(f for f in findings if f.name == "PID namespace").value == (
        "no (not used in this sandbox)"
    )


def test_runc_without_any_memory_mechanism_is_weak() -> None:
    obs = _vm_observations(
        sandbox=Sandbox.RUNC.value,
        contracts={"processes": "prevented", "files": "prevented"},
    )
    assert _weak(obs) == {"memory limit"}


def test_a_tolerated_firewall_is_information() -> None:
    obs = _vm_observations(
        sandbox=Sandbox.RUNC.value, firewall_took=False, firewall_tolerated=True
    )
    finding = next(f for f in evaluate(obs) if f.name == "firewall took")
    assert finding.ok is None
    assert "KAROTTE_FIREWALL_TOLERATE" in finding.value


def test_gvisor_expects_no_firewall() -> None:
    obs = _vm_observations(
        sandbox=Sandbox.GVISOR.value,
        cgroup_version=None,
        cgroup_writable=False,
        firewall_took=False,
        contracts={
            "memory": Contract.REAPED.value,
            "processes": Contract.REAPED.value,
            "files": Contract.REAPED.value,
        },
        student_cgroup=None,
    )
    assert passed(evaluate(obs))


def test_the_table_marks_weak_rows() -> None:
    console = Console(width=200, record=True)
    console.print(render_table(evaluate(_vm_observations(pidns_available=False))))
    text = console.export_text()
    assert "PID namespace" in text
    assert "WEAK" in text


def test_the_json_report_carries_findings_and_observations() -> None:
    obs = _vm_observations(firewall_took=False)
    report = json.loads(json.dumps(report_json(obs, evaluate(obs))))
    assert report["ok"] is False
    assert report["observations"]["session"]["memory_max"] == str(4 * GIB)
    assert {"name": "firewall took", "value": "no", "ok": False} in report["findings"]


class TestParseSessionOutput:
    def test_v2(self) -> None:
        report = parse_session_output(
            "cgroup=0::/karotte_uid_1000\n"
            + "status=Uid:\t1000\t1000\t1000\t1000\n"
            + "oom_score_adj=1000\n"
            + "pid=1\n"
        )
        assert report == SessionReport(
            uid=1000, pid=1, cgroup="/karotte_uid_1000", oom_score_adj=1000
        )

    def test_v1_uses_the_memory_hierarchy(self) -> None:
        report = parse_session_output(
            "cgroup=4:pids:/karotte_uid_1000\ncgroup=3:memory:/karotte_uid_1000\n"
            + "cgroup=1:name=systemd:/\npid=7\n"
        )
        assert report.cgroup == "/karotte_uid_1000"
        assert report.error is None

    def test_a_hybrid_host_uses_the_v1_memory_hierarchy(self) -> None:
        """Controllers on v1 and a bare cgroup2: the 0:: group holds no limits."""
        text = "cgroup=3:memory:/karotte_uid_1000\ncgroup=0::/user.slice\npid=7\n"
        assert (
            parse_session_output(text, cgroup_version=1).cgroup == "/karotte_uid_1000"
        )
        assert parse_session_output(text).cgroup == "/user.slice"

    def test_no_cgroup_is_an_error(self) -> None:
        assert parse_session_output("pid=1\n").error is not None


def _reach_everything(
    uid: int, targets: list[tuple[str, int]]
) -> list[tuple[str, int]]:
    del uid
    return targets


class FakeFirewall:
    def __init__(self, sandbox: Sandbox) -> None:
        del sandbox
        self.lifted: list[int] = []
        FakeFirewall.last = self

    last: "FakeFirewall"

    def restrict_to_internal_network(self, uid: int, **_: object) -> bool:
        raise RuntimeError(f"failed to apply firewall rule for {uid}")

    def lift_network_rules(self, uid: int) -> bool:
        self.lifted.append(uid)
        return False


def test_the_firewall_probe_lifts_its_rules_even_when_one_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(confinement_check, "Confinement", FakeFirewall)
    monkeypatch.setattr(
        confinement_check, "firewall_canaries", lambda: [("1.1.1.1", 80)]
    )
    monkeypatch.setattr(confinement_check, "reachable_as", _reach_everything)
    obs = _vm_observations(firewall_took=False, reachable_without_rules=[])

    confinement_check._probe_firewall(obs, Sandbox.VM, 60000)  # pyright: ignore[reportPrivateUsage]

    assert FakeFirewall.last.lifted == [60000]
    assert obs.reachable_without_rules == ["1.1.1.1:80"]
    assert obs.firewall_error == "failed to apply firewall rule for 60000"
    assert obs.left_behind == ["firewall rules for probe uid 60000"]
    assert "firewall took" in _weak(obs)


class FakeFirewallThatTakes(FakeFirewall):
    def restrict_to_internal_network(self, uid: int, **_: object) -> bool:
        del uid
        return True


def test_a_firewall_probe_whose_second_run_fails_is_not_a_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(confinement_check, "Confinement", FakeFirewallThatTakes)
    monkeypatch.setattr(
        confinement_check, "firewall_canaries", lambda: [("1.1.1.1", 80)]
    )
    runs: list[int] = []

    def reach(uid: int, targets: list[tuple[str, int]]) -> list[tuple[str, int]]:
        runs.append(uid)
        if len(runs) > 1:
            raise RuntimeError("The firewall self-test as uid 60000 could not run")
        return targets

    monkeypatch.setattr(confinement_check, "reachable_as", reach)
    obs = _vm_observations(firewall_took=False, reachable_without_rules=[])

    confinement_check._probe_firewall(obs, Sandbox.VM, 60000)  # pyright: ignore[reportPrivateUsage]

    assert obs.firewall_took
    assert obs.firewall_check_error is not None
    assert "reached with rules" in _weak(obs)


def test_it_is_a_subcommand_of_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """`karotte check confinement`; plain `karotte check` still checks the
    environment."""
    from typer.testing import CliRunner

    from karotte.cli import app

    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    result = CliRunner().invoke(app, ["check", "confinement", "--json"])
    assert result.exit_code == 1
    assert "must run as root" in result.output


class TestCli:
    @pytest.fixture
    def as_root(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(os, "geteuid", lambda: 0)

    def test_refuses_without_root(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(os, "geteuid", lambda: 1000)
        with pytest.raises(typer.Exit) as exc_info:
            confinement()
        assert exc_info.value.exit_code == 1

    @pytest.mark.usefixtures("as_root")
    def test_exits_nonzero_when_weak(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(
            confinement_check,
            "gather",
            lambda _hardware: _vm_observations(pidns_available=False),  # pyright: ignore[reportUnknownLambdaType]
        )
        with pytest.raises(typer.Exit) as exc_info:
            confinement(json_output=True)
        assert exc_info.value.exit_code == 1
        assert json.loads(capsys.readouterr().out)["ok"] is False

    @pytest.mark.usefixtures("as_root")
    def test_passes_with_the_hardware_passed_through(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        register_hardware_plugins(
            monkeypatch,
            limits={"a": lambda hw: HardwareLimits() if hw == "cpu-4" else None},  # pyright: ignore[reportUnknownLambdaType]
        )
        asked: list[str] = []

        def fake_gather(hardware: str) -> Observations:
            asked.append(hardware)
            return _vm_observations()

        monkeypatch.setattr(confinement_check, "gather", fake_gather)
        confinement(hardware="cpu-4")
        assert asked == ["cpu-4"]
        assert "Student confinement" in capsys.readouterr().out

    @pytest.mark.usefixtures("as_root")
    def test_unknown_hardware_is_refused_before_probing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A typo would otherwise check against a default nobody asked for."""
        register_hardware_plugins(
            monkeypatch,
            limits={"a": lambda hw: HardwareLimits() if hw == "cpu-4" else None},  # pyright: ignore[reportUnknownLambdaType]
        )
        monkeypatch.setattr(confinement_check, "gather", pytest.fail)
        with pytest.raises(typer.Exit) as exc_info:
            confinement(hardware="cpu-2")
        assert exc_info.value.exit_code == 1


def test_the_layout_follows_the_backend_karotte_confines_with(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hybrid host with a writable bare cgroup2 confines with v1."""
    monkeypatch.setattr(confinement_check, "detect_cgroups", lambda _m: V1Cgroup({}))  # pyright: ignore[reportUnknownLambdaType]
    v2 = Mount(Path("/sys/fs/cgroup/unified"), "cgroup2", frozenset(), False)
    v1 = Mount(Path("/sys/fs/cgroup/memory"), "cgroup", frozenset({"memory"}), False)
    assert confinement_check._cgroup_layout([v2, v1]) == (1, True)  # pyright: ignore[reportPrivateUsage]


def test_left_behind_is_reported(tmp_path: Path) -> None:
    obs = _vm_observations(left_behind=[f"probe quota mounts under {tmp_path}"])
    finding = next(f for f in evaluate(obs) if f.name == "left behind")
    assert str(tmp_path) in finding.value
