"""`karotte check confinement`: what this sandbox does to the student, measured.

Every probe undoes what it did. Limits and firewall rules are tried on an
unused uid, never the student's; the one real student session runs under the
student's own group with the default limits, which are put back afterwards.
"""

from __future__ import annotations

import os
import pwd
import resource
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import psutil
from loguru import logger
from rich.markup import escape
from rich.table import Table

from karotte import file_quota
from karotte import subprocess as student_subprocess
from karotte.cgroups import (
    HARNESS_LEAF,
    Mount,
    V1Cgroup,
    V2Cgroup,
    detect_cgroups,
    own_cgroup_dir,
    read_mounts,
    unregister_student_cgroup,
)
from karotte.confinement import (
    FIREWALL_TOLERATE_ENV_VAR,
    HARNESS_RESERVE_BYTES,
    STUDENT_PROCESS_LIMIT,
    VM_LAUNCHER_ENV_VAR,
    CgroupConfinement,
    Confinement,
    Contract,
    Sandbox,
    _confinement_for,  # pyright: ignore[reportPrivateUsage]
    build_confinement,
    current_sandbox,
    firewall_canaries,
    get_confinement,
    prepare_vm_guest,
    reachable_as,
    sandbox_memory_bytes,
)
from karotte.container import demoted_uid_gid, is_containerized
from karotte.hardware import VM_MEMORY_HEADROOM_BYTES
from karotte.subprocess import (
    STUDENT_OOM_SCORE_ADJ,
    student_env,
    student_session_command,
)
from karotte.trusted_bin import trusted_binary

PROBE_MEMORY_BYTES = 64 << 20
PROBE_PROCESS_COUNT = 64
PROBE_QUOTA_BYTES = 16 << 20
PROBE_QUOTA_FILES = 256
_SESSION_TIMEOUT_SECONDS = 30

_PROBE_UIDS = range(60000, 61000)

_GUEST_KERNEL_RESERVED_BYTES = 256 << 20
"""Allowance for what a guest kernel keeps out of the RAM it reports: a 5
GiB Apple `container` VM shows 5053 MiB."""

# Only shell builtins: the student may not be able to read the harness's
# Python, and nothing on disk needs to be trusted to echo these back.
_SESSION_SCRIPT = """\
while IFS= read -r l; do echo "cgroup=$l"; done < /proc/self/cgroup
while IFS= read -r l; do case "$l" in Uid:*) echo "status=$l";; esac; done < /proc/self/status
read -r o < /proc/self/oom_score_adj; echo "oom_score_adj=$o"
echo "pid=$$"
"""


@dataclass
class SessionReport:
    """What one real student session saw of itself."""

    uid: int | None = None
    pid: int | None = None
    cgroup: str | None = None
    """The session's cgroup, as its ``/proc/self/cgroup`` names it."""
    oom_score_adj: int | None = None
    memory_max: str | None = None
    pids_max: str | None = None
    oom_group: str | None = None
    error: str | None = None


@dataclass
class Observations:
    """Everything the probes measured, before any judgement."""

    sandbox: str
    containerized: bool
    student_uid: int | None
    cgroup_version: int | None
    cgroup_writable: bool
    contracts: dict[str, str] = field(default_factory=dict)
    """The contract each limit got when applied to a probe uid."""
    readback: dict[str, int | None] = field(default_factory=dict)
    """The probe limits as read back from the enforcing mechanism."""
    quota_backing: str = "none"
    firewall_took: bool = False
    firewall_tolerated: bool = False
    firewall_error: str | None = None
    firewall_check_error: str | None = None
    """Why the probe with the rules in place couldn't run, if it couldn't."""
    reachable_without_rules: list[str] = field(default_factory=list)
    reachable_with_rules: list[str] = field(default_factory=list)
    pidns_allowed: bool = False
    pidns_available: bool = False
    student_cgroup: str | None = None
    """The group a student session must land in, ``None`` without one."""
    session_limits: dict[str, int | None] = field(default_factory=dict)
    session: SessionReport = field(default_factory=SessionReport)
    ram_bytes: int | None = None
    student_memory_default: int | None = None
    vm_launcher: str | None = None
    """The karotte runtime that launched this VM, ``None`` for any other launcher."""
    left_behind: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Finding:
    name: str
    value: str
    ok: bool | None
    """``False`` is weaker than the sandbox should give; ``None`` is information."""


@dataclass(frozen=True)
class Expectations:
    """What a sandbox must give the student."""

    cgroup_v2_writable: bool
    prevented: frozenset[str]
    """Limits the kernel must enforce; the others must at least be reaped."""
    firewall: bool
    pid_namespace: bool
    ram_headroom: bool
    oom_score_adj: bool


EXPECTATIONS = {
    Sandbox.VM: Expectations(
        cgroup_v2_writable=True,
        prevented=frozenset({"memory", "processes", "files"}),
        firewall=True,
        pid_namespace=True,
        ram_headroom=True,
        oom_score_adj=True,
    ),
    Sandbox.RUNC: Expectations(
        cgroup_v2_writable=False,
        prevented=frozenset(),
        firewall=True,
        pid_namespace=False,
        ram_headroom=False,
        oom_score_adj=True,
    ),
    # gVisor accepts iptables rules and cgroup writes without enforcing them;
    # student sessions get network and PID namespaces instead.
    Sandbox.GVISOR: Expectations(
        cgroup_v2_writable=False,
        prevented=frozenset(),
        firewall=False,
        pid_namespace=True,
        ram_headroom=False,
        oom_score_adj=False,
    ),
}

LIMITS = ("memory", "processes", "files")


def evaluate(obs: Observations) -> list[Finding]:
    """Judge the observations against what their sandbox should give."""
    expect = EXPECTATIONS[Sandbox(obs.sandbox)]
    findings = [
        Finding("sandbox", obs.sandbox, None),
        Finding("containerized", _yes_no(obs.containerized), obs.containerized),
        Finding(
            "student uid",
            str(obs.student_uid) if obs.student_uid is not None else "none",
            # A student running as root is no student.
            obs.student_uid is not None and obs.student_uid != 0,
        ),
    ]

    cgroup = (
        f"v{obs.cgroup_version}, {'writable' if obs.cgroup_writable else 'read-only'}"
        if obs.cgroup_version is not None
        else "none"
    )
    findings.append(
        Finding(
            "cgroup",
            cgroup,
            (obs.cgroup_version == 2 and obs.cgroup_writable)
            if expect.cgroup_v2_writable
            else None,
        )
    )

    for limit in LIMITS:
        contract = obs.contracts.get(limit, Contract.UNSUPPORTED.value)
        if limit in expect.prevented:
            ok = contract == Contract.PREVENTED
        else:
            ok = contract != Contract.UNSUPPORTED
        value = contract
        if limit in obs.readback:
            readback = obs.readback[limit]
            shown = _amount(readback) if limit == "memory" else str(readback)
            value += f" (read back {shown if readback is not None else 'no limit'})"
        findings.append(Finding(f"{limit} limit", value, ok))
    for limit, applied in (
        ("memory", PROBE_MEMORY_BYTES),
        ("processes", PROBE_PROCESS_COUNT),
    ):
        if (
            obs.contracts.get(limit) == Contract.PREVENTED
            and limit in obs.readback
            and obs.readback[limit] != applied
        ):
            findings.append(
                Finding(
                    f"{limit} readback",
                    f"asked {applied}, got {obs.readback[limit]}",
                    False,
                )
            )
    findings.append(Finding("file quota backing", obs.quota_backing, None))

    firewall_needed = expect.firewall and not obs.firewall_tolerated
    firewall = _yes_no(obs.firewall_took)
    if obs.firewall_error:
        firewall += f" ({obs.firewall_error})"
    elif obs.firewall_tolerated and not obs.firewall_took:
        firewall += f" ({FIREWALL_TOLERATE_ENV_VAR} set)"
    findings.append(
        Finding(
            "firewall took", firewall, obs.firewall_took if firewall_needed else None
        )
    )
    findings.append(
        Finding(
            "reached without rules",
            ", ".join(obs.reachable_without_rules) or "nothing",
            None,
        )
    )
    if obs.firewall_took and obs.firewall_check_error is not None:
        # Nothing reached because nothing was tried proves nothing.
        findings.append(
            Finding(
                "reached with rules",
                f"not checked ({obs.firewall_check_error})",
                False,
            )
        )
    elif obs.firewall_took:
        findings.append(
            Finding(
                "reached with rules",
                ", ".join(obs.reachable_with_rules) or "nothing",
                not obs.reachable_with_rules,
            )
        )

    pidns = _yes_no(obs.pidns_available)
    if not obs.pidns_allowed:
        pidns += " (not used in this sandbox)"
    findings.append(
        Finding(
            "PID namespace",
            pidns,
            obs.pidns_available if expect.pid_namespace else None,
        )
    )

    findings.extend(_session_findings(obs, expect))

    if obs.ram_bytes is not None and obs.student_memory_default is not None:
        findings.append(
            _ram_finding(obs, obs.ram_bytes, obs.student_memory_default, expect)
        )

    findings.append(
        Finding("left behind", "; ".join(obs.left_behind) or "nothing", None)
    )
    return findings


def _ram_finding(
    obs: Observations, ram: int, student_default: int, expect: Expectations
) -> Finding:
    """A VM needs RAM for the student, the harness reserve and the guest
    kernel: sized to exactly the first two, a student spread over many
    processes pushes the guest into a global OOM that can kill the harness.
    karotte's launchers hold the headroom, so a VM one of them launched fails
    without it. A VM someone else launched and sized only gets a warning."""
    needed = student_default + HARNESS_RESERVE_BYTES
    value = f"{_gib(ram)} vs {_gib(student_default)}"
    if not expect.ram_headroom:
        return Finding(
            "RAM vs student memory.max",
            value + f" (needs {_gib(needed)} with the harness reserve)",
            None,
        )
    with_headroom = needed + VM_MEMORY_HEADROOM_BYTES
    value += (
        f" (needs {_gib(with_headroom)} with the harness reserve and"
        + f" {_gib(VM_MEMORY_HEADROOM_BYTES)} VM headroom)"
    )
    if ram < needed:
        return Finding("RAM vs student memory.max", value, False)
    if ram >= with_headroom - _GUEST_KERNEL_RESERVED_BYTES:
        return Finding("RAM vs student memory.max", value, True)
    if obs.vm_launcher is not None:
        return Finding("RAM vs student memory.max", value, False)
    warning = "no VM headroom; a student spread over many processes can get the harness OOM-killed"
    logger.warning(f"{warning} ({value})")
    return Finding("RAM vs student memory.max", f"{value}; warning: {warning}", None)


def _session_findings(obs: Observations, expect: Expectations) -> list[Finding]:
    session = obs.session
    if session.error is not None:
        return [Finding("student session", session.error, False)]
    findings = [
        Finding(
            "session uid",
            str(session.uid),
            session.uid == obs.student_uid and session.uid != 0,
        ),
        Finding(
            "session cgroup",
            session.cgroup or "unknown",
            session.cgroup == obs.student_cgroup
            if obs.student_cgroup is not None
            else None,
        ),
        Finding("session pid", str(session.pid), None),
    ]
    applied_memory = obs.session_limits.get("memory")
    applied_processes = obs.session_limits.get("processes")
    in_group = obs.student_cgroup is not None
    findings += [
        Finding(
            "session memory.max",
            session.memory_max or "unknown",
            session.memory_max == str(_page_floor(applied_memory))
            if in_group and applied_memory is not None
            else None,
        ),
        Finding(
            "session pids.max",
            session.pids_max or "unknown",
            session.pids_max == str(applied_processes)
            if in_group and applied_processes is not None
            else None,
        ),
        Finding(
            "session oom_score_adj",
            str(session.oom_score_adj),
            session.oom_score_adj == STUDENT_OOM_SCORE_ADJ
            if expect.oom_score_adj
            else None,
        ),
        Finding(
            "session memory.oom.group",
            session.oom_group or "unknown",
            session.oom_group == "1" if in_group and obs.cgroup_version == 2 else None,
        ),
    ]
    return findings


def _page_floor(nbytes: int) -> int:
    """``nbytes`` as the kernel stores a memory cap: rounded down to a page."""
    page = resource.getpagesize()
    return nbytes // page * page


def passed(findings: list[Finding]) -> bool:
    return all(f.ok is not False for f in findings)


def gather(hardware: str | None) -> Observations:
    """Run every probe, undoing each one; see the module docstring."""
    sandbox = current_sandbox()
    # What a run does first, so the check sees the cgroups a run would.
    guest_setup = prepare_vm_guest()
    try:
        student_uid = demoted_uid_gid()
    except RuntimeError as exc:
        logger.error(f"No student uid: {exc}")
        student_uid = None
    # Before anything builds the student's confinement (reading the sandbox's
    # memory may), which creates its group.
    group_existed = student_uid is not None and _student_group_exists(student_uid)
    mounts = read_mounts()
    # Before the same can happen for the harness: building a confinement moves
    # it into its leaf.
    harness_leaves_before = _harness_leaves(mounts)
    version, writable = _cgroup_layout(mounts)
    obs = Observations(
        sandbox=sandbox.value,
        containerized=is_containerized(),
        student_uid=student_uid,
        cgroup_version=version,
        cgroup_writable=writable,
        firewall_tolerated=bool(os.environ.get(FIREWALL_TOLERATE_ENV_VAR)),
        ram_bytes=psutil.virtual_memory().total,
        vm_launcher=os.environ.get(VM_LAUNCHER_ENV_VAR) or None,
    )
    ram = sandbox_memory_bytes(hardware)
    obs.student_memory_default = (
        ram - HARNESS_RESERVE_BYTES if ram is not None else None
    )

    probe_uid = _unused_uid(student_uid)
    _probe_contracts(obs, sandbox, probe_uid)
    _probe_quota(obs)
    _probe_firewall(obs, sandbox, probe_uid)
    if student_uid is not None:
        obs.pidns_allowed = student_subprocess._kernel_is_not_the_hosts()  # pyright: ignore[reportPrivateUsage]
        obs.pidns_available = student_subprocess._pid_namespace_available(  # pyright: ignore[reportPrivateUsage]
            student_uid
        )
        _probe_session(obs, student_uid, group_existed)
    else:
        obs.session.error = "no student uid to run a session as"

    obs.left_behind += [
        f"{step}, as every run in this sandbox does" for step in guest_setup
    ]
    for leaf in sorted(_harness_leaves(read_mounts()) - harness_leaves_before):
        obs.left_behind.append(
            f"processes moved into {leaf} with memory and pids delegated to its"
            + " siblings, as every run does at start"
        )
    return obs


def _cgroup_layout(mounts: list[Mount]) -> tuple[int | None, bool]:
    """The cgroup version karotte confines with, and whether it can. Follows
    :func:`detect_cgroups`: a hybrid host with the controllers on v1 and a bare
    writable cgroup2 confines with v1."""
    backend = detect_cgroups(mounts)
    if backend is not None:
        return backend.version, True
    if any(not m.is_cgroup2 for m in mounts):
        return 1, False
    if mounts:
        return 2, False
    return None, False


def _harness_leaves(mounts: list[Mount]) -> set[Path]:
    leaves = {
        own_cgroup_dir(mount.path) / HARNESS_LEAF
        for mount in mounts
        if mount.is_cgroup2
    }
    return {leaf for leaf in leaves if leaf.is_dir()}


def _unused_uid(student_uid: int | None) -> int:
    """A uid with no passwd entry and no processes, so limits and rules put on
    it touch nobody."""
    busy = {
        proc.info["uids"].real
        for proc in psutil.process_iter(["uids"])
        if proc.info.get("uids") is not None
    }
    for uid in _PROBE_UIDS:
        if uid in busy or uid == student_uid:
            continue
        try:
            _ = pwd.getpwuid(uid)
        except KeyError:
            return uid
    raise RuntimeError(f"No unused uid in {_PROBE_UIDS}")


def _probe_contracts(obs: Observations, sandbox: Sandbox, uid: int) -> None:
    confinement = build_confinement(sandbox, uid)
    try:
        obs.contracts["memory"] = confinement.limit_memory(PROBE_MEMORY_BYTES).value
        obs.contracts["processes"] = confinement.limit_processes(
            PROBE_PROCESS_COUNT
        ).value
        limits = confinement.current_limits()
        obs.readback["memory"] = limits.memory_bytes
        obs.readback["processes"] = limits.process_count
    finally:
        _ = confinement.limit_memory(None)
        _ = confinement.limit_processes(None)
        confinement.close()
        if isinstance(confinement, CgroupConfinement):
            confinement.group.destroy()
            unregister_student_cgroup(uid)
            if confinement.group.path.exists():
                obs.left_behind.append(f"probe cgroup {confinement.group.path}")


def _probe_quota(obs: Observations) -> None:
    """Mount a small quota of our own, the way the run's first file limit
    would, and take it off again. Where a helper outside the container or an
    earlier run already mounted the run's quota, report that one instead."""
    run_mount = file_quota.QUOTA_DIR / "mnt"
    if os.path.ismount(run_mount):
        stats = os.statvfs(run_mount)
        obs.quota_backing = (
            f"loop ext4, already mounted ({_amount(stats.f_frsize * stats.f_blocks)})"
        )
        obs.contracts["files"] = Contract.PREVENTED.value
        return
    scratch = Path(tempfile.mkdtemp(prefix="karotte-check-"))
    target = scratch / "target"
    target.mkdir()
    quota = None
    try:
        quota = file_quota.mount_file_quota(
            (target,),
            PROBE_QUOTA_BYTES,
            PROBE_QUOTA_FILES,
            quota_dir=scratch / "quota",
        )
    finally:
        if quota is not None and not file_quota.unmount_file_quota(
            quota, scratch / "quota"
        ):
            obs.left_behind.append(f"probe quota mounts under {scratch}")
        else:
            shutil.rmtree(scratch, ignore_errors=True)
    if quota is not None:
        obs.quota_backing = "loop ext4"
        obs.contracts["files"] = Contract.PREVENTED.value
    else:
        obs.quota_backing = "watchdog"
        obs.contracts["files"] = Contract.REAPED.value


def _probe_firewall(obs: Observations, sandbox: Sandbox, uid: int) -> None:
    canaries = firewall_canaries()
    confinement = Confinement(sandbox)
    try:
        obs.reachable_without_rules = _addresses(reachable_as(uid, canaries))
        try:
            obs.firewall_took = confinement.restrict_to_internal_network(
                uid, allowed_ips=[]
            )
        except RuntimeError as exc:
            obs.firewall_error = str(exc)
        if obs.firewall_took:
            try:
                obs.reachable_with_rules = _addresses(reachable_as(uid, canaries))
            except RuntimeError as exc:
                obs.firewall_check_error = str(exc)
    except RuntimeError as exc:
        obs.firewall_error = str(exc)
    finally:
        if not confinement.lift_network_rules(uid):
            obs.left_behind.append(f"firewall rules for probe uid {uid}")


def _addresses(targets: list[tuple[str, int]]) -> list[str]:
    return [f"{host}:{port}" for host, port in targets]


def _probe_session(obs: Observations, student_uid: int, group_existed: bool) -> None:
    """Start one real student session with the default limits on its group,
    and read back what it actually got. A group an earlier run left behind
    gets its own limits back; one created during the check is removed."""
    confinement = get_confinement(student_uid)
    restore: tuple[int | None, int | None, str | None] | None = None
    try:
        if isinstance(confinement, CgroupConfinement):
            group = confinement.group
            restore = (group.memory_limit(), group.process_limit(), group.swap_limit())
            obs.session_limits["memory"] = obs.student_memory_default
            obs.session_limits["processes"] = STUDENT_PROCESS_LIMIT
            _ = confinement.limit_memory(obs.student_memory_default)
            _ = confinement.limit_processes(STUDENT_PROCESS_LIMIT)
            obs.student_cgroup = _cgroup_relative(group.path)
        obs.session = run_student_session(obs.cgroup_version)
        if obs.session.cgroup is not None and obs.session.error is None:
            _read_session_limits(obs.session, read_mounts(), obs.cgroup_version)
    finally:
        if isinstance(confinement, CgroupConfinement) and restore is not None:
            group = confinement.group
            if group_existed:
                _ = group.set_memory_limit(restore[0])
                _ = group.set_process_limit(restore[1])
                # set_memory_limit sets swap to match; put back what was there.
                if restore[2] is not None:
                    _ = group.set_swap_limit(restore[2])
            else:
                _ = confinement.limit_memory(None)
                _ = confinement.limit_processes(None)
                group.destroy()
                unregister_student_cgroup(student_uid)
                if group.path.exists():
                    obs.left_behind.append(f"student cgroup {group.path} (no limits)")
        confinement.close()
        _confinement_for.cache_clear()


def _student_group_exists(uid: int) -> bool:
    backend = detect_cgroups()
    name = f"karotte_uid_{uid}"
    if isinstance(backend, V2Cgroup):
        return (backend.parent / name).is_dir()
    if isinstance(backend, V1Cgroup):
        return any((root / name).is_dir() for root in backend.roots.values())
    return False


def run_student_session(cgroup_version: int | None = None) -> SessionReport:
    argv, preexec = student_session_command(
        [trusted_binary("sh"), "-c", _SESSION_SCRIPT], disable_networking=False
    )
    try:
        result = subprocess.run(
            argv,
            preexec_fn=preexec,
            capture_output=True,
            text=True,
            timeout=_SESSION_TIMEOUT_SECONDS,
            env=student_env(),
            cwd="/",
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return SessionReport(error=f"could not start: {exc}")
    if result.returncode != 0:
        return SessionReport(
            error=f"exited {result.returncode}: {result.stderr.strip()[:200]}"
        )
    return parse_session_output(result.stdout, cgroup_version)


def parse_session_output(text: str, cgroup_version: int | None = None) -> SessionReport:
    """``cgroup_version`` is the hierarchy karotte confines with: on a hybrid
    host the ``0::`` line names a cgroup2 group that holds no limits."""
    report = SessionReport()
    v1: dict[str, str] = {}
    v2: str | None = None
    for line in text.splitlines():
        key, _, value = line.partition("=")
        if key == "cgroup":
            hierarchy, _, rest = value.partition(":")
            controllers, _, path = rest.partition(":")
            if hierarchy == "0" and not controllers:
                v2 = path
            else:
                for controller in controllers.split(","):
                    v1[controller] = path
        elif key == "status":
            fields = value.split()
            if len(fields) > 1 and fields[1].isdigit():
                report.uid = int(fields[1])
        elif key == "oom_score_adj" and value.lstrip("-").isdigit():
            report.oom_score_adj = int(value)
        elif key == "pid" and value.isdigit():
            report.pid = int(value)
    if cgroup_version == 1:
        report.cgroup = v1.get("memory")
    else:
        report.cgroup = v2 if v2 is not None else v1.get("memory")
    if report.cgroup is None:
        report.error = f"no cgroup in the session's output: {text.strip()[:200]!r}"
    return report


def _read_session_limits(
    session: SessionReport, mounts: list[Mount], cgroup_version: int | None = None
) -> None:
    """Read the limits of the session's group as root, after it exited: the
    group outlives the session, and root can read every file in it."""
    assert session.cgroup is not None
    relative = session.cgroup.lstrip("/")
    for mount in mounts:
        if mount.is_cgroup2 and cgroup_version != 1:
            base = mount.path / relative
            session.memory_max = _read(base / "memory.max")
            session.pids_max = _read(base / "pids.max")
            session.oom_group = _read(base / "memory.oom.group")
            if session.memory_max is not None:
                return
    for mount in mounts:
        if "memory" in mount.controllers:
            session.memory_max = _read(mount.path / relative / "memory.limit_in_bytes")
        if "pids" in mount.controllers:
            session.pids_max = _read(mount.path / relative / "pids.max")


def _cgroup_relative(path: Path) -> str:
    """``path`` the way ``/proc/self/cgroup`` names it."""
    for mount in read_mounts():
        if path.is_relative_to(mount.path):
            return "/" + str(path.relative_to(mount.path))
    return str(path)


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _gib(nbytes: int) -> str:
    return f"{nbytes / (1 << 30):.1f} GiB"


def _amount(value: int | None) -> str:
    if value is None:
        return "no limit"
    if value >= 1 << 20:
        return f"{value >> 20} MiB"
    return str(value)


def report_json(obs: Observations, findings: list[Finding]) -> dict[str, object]:
    return {
        "ok": passed(findings),
        "findings": [asdict(f) for f in findings],
        "observations": asdict(obs),
    }


_VERDICTS = {True: "[green]ok[/green]", False: "[bold red]WEAK[/bold red]", None: ""}


def render_table(findings: list[Finding]) -> Table:
    table = Table(title="Student confinement")
    table.add_column("Check")
    table.add_column("Value", overflow="fold")
    table.add_column("Verdict")
    for finding in findings:
        verdict = _VERDICTS[finding.ok]
        table.add_row(finding.name, escape(finding.value), verdict)
    return table
