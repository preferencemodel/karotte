"""The two cgroup layouts can't both exist on one host, so both are driven
through a faked mount table and a temporary tree."""

from pathlib import Path

import pytest

from karotte.cgroups import (
    HARNESS_LEAF,
    Mount,
    V1Cgroup,
    V2Cgroup,
    detect_cgroups,
    own_cgroup_dir,
    parse_mounts,
)

PROC_MOUNTS_V2 = """\
proc /proc proc rw,nosuid,nodev,noexec,relatime 0 0
cgroup /sys/fs/cgroup cgroup2 rw,nosuid,nodev,noexec,relatime 0 0
"""

PROC_MOUNTS_V2_READONLY = """\
cgroup /sys/fs/cgroup cgroup2 ro,nosuid,nodev,noexec,relatime 0 0
"""

# A hybrid layout (seen in Firecracker VMs): cgroup2 mounted but nearly empty,
# controllers on v1.
PROC_MOUNTS_HYBRID = """\
tmpfs /sys/fs/cgroup tmpfs rw,relatime 0 0
cgroup /sys/fs/cgroup/cpu cgroup rw,relatime,cpu 0 0
cgroup /sys/fs/cgroup/memory cgroup rw,relatime,memory 0 0
cgroup /sys/fs/cgroup/freezer cgroup rw,relatime,freezer 0 0
cgroup /sys/fs/cgroup/pids cgroup rw,relatime,pids 0 0
cgroup2 /sys/fs/cgroup/unified cgroup2 rw,relatime 0 0
"""


def test_parse_mounts_reads_v2() -> None:
    mounts = parse_mounts(PROC_MOUNTS_V2)
    assert Mount(Path("/sys/fs/cgroup"), "cgroup2", frozenset(), False) in mounts


def test_parse_mounts_records_readonly() -> None:
    (mount,) = [m for m in parse_mounts(PROC_MOUNTS_V2_READONLY) if m.is_cgroup2]
    assert mount.read_only


def test_parse_mounts_reads_v1_controllers() -> None:
    mounts = parse_mounts(PROC_MOUNTS_HYBRID)
    by_controller = {c: m for m in mounts for c in m.controllers}
    assert by_controller["memory"].path == Path("/sys/fs/cgroup/memory")
    assert by_controller["freezer"].path == Path("/sys/fs/cgroup/freezer")


def test_parse_mounts_does_not_treat_cgroup2_as_a_controller() -> None:
    """A cgroup2 line has no controller list; its options must not become one."""
    mounts = parse_mounts(PROC_MOUNTS_HYBRID)
    v2 = [m for m in mounts if m.is_cgroup2]
    assert len(v2) == 1
    assert v2[0].controllers == frozenset()


def _write_v2_tree(root: Path, controllers: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "cgroup.controllers").write_text(controllers)
    (root / "cgroup.subtree_control").write_text("")
    (root / "cgroup.procs").write_text("")


def test_detect_prefers_v2_when_it_has_the_controllers(tmp_path: Path) -> None:
    root = tmp_path / "sys/fs/cgroup"
    _write_v2_tree(root, "cpuset cpu memory pids")
    mounts = [Mount(root, "cgroup2", frozenset(), False)]

    backend = detect_cgroups(mounts=mounts, own_cgroup=root)

    assert isinstance(backend, V2Cgroup)


def test_detect_falls_back_to_v1_when_v2_lacks_controllers(tmp_path: Path) -> None:
    """The hybrid case: cgroup2 is mounted but carries only hugetlb."""
    unified = tmp_path / "sys/fs/cgroup/unified"
    _write_v2_tree(unified, "hugetlb")
    mounts = [Mount(unified, "cgroup2", frozenset(), False)]
    for controller in ("memory", "pids", "freezer"):
        path = tmp_path / "sys/fs/cgroup" / controller
        path.mkdir(parents=True)
        mounts.append(Mount(path, "cgroup", frozenset({controller}), False))

    backend = detect_cgroups(mounts=mounts, own_cgroup=unified)

    assert isinstance(backend, V1Cgroup)


def test_detect_returns_none_without_memory_or_pids(tmp_path: Path) -> None:
    root = tmp_path / "cgroup"
    _write_v2_tree(root, "hugetlb")
    mounts = [Mount(root, "cgroup2", frozenset(), False)]

    assert detect_cgroups(mounts=mounts, own_cgroup=root) is None


def test_detect_ignores_readonly_v2(tmp_path: Path) -> None:
    """An unprivileged runc pod: the controllers are all there, but we cannot write.

    Remounting read-write would need CAP_SYS_ADMIN, which runc pods are
    deliberately not given, so this must stay a "no" rather than become a
    remount.
    """
    root = tmp_path / "cgroup"
    _write_v2_tree(root, "memory pids")
    mounts = [Mount(root, "cgroup2", frozenset(), True)]

    assert detect_cgroups(mounts=mounts, own_cgroup=root) is None


def test_detect_never_shells_out(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guard on the CAP_SYS_ADMIN decision: no mount/remount, ever."""
    import subprocess

    def explode(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"detect_cgroups ran a subprocess: {args} {kwargs}")

    monkeypatch.setattr(subprocess, "run", explode)
    monkeypatch.setattr(subprocess, "Popen", explode)

    detect_cgroups(mounts=[], own_cgroup=None)


def test_v2_moves_harness_into_a_leaf_before_delegating(tmp_path: Path) -> None:
    """cgroup v2 forbids a group that holds both processes and subtree_control,
    so the harness has to vacate before controllers can reach a child.

    Membership is asserted by the writes rather than the file contents: writing
    a pid to a real ``cgroup.procs`` adds a member, where writing to the plain
    file standing in for it here would just overwrite.
    """
    root = tmp_path / "cgroup"
    _write_v2_tree(root, "memory pids")
    (root / "cgroup.procs").write_text("101\n102\n")

    writes: list[tuple[Path, str]] = []
    real_write_text = Path.write_text

    def spy(self: Path, data: str) -> int:
        writes.append((self, data))
        return real_write_text(self, data)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Path, "write_text", spy)
        student = V2Cgroup(root).create("student")

    leaf_procs = root / HARNESS_LEAF / "cgroup.procs"
    assert [data for path, data in writes if path == leaf_procs] == ["101", "102"]
    assert "memory" in (root / "cgroup.subtree_control").read_text()
    assert student.path == root / "student"


def test_v2_does_not_move_the_harness_when_the_group_is_empty(tmp_path: Path) -> None:
    """The real root may delegate while holding processes; an empty group needs
    no leaf at all."""
    root = tmp_path / "cgroup"
    _write_v2_tree(root, "memory pids")

    V2Cgroup(root).create("student")

    assert not (root / HARNESS_LEAF).exists()


@pytest.mark.parametrize(
    ("proc_self_cgroup", "expected"),
    [
        ("0::/\n", ""),
        ("0::/karotte_harness\n", ""),
        ("0::/karotte_harness/karotte_harness\n", ""),
        ("0::/some/pod\n", "some/pod"),
        ("0::/some/pod/karotte_harness\n", "some/pod"),
    ],
)
def test_own_cgroup_dir_steps_out_of_the_harness_leaf(
    tmp_path: Path, proc_self_cgroup: str, expected: str
) -> None:
    """A process already moved into the harness leaf must find the group that
    delegates, not the leaf."""
    assert own_cgroup_dir(tmp_path, proc_self_cgroup) == tmp_path / expected


def test_v2_second_process_shares_the_limited_group(tmp_path: Path) -> None:
    """The MCP server confines bash sessions from its own process, after the
    main harness moved it into the leaf. It must join the group the main
    process limited, not create an unlimited one under the leaf."""
    root = tmp_path / "cgroup"
    _write_v2_tree(root, "memory pids")
    (root / "cgroup.procs").write_text("101\n102\n")

    main = V2Cgroup(own_cgroup_dir(root, "0::/\n")).create("karotte_uid_1000")
    assert main.set_memory_limit(4 * 1024**3)
    assert main.set_process_limit(2048)
    # The kernel empties the root as the pids move; the fake tree does not.
    (root / "cgroup.procs").write_text("")

    second = V2Cgroup(own_cgroup_dir(root, "0::/karotte_harness\n")).create(
        "karotte_uid_1000"
    )

    assert second.path == root / "karotte_uid_1000"
    assert not (root / HARNESS_LEAF / "karotte_uid_1000").exists()
    assert not (root / HARNESS_LEAF / HARNESS_LEAF).exists()
    assert second.memory_limit() == 4 * 1024**3
    assert second.process_limit() == 2048


def test_v2_student_group_is_oom_killed_as_a_whole(tmp_path: Path) -> None:
    root = tmp_path / "cgroup"
    _write_v2_tree(root, "memory pids")

    student = V2Cgroup(root).create("student")

    assert (student.path / "memory.oom.group").read_text() == "1"


def test_v2_writes_limits_with_v2_filenames(tmp_path: Path) -> None:
    root = tmp_path / "cgroup"
    _write_v2_tree(root, "memory pids")

    student = V2Cgroup(root).create("student")
    assert student.set_memory_limit(256 * 1024 * 1024)
    assert student.set_process_limit(64)

    assert (student.path / "memory.max").read_text() == str(256 * 1024 * 1024)
    assert (student.path / "pids.max").read_text() == "64"


def test_v2_lifts_limits_with_the_max_token(tmp_path: Path) -> None:
    root = tmp_path / "cgroup"
    _write_v2_tree(root, "memory pids")

    student = V2Cgroup(root).create("student")
    assert student.set_memory_limit(None)
    assert student.set_process_limit(None)

    assert (student.path / "memory.max").read_text() == "max"
    assert (student.path / "pids.max").read_text() == "max"


def test_v2_memory_limit_turns_off_swap_while_it_holds(tmp_path: Path) -> None:
    """On a host with swap, the group would page out past memory.max instead
    of being OOM-killed."""
    root = tmp_path / "cgroup"
    _write_v2_tree(root, "memory pids")
    student = V2Cgroup(root).create("student")
    (student.path / "memory.swap.max").write_text("max")

    assert student.set_memory_limit(256 * 1024 * 1024)
    assert (student.path / "memory.swap.max").read_text() == "0"

    assert student.set_memory_limit(None)
    assert (student.path / "memory.swap.max").read_text() == "max"


def test_v2_swap_limit_reads_back_and_restores(tmp_path: Path) -> None:
    """So a probe that set a memory limit can put back the swap cap it found."""
    root = tmp_path / "cgroup"
    _write_v2_tree(root, "memory pids")
    student = V2Cgroup(root).create("student")
    (student.path / "memory.swap.max").write_text(str(512 << 20))

    saved = student.swap_limit()
    _ = student.set_memory_limit(256 << 20)
    assert student.set_swap_limit(saved or "")

    assert saved == str(512 << 20)
    assert (student.path / "memory.swap.max").read_text() == str(512 << 20)


def test_v2_memory_limit_without_swap_accounting(tmp_path: Path) -> None:
    root = tmp_path / "cgroup"
    _write_v2_tree(root, "memory pids")
    student = V2Cgroup(root).create("student")

    assert student.set_memory_limit(256 * 1024 * 1024)
    assert not (student.path / "memory.swap.max").exists()


def test_v2_reads_its_limits_back(tmp_path: Path) -> None:
    root = tmp_path / "cgroup"
    _write_v2_tree(root, "memory pids")

    student = V2Cgroup(root).create("student")
    _ = student.set_memory_limit(256 * 1024 * 1024)
    _ = student.set_process_limit(64)

    assert student.memory_limit() == 256 * 1024 * 1024
    assert student.process_limit() == 64

    _ = student.set_memory_limit(None)
    _ = student.set_process_limit(None)

    assert student.memory_limit() is None
    assert student.process_limit() is None


def test_v1_reads_unlimited_memory_back_as_none(tmp_path: Path) -> None:
    """A real kernel echoes ``-1`` back as PAGE_COUNTER_MAX, so both the token
    we wrote and the huge readback mean "no limit"."""
    roots = {}
    for controller in ("memory", "pids", "freezer"):
        roots[controller] = tmp_path / controller
        roots[controller].mkdir()

    student = V1Cgroup(roots).create("student")
    _ = student.set_memory_limit(None)
    assert student.memory_limit() is None

    (roots["memory"] / "student" / "memory.limit_in_bytes").write_text(
        "9223372036854771712"
    )
    assert student.memory_limit() is None


def test_v1_lifts_the_memory_limit_with_minus_one(tmp_path: Path) -> None:
    """v1 has no ``max`` token for memory; ``-1`` is its unlimited spelling."""
    roots = {}
    for controller in ("memory", "pids", "freezer"):
        roots[controller] = tmp_path / controller
        roots[controller].mkdir()

    student = V1Cgroup(roots).create("student")
    assert student.set_memory_limit(None)
    assert student.set_process_limit(None)

    assert (roots["memory"] / "student" / "memory.limit_in_bytes").read_text() == "-1"
    assert (roots["pids"] / "student" / "pids.max").read_text() == "max"


def test_v1_writes_limits_with_v1_filenames(tmp_path: Path) -> None:
    roots = {}
    for controller in ("memory", "pids", "freezer"):
        roots[controller] = tmp_path / controller
        roots[controller].mkdir()

    student = V1Cgroup(roots).create("student")
    assert student.set_memory_limit(256 * 1024 * 1024)
    assert student.set_process_limit(64)

    assert (roots["memory"] / "student" / "memory.limit_in_bytes").read_text() == str(
        256 * 1024 * 1024
    )
    assert (roots["pids"] / "student" / "pids.max").read_text() == "64"


def test_v1_join_paths_cover_every_controller(tmp_path: Path) -> None:
    """v1 puts each controller in its own hierarchy, so a process has to be
    written into all of them, not just one."""
    roots = {}
    for controller in ("memory", "pids", "freezer"):
        roots[controller] = tmp_path / controller
        roots[controller].mkdir()

    student = V1Cgroup(roots).create("student")

    assert sorted(p.name for p in student.join_paths) == ["student"] * 3
    assert len(student.join_paths) == 3


def test_v1_kill_freezes_before_enumerating(tmp_path: Path) -> None:
    """A frozen process cannot fork, which is what makes the pid list safe to
    walk. Freezing after reading it would race a fork-and-die chain."""
    order: list[str] = []
    roots = {}
    for controller in ("memory", "pids", "freezer"):
        roots[controller] = tmp_path / controller
        roots[controller].mkdir()

    student = V1Cgroup(roots).create("student")
    freezer_dir = roots["freezer"] / "student"
    (freezer_dir / "freezer.state").write_text("FROZEN")
    (freezer_dir / "cgroup.procs").write_text("")

    real_read_text = Path.read_text

    def spy_read(self: Path) -> str:
        if self.name == "cgroup.procs":
            order.append("read_procs")
        return real_read_text(self)

    real_write_text = Path.write_text

    def spy_write(self: Path, data: str) -> int:
        if self.name == "freezer.state" and data == "FROZEN":
            order.append("freeze")
        return real_write_text(self, data)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Path, "read_text", spy_read)
        mp.setattr(Path, "write_text", spy_write)
        student.kill_all()

    assert order.index("freeze") < order.index("read_procs")


def test_v2_inherited_memory_limit_is_the_tightest_above_the_student(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cgroup"
    _write_v2_tree(root, "memory pids")
    (root / "memory.max").write_text(str(8 << 30))
    pod = root / "pod"
    _write_v2_tree(pod, "memory pids")
    (pod / "memory.max").write_text(str(6 << 30))
    container = pod / "container"
    _write_v2_tree(container, "memory pids")
    (container / "memory.max").write_text("max")

    student = V2Cgroup(container).create("student")
    _ = student.set_memory_limit(1 << 30)

    assert student.inherited_memory_limit() == 6 << 30


def test_v2_root_has_no_inherited_memory_limit(tmp_path: Path) -> None:
    """The real root cgroup has no ``memory.max`` at all, as in a VM."""
    root = tmp_path / "cgroup"
    _write_v2_tree(root, "memory pids")

    student = V2Cgroup(root).create("student")

    assert student.inherited_memory_limit() is None


def test_v1_inherited_memory_limit_reads_the_hierarchy_root(tmp_path: Path) -> None:
    roots = {}
    for controller in ("memory", "pids", "freezer"):
        roots[controller] = tmp_path / controller
        roots[controller].mkdir()
    (roots["memory"] / "memory.limit_in_bytes").write_text(str(6 << 30))

    student = V1Cgroup(roots).create("student")

    assert student.inherited_memory_limit() == 6 << 30

    (roots["memory"] / "memory.limit_in_bytes").write_text("9223372036854771712")
    assert student.inherited_memory_limit() is None
