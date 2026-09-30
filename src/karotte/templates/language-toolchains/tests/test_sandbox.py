"""Tests for the sandboxed-run machinery: the seccomp filter's answers, the
read-only-root gate, and the grading helpers that put them in front of a run.
What needs a real kernel runs only on Linux; the filter itself is checked by
interpreting its BPF the way the kernel would."""

import ast
import errno
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from environment import elf, sandbox, toolchain_grading
from environment.toolchains import Language, RunConfinement

SECCOMP_DATA = struct.Struct("<iIQ6Q")


def run_filter(program: bytes, arch: int, nr: int, *args: int) -> int:
    """The kernel's side of a seccomp filter, enough of it to say what the
    program answers for one call: a word load, the three jumps the filter uses
    and a return."""
    data = SECCOMP_DATA.pack(nr, arch, 0, *args, *(0,) * (6 - len(args)))
    accumulator = 0
    at = 0
    while True:
        code, jt, jf, k = struct.unpack_from("<HBBI", program, at * 8)
        at += 1
        if code == sandbox.LOAD:
            (accumulator,) = struct.unpack_from("<I", data, k)
        elif code == sandbox.RET:
            return k
        else:
            taken = {
                sandbox.JEQ: accumulator == k,
                sandbox.JGE: accumulator >= k,
                sandbox.JSET: bool(accumulator & k),
            }[code]
            at += jt if taken else jf


REFUSE = sandbox.SECCOMP_RET_ERRNO | errno.EPERM

READ_NR = {"x86_64": 0, "aarch64": 63}


@pytest.mark.parametrize("machine", ["x86_64", "aarch64"])
class TestTheSyscallFilterAnswers:
    def answer(self, machine: str, nr: int, *args: int) -> int:
        return run_filter(
            sandbox.build_filter(machine),
            sandbox.ARCHITECTURES[machine],
            nr,
            *args,
        )

    def test_a_call_it_has_nothing_against_goes_through(self, machine):
        assert self.answer(machine, READ_NR[machine]) == sandbox.SECCOMP_RET_ALLOW

    def test_every_named_syscall_is_refused(self, machine):
        for name in sandbox.REFUSED:
            assert self.answer(machine, sandbox.SYSCALLS[machine][name]) == REFUSE

    def test_only_executable_mappings_are_refused(self, machine):
        for name in sandbox.PROT_CHECKED:
            nr = sandbox.SYSCALLS[machine][name]
            assert (
                self.answer(machine, nr, 0, 0, 0x1 | 0x2) == sandbox.SECCOMP_RET_ALLOW
            )
            assert self.answer(machine, nr, 0, 0, 0x1 | 0x4) == REFUSE
            assert self.answer(machine, nr, 0, 0, 0x7) == REFUSE

    def test_only_the_dumpable_prctl_is_refused(self, machine):
        nr = sandbox.SYSCALLS[machine]["prctl"]
        assert self.answer(machine, nr, sandbox.PR_SET_DUMPABLE, 1) == REFUSE
        assert self.answer(machine, nr, sandbox.PR_SET_DUMPABLE, 0) == REFUSE
        assert self.answer(machine, nr, sandbox.PR_GET_DUMPABLE) == (
            sandbox.SECCOMP_RET_ALLOW
        )

    def test_another_abi_is_killed_rather_than_filtered(self, machine):
        program = sandbox.build_filter(machine)
        for foreign in (0x40000003, 0x40000028, 0xC00000B7 ^ 0xC000003E):
            if foreign == sandbox.ARCHITECTURES[machine]:
                continue
            answer = run_filter(program, foreign, READ_NR[machine])
            assert answer == sandbox.SECCOMP_RET_KILL_PROCESS


@pytest.mark.parametrize("machine", ["x86_64", "aarch64"])
class TestTheFilterAlaunchCanInstallBeforeExec:
    def answer(self, machine: str, nr: int, *args: int) -> int:
        return run_filter(
            sandbox.build_preexec_filter(machine),
            sandbox.ARCHITECTURES[machine],
            nr,
            *args,
        )

    def test_it_refuses_everything_the_shim_does_but_the_two_it_cannot(self, machine):
        for name in sandbox.PREEXEC_REFUSED:
            assert self.answer(machine, sandbox.SYSCALLS[machine][name]) == REFUSE

    def test_the_launch_can_still_exec_the_artifact(self, machine):
        """The next thing this child does is exec, so the ban on that waits
        for the shim."""
        for name in ("execve", "execveat"):
            assert self.answer(machine, sandbox.SYSCALLS[machine][name]) == (
                sandbox.SECCOMP_RET_ALLOW
            )

    def test_the_loader_can_still_map_the_artifacts_own_text(self, machine):
        """ld.so maps the program and its libraries executable; refusing that
        here would leave nothing able to start."""
        for name in sandbox.PROT_CHECKED:
            nr = sandbox.SYSCALLS[machine][name]
            assert self.answer(machine, nr, 0, 0, 0x1 | 0x4) == (
                sandbox.SECCOMP_RET_ALLOW
            )

    def test_the_shim_can_still_take_the_address_space_away(self, machine):
        """The kernel resets dumpable across execve, so the shim is what sets
        it and this must not refuse the call."""
        nr = sandbox.SYSCALLS[machine]["prctl"]
        assert self.answer(machine, nr, sandbox.PR_SET_DUMPABLE, 0) == (
            sandbox.SECCOMP_RET_ALLOW
        )

    def test_another_abi_is_killed_here_too(self, machine):
        answer = run_filter(
            sandbox.build_preexec_filter(machine), 0x40000003, READ_NR[machine]
        )

        assert answer == sandbox.SECCOMP_RET_KILL_PROCESS


class TestTheProgramGetsItsOwnArguments:
    """`sandbox.py` runs the program named on its command line; everything
    after that name is the program's. A task whose submission is invoked with
    arguments -- an input to read, a path to write -- hands them through here,
    and a sandbox that kept only the program name would give it none."""

    def test_the_program_sees_what_followed_it(self, tmp_path, monkeypatch):
        target = tmp_path / "program.py"
        target.write_text(
            "import sys, json, pathlib\n"
            "pathlib.Path(sys.argv[2]).write_text(json.dumps(sys.argv))\n"
        )
        seen = tmp_path / "argv.json"

        monkeypatch.setattr(sandbox, "confine", lambda: None)
        monkeypatch.setattr(sys, "argv", ["sandbox.py"])
        sandbox.main(["sandbox.py", str(target), "input.txt", str(seen)])

        assert json.loads(seen.read_text()) == [str(target), "input.txt", str(seen)]

    def test_a_program_with_no_arguments_still_sees_its_own_name(
        self, tmp_path, monkeypatch
    ):
        target = tmp_path / "program.py"
        seen = tmp_path / "argv.json"
        target.write_text(
            "import sys, json, pathlib\n"
            f"pathlib.Path({str(seen)!r}).write_text(json.dumps(sys.argv))\n"
        )

        monkeypatch.setattr(sandbox, "confine", lambda: None)
        monkeypatch.setattr(sys, "argv", ["sandbox.py"])
        sandbox.main(["sandbox.py", str(target)])

        assert json.loads(seen.read_text()) == [str(target)]

    def test_a_bare_invocation_is_a_usage_error(self):
        with pytest.raises(SystemExit):
            sandbox.main(["sandbox.py"])


def test_the_two_halves_of_the_filter_cover_the_same_ground():
    """What waits for the shim is exactly the two rules a launch cannot
    install before exec'ing, so adding a refusal to one half cannot silently
    skip the other."""
    assert set(sandbox.REFUSED) - set(sandbox.PREEXEC_REFUSED) == {
        "execve",
        "execveat",
    }


def test_the_x32_calling_convention_is_killed():
    program = sandbox.build_filter("x86_64")
    answer = run_filter(program, sandbox.AUDIT_ARCH_X86_64, 0x40000000 | 9)
    assert answer == sandbox.SECCOMP_RET_KILL_PROCESS


def test_the_restrictions_text_names_what_the_filter_refuses():
    """The prompt paragraph and the filter live in one file so they cannot
    drift apart silently; this is the alarm if someone edits only one."""
    for name in ("execve", "ptrace", "memfd_create", "io_uring", "PROT_EXEC", "EPERM"):
        assert name in sandbox.RESTRICTIONS


class TestTheReadOnlyRootGate:
    def refuse_unshare(self, monkeypatch, code: int) -> None:
        def refuse(flags: int) -> None:
            raise OSError(code, "unshare")

        monkeypatch.setattr(sandbox.os, "unshare", refuse, raising=False)

    def test_a_kernel_without_mount_setattr_is_skipped(self, monkeypatch):
        """gVisor: the launch goes ahead unsealed and `check_sandbox` reports
        the writable filesystem instead."""
        monkeypatch.setattr(sandbox.os, "unshare", lambda flags: None, raising=False)

        def refuse(*args, **attributes):
            raise OSError(errno.ENOSYS, "mount_setattr")

        monkeypatch.setattr(sandbox, "_mount_setattr", refuse)

        sandbox._lock_root_read_only()

    def test_a_kernel_without_mount_namespaces_is_skipped(self, monkeypatch):
        self.refuse_unshare(monkeypatch, errno.ENOSYS)

        sandbox._lock_root_read_only()

    def test_a_denied_namespace_kills_the_launch(self, monkeypatch, capfd):
        self.refuse_unshare(monkeypatch, errno.EPERM)

        with pytest.raises(OSError, match="CAP_SYS_ADMIN"):
            sandbox._lock_root_read_only()

        assert "sandbox seal:" in capfd.readouterr().err

    def test_a_denied_seal_kills_the_launch(self, monkeypatch, capfd):
        monkeypatch.setattr(sandbox.os, "unshare", lambda flags: None, raising=False)

        def refuse(*args, **attributes):
            raise OSError(errno.EPERM, "mount_setattr /")

        monkeypatch.setattr(sandbox, "_mount_setattr", refuse)

        with pytest.raises(OSError, match="mount_setattr"):
            sandbox._lock_root_read_only()

        assert "sandbox seal:" in capfd.readouterr().err

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_the_reason_survives_where_the_exception_does_not(self, monkeypatch, capfd):
        """An exception raised in a preexec reaches the parent flattened to a
        generic `SubprocessError` — never the `OSError` a launch path reads as
        "would not start" — and the reason lands on the captured stderr."""
        self.refuse_unshare(monkeypatch, errno.EPERM)

        with pytest.raises(subprocess.SubprocessError, match="preexec_fn"):
            subprocess.run(
                [sys.executable, "-c", "pass"],
                preexec_fn=sandbox._lock_root_read_only,
            )

        assert "sandbox seal:" in capfd.readouterr().err


class TestTheStudentCanReadEveryRule:
    """Nothing the graded launch imposes is enforced from a file the student
    cannot read: the seal and the pre-exec filter live in the module that
    ships to /opt/grader, and the image copies it there beside the shim's
    source."""

    @pytest.fixture
    def containerfile(self) -> str:
        return (Path(__file__).resolve().parent.parent / "Containerfile").read_text()

    def test_the_grader_installs_the_students_copy(self):
        assert (
            toolchain_grading.make_read_only_root_fn is sandbox.make_read_only_root_fn
        )
        assert (
            toolchain_grading.make_preexec_filter_fn is sandbox.make_preexec_filter_fn
        )

    def test_the_sources_ship_beside_the_binaries(self, containerfile: str):
        assert (
            "COPY src/environment/shim.c src/environment/sandbox.py /opt/grader/"
            in containerfile
        )
        assert "chmod 0444 /opt/grader/shim.c /opt/grader/sandbox.py" in containerfile

    def test_it_imports_nothing_but_the_stdlib(self):
        """It is staged on its own beside a submission and bootstraps a run as
        `python sandbox.py <file>`, so an import of the environment package —
        or of anything installed — would be a file that is not there."""
        tree = ast.parse(Path(sandbox.__file__).read_text())
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                assert node.level == 0, "a relative import needs the package"
                imported.add((node.module or "").split(".")[0])

        assert imported <= sys.stdlib_module_names, imported - sys.stdlib_module_names


class TestTheGradingHelpers:
    def test_the_sandbox_is_staged_beside_the_source(self, tmp_path):
        staged = toolchain_grading.stage_sandbox(tmp_path)

        assert staged == tmp_path / "sandbox.py"
        assert staged.read_bytes() == Path(sandbox.__file__).read_bytes()

    def test_the_confinement_runs_in_front_of_the_file(self):
        argv = toolchain_grading.sandboxed_argv(
            ["/python3", "/build/server.py"], Path("/build/sandbox.py")
        )

        assert argv == ["/python3", "/build/sandbox.py", "/build/server.py"]

    def test_the_filesystem_is_sealed_after_the_cgroup_and_before_the_drop(
        self, monkeypatch
    ):
        order: list[str] = []
        monkeypatch.setattr(
            toolchain_grading,
            "make_demote_fn",
            lambda *fds, **_: lambda: order.append("demote"),
        )
        monkeypatch.setattr(
            toolchain_grading,
            "get_confinement",
            lambda: SimpleNamespace(
                student_preexec=lambda: lambda: order.append("join")
            ),
        )
        monkeypatch.setattr(
            toolchain_grading,
            "make_read_only_root_fn",
            lambda: lambda: order.append("seal"),
        )

        toolchain_grading.make_run_preexec(0, 1, 2, seal_root=True)()

        assert order == ["join", "seal", "demote"]

    def test_a_compiled_run_is_filtered_after_the_seal_and_before_the_drop(
        self, monkeypatch
    ):
        order: list[str] = []
        monkeypatch.setattr(
            toolchain_grading,
            "make_demote_fn",
            lambda *fds, **_: lambda: order.append("demote"),
        )
        monkeypatch.setattr(
            toolchain_grading,
            "get_confinement",
            lambda: SimpleNamespace(
                student_preexec=lambda: lambda: order.append("join")
            ),
        )
        monkeypatch.setattr(
            toolchain_grading,
            "make_read_only_root_fn",
            lambda: lambda: order.append("seal"),
        )
        monkeypatch.setattr(
            toolchain_grading,
            "make_preexec_filter_fn",
            lambda: lambda: order.append("filter"),
        )

        toolchain_grading.make_run_preexec(
            0, 1, 2, seal_root=True, filter_syscalls=True
        )()

        assert order == ["join", "seal", "filter", "demote"]

    def test_an_interpreted_run_is_not_filtered_before_exec(self, monkeypatch):
        """Its confinement goes on inside the interpreter, which is already
        past the exec this would have to allow."""
        monkeypatch.setattr(
            toolchain_grading, "make_demote_fn", lambda *fds, **_: lambda: None
        )
        monkeypatch.setattr(
            toolchain_grading,
            "get_confinement",
            lambda: SimpleNamespace(student_preexec=lambda: None),
        )
        monkeypatch.setattr(
            toolchain_grading,
            "make_preexec_filter_fn",
            lambda: pytest.fail("only a compiled run is filtered before exec"),
        )

        toolchain_grading.make_run_preexec(0, 1, 2)()

    def test_an_unsealed_run_is_left_alone(self, monkeypatch):
        monkeypatch.setattr(
            toolchain_grading, "make_demote_fn", lambda *fds, **_: lambda: None
        )
        monkeypatch.setattr(
            toolchain_grading,
            "get_confinement",
            lambda: SimpleNamespace(student_preexec=lambda: None),
        )
        monkeypatch.setattr(
            toolchain_grading,
            "make_read_only_root_fn",
            lambda: pytest.fail("only a sandboxed run seals the filesystem"),
        )

        toolchain_grading.make_run_preexec(0, 1, 2)()

    def test_off_the_container_there_is_nothing_to_do(self, monkeypatch):
        monkeypatch.setattr(toolchain_grading, "make_demote_fn", lambda *fds, **_: None)

        assert toolchain_grading.make_run_preexec(0, 1, 2, seal_root=True) is None

    def _capture_policy(self, monkeypatch) -> dict[str, object]:
        captured: dict[str, object] = {}

        def fake_demote(*fds, **policy):
            captured.update(policy)
            return lambda: None

        monkeypatch.setattr(toolchain_grading, "make_demote_fn", fake_demote)
        monkeypatch.setattr(
            toolchain_grading,
            "get_confinement",
            lambda: SimpleNamespace(student_preexec=lambda: None),
        )
        return captured

    def test_the_platforms_own_mounts_are_covered_for_the_run(self, monkeypatch):
        """The image seals the interpreters it installed; it cannot chmod away
        one the platform mounts in read-only beside them, and a run that can
        `dlopen` loads it without exec'ing anything."""
        captured = self._capture_policy(monkeypatch)
        monkeypatch.setenv("KAROTTE_CONTAINERIZED", "1")
        monkeypatch.setattr(
            toolchain_grading,
            "platform_tooling_dirs",
            lambda: ("/opt/platform-tooling",),
        )

        toolchain_grading.make_run_preexec(0, 1, 2)

        assert captured["covered_dirs"] == ("/opt/platform-tooling",)

    def test_a_run_covers_nothing_off_the_container(self, monkeypatch):
        """Covering fails closed, so a dev box has to name nothing at all."""
        captured = self._capture_policy(monkeypatch)
        monkeypatch.delenv("KAROTTE_CONTAINERIZED", raising=False)

        toolchain_grading.make_run_preexec(0, 1, 2)

        assert captured["covered_dirs"] == ()

    def test_the_staging_dirs_are_bound_noexec_for_the_run(self, monkeypatch):
        """The read-only seal is not noexec, so a mapped-code run could map its
        own source executable; the staging dirs are bound noexec to close it."""
        captured = self._capture_policy(monkeypatch)

        toolchain_grading.make_run_preexec(
            0, 1, 2, noexec_dirs=["/opt/grader/build/x", "/workdir"]
        )

        assert captured["read_only_dirs"] == ["/opt/grader/build/x", "/workdir"]

    def test_a_mapped_code_interpreted_run_noexecs_its_staging_dirs(self):
        """A run whose program is an interpreter only reads the staged file as
        data, so its location can be noexec without refusing an exec."""
        dirs = toolchain_grading.run_noexec_dirs(
            Language.RUBY,
            ["ruby", "-E", "UTF-8", "/opt/grader/build/x/server.rb"],
            Path("/opt/grader/build/x/artifact"),
            Path("/opt/grader/build/x"),
        )

        assert dirs == ["/opt/grader/build/x", str(toolchain_grading.STUDENT_WORKDIR)]

    def test_a_compiled_run_noexecs_nothing(self):
        """Its program is the artifact it execs out of the build directory, and
        noexec there would refuse the exec that starts it."""
        artifact = Path("/opt/grader/build/x/main")
        dirs = toolchain_grading.run_noexec_dirs(
            Language.DART, [str(artifact)], artifact, Path("/opt/grader/build/x")
        )

        assert dirs == []

    def test_a_jitting_run_noexecs_nothing(self):
        """Its shim hands out anonymous executable pages, so a submission that
        wants its own machine code copies it into one and the bind buys
        nothing — while still being one more way for the runtime not to
        start."""
        dirs = toolchain_grading.run_noexec_dirs(
            Language.KOTLIN,
            KOTLIN_RUN_ARGV,
            Path("/opt/grader/build/x/main.jar"),
            Path("/opt/grader/build/x"),
        )

        assert dirs == []

    def test_a_strictly_confined_run_noexecs_nothing(self):
        """Its shim refuses every executable mapping of a file already."""
        dirs = toolchain_grading.run_noexec_dirs(
            Language.PYTHON,
            ["/opt/python/bin/python3", "sandbox.py", "/opt/grader/build/x/sol.py"],
            Path("/opt/grader/build/x/sol.py"),
            Path("/opt/grader/build/x"),
        )

        assert dirs == []

    def _stop_at_the_probe(self, monkeypatch) -> dict:
        seen: dict = {}

        def record(*args, **kwargs):
            seen.update(kwargs)
            raise OSError("reached the probe")

        monkeypatch.setattr(
            toolchain_grading,
            "subprocess",
            SimpleNamespace(run=record, SubprocessError=subprocess.SubprocessError),
        )
        return seen

    def test_the_check_stages_a_file_in_every_dir_it_was_told_to_bind(
        self, monkeypatch, tmp_path
    ):
        """The probe cannot see the bind by itself, and the file a submission
        runs as `$0` is the staged copy in the build directory, so every bound
        directory gets a file the mapped-code probe must fail to map."""
        seen = self._stop_at_the_probe(monkeypatch)
        build, workdir = tmp_path / "build", tmp_path / "workdir"
        build.mkdir()
        workdir.mkdir()

        toolchain_grading.check_native_sandbox(
            None,
            Language.RUBY,
            ["ruby", "x.rb"],
            noexec_dirs=[str(build), str(workdir)],
        )

        name = toolchain_grading.SUBMISSION_PROBE_NAME
        staged = [build / name, workdir / name]
        assert seen["env"]["KAROTTE_SUBMISSION_PROBE_FILE"] == os.pathsep.join(
            str(path) for path in staged
        )
        assert not [path for path in staged if path.exists()]

    def test_a_run_with_no_bind_to_check_stages_nothing(self, monkeypatch):
        """A compiled mapped-code run says so with an empty list, and the probe
        must not be told to assert a noexec that is not there."""
        seen = self._stop_at_the_probe(monkeypatch)

        toolchain_grading.check_native_sandbox(
            None, Language.DART, ["/opt/grader/build/x/main"], noexec_dirs=[]
        )

        assert "KAROTTE_SUBMISSION_PROBE_FILE" not in seen["env"]

    def test_a_mapped_code_launch_that_never_says_is_refused(self, monkeypatch):
        """Forgetting the bind is the failure to make loud: the probe would
        otherwise pass and the run would go ahead with the hole open."""
        self._stop_at_the_probe(monkeypatch)

        answer = toolchain_grading.check_native_sandbox(
            None, Language.RUBY, ["ruby", "x.rb"]
        )

        assert answer is not None and "run_noexec_dirs" in answer
        assert "reached the probe" not in answer

    def test_a_dir_the_check_cannot_stage_in_is_our_problem(self, monkeypatch):
        """Staging is the machinery's own step, so a failure is a run to report
        unreliable rather than a traceback out of the check."""
        answer = toolchain_grading.check_native_sandbox(
            None, Language.RUBY, ["ruby", "x.rb"], noexec_dirs=["/nonexistent"]
        )

        assert answer is not None and "FileNotFoundError" in answer

    def test_a_mount_the_platform_did_not_make_is_not_named(self, monkeypatch):
        """Naming an absent path would fail the mount and take every launch on
        a runner that mounts nothing."""
        monkeypatch.setenv("KAROTTE_CONTAINERIZED", "1")
        monkeypatch.setattr(toolchain_grading, "platform_tooling_dirs", lambda: ())

        assert toolchain_grading.dirs_to_cover() == ()

    def test_a_machine_that_cannot_confine_one_is_our_problem(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(
            toolchain_grading, "student_python", lambda: Path(sys.executable)
        )
        monkeypatch.setattr(
            toolchain_grading,
            "subprocess",
            SimpleNamespace(
                run=lambda *a, **k: (_ for _ in ()).throw(OSError("no namespaces")),
                SubprocessError=subprocess.SubprocessError,
            ),
        )

        answer = toolchain_grading.check_sandbox(tmp_path / "sandbox.py", None)

        assert answer is not None and "no namespaces" in answer

    def test_the_probe_runs_with_the_launch_environment(self, monkeypatch, tmp_path):
        seen: dict = {}

        def record(*args, **kwargs):
            seen.update(kwargs)
            raise OSError("stop here")

        monkeypatch.setattr(
            toolchain_grading, "student_python", lambda: Path(sys.executable)
        )
        monkeypatch.setattr(
            toolchain_grading,
            "subprocess",
            SimpleNamespace(run=record, SubprocessError=subprocess.SubprocessError),
        )

        toolchain_grading.check_sandbox(tmp_path / "sandbox.py", None)

        assert seen["env"]["PYTHONSAFEPATH"] == "1"
        assert seen["env"]["LC_CTYPE"] == toolchain_grading.UTF8_LOCALE

    def test_a_machine_that_cannot_confine_a_compiled_run_is_our_problem(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            toolchain_grading,
            "subprocess",
            SimpleNamespace(
                run=lambda *a, **k: (_ for _ in ()).throw(OSError("no namespaces")),
                SubprocessError=subprocess.SubprocessError,
            ),
        )

        answer = toolchain_grading.check_native_sandbox(None)

        assert answer is not None and "no namespaces" in answer

    def test_the_native_probe_runs_with_the_confinement_preloaded(self, monkeypatch):
        seen: dict = {}

        def record(*args, **kwargs):
            seen.update(kwargs)
            raise OSError("stop here")

        monkeypatch.setattr(
            toolchain_grading,
            "subprocess",
            SimpleNamespace(run=record, SubprocessError=subprocess.SubprocessError),
        )

        toolchain_grading.check_native_sandbox(None)

        assert seen["env"]["LD_PRELOAD"] == str(toolchain_grading.SHIM)
        assert seen["env"]["LC_CTYPE"] == toolchain_grading.UTF8_LOCALE

    def test_the_native_probe_is_asked_about_the_cells_own_preloads(self, monkeypatch):
        """The probe answers about the process it is in, so a cell that loads
        more than the shim has to be probed with all of it."""
        seen: dict = {}

        def record(*args, **kwargs):
            seen.update(kwargs)
            raise OSError("stop here")

        monkeypatch.setattr(
            toolchain_grading,
            "subprocess",
            SimpleNamespace(run=record, SubprocessError=subprocess.SubprocessError),
        )

        toolchain_grading.check_native_sandbox(None, Language.PASCAL)

        assert (
            seen["env"]["LD_PRELOAD"]
            == toolchain_grading.sandboxed_env({}, Language.PASCAL)["LD_PRELOAD"]
        )
        assert "libgcc_s.so.1" in seen["env"]["LD_PRELOAD"]


@pytest.mark.skipif(sys.platform != "linux", reason="seccomp is a Linux interface")
class TestTheConfinementHoldsOnThisMachine:
    def sandbox_run(self, tmp_path: Path, *argv: str) -> subprocess.CompletedProcess:
        staged = tmp_path / "sandbox.py"
        shutil.copyfile(sandbox.__file__, staged)
        return subprocess.run(
            [sys.executable, str(staged), *argv],
            capture_output=True,
            text=True,
            cwd=tmp_path,
            timeout=60,
        )

    def test_everything_but_the_filesystem_holds_without_a_mount_namespace(
        self, tmp_path
    ):
        finished = self.sandbox_run(tmp_path, "--check")

        holes = [line for line in finished.stderr.splitlines() if line]
        assert holes and all("writable" in hole for hole in holes), finished.stderr

    def test_the_file_it_is_handed_runs_as_the_program(self, tmp_path):
        target = tmp_path / "server.py"
        target.write_text(
            "import sys\nprint(__name__, sys.argv, __file__, flush=True)\n"
        )

        finished = self.sandbox_run(tmp_path, str(target))

        assert finished.returncode == 0, finished.stderr
        assert finished.stdout.split() == ["__main__", f"['{target}']", str(target)]

    def test_the_stdlib_is_still_importable_behind_it(self, tmp_path):
        target = tmp_path / "server.py"
        target.write_text(
            "import json, mmap, ctypes, unicodedata, array, threading, os\n"
            "print(json.dumps(unicodedata.normalize('NFKD', 'é')), flush=True)\n"
        )

        finished = self.sandbox_run(tmp_path, str(target))

        assert finished.returncode == 0, finished.stderr

    def preexec_filtered_run(self, code: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=60,
            preexec_fn=sandbox.make_preexec_filter_fn(),
        )

    def test_a_filter_installed_before_exec_survives_it(self):
        """Which is the whole point: it is live for the loader's own work, so
        a hook running ahead of the shim is already filtered."""
        finished = self.preexec_filtered_run(
            "import ctypes;"
            "libc = ctypes.CDLL(None, use_errno=True);"
            "print(libc.ptrace(0, 0, 0, 0), ctypes.get_errno())"
        )

        assert finished.returncode == 0, finished.stderr
        answer, code = finished.stdout.split()
        assert answer == "-1" and int(code) == errno.EPERM

    def test_the_interpreter_still_starts_under_it(self):
        """The exec itself and the loader's executable mappings are what this
        filter deliberately leaves alone."""
        finished = self.preexec_filtered_run("import json, ctypes; print('ok')")

        assert finished.returncode == 0, finished.stderr
        assert finished.stdout.strip() == "ok"

    def test_the_shim_can_still_finish_the_job_behind_it(self):
        """The strict filter stacks on top of this one rather than replacing
        it, and the calls the shim needs to get there are still open."""
        finished = self.preexec_filtered_run(
            "import sys;"
            f"sys.path.insert(0, {str(Path(sandbox.__file__).parent)!r});"
            "from environment import sandbox;"
            "sandbox.confine();"
            "print(sandbox.refusals())"
        )

        assert finished.returncode == 0, finished.stderr
        holes = finished.stdout.strip()
        assert "writable" in holes or holes == "[]", holes


def test_the_shim_refuses_what_the_interpreter_confinement_refuses():
    """One confinement in two halves — sandbox.py for an interpreted run, the
    preloaded shim for a compiled one. This is the alarm if only one is
    edited."""
    source = (Path(sandbox.__file__).parent / "shim.c").read_text()

    for name in sandbox.REFUSED:
        assert f"REFUSE(SYS_{name})" in source, name
    for name in sandbox.PROT_CHECKED:
        assert f"SYS_{name}" in source, name


def test_no_shim_variant_drops_a_refusal():
    """Every REFUSE sits above the first conditional, so the three builds
    differ in what they say about executable pages and in nothing else."""
    source = (Path(sandbox.__file__).parent / "shim.c").read_text()
    common = source.split("#ifndef ALLOW_JIT")[0]

    for name in sandbox.REFUSED:
        assert f"REFUSE(SYS_{name})" in common, name
    assert "PROT_EXEC" not in common


def test_the_probe_checks_a_submission_file_cannot_map_executable():
    """The mapped-code shim lets a file map as code, so the probe has to prove
    the runtime's own file still does while a submission's does not. This is the
    alarm if the second half is dropped."""
    source = (Path(sandbox.__file__).parent / "sandbox_probe.c").read_text()

    assert "KAROTTE_SUBMISSION_PROBE_FILE" in source
    assert "no_staged_file_maps_executable" in source
    assert "/proc/self/exe" in source
    assert "EPERM" in source and "EACCES" in source


def test_the_mapped_code_shim_is_the_same_confinement():
    """The second shim differs in one rule. This is the alarm if a refusal is
    added to the source in a way the ifdef leaves out of one of the two."""
    source = (Path(sandbox.__file__).parent / "shim.c").read_text()
    strict, mapped = source.split("#ifdef ALLOW_MAPPED_CODE")[0], source

    for name in sandbox.REFUSED:
        assert f"REFUSE(SYS_{name})" in strict, name
    assert "MAP_ANONYMOUS" in mapped.split("#ifdef ALLOW_MAPPED_CODE")[1]


class TestClosingTheWayOutOfTheLanguage:
    def argv(self, *argv: str, allow_unsafe: bool = False) -> list[str]:
        return toolchain_grading.harden_build_argv(list(argv), allow_unsafe)

    def test_a_rust_build_forbids_unsafe_unless_the_task_says_otherwise(self):
        assert self.argv("rustc", "-O", "main.rs", "-o", "main") == [
            "rustc",
            "-F",
            "unsafe-code",
            "-O",
            "main.rs",
            "-o",
            "main",
        ]

    def test_a_task_that_grades_speed_can_open_it(self):
        argv = ["rustc", "-O", "main.rs", "-o", "main"]

        assert self.argv(*argv, allow_unsafe=True) == argv

    def test_the_flag_is_found_behind_an_absolute_path(self):
        assert self.argv("/usr/local/bin/rustc", "main.rs")[1:3] == [
            "-F",
            "unsafe-code",
        ]

    def test_a_swift_build_forbids_the_unsafe_pointer_family(self):
        assert self.argv("swiftc", "-O", "-o", "main", "main.swift")[1:4] == [
            "-strict-memory-safety",
            "-Werror",
            "StrictMemorySafety",
        ]

    def test_a_csharp_build_cannot_have_its_unsafe_turned_back_on(self):
        """The source can carry `#:property AllowUnsafeBlocks=true`; a
        command-line `-p:` is what beats it."""
        argv = self.argv("dotnet", "publish", "main.cs", "-o", ".")

        assert "-p:AllowUnsafeBlocks=false" in argv
        assert argv.index("-p:AllowUnsafeBlocks=false") < argv.index("main.cs")

    def test_a_haskell_build_evaluates_no_template_haskell_splice(self):
        """Not `-XNoTemplateHaskell`, which an in-file pragma overrides: the
        splice is sent to a `ghc-iserv` the image leaves unrunnable."""
        assert self.argv("ghc", "-O2", "-c", "main.hs") == [
            "ghc",
            "-fexternal-interpreter",
            "-O2",
            "-c",
            "main.hs",
        ]

    def test_a_haskell_task_that_wants_splices_can_open_it(self):
        argv = ["ghc", "-O2", "-c", "main.hs"]

        assert self.argv(*argv, allow_unsafe=True) == argv

    def test_a_compiler_with_no_rule_yet_is_left_alone(self):
        argv = ["dart", "compile", "exe", "main.dart", "-o", "main"]

        assert self.argv(*argv) == argv


class TestLinkingSomethingTheRunCanConfine:
    """A statically linked artifact has no loader to read `LD_PRELOAD`, and
    three of the compilers link one unless told otherwise."""

    def argv(self, *argv: str) -> list[str]:
        return toolchain_grading.harden_build_argv(list(argv), allow_unsafe=False)

    def test_go_builds_a_position_independent_executable(self):
        assert self.argv("go", "build", "-o", "main", "main.go") == [
            "go",
            "build",
            "-buildmode=pie",
            "-o",
            "main",
            "main.go",
        ]

    def test_zig_links_the_one_shared_library_every_cell_has(self):
        assert self.argv("zig", "build-exe", "-femit-bin=main", "main.zig") == [
            "zig",
            "build-exe",
            "-lc",
            "-femit-bin=main",
            "main.zig",
        ]

    def test_a_subcommand_flag_never_lands_in_front_of_the_subcommand(self):
        """`go -buildmode=pie build` is not a command."""
        assert self.argv("go", "build", "main.go")[1] == "build"

    def test_a_compiler_whose_subcommand_is_missing_is_left_alone(self):
        argv = ["go", "version"]

        assert self.argv(*argv) == argv

    def test_the_freestanding_cells_link_against_a_loader(self):
        argv = self.argv("ld", "-o", "main", "main.o")

        assert argv[1:3] == ["-pie", "-dynamic-linker"]
        assert argv[3] == toolchain_grading.loader()
        assert argv[3] in elf.LOADER.values()

    def test_llvm_ir_is_compiled_for_that_link(self):
        assert self.argv("llc", "-filetype=obj", "-o", "main.o", "main.ll")[1] == (
            "-relocation-model=pic"
        )

    def test_linkage_flags_are_not_the_task_s_to_open(self):
        """`allow_unsafe` opens the language, not the confinement."""
        opened = toolchain_grading.harden_build_argv(
            ["go", "build", "-o", "main", "main.go"], allow_unsafe=True
        )

        assert "-buildmode=pie" in opened


class TestRaisingLimitsASubmissionCannotRaiseItself:
    """A runtime whose defaults are too small for real work, where neither the
    pinned argv nor the run's scrubbed environment leaves a way to say so."""

    def argv(self, *argv: str) -> list[str]:
        return toolchain_grading.harden_build_argv(list(argv), allow_unsafe=False)

    def test_gnu_prolog_gets_stacks_it_can_finish_a_workload_on(self):
        argv = self.argv("gplc", "-o", "server", "server.pl")

        assert argv[1:7] == [
            "--global-size",
            "524288",
            "--local-size",
            "131072",
            "--trail-size",
            "131072",
        ]
        assert argv[7:] == ["-o", "server", "server.pl"]

    def test_the_task_cannot_open_them_by_grading_speed(self):
        """`allow_unsafe` opens the language, not a runtime's limits."""
        argv = toolchain_grading.harden_build_argv(
            ["gplc", "-o", "server", "server.pl"], allow_unsafe=True
        )

        assert "--global-size" in argv


class TestHardeningTheRunTheWayTheBuildIsHardened:
    """A cell whose run starts a kept interpreter cannot be confined by the
    shim alone: the runtime has to be told not to do the things the filter
    refuses."""

    def test_a_node_run_is_given_the_interpreter_only_v8(self):
        """V8 compiles bytecode to machine code as it runs and dies on the
        first executable page; `--jitless` is the whole of what makes a JS run
        survive the filter."""
        assert toolchain_grading.harden_run_argv(
            ["node", "main.js"], Language.JS_TS
        ) == ["node", "--jitless", "--no-expose-wasm", "main.js"]

    def test_the_flags_go_behind_the_interpreter(self):
        """`--jitless node` is not a command."""
        assert (
            toolchain_grading.harden_run_argv(["node", "main.js"], Language.JS_TS)[0]
            == "node"
        )

    def test_a_cell_with_nothing_to_add_is_left_alone(self):
        argv = ["./main"]

        assert toolchain_grading.harden_run_argv(argv, Language.RUST) == argv

    def test_a_jvm_run_is_denied_native_access(self):
        """The JIT shim must let any page become executable, so the seccomp
        filter cannot stop FFM from running machine code the submission carries
        as bytes; `--illegal-native-access=deny` is what closes it, at the JVM."""
        for language in (Language.KOTLIN, Language.JAVA, Language.SCALA):
            argv = toolchain_grading.harden_run_argv(
                ["java", "-jar", "main.jar"], language
            )

            assert argv == ["java", "--illegal-native-access=deny", "-jar", "main.jar"]

    def test_each_cell_gets_the_weakest_rules_its_runtime_needs(self):
        """Dart's AOT runtime maps the snapshot inside the artifact and Ruby
        dlopens the stdlib extensions it is `require`d, so both need a file to
        be mappable as code; the JVM, BeamAsm and Julia all compile as they run,
        and die under either of the other two. Everything else takes the strict
        shim."""
        by_confinement: dict[str, set[Language]] = {}
        for language in Language:
            confinement = toolchain_grading.run_confinement(language)
            by_confinement.setdefault(confinement, set()).add(language)

        assert by_confinement[RunConfinement.MAPPED_CODE] == {
            Language.DART,
            Language.RUBY,
        }
        assert by_confinement[RunConfinement.JIT] == {
            Language.CLOJURE,
            Language.ERLANG_ELIXIR,
            Language.JAVA,
            Language.JULIA,
            Language.KOTLIN,
            Language.SCALA,
        }

    def test_a_ruby_run_gets_the_shim_that_lets_its_stdlib_load(self):
        env = toolchain_grading.sandboxed_env({}, Language.RUBY)

        assert env["LD_PRELOAD"] == str(toolchain_grading.MAPPED_CODE_SHIM)

    def test_a_node_run_gets_the_strict_one(self):
        """`--jitless` is what buys JS the strict shim: nothing it does needs
        an executable mapping the loader did not make."""
        env = toolchain_grading.sandboxed_env({}, Language.JS_TS)

        assert env["LD_PRELOAD"] == str(toolchain_grading.SHIM)

    def test_a_jvm_run_gets_the_one_that_allows_a_jit(self):
        env = toolchain_grading.sandboxed_env({}, Language.KOTLIN)

        assert env["LD_PRELOAD"] == str(toolchain_grading.JIT_SHIM)

    def test_a_launch_that_names_no_language_gets_the_strict_rules(self):
        """The failure to have: a cell that needed weaker ones dies at startup
        rather than running unconfined."""
        env = toolchain_grading.sandboxed_env({})

        assert env["LD_PRELOAD"] == str(toolchain_grading.SHIM)


def elf_image(
    *,
    interp: bool = True,
    interp_path: bytes | None = None,
    preinit: bool = False,
    relocations: tuple[int, ...] = (),
    machine: int = elf.EM_X86_64,
) -> bytes:
    """A minimal 64-bit little-endian ELF carrying the program headers the
    artifact gate reads; its one PT_LOAD identity-maps the file, so a virtual
    address is its own file offset. `relocations` are r_info type fields for a
    DT_RELA table, and `interp_path` the raw PT_INTERP bytes, terminator
    included."""
    kinds = [elf.PT_LOAD, *([elf.PT_INTERP] if interp else []), elf.PT_DYNAMIC]
    phoff, size = 64, elf.PHDR.size
    dynamic_at = phoff + size * len(kinds)

    entries = 1 + (1 if preinit else 0) + (3 if relocations else 0)
    rela_at = dynamic_at + entries * elf.DYN.size
    rela = b"".join(elf.RELA.pack(0, kind, 0) for kind in relocations)

    tags = [(elf.DT_PREINIT_ARRAY, 0x1000)] if preinit else []
    if relocations:
        tags += [
            (elf.DT_RELA, rela_at),
            (elf.DT_RELASZ, len(rela)),
            (elf.DT_RELAENT, elf.RELA.size),
        ]
    dynamic = b"".join(
        elf.DYN.pack(tag, value) for tag, value in [*tags, (elf.DT_NULL, 0)]
    )

    if interp_path is None:
        interp_path = elf.LOADER[elf.MACHINES[machine]].encode() + b"\0"
    interp_at = rela_at + len(rela)

    image = bytearray(interp_at + len(interp_path))
    image[:4] = elf.ELF_MAGIC
    image[4] = elf.ELFCLASS64
    image[5] = elf.ELFDATA2LSB
    struct.pack_into("<H", image, elf.MACHINE, machine)
    struct.pack_into("<Q", image, elf.PHOFF, phoff)
    struct.pack_into("<H", image, elf.PHENTSIZE, size)
    struct.pack_into("<H", image, elf.PHNUM, len(kinds))
    for index, kind in enumerate(kinds):
        at, length = {
            elf.PT_DYNAMIC: (dynamic_at, len(dynamic)),
            elf.PT_INTERP: (interp_at, len(interp_path)),
            elf.PT_LOAD: (0, len(image)),
        }.get(kind, (0, 0))
        elf.PHDR.pack_into(image, phoff + size * index, kind, 0, at, 0, 0, length, 0, 0)
    image[dynamic_at : dynamic_at + len(dynamic)] = dynamic
    image[rela_at:interp_at] = rela
    image[interp_at:] = interp_path
    return bytes(image)


class TestTheCompiledArtifactGate:
    def artifact(self, tmp_path: Path, image: bytes) -> Path:
        path = tmp_path / "artifact"
        path.write_bytes(image)
        return path

    def test_an_ordinary_dynamic_executable_passes(self, tmp_path):
        path = self.artifact(tmp_path, elf_image())

        assert toolchain_grading.check_compiled_artifact(path) is None

    def test_a_static_executable_is_refused(self, tmp_path):
        """No loader means nothing reads LD_PRELOAD, so the run would have no
        confinement at all."""
        path = self.artifact(tmp_path, elf_image(interp=False))

        answer = toolchain_grading.check_compiled_artifact(path)

        assert answer is not None and "statically linked" in answer

    def test_a_preinit_array_hook_is_refused(self, tmp_path):
        """It runs before the shim's constructor, which is long enough to take
        an executable page and keep it."""
        path = self.artifact(tmp_path, elf_image(preinit=True))

        answer = toolchain_grading.check_compiled_artifact(path)

        assert answer is not None and "preinit_array" in answer

    def test_something_that_is_not_an_elf_is_not_the_runs_own_program(self, tmp_path):
        """A jar or a script is loaded by an interpreter of ours, and that is
        what the shim is preloaded into."""
        path = self.artifact(tmp_path, b"PK\x03\x04" + bytes(64))

        assert toolchain_grading.check_compiled_artifact(path) is None

    def test_a_truncated_elf_is_refused_rather_than_crashing(self, tmp_path):
        path = self.artifact(tmp_path, elf_image()[:40])

        answer = toolchain_grading.check_compiled_artifact(path)

        assert answer is not None and "ELF" in answer

    def test_a_few_byte_file_with_elf_magic_is_refused(self, tmp_path):
        path = self.artifact(tmp_path, elf.ELF_MAGIC + b"\x02")

        answer = toolchain_grading.check_compiled_artifact(path)

        assert answer is not None and "ELF" in answer

    def test_an_ifunc_resolver_is_refused(self, tmp_path):
        """The loader runs a resolver while applying relocations, which is
        before any constructor including the shim's — the same window
        .preinit_array opens, by a different door."""
        path = self.artifact(
            tmp_path, elf_image(relocations=(elf.R_IRELATIVE["x86_64"],))
        )

        answer = toolchain_grading.check_compiled_artifact(path)

        assert answer is not None and "ifunc" in answer

    def test_an_ifunc_resolver_is_refused_on_aarch64(self, tmp_path):
        """The relocation type is numbered per architecture, so reading the
        wrong table would miss it."""
        path = self.artifact(
            tmp_path,
            elf_image(
                relocations=(elf.R_IRELATIVE["aarch64"],),
                machine=elf.EM_AARCH64,
            ),
        )

        answer = toolchain_grading.check_compiled_artifact(path)

        assert answer is not None and "ifunc" in answer

    def test_ordinary_relocations_are_left_alone(self, tmp_path):
        """Every dynamic executable carries relocations; only the IRELATIVE
        ones run code."""
        path = self.artifact(tmp_path, elf_image(relocations=(0, 1, 8)))

        assert toolchain_grading.check_compiled_artifact(path) is None

    def test_a_relocation_table_pointing_nowhere_is_refused(self, tmp_path):
        """If no PT_LOAD covers the table this cannot say what the loader
        would run, and guessing in the student's favour is how a gate is
        walked past."""
        image = bytearray(elf_image(relocations=(0,)))
        elf.PHDR.pack_into(image, 64, elf.PT_LOAD, 0, 0, 0, 0, 8, 0, 0)
        path = self.artifact(tmp_path, bytes(image))

        answer = toolchain_grading.check_compiled_artifact(path)

        assert answer is not None and "ELF" in answer

    def test_a_dynamic_section_past_the_end_of_the_file_is_refused(self, tmp_path):
        image = bytearray(elf_image())
        phoff, size = 64, elf.PHDR.size
        elf.PHDR.pack_into(
            image, phoff + size, elf.PT_DYNAMIC, 0, len(image) - 4, 0, 0, 64, 0, 0
        )
        path = self.artifact(tmp_path, bytes(image))

        answer = toolchain_grading.check_compiled_artifact(path)

        assert answer is not None and "ELF" in answer

    def test_an_interpreter_of_the_submissions_own_is_refused(self, tmp_path):
        """The kernel starts what PT_INTERP names, so a submission that links
        its own `.interp` section replaces the loader rather than being
        confined by it."""
        path = self.artifact(tmp_path, elf_image(interp_path=b"/workdir/loader\0"))

        answer = toolchain_grading.check_compiled_artifact(path)

        assert answer is not None and "/workdir/loader" in answer

    def test_the_loaders_name_in_front_of_the_submissions_does_not_help(self, tmp_path):
        """Linking a `.interp` of one's own leaves both strings in the segment,
        and which one runs is whichever the linker put first."""
        forged = b"/workdir/loader\0" + elf.LOADER["x86_64"].encode() + b"\0"
        path = self.artifact(tmp_path, elf_image(interp_path=forged))

        answer = toolchain_grading.check_compiled_artifact(path)

        assert answer is not None and "/workdir/loader" in answer

    def test_the_aarch64_loader_is_a_different_path(self, tmp_path):
        """Checking against one machine's loader would refuse every artifact on
        the other."""
        path = self.artifact(tmp_path, elf_image(machine=elf.EM_AARCH64))

        assert toolchain_grading.check_compiled_artifact(path) is None

    def test_an_unterminated_interpreter_is_refused_rather_than_crashing(
        self, tmp_path
    ):
        path = self.artifact(tmp_path, elf_image(interp_path=b"/lib64/ld"))

        answer = toolchain_grading.check_compiled_artifact(path)

        assert answer is not None and "ELF" in answer


def test_a_compiled_run_gets_the_confinement_preloaded():
    env = toolchain_grading.sandboxed_env({"PATH": "/usr/bin"})

    assert env["LD_PRELOAD"] == str(toolchain_grading.SHIM)
    assert env["PATH"] == "/usr/bin"


def test_the_three_shims_are_three_different_files():
    shims = set(toolchain_grading.SHIMS.values())

    assert len(shims) == len(RunConfinement)


def test_the_probe_is_asked_about_the_confinement_the_run_will_have(monkeypatch):
    """A probe run under the strict rules says nothing about a launch that will
    not have them."""
    seen: dict = {}

    def record(argv, **kwargs):
        seen.update(argv=argv, **kwargs)
        raise OSError("stop here")

    monkeypatch.setattr(
        toolchain_grading,
        "subprocess",
        SimpleNamespace(run=record, SubprocessError=subprocess.SubprocessError),
    )

    toolchain_grading.check_native_sandbox(None, Language.DART, noexec_dirs=[])

    assert seen["argv"][1:] == ["--mapped-code"]
    assert seen["env"]["LD_PRELOAD"] == str(toolchain_grading.MAPPED_CODE_SHIM)


KOTLIN_RUN_ARGV = toolchain_grading.harden_run_argv(
    ["/opt/toolchains/kotlin/jre/bin/java", "-jar", "main.jar"], Language.KOTLIN
)


def test_the_probe_is_told_when_the_run_may_jit(monkeypatch):
    seen: dict = {}

    def record(argv, **kwargs):
        seen.update(argv=argv, **kwargs)
        raise OSError("stop here")

    monkeypatch.setattr(
        toolchain_grading,
        "subprocess",
        SimpleNamespace(run=record, SubprocessError=subprocess.SubprocessError),
    )

    toolchain_grading.check_native_sandbox(None, Language.KOTLIN, KOTLIN_RUN_ARGV)

    assert seen["argv"][1:] == ["--jit"]
    assert seen["env"]["LD_PRELOAD"] == str(toolchain_grading.JIT_SHIM)


class TestTheRunArgvIsCheckedToo:
    """The flags `harden_run_argv` injects are the half of the confinement no
    probe can see, since they are the launch's argv rather than its rules. A
    JVM run that lost `--illegal-native-access=deny` reaches native code and
    nothing at run time notices, so the question is asked here."""

    def stop_at_the_probe(self, monkeypatch) -> None:
        monkeypatch.setattr(
            toolchain_grading,
            "subprocess",
            SimpleNamespace(
                run=lambda *a, **k: (_ for _ in ()).throw(OSError("reached the probe")),
                SubprocessError=subprocess.SubprocessError,
            ),
        )

    def test_a_launch_that_kept_the_flags_reaches_the_probe(self, monkeypatch):
        self.stop_at_the_probe(monkeypatch)

        answer = toolchain_grading.check_native_sandbox(
            None, Language.KOTLIN, KOTLIN_RUN_ARGV
        )

        assert answer is not None and "reached the probe" in answer

    def test_a_launch_that_dropped_them_is_refused(self):
        answer = toolchain_grading.check_native_sandbox(
            None, Language.KOTLIN, ["/opt/toolchains/kotlin/jre/bin/java", "-jar", "x"]
        )

        assert answer is not None
        assert "--illegal-native-access=deny" in answer
        assert "reached the probe" not in answer

    def test_a_launch_that_never_says_what_it_runs_is_refused(self, monkeypatch):
        """The default is the forgetful one, so it has to be the loud one."""
        self.stop_at_the_probe(monkeypatch)

        answer = toolchain_grading.check_native_sandbox(None, Language.JS_TS)

        assert answer is not None and "--jitless" in answer

    def test_a_cell_with_nothing_to_inject_need_not_say(self, monkeypatch):
        self.stop_at_the_probe(monkeypatch)

        answer = toolchain_grading.check_native_sandbox(None, Language.RUST)

        assert answer is not None and "reached the probe" in answer
