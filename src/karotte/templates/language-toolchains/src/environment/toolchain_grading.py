"""Build a single-file student submission safely, in a sealed-toolchain image.

A task pins one source filename per language and the argv that builds and runs
it. The `build` MCP tool and the grader both run:

    submission, why_not = chosen_submission(candidates)
    build_dir = make_build_dir()
    source = stage_submission(submission, build_dir)
    result = build_submission(submission, source, build_dir)

which is what makes the tool's build the grading build. The grader then:

    seal_toolchain(language, keep_student_python=...)
    # launch `resolve_argv(submission.run, ...)` demoted, with an env built
    # from scratch including `"LC_CTYPE": UTF8_LOCALE`, and `make_demote_fn`
    # from karotte.subprocess as preexec_fn.

Everything here assumes the caller runs as root inside the image; builds are
demoted to the builder uid, runs to the student.
"""

import contextlib
import fnmatch
import os
import platform
import shutil
import stat
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

from karotte.confinement import get_confinement
from karotte.process_utils import kill_processes
from karotte.subprocess import make_demote_fn, platform_tooling_dirs
from karotte.untrusted_paths import probe

from environment import elf
from environment.paths import STUDENT_WORKDIR
from environment.sandbox import make_preexec_filter_fn, make_read_only_root_fn
from environment.toolchains import (
    BUILDER_UID,
    TOOLCHAINS,
    Language,
    RunConfinement,
    student_python,
    toolchain_dir,
)

# Not /workdir or /tmp: the student must not be able to reach the artifact or
# plant anything at a name the grader will use. Root-owned and unlistable (the
# Containerfile makes it 0711), with a random name per run inside it.
GRADER_BIN_DIR = (
    Path("/opt/grader")
    if "KAROTTE_CONTAINERIZED" in os.environ
    else Path(__file__).resolve().parent.parent.parent / ".grader_bin"
)
BUILD_ROOT = GRADER_BIN_DIR / "build"

# An inherited PATH starts with student-owned /workdir/.venv/bin, where a shim
# named `rustc` would win. /usr/local/bin holds the toolchain wrappers.
SAFE_PATH = "/usr/local/bin:/usr/bin:/bin"

# With no locale set, Ruby comes up US-ASCII, the BEAM latin1 and the JVM ASCII.
UTF8_LOCALE = "C.UTF-8"

# Generous for a single-file compile, and off the clock.
BUILD_TIMEOUT_S = 300.0

_STDERR_TAIL_CHARS = 2000


# What the build's own mount namespace replaces with an empty tmpfs. The first
# three are the image's 1777 directories; the last is the student's workdir,
# which the build must not be able to read.
EPHEMERAL_BUILD_DIRS = ("/tmp", "/var/tmp", "/dev/shm", str(STUDENT_WORKDIR))
READ_ONLY_BUILD_DIRS: tuple[str, ...] = ()


def dirs_to_cover() -> tuple[str, ...]:
    """The platform's own mounts, covered for both the build and the run.

    The image can seal away every interpreter it installed, but not one the
    platform mounts in beside them: its tooling can carry a whole CPython
    that a run able to `dlopen` can load without exec'ing anything. Covered for
    the build too, or a compile-time hook copies it somewhere the run keeps.

    Empty off the container and on any runner without the mount, which is what
    keeps a dev box launching: covering fails closed.
    """
    if "KAROTTE_CONTAINERIZED" not in os.environ:
        return ()
    return platform_tooling_dirs()


def _build_mount_policy() -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """The directories the build's mount namespace replaces, or nothing at all
    off the container, where the mounts in question are the dev box's own."""
    if "KAROTTE_CONTAINERIZED" not in os.environ:
        return (), (), ()
    return EPHEMERAL_BUILD_DIRS, READ_ONLY_BUILD_DIRS, dirs_to_cover()


def _tail(output: bytes) -> str:
    """The end of a compiler's output, as text. Decoded leniently: a compiler
    quoting the student's source back can emit bytes that are not UTF-8."""
    return output.decode(errors="replace")[-_STDERR_TAIL_CHARS:]


@dataclass(frozen=True)
class Submission:
    """One file a task asks for, and what the grader does with it.

    `build` is argv templates run in order, each of which must succeed; empty
    means nothing to compile. `run` is one template. `resolve_argv` fills in
    `{source}`, `{artifact}`, `{python}` and `{build_dir}`.
    """

    source: str
    build: tuple[tuple[str, ...], ...] = ()
    run: tuple[str, ...] = ("{artifact}",)

    # A bare name inside the build directory. `kotlinc -d` writes a jar only if
    # the name ends in `.jar`.
    artifact: str = "artifact"

    # For a compiler that decides something from the machine rather than from
    # its argv, where the decision should be the task's.
    build_env: dict[str, str] = field(default_factory=dict)

    # Glob patterns for build output the run still loads beside the artifact.
    # Everything else the build wrote is deleted. The BEAM needs `("*.beam",)`
    # for its code path, and a native-AOT publish for its companion files.
    keep_built: tuple[str, ...] = ()

    # For a task whose point is what unsafe code can do: skips the forbid
    # `harden_build_argv` would inject.
    allow_unsafe: bool = False


def is_regular_file(path: Path) -> bool:
    """A real file at this exact path, not a link. As root under
    `fs.protected_symlinks`, `is_file` raises on a planted link instead of
    returning False; anything the probe cannot see is treated as no submission."""
    st = probe(path)
    return st is not None and stat.S_ISREG(st.st_mode)


def student_source(submission: Submission) -> Path:
    return STUDENT_WORKDIR / submission.source


def submitted(candidates: list[Submission]) -> list[Submission]:
    """Which of a task's accepted submissions the student actually wrote."""
    return [s for s in candidates if is_regular_file(student_source(s))]


def chosen_submission(
    candidates: list[Submission],
) -> tuple[Submission | None, str | None]:
    """The one submission to grade, or why there is none. Refuses more than one
    rather than picking on the student's behalf."""
    found = submitted(candidates)
    names = " or ".join(s.source for s in candidates)
    if not found:
        return None, f"no {names} in {STUDENT_WORKDIR}"
    if len(found) > 1:
        written = ", ".join(s.source for s in found)
        return None, f"one file: found {written}"
    return found[0], None


def make_build_dir() -> Path:
    """A fresh directory for this build, owned by the builder uid."""
    BUILD_ROOT.mkdir(parents=True, exist_ok=True)
    _chown(BUILD_ROOT, 0)
    # Traversable so the build and the run can reach what is inside, unlistable
    # so neither can find out what it is called.
    BUILD_ROOT.chmod(0o711)

    build_dir = Path(tempfile.mkdtemp(dir=BUILD_ROOT))
    _chown(build_dir, BUILDER_UID)
    build_dir.chmod(0o700)
    return build_dir


def stage_submission(submission: Submission, build_dir: Path) -> Path:
    """Copy the one file the task asked for into the build directory, and
    return where it landed.

    Building in place would let `mod helper;` or `include!("helper.rs")` pull a
    second file into the binary being graded; interpreted submissions are
    staged too, since they run `{source}` and could rewrite it mid-run. Nothing
    else lands here and nothing links back out, because the build's mount
    namespace is what makes the second file unreachable and a link out of the
    directory would walk around it.
    """
    staged = build_dir / Path(submission.source).name
    shutil.copyfile(student_source(submission), staged)
    return staged


def artifact_path(build_dir: Path, submission: Submission) -> Path:
    return build_dir / submission.artifact


def resolve_argv(
    argv: tuple[str, ...], source: Path, artifact: Path, build_dir: Path
) -> list[str]:
    """Fill a `Submission` template in. The grader is the half that knows paths."""
    # Only where a template asks for it: `student_python` refuses an image
    # holding more than one managed CPython, which a compiled cell need not care
    # about.
    python = str(student_python()) if any("{python}" in arg for arg in argv) else ""
    # Not `str.format`: an argv element can carry braces of its own — an Erlang
    # term, a JSON literal, an awk program — and must come through untouched.
    values = {
        "{source}": str(source),
        "{artifact}": str(artifact),
        "{python}": python,
        "{build_dir}": str(build_dir),
    }
    resolved = []
    for arg in argv:
        for token, value in values.items():
            arg = arg.replace(token, value)
        resolved.append(arg)
    return resolved


class Hardening(NamedTuple):
    """What the template adds to one compiler's argv."""

    linkage: tuple[str, ...] = ()

    forbid_unsafe: tuple[str, ...] = ()

    limits: tuple[str, ...] = ()

    after: str | None = None


HARDENING = {
    # `-F unsafe-code` refuses `asm!`/`global_asm!` and the unsafe blocks that reach FFI.
    "rustc": Hardening(forbid_unsafe=("-F", "unsafe-code")),
    "swiftc": Hardening(
        forbid_unsafe=("-strict-memory-safety", "-Werror", "StrictMemorySafety")
    ),
    "ghc": Hardening(forbid_unsafe=("-fexternal-interpreter",)),
    "go": Hardening(linkage=("-buildmode=pie",), after="build"),
    "zig": Hardening(linkage=("-lc",), after="build-exe"),
    # The freestanding cells link with `ld` directly, which defaults to a static binary.
    "ld": Hardening(linkage=("-pie", "-dynamic-linker", "{loader}")),
    "llc": Hardening(linkage=("-relocation-model=pic",)),
    "fpc": Hardening(linkage=("-k-lc", "-k--dynamic-linker={loader}")),
    "dotnet": Hardening(
        linkage=("-p:StaticExecutable=false", "-p:PublishAot=true"),
        forbid_unsafe=("-p:AllowUnsafeBlocks=false",),
        after="publish",
    ),
    # GNU Prolog defaults to a 32MB global stack it never garbage collects
    "gplc": Hardening(
        limits=(
            "--global-size",
            "524288",
            "--local-size",
            "131072",
            "--trail-size",
            "131072",
        )
    ),
}


def loader() -> str:
    """The one interpreter an artifact may ask for on this machine."""
    machine = {"arm64": "aarch64", "amd64": "x86_64"}.get(
        platform.machine(), platform.machine()
    )
    if machine not in elf.LOADER:
        raise RuntimeError(f"no loader path for {machine}")
    return elf.LOADER[machine]


def harden_build_argv(argv: list[str], allow_unsafe: bool) -> list[str]:
    """The build argv with the language's escape hatch closed, its artifact
    linked so the scored run can confine it, and any runtime limits raised that
    the submission has no way to raise itself. Injected here rather than left to
    each task's pinned argv, so forgetting it is not a way to grade an
    unenforced submission."""
    hardening = HARDENING.get(Path(argv[0]).name)
    if hardening is None:
        return argv
    flags = hardening.limits + hardening.linkage
    if not allow_unsafe:
        flags += hardening.forbid_unsafe
    if not flags:
        return argv
    if any("{loader}" in flag for flag in flags):
        flags = tuple(flag.replace("{loader}", loader()) for flag in flags)

    at = 1
    if hardening.after is not None:
        if hardening.after not in argv:
            return argv
        at = argv.index(hardening.after) + 1
    return [*argv[:at], *flags, *argv[at:]]


def harden_run_argv(argv: list[str], language: Language) -> list[str]:
    """The run argv with the flags the cell's runtime needs for the
    confinement to hold over it. Injected for `harden_build_argv`'s reason:
    forgetting one is a scored run that dies at startup, or one that kept the
    JIT the filter was supposed to take."""
    flags = TOOLCHAINS[language].run_flags
    return [argv[0], *flags, *argv[1:]] if flags else argv


class BuildResult(NamedTuple):
    """The artifact, why the build failed (None on success), and everything
    the build commands wrote."""

    artifact: Path
    error: str | None
    stdout: bytes = b""
    stderr: bytes = b""


def build_submission(
    submission: Submission, source: Path, build_dir: Path
) -> BuildResult:
    """Compile the staged copy of the student's file.

    The build runs demoted to the builder uid, so a submission cannot bake
    root-only data into the artifact with something like Rust's
    `include_bytes!`. Once it succeeds the directory is cleaned up and handed
    back to root.

    It also runs in a mount namespace of its own, because a compiler can be
    talked into running the student's code at compile time without the pinned
    argv changing — GHC's `-pgmF` pragma names a program to run from inside
    the source file. Rather than chase each such hook, `_build_mount_policy`
    leaves the build nowhere to put anything: the build directory is under
    the root-owned unlistable `BUILD_ROOT`, which is neither ephemeral nor
    read-only, so the artifact is unaffected and `TMPDIR` points there.
    """
    assert submission.build

    artifact = artifact_path(build_dir, submission)
    # The build owns this directory and can rewrite the staged copy; these are
    # the bytes that go back.
    pristine = source.read_bytes()

    stdout = bytearray()
    stderr = bytearray()
    ephemeral_dirs, read_only_dirs, covered = _build_mount_policy()

    def result(error: str | None) -> BuildResult:
        return BuildResult(artifact, error, bytes(stdout), bytes(stderr))

    # Each must succeed; javac then jar is two commands.
    for command in submission.build:
        argv = harden_build_argv(
            resolve_argv(command, source, artifact, build_dir), submission.allow_unsafe
        )
        try:
            finished = subprocess.run(
                argv,
                cwd=build_dir,
                # Nothing inherited. HOME and TMPDIR are the build directory
                # because a compiler with nowhere for scratch work picks
                # somewhere — `go build` refuses to start without either.
                env={
                    "PATH": SAFE_PATH,
                    "HOME": str(build_dir),
                    "TMPDIR": str(build_dir),
                    "LC_CTYPE": UTF8_LOCALE,
                    **submission.build_env,
                },
                capture_output=True,
                timeout=BUILD_TIMEOUT_S,
                # stdout and stderr only: fd 0 is the grader's own stdin.
                preexec_fn=make_demote_fn(
                    1,
                    2,
                    uid_gid=BUILDER_UID,
                    ephemeral_dirs=ephemeral_dirs,
                    read_only_dirs=read_only_dirs,
                    covered_dirs=covered,
                ),
            )
        except subprocess.TimeoutExpired as expired:
            stdout += expired.stdout or b""
            stderr += expired.stderr or b""
            return result(f"did not finish within {BUILD_TIMEOUT_S:.0f}s")

        stdout += finished.stdout or b""
        stderr += finished.stderr or b""
        if finished.returncode != 0:
            tail = _tail(finished.stderr or finished.stdout)
            return result(f"`{argv[0]}` exited {finished.returncode}\n{tail}")

    # A compiler can run student code at compile time (GHC evaluates
    # `$(runIO ...)`) and leave a process behind that would still be able to
    # swap the artifact after the checks below. Outside the container nothing
    # was demoted, and the uid may be a real user of the machine.
    if "KAROTTE_CONTAINERIZED" in os.environ:
        try:
            kill_processes(BUILDER_UID)
        except RuntimeError as e:
            return result(f"left processes that would not be reaped: {e}")

    # `is_file` follows links, and so would the chmods below: root would set
    # modes on whatever the link points at.
    if artifact.is_symlink():
        return result(f"left a symlink at {artifact.name}")
    if not artifact.is_file():
        return result(f"exited 0 but produced no {artifact.name}")
    unconfined = check_compiled_artifact(artifact)
    if unconfined is not None:
        return result(f"{artifact.name} {unconfined}")

    clear_built_scratch(build_dir, artifact, submission.keep_built)
    source.write_bytes(pristine)
    seal_build_dir(build_dir)
    artifact.chmod(0o555)
    return result(None)


def clear_built_scratch(build_dir: Path, artifact: Path, keep: tuple[str, ...]) -> None:
    """Delete everything the build left beside the artifact, apart from `keep`.

    The run can reach this directory, and the task promises it a single file
    with nothing else of the student's to read. The staged source goes too and
    is written back afterwards.
    """
    for entry in sorted(build_dir.iterdir()):
        if entry == artifact:
            continue
        if any(fnmatch.fnmatch(entry.name, pattern) for pattern in keep):
            continue
        if entry.is_symlink() or not entry.is_dir():
            entry.unlink()
        else:
            # A build can leave directories unwritable — Go's module cache is
            # one — and rmtree needs the write bit to empty them.
            for root, _dirs, _files in os.walk(entry):
                os.chmod(root, 0o700)
            shutil.rmtree(entry)


def seal_build_dir(build_dir: Path) -> None:
    """Hand the finished build tree back to root, read-only.

    The whole tree rather than the artifact alone: while the student owns the
    directory they can unlink a file and leave something else at its name. It
    stays readable and traversable because the run is launched as the student
    and has to reach the artifact.
    """
    for path in (build_dir, *sorted(build_dir.rglob("*"))):
        # chmod and chown both follow links, and a link the build left is not
        # the grader's to follow out of here.
        if path.is_symlink():
            continue
        _chown(path, 0)
        if path.is_dir():
            path.chmod(0o755)
        else:
            # Unwritable but readable, keeping whatever execute bits the build
            # chose: a compiler that wrote its output 0600 would otherwise leave
            # the run unable to open it.
            path.chmod((path.stat().st_mode & 0o555) | 0o444)


def _chown(path: Path, uid: int) -> None:
    """Hand `path` to `uid`, where this process can. Off the container it is
    not root and there is no student uid to hand anything to."""
    if os.geteuid() == 0:
        os.chown(path, uid, uid)


SANDBOX_MODULE = Path(__file__).with_name("sandbox.py")

SANDBOX_CHECK_TIMEOUT_S = 30.0


def stage_sandbox(build_dir: Path) -> Path:
    """Copy the confinement module beside the staged source, and return where
    it landed."""
    staged = build_dir / SANDBOX_MODULE.name
    shutil.copyfile(SANDBOX_MODULE, staged)
    return staged


def sandboxed_argv(argv: list[str], sandbox: Path) -> list[str]:
    """The run argv with the confinement in front of the student's file: the
    interpreter stays argv[0], the sandbox module confines the process and
    then runs the file it is handed."""
    return [argv[0], str(sandbox), *argv[1:]]


def make_run_preexec(
    *chown_fds: int,
    seal_root: bool = False,
    filter_syscalls: bool = False,
    noexec_dirs: Sequence[str] = (),
) -> Callable[[], None] | None:
    """The preexec for launching a graded run: join the process confinement,
    seal the filesystem for a sandboxed cell, filter the syscalls a compiled
    run does not need, then cover the platform's mounts, bind `noexec_dirs`
    noexec and drop to the student. The order is load-bearing: the cgroup's
    files are root-owned and sit on a filesystem the seal is about to close.

    The cover comes last only because it rides along with the drop. It does not
    care where it runs — an empty read-only tmpfs is read-only whatever a later
    step does — which is what a writable one could not have said, sealed root
    or not.

    `noexec_dirs` is what `run_noexec_dirs` returned: each is bound onto itself
    on its own mount, so no exec-able path to the same bytes is left.

    A step that fails surfaces from `Popen` as `subprocess.SubprocessError`,
    never as the `OSError` of a run that would not start. That is the
    machinery failing, not the student's file: route it to unreliable, and
    look in the captured stderr for the reason."""
    demote = make_demote_fn(
        *chown_fds, read_only_dirs=noexec_dirs, covered_dirs=dirs_to_cover()
    )
    if demote is None:
        return None
    steps = [
        step
        for step in (
            get_confinement().student_preexec(),
            make_read_only_root_fn() if seal_root else None,
            make_preexec_filter_fn() if filter_syscalls else None,
            demote,
        )
        if step is not None
    ]
    if steps == [demote]:
        return demote

    def _preexec() -> None:
        for step in steps:
            step()

    return _preexec


def run_noexec_dirs(
    language: Language | None, run_argv: list[str], artifact: Path, build_dir: Path
) -> list[str]:
    """Where this run's submission is staged, to bind noexec, or nothing.

    Only a mapped-code cell whose program is an interpreter gets them. That is
    the one shim under which a submission can `mmap` its own `$0` `PROT_EXEC`
    and run machine code it carried past `__END__`, which the read-only seal
    leaves open because read-only is not noexec. The runtime's own `.so` files
    live elsewhere and keep mapping as code.

    Everything else gets nothing: the strict shim already refuses every
    executable mapping of a file, the JIT one hands out anonymous executable
    pages that a submission can copy itself into anyway, and a compiled run
    `execve`s its artifact out of the build directory, which noexec there would
    refuse.
    """
    if run_confinement(language) != RunConfinement.MAPPED_CODE:
        return []
    if not run_argv or run_argv[0] == str(artifact):
        return []
    return [str(build_dir), str(STUDENT_WORKDIR)]


def check_sandbox(sandbox: Path, preexec_fn: Callable[[], None] | None) -> str | None:
    """Run the confinement's self-check through the run's own preexec and with
    the run's environment, and return why this machine could not confine one,
    or None. A hole is the environment's problem, not the student's: report
    the run unreliable rather than grading the submission."""
    argv = [str(student_python()), str(sandbox), "--check"]
    try:
        finished = subprocess.run(
            argv,
            cwd=STUDENT_WORKDIR,
            env={
                "PATH": SAFE_PATH,
                "HOME": str(STUDENT_WORKDIR),
                "PYTHONSAFEPATH": "1",
                "LC_CTYPE": UTF8_LOCALE,
            },
            input=b"",
            capture_output=True,
            timeout=SANDBOX_CHECK_TIMEOUT_S,
            preexec_fn=preexec_fn,
        )
    except (OSError, subprocess.SubprocessError) as e:
        return f"{type(e).__name__}: {e}"
    if finished.returncode == 0:
        return None
    tail = _tail(finished.stderr or finished.stdout)
    return f"exited {finished.returncode}\n{tail}"


SHIM = GRADER_BIN_DIR / "sandbox_shim.so"

MAPPED_CODE_SHIM = GRADER_BIN_DIR / "sandbox_shim_mapped.so"

JIT_SHIM = GRADER_BIN_DIR / "sandbox_shim_jit.so"

SANDBOX_PROBE = GRADER_BIN_DIR / "sandbox_probe"

SHIMS = {
    RunConfinement.STRICT: SHIM,
    RunConfinement.MAPPED_CODE: MAPPED_CODE_SHIM,
    RunConfinement.JIT: JIT_SHIM,
}

# What the probe has to be told, so it asks about the rules the run will have.
PROBE_MODE = {
    RunConfinement.MAPPED_CODE: ["--mapped-code"],
    RunConfinement.JIT: ["--jit"],
}


def run_confinement(language: Language | None) -> RunConfinement:
    """Which shim this cell's run is launched under. Taken from the table
    rather than from the task, so `sandboxed_env` and `check_native_sandbox`
    cannot end up asking about different confinements. No language means the
    strict one: a cell that needed weaker rules dies loudly at startup, which
    is the failure to have."""
    if language is None:
        return RunConfinement.STRICT
    return TOOLCHAINS[language].run_confinement


def run_environment(language: Language | None) -> dict[str, str]:
    """What the cell's table entry exports into a run of it, with `{prefix}`
    filled in. For a runtime started directly rather than through a wrapper,
    which would have exported these itself."""
    if language is None:
        return {}
    prefix = toolchain_dir(language)
    return {
        name: value.format(prefix=prefix)
        for name, value in TOOLCHAINS[language].run_env.items()
    }


def run_preload(language: Language | None) -> tuple[str, ...]:
    """Sonames the cell's run loads beside the shim, from the table for
    `run_confinement`'s reason."""
    if language is None:
        return ()
    return TOOLCHAINS[language].run_preload


def sandboxed_env(
    env: dict[str, str], language: Language | None = None
) -> dict[str, str]:
    """The run environment with the confinement preloaded — the compiled-run
    counterpart of `sandboxed_argv`.

    A compiled artifact has no interpreter to bootstrap through, and the filter
    cannot go in the launch's preexec: it refuses execve, and the next thing
    the child does is exec the artifact. The loader installs it instead, from a
    constructor that runs once every library is mapped and before the
    artifact's own initialisers.

    Which is also what the cell's `run_preload` rides on: a library the loader
    mapped alongside the shim is one the run's own dlopen of it will find
    already loaded, rather than one the filter refuses to map. The shim stays
    first, and glibc splits the value on spaces as well as on colons.

    The cell's `run_env` rides along because this is the one place every
    scored run's environment goes through.
    """
    preloaded = (str(SHIMS[run_confinement(language)]), *run_preload(language))
    return {
        **env,
        **run_environment(language),
        "LD_PRELOAD": " ".join(preloaded),
    }


def check_compiled_artifact(artifact: Path) -> str | None:
    """Why the preloaded confinement would not hold over this artifact, or
    None. Only an ELF is the run's own program; a jar or a script is loaded by
    an interpreter of ours, which is what the shim is preloaded into."""
    with artifact.open("rb") as handle:
        if handle.read(4) != elf.ELF_MAGIC:
            return None
    return elf.why_the_confinement_would_not_hold(artifact)


def why_the_run_argv_would_not_hold(
    language: Language | None, run_argv: list[str] | None
) -> str | None:
    """Why the launch's own argv leaves a hole the probe cannot see, or None.

    What `harden_run_argv` injects is part of the confinement and none of it is
    visible from inside a probe: it is the launch's argv, not its rules. A JVM
    run that lost `--illegal-native-access=deny` reaches native code with
    nothing at run time to notice, so a launch that will not say what it runs
    is refused rather than trusted.
    """
    flags = TOOLCHAINS[language].run_flags if language is not None else ()
    if not flags:
        return None
    wanted = " ".join(flags)
    if run_argv is None:
        return (
            f"a {language} run needs {wanted}, and the launch did not say what "
            "it will run — pass the argv `harden_run_argv` returned"
        )
    missing = [flag for flag in flags if flag not in run_argv]
    if missing:
        return f"the run argv is missing {' '.join(missing)} — see harden_run_argv"
    return None


SUBMISSION_PROBE_NAME = ".sandbox-exec-probe"


def why_the_noexec_bind_would_not_hold(
    language: Language | None, noexec_dirs: Sequence[str] | None
) -> str | None:
    """Why a mapped-code launch leaves the noexec bind unanswered, or None.

    Only that cell needs the bind, and a launch that says nothing gets no bind
    and a probe with nothing to check — the hole back with a passing check in
    front of it. An empty list is an answer: it is what a compiled run gets.
    """
    if run_confinement(language) != RunConfinement.MAPPED_CODE:
        return None
    if noexec_dirs is None:
        return (
            f"a {language} run needs its staging dirs bound noexec, and the "
            "launch did not say which — pass what `run_noexec_dirs` returned"
        )
    return None


def _stage_submission_probe(path: Path) -> None:
    """A file where a submission's own would sit, for the probe to fail to map."""
    path.write_bytes(elf.ELF_MAGIC)
    _chown(path, 0)
    path.chmod(0o444)


def check_native_sandbox(
    preexec_fn: Callable[[], None] | None,
    language: Language | None = None,
    run_argv: list[str] | None = None,
    noexec_dirs: Sequence[str] | None = None,
) -> str | None:
    """The question `check_sandbox` asks of a Python run, asked the way a
    compiled one gets its confinement. A hole is the environment's problem, not
    the student's: report the run unreliable rather than grading it.

    The language has to be the launch's, or the probe is answering about a
    confinement the run will not have, and `run_argv` has to be the argv the
    launch will use, or the half of the confinement that lives there goes
    unchecked.

    `noexec_dirs` is what `make_run_preexec` was handed. The probe cannot see a
    mount, so every bound directory gets a file the mapped-code probe must fail
    to map executable — the build directory included, since the file a
    submission runs as `$0` is the staged copy there."""
    unhardened = why_the_run_argv_would_not_hold(language, run_argv)
    if unhardened is not None:
        return unhardened
    unbound = why_the_noexec_bind_would_not_hold(language, noexec_dirs)
    if unbound is not None:
        return unbound
    confinement = run_confinement(language)
    env = sandboxed_env(
        {"PATH": SAFE_PATH, "HOME": str(STUDENT_WORKDIR), "LC_CTYPE": UTF8_LOCALE},
        language,
    )
    staged = [
        Path(directory) / SUBMISSION_PROBE_NAME for directory in noexec_dirs or ()
    ]
    argv = [str(SANDBOX_PROBE), *PROBE_MODE.get(confinement, [])]
    try:
        for probe_file in staged:
            _stage_submission_probe(probe_file)
        if staged:
            env["KAROTTE_SUBMISSION_PROBE_FILE"] = os.pathsep.join(
                str(probe_file) for probe_file in staged
            )
        finished = subprocess.run(
            argv,
            cwd=STUDENT_WORKDIR,
            env=env,
            input=b"",
            capture_output=True,
            timeout=SANDBOX_CHECK_TIMEOUT_S,
            preexec_fn=preexec_fn,
        )
    except (OSError, subprocess.SubprocessError) as e:
        return f"{type(e).__name__}: {e}"
    finally:
        for probe_file in staged:
            with contextlib.suppress(OSError):
                probe_file.unlink(missing_ok=True)
    if finished.returncode == 0:
        return None
    tail = _tail(finished.stderr or finished.stdout)
    return f"exited {finished.returncode}\n{tail}"
