"""Locks this interpreter down — non-dumpable, no new executable pages, no
exec, no ptrace — and then runs the file named on its command line. The half a
graded launch installs from the outside, before it starts a run, is here too,
so every rule a confined run is under is in this one file."""

import ctypes
import errno
import mmap
import os
import platform
import runpy
import struct
import sys
from collections.abc import Callable

RESTRICTIONS = """
the interpreter confines itself before your file gets it, so a few things a
Python program can normally do are gone by then:
- nothing on the filesystem is writable, /tmp, /var/tmp and /dev/shm included.
  posix semaphores live in /dev/shm, so multiprocessing's Pool, Queue and Lock
  all fail to build; os.fork with pipes and anonymous mmap between the halves
  is the way to use the other cores
- no execve, so no subprocess, and no restarting the interpreter with other
  flags
- no page ever becomes executable: mmap and mprotect refuse PROT_EXEC, which
  rules out a JIT and out dlopen of anything not already loaded. every stdlib C
  extension but _crypt and _tkinter is compiled into this interpreter, so
  imports are unaffected, and so is ctypes against a library that is already in
  the process
- no ptrace, no writing your own memory through /proc, no memfd_create, no
  io_uring, no SysV shared memory
each of those fails with EPERM rather than being quietly ignored
""".strip()

PR_GET_DUMPABLE = 3
PR_SET_DUMPABLE = 4
PR_SET_SECCOMP = 22
PR_SET_NO_NEW_PRIVS = 38

SECCOMP_MODE_FILTER = 2

AUDIT_ARCH_X86_64 = 0xC000003E
AUDIT_ARCH_AARCH64 = 0xC00000B7

X32_SYSCALL_BIT = 0x40000000

PROT_EXEC = 0x4

PTRACE_TRACEME = 0

BPF_LD = 0x00
BPF_W = 0x00
BPF_ABS = 0x20
BPF_JMP = 0x05
BPF_JEQ = 0x10
BPF_JGE = 0x30
BPF_JSET = 0x40
BPF_RET = 0x06
BPF_K = 0x00

LOAD = BPF_LD | BPF_W | BPF_ABS
JEQ = BPF_JMP | BPF_JEQ | BPF_K
JGE = BPF_JMP | BPF_JGE | BPF_K
JSET = BPF_JMP | BPF_JSET | BPF_K
RET = BPF_RET | BPF_K

SECCOMP_RET_KILL_PROCESS = 0x80000000
SECCOMP_RET_ERRNO = 0x00050000
SECCOMP_RET_ALLOW = 0x7FFF0000

NR_OFFSET = 0
ARCH_OFFSET = 4
ARGS_OFFSET = 16


def arg_offset(index: int) -> int:
    """Where the low half of `seccomp_data.args[index]` sits. Little-endian
    only, which is every machine this image is built for."""
    return ARGS_OFFSET + 8 * index


ARCHITECTURES = {
    "x86_64": AUDIT_ARCH_X86_64,
    "aarch64": AUDIT_ARCH_AARCH64,
}

SYSCALLS = {
    "x86_64": {
        "mmap": 9,
        "mprotect": 10,
        "shmget": 29,
        "shmat": 30,
        "execve": 59,
        "ptrace": 101,
        "prctl": 157,
        "memfd_create": 319,
        "execveat": 322,
        "pkey_mprotect": 329,
        "io_uring_setup": 425,
        "io_uring_enter": 426,
        "io_uring_register": 427,
    },
    "aarch64": {
        "mmap": 222,
        "mprotect": 226,
        "shmget": 194,
        "shmat": 196,
        "execve": 221,
        "ptrace": 117,
        "prctl": 167,
        "memfd_create": 279,
        "execveat": 281,
        "pkey_mprotect": 288,
        "io_uring_setup": 425,
        "io_uring_enter": 426,
        "io_uring_register": 427,
    },
}

REFUSED = (
    "ptrace",
    "execve",
    "execveat",
    "memfd_create",
    "io_uring_setup",
    "io_uring_enter",
    "io_uring_register",
    "shmget",
    "shmat",
)

PROT_CHECKED = ("mmap", "mprotect", "pkey_mprotect")

PREEXEC_REFUSED = tuple(name for name in REFUSED if name not in ("execve", "execveat"))


class Program:
    """A BPF program with labels, since a jump only ever goes forward and
    counting the instructions between two of them by hand does not survive
    editing the filter."""

    def __init__(self) -> None:
        self.instructions: list[tuple[int, object, object, int]] = []
        self.labels: dict[str, int] = {}

    def emit(self, code: int, jt: object = 0, jf: object = 0, k: int = 0) -> None:
        self.instructions.append((code, jt, jf, k))

    def label(self, name: str) -> None:
        self.labels[name] = len(self.instructions)

    def assemble(self) -> bytes:
        return b"".join(
            struct.pack(
                "<HBBI", code, self._jump(jt, at), self._jump(jf, at), k & 0xFFFFFFFF
            )
            for at, (code, jt, jf, k) in enumerate(self.instructions)
        )

    def _jump(self, target: object, at: int) -> int:
        if isinstance(target, int):
            return target
        distance = self.labels[str(target)] - at - 1
        if not 0 <= distance <= 255:
            raise ValueError(f"{target} is {distance} instructions away")
        return distance


def _assemble(
    machine: str,
    refused: tuple[str, ...],
    prot_checked: tuple[str, ...],
    dumpable: bool,
) -> bytes:
    numbers = SYSCALLS[machine]
    program = Program()

    program.emit(LOAD, k=ARCH_OFFSET)
    program.emit(JEQ, jt="native", jf="kill", k=ARCHITECTURES[machine])

    program.label("native")
    program.emit(LOAD, k=NR_OFFSET)
    if machine == "x86_64":
        program.emit(JGE, jt="kill", k=X32_SYSCALL_BIT)

    for name in refused:
        program.emit(JEQ, jt="refuse", k=numbers[name])
    if dumpable:
        program.emit(JEQ, jt="dumpable", k=numbers["prctl"])
    for name in prot_checked:
        program.emit(JEQ, jt="executable", k=numbers[name])
    program.emit(RET, k=SECCOMP_RET_ALLOW)

    if dumpable:
        program.label("dumpable")
        program.emit(LOAD, k=arg_offset(0))
        program.emit(JEQ, jt="refuse", k=PR_SET_DUMPABLE)
        program.emit(RET, k=SECCOMP_RET_ALLOW)

    if prot_checked:
        program.label("executable")
        program.emit(LOAD, k=arg_offset(2))
        program.emit(JSET, jt="refuse", k=PROT_EXEC)
        program.emit(RET, k=SECCOMP_RET_ALLOW)

    program.label("refuse")
    program.emit(RET, k=SECCOMP_RET_ERRNO | errno.EPERM)

    program.label("kill")
    program.emit(RET, k=SECCOMP_RET_KILL_PROCESS)

    return program.assemble()


def build_filter(machine: str) -> bytes:
    """The filter, for one machine's native ABI. Every other ABI the kernel
    would answer on — i386 through `int 0x80`, x32 through the high bit of the
    syscall number — is killed rather than filtered, since a 64-bit program has
    no business on either and translating the rules twice is how they end up
    disagreeing."""
    return _assemble(machine, REFUSED, PROT_CHECKED, dumpable=True)


def build_preexec_filter(machine: str) -> bytes:
    """The part of the confinement a launch can install before it execs a
    compiled artifact, so that whatever runs ahead of the shim's constructor is
    not running unfiltered."""
    return _assemble(machine, PREEXEC_REFUSED, (), dumpable=False)


class SockFprog(ctypes.Structure):
    _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.c_void_p)]


def libc() -> ctypes.CDLL:
    handle = ctypes.CDLL(None, use_errno=True)
    handle.prctl.restype = ctypes.c_int
    handle.prctl.argtypes = [ctypes.c_int] + [ctypes.c_ulong] * 4
    return handle


def prctl(handle: ctypes.CDLL, option: int, *args: int) -> int:
    padded = (*args, *(0,) * (4 - len(args)))
    ctypes.set_errno(0)
    return handle.prctl(option, *padded)


def confine() -> None:
    """Take away the address space's own handles on itself, then refuse the
    syscalls that would hand one back."""
    machine = platform.machine()
    if machine not in SYSCALLS:
        raise SystemExit(f"sandbox: no syscall table for {machine}")

    handle = libc()

    def demand(option: int, name: str, *args: int) -> None:
        if prctl(handle, option, *args) != 0:
            raise SystemExit(f"sandbox: {name}: {os.strerror(ctypes.get_errno())}")

    demand(PR_SET_DUMPABLE, "PR_SET_DUMPABLE", 0)
    demand(PR_SET_NO_NEW_PRIVS, "PR_SET_NO_NEW_PRIVS", 1)

    program = build_filter(machine)
    held = ctypes.create_string_buffer(program, len(program))
    fprog = SockFprog(len(program) // 8, ctypes.addressof(held))
    demand(
        PR_SET_SECCOMP, "PR_SET_SECCOMP", SECCOMP_MODE_FILTER, ctypes.addressof(fprog)
    )


WRITE_PROBES = ("/tmp", "/var/tmp", "/dev/shm", ".")


def address_space_aliases() -> list[str]:
    """Every path in /proc that reaches this address space. They are all the
    same file to the kernel, but only if you name them all do you find out."""
    pid = os.getpid()
    tid = os.readlink("/proc/thread-self").split("/")[-1]
    return [
        "/proc/self/mem",
        f"/proc/{pid}/mem",
        "/proc/thread-self/mem",
        f"/proc/self/task/{tid}/mem",
        f"/proc/{pid}/task/{tid}/mem",
    ]


def refusals() -> list[str]:
    """What the confinement is supposed to have taken away, checked from
    inside it. Anything this returns is a hole."""
    holes: list[str] = []

    for directory in WRITE_PROBES:
        probe = os.path.join(directory, ".sandbox-write-probe")
        try:
            handle = os.open(probe, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except OSError:
            continue
        os.close(handle)
        os.unlink(probe)
        holes.append(f"{directory} is writable")

    try:
        os.execv("/nonexistent/sandbox-probe", ["sandbox-probe"])
    except PermissionError:
        pass
    except OSError as e:
        holes.append(f"execve answered {errno.errorcode.get(e.errno, e.errno)}")

    try:
        mmap.mmap(-1, mmap.PAGESIZE, prot=mmap.PROT_READ | mmap.PROT_EXEC).close()
        holes.append("mmap still hands out executable pages")
    except OSError:
        pass

    handle = libc()
    if prctl(handle, PR_GET_DUMPABLE) != 0:
        holes.append("the process is still dumpable")
    if prctl(handle, PR_SET_DUMPABLE, 1) == 0:
        holes.append("PR_SET_DUMPABLE still answers")
    handle.ptrace.restype = ctypes.c_long
    handle.ptrace.argtypes = [ctypes.c_int] + [ctypes.c_ulong] * 3
    if handle.ptrace(PTRACE_TRACEME, 0, 0, 0) == 0:
        holes.append("ptrace still answers")

    for alias in address_space_aliases():
        try:
            os.close(os.open(alias, os.O_WRONLY))
        except OSError:
            continue
        holes.append(f"{alias} opens for writing")

    return holes


CLONE_NEWNS = 0x00020000

MS_PRIVATE = 1 << 18

MOUNT_ATTR_RDONLY = 0x1
MOUNT_ATTR_NOSUID = 0x2

AT_FDCWD = -100
AT_RECURSIVE = 0x8000

SYS_MOUNT_SETATTR = {"x86_64": 442, "aarch64": 442}


class _MountAttr(ctypes.Structure):
    _fields_ = [
        ("attr_set", ctypes.c_uint64),
        ("attr_clr", ctypes.c_uint64),
        ("propagation", ctypes.c_uint64),
        ("userns_fd", ctypes.c_uint64),
    ]


_libc: ctypes.CDLL | None = None


def _ensure_libc() -> None:
    """Load libc in the parent: `CDLL(None)` takes the symbols already in the
    process, so the forked child has nothing to dlopen."""
    global _libc
    if _libc is None:
        handle = libc()
        handle.syscall.restype = ctypes.c_long
        handle.syscall.argtypes = [
            ctypes.c_long,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        _libc = handle


def _mount_setattr(target: bytes, flags: int, **attributes: int) -> None:
    assert _libc is not None, "_ensure_libc must run in the parent first"
    attribute = _MountAttr(**attributes)
    ctypes.set_errno(0)
    answered = _libc.syscall(
        SYS_MOUNT_SETATTR[platform.machine()],
        AT_FDCWD,
        target,
        flags,
        ctypes.byref(attribute),
        ctypes.sizeof(attribute),
    )
    if answered != 0:
        code = ctypes.get_errno()
        raise OSError(code, f"mount_setattr {target.decode()}: {os.strerror(code)}")


def _seal_failed(exc: OSError, reason: str) -> OSError:
    """Report a failed seal where the report can still be read. An exception
    raised in a preexec reaches the parent flattened to
    `subprocess.SubprocessError("Exception occurred in preexec_fn.")` — type,
    errno and message all lost — so the why goes to fd 2, which is already the
    launch's captured stderr pipe by preexec time."""
    message = f"sandbox seal: {reason}. Refusing to run with a writable filesystem."
    os.write(2, f"{message}\n".encode())
    return OSError(exc.errno, message)


def _lock_root_read_only() -> None:
    """Enter a mount namespace where nothing on the filesystem is writable, so
    no writable mount is left to be a noexec one. Runs in the forked child
    while still root, since CLONE_NEWNS and mount_setattr need CAP_SYS_ADMIN.

    ENOSYS is a kernel without mount_setattr — gVisor — and skips the seal:
    the launch goes ahead unsealed and the grader's check reports the writable
    filesystem, so the run is graded unreliable rather than crashing. Any
    other failure kills the launch rather than run with a writable
    filesystem."""
    try:
        os.unshare(CLONE_NEWNS)
    except OSError as exc:
        if exc.errno == errno.ENOSYS:
            return
        raise _seal_failed(
            exc,
            f"could not unshare a mount namespace ({os.strerror(exc.errno or 0)}) "
            "— is CAP_SYS_ADMIN missing?",
        ) from exc
    try:
        _mount_setattr(b"/", AT_RECURSIVE, propagation=MS_PRIVATE)
        _mount_setattr(
            b"/", AT_RECURSIVE, attr_set=MOUNT_ATTR_RDONLY | MOUNT_ATTR_NOSUID
        )
    except OSError as exc:
        if exc.errno == errno.ENOSYS:
            return
        raise _seal_failed(exc, str(exc.strerror or exc)) from exc


def make_read_only_root_fn() -> Callable[[], None]:
    machine = platform.machine()
    if machine not in SYS_MOUNT_SETATTR:
        raise RuntimeError(f"no mount_setattr number for {machine}")
    _ensure_libc()
    return _lock_root_read_only


def make_preexec_filter_fn() -> Callable[[], None]:
    """Install what the confinement can take away before the artifact is
    exec'd, so a hook running ahead of the shim's constructor — an ifunc
    resolver is the one the artifact gate cannot see coming — is filtered
    rather than free."""
    machine = platform.machine()
    if machine not in SYSCALLS:
        raise RuntimeError(f"no syscall table for {machine}")

    program = build_preexec_filter(machine)
    handle = libc()

    def _confine() -> None:
        held = ctypes.create_string_buffer(program, len(program))
        fprog = SockFprog(len(program) // 8, ctypes.addressof(held))
        for option, name, *args in (
            (PR_SET_NO_NEW_PRIVS, "PR_SET_NO_NEW_PRIVS", 1),
            (
                PR_SET_SECCOMP,
                "PR_SET_SECCOMP",
                SECCOMP_MODE_FILTER,
                ctypes.addressof(fprog),
            ),
        ):
            if prctl(handle, option, *args) != 0:
                code = ctypes.get_errno()
                raise _seal_failed(OSError(code, name), f"{name}: {os.strerror(code)}")

    return _confine


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        raise SystemExit(f"usage: {argv[0]} <program.py> [args...]|--check")

    confine()

    if argv[1] == "--check":
        holes = refusals()
        for hole in holes:
            print(f"sandbox: {hole}", file=sys.stderr)
        return 1 if holes else 0

    # Everything after the program is the program's own argv. A task whose
    # submission takes arguments -- a corpus to read and a path to write --
    # gets nothing at all if they stop here.
    sys.argv = argv[1:]
    runpy.run_path(argv[1], run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
