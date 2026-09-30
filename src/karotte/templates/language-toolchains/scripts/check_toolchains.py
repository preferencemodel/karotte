# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Compile and run a hello-world with one language's toolchain, inside the image.

    python3 scripts/check_toolchains.py rust

Installs the way a `pre_hook` does, builds as the builder uid — the only uid
the toolchain is granted to; the student must not reach it at any point — runs
what came out as the student, then seals the way a grader does and checks that
what should be gone is gone and what should still run still runs. None of that
is checkable on a dev box.

One language per invocation: installing a toolchain is not reversible, so a
second in the same container is no longer testing an image with one in it.
Bind-mounted rather than copied in — a probe of the image, not part of it.
Checking a language that `toolchain_config.py` does not enable fails, since
nothing was staged for it. See `just check-toolchains`.
"""

# The %-formatting below is deliberate (see the PROGRAMS comment): f-strings
# would need doubled braces, which Jinja eats when the template is rendered.
# ruff: noqa: UP031

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple

# In the container this is a no-op and `environment` comes from the root venv
# the script is run with. On a dev box it is what makes the import resolve.
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir():
    sys.path.insert(0, str(_SRC))

from environment.paths import STUDENT_WORKDIR  # noqa: E402
from environment.toolchain_grading import (  # noqa: E402
    PROBE_MODE,
    SAFE_PATH,
    UTF8_LOCALE,
    Submission,
    build_submission,
    harden_run_argv,
    make_build_dir,
    run_confinement,
    run_environment,
    sandboxed_env,
    seal_build_dir,
)
from environment.toolchain_grading import (  # noqa: E402
    SANDBOX_PROBE as PROBE_BINARY,
)
from environment.toolchains import (  # noqa: E402
    BASE_IMAGE_BUILD_UTILITIES,
    COBOL_LIBCOB_ARCHIVE,
    HASKELL_MAIN_STUB,
    SCALA_LIBRARY_JAR,
    TOOLCHAINS,
    TOOLCHAINS_DIR,
    UV_PYTHON_DIR,
    Language,
    beam_argv,
    clojure_argv,
    elixir_argv,
    enabled_languages,
    install_toolchain,
    julia_binary,
    jvm_java,
    seal_toolchain,
)

# The banner every program prints. Compiling is what is being tested, but a
# binary that builds and then cannot run is not a working toolchain either, so
# each one is executed and its output checked.
OK = "toolchain ok"

# (filename, source, build commands, run command) per language, one entry per
# language the cell accepts — which is one for every cell but the BEAM one.
# Deliberately plain: no package manager, no dependencies, nothing off the
# network — which is also the position a task leaves the student in. The build
# and run argvs mirror the per-language reference build and run commands.
#
# Sources are %-formatted rather than f-strings: template .py files must not
# contain doubled braces (create_env renders them through Jinja), which
# brace-heavy languages would need everywhere in an f-string.
Program = tuple[str, str, list[list[str]], list[str]]

PROGRAMS: dict[Language, list[Program]] = {
    Language.PYTHON: [
        (
            "main.py",
            'print("%s")\n' % OK,
            [],
            ["python3", "main.py"],
        )
    ],
    # Nothing to build, so the cell is the interpreter the table names, which
    # `keep` is what carries through the sealing.
    Language.RUBY: [
        (
            "main.rb",
            'puts "%s"\n' % OK,
            [],
            ["ruby", "main.rb"],
        )
    ],
    # Without `PPC_CONFIG_PATH` (exported by the wrapper) the compiler looks
    # for a /etc/fpc.cfg this image does not have and finds none of its units.
    Language.PASCAL: [
        (
            "main.pas",
            "program main;\nbegin\n  writeln('%s');\nend.\n" % OK,
            [["fpc", "-O3", "-omain", "main.pas"]],
            ["./main"],
        )
    ],
    # The only probe whose toolchain the base image already had. Freestanding:
    # `_start` rather than `main`, and the write and exit syscalls rather than
    # a libc this cell does not link.
    Language.ASSEMBLY: [
        (
            "main.s",
            ".section .rodata\n"
            'msg: .ascii "%s\\n"\n' % OK + "        .set msg_len, . - msg\n"
            ".section .text\n"
            ".globl _start\n"
            "_start:\n"
            "        movq $1, %rax\n"
            "        movq $1, %rdi\n"
            "        leaq msg(%rip), %rsi\n"
            "        movq $msg_len, %rdx\n"
            "        syscall\n"
            "        movq $60, %rax\n"
            "        xorq %rdi, %rdi\n"
            "        syscall\n",
            [["as", "-o", "main.o", "main.s"], ["ld", "-o", "main", "main.o"]],
            ["./main"],
        )
    ],
    # Nothing to build here either: Julia's compiler is not separable from its
    # runtime, so the cell is the run argv and the `keep` that carries `julia`
    # through the sealing of its own directory.
    Language.JULIA: [
        (
            "main.jl",
            'println("%s")\n' % OK,
            [],
            [
                str(julia_binary()),
                "--startup-file=no",
                "--compiled-modules=existing",
                "-O3",
                "main.jl",
            ],
        )
    ],
    # A file list rather than `go mod init` and `go build .`: naming the file
    # is the one build mode that needs no module.
    Language.GO: [
        (
            "main.go",
            'package main\n\nimport "fmt"\n\nfunc main() { fmt.Println("%s") }\n' % OK,
            [["go", "build", "-o", "main", "main.go"]],
            ["./main"],
        )
    ],
    Language.RUST: [
        (
            "main.rs",
            'fn main() { println!("%s"); }\n' % OK,
            [["rustc", "-O", "main.rs", "-o", "main"]],
            ["./main"],
        )
    ],
    # `std.debug.print` rather than a write to stdout: as of 0.16 the stdout
    # handle wants an `Io` instance threaded through it, which is a lot of
    # ceremony for a probe. It prints on stderr, which is why both streams
    # count. `-femit-bin` because the binary would otherwise land at the
    # source's name.
    Language.ZIG: [
        (
            "main.zig",
            'const std = @import("std");\n'
            'pub fn main() void { std.debug.print("%s\\n", .{}); }\n' % OK,
            [["zig", "build-exe", "-O", "ReleaseFast", "-femit-bin=main", "main.zig"]],
            ["./main"],
        )
    ],
    # Two commands, because GHC's usual link compiles a generated C wrapper
    # and this cell's C frontend is sealed: `-c` compiles the module, and the
    # link takes `-no-hs-main` plus the wrapper the builder precompiled. The
    # stub bakes in `-with-rtsopts=-N`, whose RTS is the threaded one — which
    # is why `-threaded` stays on the link.
    Language.HASKELL: [
        (
            "main.hs",
            'main :: IO ()\nmain = putStrLn "%s"\n' % OK,
            [
                ["ghc", "-O2", "-c", "main.hs"],
                [
                    "ghc",
                    "-O2",
                    "-threaded",
                    "-no-hs-main",
                    "-o",
                    "main",
                    "main.o",
                    HASKELL_MAIN_STUB,
                ],
            ],
            ["./main"],
        ),
    ],
    # `-static-stdlib` is what lets the binary outlive its toolchain: linked
    # dynamically, the Swift runtime is inside the directory the sealing closes.
    Language.SWIFT: [
        (
            "main.swift",
            'print("%s")\n' % OK,
            [["swiftc", "-O", "-static-stdlib", "-o", "main", "main.swift"]],
            ["./main"],
        )
    ],
    # `ocamlopt` directly rather than through ocamlfind or dune: neither is in
    # the distribution's `ocaml` package, so neither is what a task installs.
    Language.OCAML: [
        (
            "main.ml",
            'let () = print_endline "%s"\n' % OK,
            [
                [
                    "ocamlopt",
                    "-I",
                    "+threads",
                    "unix.cmxa",
                    "threads.cmxa",
                    "-o",
                    "main",
                    "main.ml",
                ]
            ],
            ["./main"],
        )
    ],
    Language.KOTLIN: [
        (
            "main.kt",
            'fun main() { println("%s") }\n' % OK,
            [["kotlinc", "main.kt", "-include-runtime", "-d", "main.jar"]],
            [str(jvm_java(Language.KOTLIN)), "-jar", "main.jar"],
        )
    ],
    # javac cannot package what it produces, so jar folds the class files into
    # the one artifact the run starts. The public class is named after the
    # file, as javac requires.
    Language.JAVA: [
        (
            "main.java",
            "public class main {\n"
            '    public static void main(String[] args) { System.out.println("%s"); }\n'
            % OK
            + "}\n",
            [
                ["javac", "-d", "classes", "main.java"],
                [
                    "jar",
                    "--create",
                    "--file",
                    "main.jar",
                    "--main-class=main",
                    "-C",
                    "classes",
                    ".",
                ],
            ],
            [str(jvm_java(Language.JAVA)), "-jar", "main.jar"],
        )
    ],
    Language.DART: [
        (
            "main.dart",
            'void main() { print("%s"); }\n' % OK,
            [["dart", "compile", "exe", "main.dart", "-o", "main"]],
            ["./main"],
        )
    ],
    # The one cell that accepts two languages, so the probe builds both: each
    # compiler refuses the other's file, and a cell where only one half works
    # is a cell that fails half its tasks. `erlc` names the beam after the
    # module, so the run names the module rather than a file — and needs the
    # directory it landed in on the code path. Both runs name the emulator
    # rather than `erl`/`elixir`, which are scripts that exec their way to it.
    Language.ERLANG_ELIXIR: [
        (
            "main.erl",
            '-module(main).\n-export([main/0]).\nmain() -> io:put_chars("%s\\n").\n'
            % OK,
            [["erlc", "main.erl"]],
            list(
                beam_argv(
                    ".",
                    "-noshell",
                    "-pa",
                    ".",
                    "-s",
                    "main",
                    "main",
                    "-s",
                    "init",
                    "stop",
                    "--",
                )
            ),
        ),
        (
            "main.ex",
            'defmodule Server do\n  def main, do: IO.puts("%s")\nend\n' % OK,
            [["elixirc", "main.ex"]],
            list(elixir_argv(".", "-pa", ".", "-e", "Server.main()")),
        ),
    ],
    # `-o .` because the publish otherwise writes the binary into a hashed
    # directory under HOME. Native AOT, so what comes out needs neither the SDK
    # nor a `keep`.
    Language.CSHARP: [
        (
            "main.cs",
            'class Server { static void Main() { System.Console.WriteLine("%s"); } }\n'
            % OK,
            [["dotnet", "publish", "main.cs", "-o", ".", "-p:NuGetAudit=false"]],
            ["./main"],
        )
    ],
    # A code generator and the base image's linker, with no frontend anywhere
    # between them. Freestanding for the same reason the Assembly cell is — no
    # libc to link against — so this is the Assembly probe's program written as
    # what a compiler would have emitted.
    Language.LLVM_IR: [
        (
            "main.ll",
            'target triple = "x86_64-unknown-linux-gnu"\n'
            f'@msg = private constant [{len(OK) + 1} x i8] c"{OK}\\0A"\n'
            "define void @_start() noreturn {\n"
            '  call i64 asm sideeffect "syscall", '
            '"={ax},{ax},{di},{si},{dx},~{rcx},~{r11},~{memory}"'
            f"(i64 1, i64 1, ptr @msg, i64 {len(OK) + 1})\n"
            '  call i64 asm sideeffect "syscall", '
            '"={ax},{ax},{di},~{rcx},~{r11},~{memory}"(i64 60, i64 0)\n'
            "  unreachable\n"
            "}\n",
            [
                ["llc", "-O3", "-filetype=obj", "-o", "main.o", "main.ll"],
                ["ld", "-o", "main", "main.o"],
            ],
            ["./main"],
        )
    ],
    # scalac writes a jar and names the entry point itself, and the two
    # commands after it fold in the standard library the jar would otherwise be
    # missing once the toolchain is sealed.
    Language.SCALA: [
        (
            "main.scala",
            'object main:\n  def main(args: Array[String]): Unit = println("%s")\n'
            % OK,
            [
                ["scalac", "-d", "main.jar", "main.scala"],
                [
                    "unzip",
                    "-oq",
                    SCALA_LIBRARY_JAR,
                    "-d",
                    "library",
                    "-x",
                    "META-INF/*",
                ],
                ["jar", "--update", "--file", "main.jar", "-C", "library", "."],
            ],
            [str(jvm_java(Language.SCALA)), "-jar", "main.jar"],
        )
    ],
    # The type annotation is what makes the file TypeScript; tsc names the
    # output after the source, which is why the run starts main.js without a
    # build flag saying so. `--noCheck` because checking `import "node:net"`
    # needs @types/node and the image has no registry.
    Language.JS_TS: [
        (
            "main.ts",
            'const banner: string = "%s";\nconsole.log(banner);\n' % OK,
            [
                [
                    "tsc",
                    "--noCheck",
                    "--target",
                    "es2023",
                    "--module",
                    "commonjs",
                    "--esModuleInterop",
                    "--outDir",
                    ".",
                    "main.ts",
                ]
            ],
            ["node", "main.js"],
        )
    ],
    # g++ over the C++ name, because it is the one compiler that accepts both
    # of the cell's languages. The source is deliberately C-flavored — that it
    # builds under g++ is the grouping the cell rests on.
    Language.C_CPP: [
        (
            "main.cpp",
            '#include <cstdio>\nint main(void) { std::puts("%s"); return 0; }\n' % OK,
            [["g++", "-O3", "-pthread", "-o", "main", "main.cpp"]],
            ["./main"],
        )
    ],
    Language.FORTRAN: [
        (
            "main.f90",
            "program main\n  print '(a)', '%s'\nend program main\n" % OK,
            [["gfortran", "-O2", "-o", "main", "main.f90"]],
            ["./main"],
        )
    ],
    # Nothing to build: Clojure's compiler is its runtime, so the cell is the
    # kept `clojure` launcher plus the jar the file-by-file sealing leaves
    # readable.
    Language.CLOJURE: [
        (
            "main.clj",
            '(println "%s")\n' % OK,
            [],
            list(clojure_argv("main.clj")),
        )
    ],
    # `-free` because the task hands the student a bare file, and fixed-form
    # column rules are 1968's problem. COB_LIBS (see BUILD_ENV) swaps the
    # default shared libcob for the static archive, which is what lets the
    # artifact outlive the sealed directory.
    Language.COBOL: [
        (
            "main.cob",
            "IDENTIFICATION DIVISION.\n"
            "PROGRAM-ID. MAIN.\n"
            "PROCEDURE DIVISION.\n"
            '    DISPLAY "%s".\n' % OK + "    STOP RUN.\n",
            [["cobc", "-x", "-free", "-O2", "-o", "main", "main.cob"]],
            ["./main"],
        )
    ],
    # gplc links the Prolog engine in statically, so the binary is standalone;
    # `halt` because a gplc executable otherwise falls into the top-level REPL
    # after its initialization goal.
    Language.PROLOG: [
        (
            "main.pl",
            ":- initialization(main).\nmain :- write('%s'), nl, halt.\n" % OK,
            [["gplc", "-o", "main", "main.pl"]],
            ["./main"],
        )
    ],
    Language.MOJO: [
        (
            "main.mojo",
            'def main():\n    print("%s")\n' % OK,
            [["mojo", "build", "-o", "main", "main.mojo"]],
            ["./main"],
        )
    ],
}

# What a grader adds to a build's environment where the wrapper alone is not
# enough, keyed by probe filename. `ELIXIR_ERL_OPTIONS=+fnu` because the VM's
# filename encoding follows the locale: without one it comes up latin1 and
# warns on stderr. Kept alongside the image's `LC_CTYPE` as what holds if the
# locale ever goes missing.
BUILD_ENV: dict[str, dict[str, str]] = {
    "main.pas": {"PPC_CONFIG_PATH": f"{TOOLCHAINS_DIR}/pascal/etc"},
    "main.cs": {
        "NUGET_PACKAGES": f"{TOOLCHAINS_DIR}/csharp/nuget",
        "DOTNET_CLI_TELEMETRY_OPTOUT": "1",
        "DOTNET_NOLOGO": "1",
    },
    "main.ex": {"ELIXIR_ERL_OPTIONS": "+fnu"},
    # The static libcob in place of the default `-lcob`, so the binary
    # outlives the sealed directory; gmp is libcob's own arithmetic.
    "main.cob": {"COB_LIBS": f"{COBOL_LIBCOB_ARCHIVE} -lgmp -lm"},
}


def _as_user(user: str, argv: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["runuser", "-u", user, "--", *argv],
        cwd=cwd,
        capture_output=True,
        text=True,
    )


def _as_student(argv: list[str], cwd: Path) -> subprocess.CompletedProcess:
    """Run a command the way the agent would: as the student, not as root. A
    runtime root can reach but the student cannot is one the task lacks."""
    return _as_user("student", argv, cwd)


def _as_builder(argv: list[str], cwd: Path) -> subprocess.CompletedProcess:
    """Run a command the way the build MCP tool would: as the builder uid, the
    only uid the toolchain is granted to."""
    return _as_user("builder", argv, cwd)


def _run_env(language: Language) -> list[str]:
    """The cell's `run_env`, as `env` arguments. Prepended rather than passed
    through, since `runuser` sits between this and the run."""
    return [f"{name}={value}" for name, value in run_environment(language).items()]


def check(language: Language) -> None:
    install_toolchain(language)

    # Both uids work here: the student writes sources and runs artifacts, the
    # builder compiles between the two.
    workdir = Path(tempfile.mkdtemp(prefix=f"probe-{language.value}-", dir="/tmp"))
    workdir.chmod(0o777)

    check_student_never_reaches_the_toolchain(language, PROGRAMS[language], workdir)
    check_build_sandbox(language)
    if language is Language.HASKELL:
        check_ghc_preprocessor_pragma(language)
    if language in CONFINED_CELLS:
        for program in (
            PROGRAMS[language][0],
            *CONFINED_ONLY_PROGRAMS.get(language, []),
        ):
            check_the_scored_run_is_confined(language, program)
    if TOOLCHAINS[language].run_preload:
        check_the_run_preload_keeps_the_confinement(language)
    if TOOLCHAINS[language].run_flags:
        check_the_run_flags_are_load_bearing(language, PROGRAMS[language][0])
    if language in ESCAPES:
        check_escapes_are_refused(language, PROGRAMS[language][0])
    if language in RUNTIME_ESCAPES:
        check_runtime_escapes_are_refused(language, PROGRAMS[language][0])

    # Everything is built before anything is sealed, which is both the order a
    # real run has and the only order available: a sealed cell cannot be
    # reopened in one container, since dnf will not reinstall over packages it
    # considers present.
    for program in PROGRAMS[language]:
        build_one(language, workdir, program)

    for program in PROGRAMS[language]:
        print(f"PASS {language}: built and ran {program[0]} with its own toolchain")

    check_c_frontend(language, workdir)
    check_sealing(language, workdir, PROGRAMS[language])


# Where a compile-time hook would drop something for the scored run to find.
# The first three are the image's 1777 directories; the last is the student's
# workdir, which the build must not be able to read.
SANDBOX_DIRS = ("/tmp", "/var/tmp", "/dev/shm", str(STUDENT_WORKDIR))

PAYLOAD_NAME = "karotte_probe_payload"

# What a compiler talked into running a program at compile time would run.
PLANTED_NAME = "karotte_probe_planted.sh"


def _plant(path: Path) -> None:
    """A stand-in for the program a `-pgmF` pragma names: it writes a payload
    everywhere the student could later look, then fails."""
    writes = "\n".join(
        f"echo payload > {directory}/{PAYLOAD_NAME} 2>/dev/null"
        for directory in SANDBOX_DIRS
    )
    path.write_text(f"#!/bin/sh\n{writes}\necho EXECUTED\nexit 1\n")
    path.chmod(0o755)
    shutil.chown(path, user="student", group="student")


def _payloads_outside_the_build() -> list[str]:
    return [
        f"{directory}/{PAYLOAD_NAME}"
        for directory in SANDBOX_DIRS
        if Path(directory, PAYLOAD_NAME).exists()
    ]


# Runs as the builder inside the build's own mount namespace. %-formatted so
# the `{artifact}` `resolve_argv` fills in stays a single pair of braces (see
# the PROGRAMS comment).
SANDBOX_PROBE = """
for planted in %(workdir)s/%(planted)s /tmp/%(planted)s; do
  if "$planted" > /dev/null 2>&1; then echo "RAN $planted"; else echo "BLOCKED $planted"; fi
done
if [ -n "$(ls -A %(workdir)s 2>/dev/null)" ]; then
  echo "READ %(workdir)s"
else
  echo "NOREAD %(workdir)s"
fi
for directory in %(dirs)s; do
  if echo payload > "$directory/%(payload)s" 2>/dev/null; then
    echo "WROTE $directory"
  else
    echo "NOWRITE $directory"
  fi
done
touch "{artifact}"
""" % {
    "workdir": STUDENT_WORKDIR,
    "planted": PLANTED_NAME,
    "dirs": " ".join(SANDBOX_DIRS),
    "payload": PAYLOAD_NAME,
}


def why_the_build_sandbox_leaks(report: str) -> str | None:
    """What the probe's report says the build could still do, or None.

    Read as the absence of the good answer rather than the presence of the bad
    one: `NOREAD` contains `READ`, and looking for the second is a check that
    fails whichever the probe reports.
    """
    if "RAN " in report:
        return "the build executed a planted script"
    if f"NOREAD {STUDENT_WORKDIR}" not in report:
        return (
            "the build can read the student's workdir, so a source file can "
            "name a second one there"
        )
    return None


def check_build_sandbox(language: Language) -> None:
    """A build can be made to run the student's code — GHC's `-pgmF` names a
    program from inside the source file, and the pinned argv never sees it.
    What that must not buy is a payload the scored run can pick up, so the
    build's filesystem is checked rather than the compilers' behaviour: every
    shared writable directory is its own tmpfs, and so is the workdir, which
    the build must not be able to read either — a source file naming a second
    file there by absolute path is how a compiler with an include directive
    gets around the one staged file.
    """
    for directory in SANDBOX_DIRS:
        Path(directory, PAYLOAD_NAME).unlink(missing_ok=True)
    planted = [STUDENT_WORKDIR / PLANTED_NAME, Path("/tmp") / PLANTED_NAME]
    for path in planted:
        _plant(path)

    build_dir = make_build_dir()
    source = build_dir / "probe.sh"
    source.write_text("")
    submission = Submission(
        source="probe.sh", build=(("/bin/sh", "-c", SANDBOX_PROBE),)
    )
    result = build_submission(submission, source, build_dir)
    if result.error is not None:
        raise SystemExit(
            f"FAIL {language}: sandbox probe did not build: {result.error}"
        )

    report = result.stdout.decode(errors="replace")
    leak = why_the_build_sandbox_leaks(report)
    if leak is not None:
        raise SystemExit(f"FAIL {language}: {leak}\n{report}")
    escaped = _payloads_outside_the_build()
    if escaped:
        raise SystemExit(
            f"FAIL {language}: what the build wrote outlived it: {', '.join(escaped)}"
        )

    for path in planted:
        path.unlink(missing_ok=True)
    print(f"PASS {language}: nothing a build writes survives it or is student-visible")


def check_ghc_preprocessor_pragma(language: Language) -> None:
    """The vector the sandbox exists for, run for real: a `-pgmF` pragma in
    the submitted source names a program for GHC to run at compile time,
    without the pinned argv changing. The build may fail; what must hold is
    that the program left nothing behind."""
    for directory in SANDBOX_DIRS:
        Path(directory, PAYLOAD_NAME).unlink(missing_ok=True)
    planted = STUDENT_WORKDIR / PLANTED_NAME
    _plant(planted)

    build_dir = make_build_dir()
    source = build_dir / "main.hs"
    source.write_text(
        '{-# OPTIONS_GHC -F -pgmF %s #-}\nmain :: IO ()\nmain = putStrLn "x"\n'
        % planted
    )
    submission = Submission(
        source="main.hs", build=(("ghc", "-O0", "-o", "{artifact}", "{source}"),)
    )
    result = build_submission(submission, source, build_dir)

    escaped = _payloads_outside_the_build()
    if escaped:
        raise SystemExit(
            f"FAIL {language}: a -pgmF pragma left {', '.join(escaped)} behind"
        )
    planted.unlink(missing_ok=True)
    verdict = "was refused" if result.error is not None else "ran but reached nothing"
    print(f"PASS {language}: the -pgmF pragma {verdict}")


UNCONFINED_CELLS = frozenset({Language.PYTHON})

CONFINED_CELLS = frozenset(Language) - UNCONFINED_CELLS

# Programs that go through the confined run alone: a hello-world says nothing
# about a runtime that reaches for a library after the shim's constructor.
CONFINED_ONLY_PROGRAMS: dict[Language, list[Program]] = {
    Language.PASCAL: [
        (
            "threads.pas",
            "program threads;\n"
            "{$mode objfpc}\n"
            "uses cthreads;\n"
            "var\n"
            "  ran: boolean;\n"
            "function worker(parameter: pointer): ptrint;\n"
            "begin\n"
            "  ran := true;\n"
            "  result := 0;\n"
            "end;\n"
            "var\n"
            "  id: TThreadID;\n"
            "begin\n"
            "  ran := false;\n"
            "  id := BeginThread(@worker);\n"
            "  WaitForThreadTerminate(id, 5000);\n"
            "  if ran then writeln('%s');\n"
            "end.\n" % OK,
            [["fpc", "-O3", "-othreads", "threads.pas"]],
            ["./threads"],
        )
    ],
}

# An interpreter's argv names the file it is handed, not the build's output —
# and the BEAM's names neither, since the emulator is argv[0] and the module it
# runs is found on the code path.
RUN_ARTIFACT = {
    Language.ERLANG_ELIXIR: "main.beam",
    Language.JAVA: "main.jar",
    Language.JS_TS: "main.js",
    Language.KOTLIN: "main.jar",
    Language.SCALA: "main.jar",
}


def artifact_name(language: Language, run_command: list[str]) -> str:
    """What the cell's run starts, as a name inside the build directory."""
    return RUN_ARTIFACT.get(language) or run_command[0].removeprefix("./")


_GRADING_BUILDS: dict[tuple[Language, str, str, bool], tuple[Path, str | None]] = {}


def _build_as_grading_would(
    language: Language, program: Program, source: str, allow_unsafe: bool = False
) -> tuple[Path, str | None]:
    """One source through the cell's grading build, or staged where the cell
    has nothing to build, plus why the build failed; the returned directory is
    the one the run starts in. Built once per (source, flag): the sealed dir is
    read-only, so checks that compile the same program share it safely."""
    key = (language, program[0], source, allow_unsafe)
    if key not in _GRADING_BUILDS:
        _GRADING_BUILDS[key] = _build_as_grading_would_uncached(
            language, program, source, allow_unsafe
        )
    return _GRADING_BUILDS[key]


def _build_as_grading_would_uncached(
    language: Language, program: Program, source: str, allow_unsafe: bool
) -> tuple[Path, str | None]:
    filename, _, build_commands, run_command = program
    build_dir = make_build_dir()
    staged = build_dir / filename
    staged.write_text(source)
    if not build_commands:
        # The run is launched as the student and has to be able to read the file.
        seal_build_dir(build_dir)
        return build_dir, None

    submission = Submission(
        source=filename,
        build=tuple(tuple(command) for command in build_commands),
        artifact=artifact_name(language, run_command),
        build_env=BUILD_ENV.get(filename, {}),
        allow_unsafe=allow_unsafe,
    )
    return build_dir, build_submission(submission, staged, build_dir).error


def _run_confined(
    language: Language, build_dir: Path, run_command: list[str], harden: bool = True
) -> subprocess.CompletedProcess:
    """Launch the cell's run the way a scored one is launched: as the student,
    with the shim the cell's runtime needs and the flags the template puts in
    front of it."""
    env = sandboxed_env(
        {"PATH": SAFE_PATH, "HOME": str(build_dir), "LC_CTYPE": UTF8_LOCALE},
        language,
    )
    argv = harden_run_argv(list(run_command), language) if harden else list(run_command)
    return _as_student(
        ["env", *(f"{name}={value}" for name, value in env.items()), *argv],
        build_dir,
    )


def check_the_scored_run_is_confined(language: Language, program: Program) -> None:
    """Build the cell's own hello-world the way grading does and run it the way
    a scored run does.

    Two things this is the only place to learn: whether the compiler, with what
    the template injects, links an artifact the gate accepts — a statically
    linked one has no loader to read `LD_PRELOAD` — and whether the language's
    runtime still starts with the confinement in front of it.
    """
    _, source, _, run_command = program
    build_dir, error = _build_as_grading_would(language, program, source)
    if error is not None:
        raise SystemExit(
            f"FAIL {language}: the grading build produced nothing the scored run "
            f"could confine: {error}"
        )

    finished = _run_confined(language, build_dir, run_command)
    if finished.returncode != 0 or OK not in finished.stdout + finished.stderr:
        raise SystemExit(
            f"FAIL {language}: what the run starts does not run under the "
            f"confinement, exited {finished.returncode}\n"
            f"stdout:\n{finished.stdout}\nstderr:\n{finished.stderr}"
        )
    print(f"PASS {language}: what the grading build leaves runs confined")


def check_the_run_preload_keeps_the_confinement(language: Language) -> None:
    """A cell that loads more than the shim has to be probed with all of it.
    The probe answers about the process it is in, and what the run's own
    `LD_PRELOAD` puts there is part of that process.

    Nothing is expected but the writable directories the run's mount namespace
    closes, which this has none of — and their absence means the probe never
    ran, so it is required rather than merely allowed.
    """
    workdir = Path(tempfile.mkdtemp(prefix=f"preload-{language.value}-", dir="/tmp"))
    workdir.chmod(0o777)
    env = sandboxed_env(
        {"PATH": SAFE_PATH, "HOME": str(workdir), "LC_CTYPE": UTF8_LOCALE}, language
    )
    argv = [str(PROBE_BINARY), *PROBE_MODE.get(run_confinement(language), [])]
    finished = _as_student(
        ["env", *(f"{name}={value}" for name, value in env.items()), *argv], workdir
    )

    reported = (finished.stdout + finished.stderr).splitlines()
    if not any("is writable" in line for line in reported):
        raise SystemExit(
            f"FAIL {language}: the probe never ran with {env['LD_PRELOAD']}\n"
            f"stdout:\n{finished.stdout}\nstderr:\n{finished.stderr}"
        )
    unexpected = [line for line in reported if "is writable" not in line]
    if unexpected:
        raise SystemExit(
            f"FAIL {language}: the cell's preloads left something open:\n"
            + "\n".join(unexpected)
        )
    print(f"PASS {language}: the confinement holds with the cell's preloads in front")


def check_the_run_flags_are_load_bearing(language: Language, program: Program) -> None:
    """What the template injects into the run costs the student something, so
    this is the alarm for the day it stops buying anything. A flag earns its
    place one of two ways: the run does not start without it (JS's `--jitless`),
    or an escape the hardened run refuses is open without it (the JVM's deny)."""
    _, source, _, run_command = program
    build_dir, error = _build_as_grading_would(language, program, source)
    if error is not None:
        raise SystemExit(f"FAIL {language}: the grading build failed: {error}")

    if _run_confined(language, build_dir, run_command, harden=False).returncode != 0:
        print(f"PASS {language}: the injected run flags are what lets the run start")
        return

    for name, escape in RUNTIME_ESCAPES.get(language, {}).items():
        escape_dir, escape_error = _build_as_grading_would(
            language, program, escape.source
        )
        if escape_error is not None:
            continue
        finished = _run_confined(language, escape_dir, run_command, harden=False)
        if "ESCAPED" in (finished.stdout + finished.stderr):
            print(f"PASS {language}: without the run flags, {name} is open")
            return

    flags = " ".join(TOOLCHAINS[language].run_flags)
    raise SystemExit(
        f"FAIL {language}: the run survives without {flags} and no escape opens, "
        "so the flags cost the student something and buy nothing"
    )


class Escape(NamedTuple):
    """A submission that would run its own machine code before, or instead of,
    the confinement the scored run installs."""

    source: str

    caught_by_the_gate: bool = True


ESCAPES: dict[Language, dict[str, Escape]] = {
    Language.RUST: {
        "preinit": Escape(
            "#[used]\n"
            '#[link_section = ".preinit_array"]\n'
            'static HOOK: extern "C" fn() = { extern "C" fn h() {} h };\n'
            "fn main() {}\n"
        ),
        "ifunc": Escape(
            'extern "C" fn real() {}\n'
            "#[no_mangle]\n"
            'pub extern "C" fn resolver() -> *const () { real as *const () }\n'
            "core::arch::global_asm!(\n"
            '    ".globl chosen",\n'
            '    ".type chosen, %gnu_indirect_function",\n'
            '    ".set chosen, resolver"\n'
            ");\n"
            'extern "C" { fn chosen(); }\n'
            "fn main() { unsafe { chosen() } }\n"
        ),
    },
    Language.C_CPP: {
        "preinit": Escape(
            'extern "C" void hook(void) {}\n'
            '__attribute__((section(".preinit_array"), used))\n'
            "static void (*entry)(void) = hook;\n"
            "int main() { return 0; }\n"
        ),
        "ifunc": Escape(
            'extern "C" void real(void) {}\n'
            'extern "C" void (*resolver(void))(void) { return real; }\n'
            'extern "C" void chosen(void) __attribute__((ifunc("resolver")));\n'
            "int main() { chosen(); return 0; }\n"
        ),
        "interp": Escape(
            'extern "C" const char loader[] __attribute__((section(".interp"), used))\n'
            '    = "/workdir/loader";\n'
            "int main() { return 0; }\n"
        ),
    },
    Language.ZIG: {
        "preinit": Escape(
            "fn hook() callconv(.c) void {}\n"
            'export const HOOK linksection(".preinit_array") = &hook;\n'
            "pub fn main() void {}\n"
        ),
    },
    Language.ASSEMBLY: {
        "interp": Escape(
            '.section .interp, "a"\n'
            '        .asciz "/workdir/loader"\n'
            ".section .text\n"
            ".globl _start\n"
            "_start:\n"
            "        movq $60, %rax\n"
            "        xorq %rdi, %rdi\n"
            "        syscall\n"
        ),
    },
    Language.LLVM_IR: {
        "interp": Escape(
            'target triple = "x86_64-unknown-linux-gnu"\n'
            'module asm ".section .interp,\\22a\\22"\n'
            'module asm ".asciz \\22/workdir/loader\\22"\n'
            'module asm ".text"\n'
            "define void @_start() noreturn {\n"
            '  call i64 asm sideeffect "syscall", '
            '"={ax},{ax},{di},~{rcx},~{r11},~{memory}"(i64 60, i64 0)\n'
            "  unreachable\n"
            "}\n"
        ),
    },
    Language.SWIFT: {
        "unsafe": Escape(
            "let p = UnsafeMutablePointer<Int>.allocate(capacity: 1)\n"
            "p.pointee = 5\n"
            "print(p.pointee)\n",
            caught_by_the_gate=False,
        ),
    },
    # Both are Template Haskell, which runs student code at compile time.
    # Neither is the gate's to catch — the artifact is an ordinary dynamic PIE
    # — so the build has to refuse them, and it does by leaving a splice
    # nowhere to be evaluated.
    #
    # The first is the load-bearing one: it builds under `allow_unsafe`, so the
    # injected flag is the only thing refusing it, and dropping that flag fails
    # this check. The second is the vector worth having written down — machine
    # code assembled straight into `.text` — but it is also blocked here by the
    # sealed C frontend, so on its own it would not notice the flag going away.
    Language.HASKELL: {
        "template haskell": Escape(
            "{-# LANGUAGE TemplateHaskell #-}\n"
            "import Language.Haskell.TH\n"
            "main :: IO ()\n"
            'main = putStrLn $( litE (stringL "spliced") )\n',
            caught_by_the_gate=False,
        ),
        "machine code via addForeignSource": Escape(
            "{-# LANGUAGE TemplateHaskell #-}\n"
            "{-# LANGUAGE ForeignFunctionInterface #-}\n"
            "import Language.Haskell.TH\n"
            "import Language.Haskell.TH.Syntax\n"
            "$(do addForeignSource LangAsm\n"
            '       ".globl injected\\n.type injected,@function\\ninjected:\\n  movl $3735928559, %eax\\n  ret\\n"\n'
            "     return [])\n"
            'foreign import ccall unsafe "injected" injected :: IO Int\n'
            "main :: IO ()\n"
            "main = injected >>= print\n",
            caught_by_the_gate=False,
        ),
    },
    # `#:property AllowUnsafeBlocks=true` in the source is the interesting
    # half: it is how a file-based app turns its own unsafe on, and the
    # command-line `-p:` the template injects beats it (CS0227).
    Language.CSHARP: {
        "unsafe": Escape(
            "#:property AllowUnsafeBlocks=true\n"
            "class Server {\n"
            "    static unsafe void Main() {\n"
            "        int x = 5; int* p = &x;\n"
            '        System.Console.WriteLine("unsafe " + *p);\n'
            "    }\n"
            "}\n",
            caught_by_the_gate=False,
        ),
    },
}


def check_escapes_are_refused(language: Language, program: Program) -> None:
    """Every way out of this cell that has been thought of, tried for real
    against the grading build."""
    _, ordinary = _build_as_grading_would(language, program, program[1])
    if ordinary is not None:
        raise SystemExit(
            f"FAIL {language}: the hardened build refused an ordinary program, so "
            f"nothing below is a test of anything: {ordinary}"
        )

    for name, escape in ESCAPES[language].items():
        _, refused = _build_as_grading_would(language, program, escape.source)
        if refused is None:
            raise SystemExit(f"FAIL {language}: the {name} escape was built anyway")

        if not escape.caught_by_the_gate:
            continue
        _, opened = _build_as_grading_would(
            language, program, escape.source, allow_unsafe=True
        )
        if opened is None:
            raise SystemExit(
                f"FAIL {language}: with the language's forbid opened, the {name} "
                "escape produced an artifact the gate let through"
            )
        if opened.startswith("`"):
            raise SystemExit(
                f"FAIL {language}: the {name} escape was refused by the compiler, "
                f"not the artifact gate, so the gate is untested: {opened}"
            )

    print(f"PASS {language}: {', '.join(ESCAPES[language])} refused")


# The two ways a JVM run reaches a Java compiler: the service the JDK publishes,
# and the class file behind it, which a run can read out of the runtime image
# and define into a loader of its own — so `--limit-modules` proves nothing and
# only a runtime linked without those modules answers no to both.
JAVAC_CLASS = "/modules/jdk.compiler/com/sun/tools/javac/api/JavacTool.class"

KOTLIN_SMUGGLE = (
    """fun main() {
    val service = try {
        Class.forName("javax.tools.ToolProvider")
            .getMethod("getSystemJavaCompiler").invoke(null) != null
    } catch (e: Throwable) { false }
    val onDisk = try {
        val jrt = java.nio.file.FileSystems.getFileSystem(java.net.URI.create("jrt:/"))
        java.nio.file.Files.exists(jrt.getPath("%s"))
    } catch (e: Throwable) { false }
    println(if (service || onDisk) "ESCAPED" else "REFUSED")
}
"""
    % JAVAC_CLASS
)

JAVA_SMUGGLE = (
    """public class main {
    public static void main(String[] args) {
        boolean service;
        try {
            service = Class.forName("javax.tools.ToolProvider")
                .getMethod("getSystemJavaCompiler").invoke(null) != null;
        } catch (Throwable t) { service = false; }
        boolean onDisk;
        try {
            java.nio.file.FileSystem jrt =
                java.nio.file.FileSystems.getFileSystem(java.net.URI.create("jrt:/"));
            onDisk = java.nio.file.Files.exists(jrt.getPath("%s"));
        } catch (Throwable t) { onDisk = false; }
        System.out.println(service || onDisk ? "ESCAPED" : "REFUSED");
    }
}
"""
    % JAVAC_CLASS
)

SCALA_SMUGGLE = (
    """object main:
  def main(args: Array[String]): Unit =
    val service =
      try
        Class.forName("javax.tools.ToolProvider")
          .getMethod("getSystemJavaCompiler").invoke(null) != null
      catch
        case _: Throwable => false
    val onDisk =
      try
        val jrt = java.nio.file.FileSystems.getFileSystem(java.net.URI.create("jrt:/"))
        java.nio.file.Files.exists(jrt.getPath("%s"))
      catch
        case _: Throwable => false
    println(if service || onDisk then "ESCAPED" else "REFUSED")
"""
    % JAVAC_CLASS
)

KOTLIN_FFM = """import java.lang.foreign.*
fun main() {
    try {
        val linker = Linker.nativeLinker()
        val strlen = linker.downcallHandle(
            linker.defaultLookup().find("strlen").get(),
            FunctionDescriptor.of(ValueLayout.JAVA_LONG, ValueLayout.ADDRESS))
        Arena.ofConfined().use { a ->
            val n = strlen.invoke(a.allocateFrom("hi")) as Long
            println(if (n == 2L) "ESCAPED" else "REFUSED")
        }
    } catch (t: Throwable) { println("REFUSED") }
}
"""

SCALA_FFM = """import java.lang.foreign.*
object main:
  def main(args: Array[String]): Unit =
    try
      val linker = Linker.nativeLinker()
      val strlen = linker.downcallHandle(
        linker.defaultLookup().find("strlen").get(),
        FunctionDescriptor.of(ValueLayout.JAVA_LONG, ValueLayout.ADDRESS))
      val a = Arena.ofConfined()
      val n = strlen.invoke(a.allocateFrom("hi")).asInstanceOf[Long]
      println(if n == 2L then "ESCAPED" else "REFUSED")
    catch case _: Throwable => println("REFUSED")
"""

JAVA_FFM = """public class main {
    public static void main(String[] args) {
        try {
            Class<?> L = Class.forName("java.lang.foreign.Linker");
            Class<?> SL = Class.forName("java.lang.foreign.SymbolLookup");
            Class<?> FD = Class.forName("java.lang.foreign.FunctionDescriptor");
            Class<?> ML = Class.forName("java.lang.foreign.MemoryLayout");
            Class<?> VL = Class.forName("java.lang.foreign.ValueLayout");
            Class<?> MS = Class.forName("java.lang.foreign.MemorySegment");
            Class<?> AR = Class.forName("java.lang.foreign.Arena");
            Class<?> OPT = Class.forName("java.lang.foreign.Linker$Option");
            Object linker = L.getMethod("nativeLinker").invoke(null);
            Object lookup = L.getMethod("defaultLookup").invoke(linker);
            Object jlong = VL.getField("JAVA_LONG").get(null);
            Object addr = VL.getField("ADDRESS").get(null);
            Object mls = java.lang.reflect.Array.newInstance(ML, 1);
            java.lang.reflect.Array.set(mls, 0, addr);
            Object fd = FD.getMethod("of", ML, mls.getClass()).invoke(null, jlong, mls);
            Object sym = ((java.util.Optional<?>) SL.getMethod("find", String.class)
                .invoke(lookup, "strlen")).get();
            Object opts = java.lang.reflect.Array.newInstance(OPT, 0);
            Object handle = L.getMethod("downcallHandle", MS, FD, opts.getClass())
                .invoke(linker, sym, fd, opts);
            Object arena = AR.getMethod("ofConfined").invoke(null);
            Object s = AR.getMethod("allocateFrom", String.class).invoke(arena, "hi");
            Object n = ((java.lang.invoke.MethodHandle) handle).invokeWithArguments(s);
            System.out.println(((Long) n) == 2L ? "ESCAPED" : "REFUSED");
        } catch (Throwable t) {
            System.out.println("REFUSED");
        }
    }
}
"""

JULIA_SPAWN = """try
    run(`id`)
    println("ESCAPED")
catch e
    println("REFUSED")
end
"""

ERLANG_LOAD_NIF = """-module(main).
-export([main/0]).
main() ->
    case catch erlang:load_nif("/tmp/karotte_probe_nif", 0) of
        {error, _} -> io:put_chars("REFUSED\\n");
        ok -> io:put_chars("ESCAPED\\n");
        _ -> io:put_chars("REFUSED\\n")
    end.
"""

ERLANG_DDLL = """-module(main).
-export([main/0]).
main() ->
    case catch erl_ddll:load_driver("/tmp", "karotte_probe_drv") of
        {error, _} -> io:put_chars("REFUSED\\n");
        ok -> io:put_chars("ESCAPED\\n");
        _ -> io:put_chars("REFUSED\\n")
    end.
"""

ERLANG_PORT_SPAWN = """-module(main).
-export([main/0]).
main() ->
    io:put_chars(os:cmd("id")),
    io:put_chars("ESCAPED\\n").
"""


class RuntimeEscape(NamedTuple):
    """Something a cell's runtime offers that the confinement has to refuse,
    which no artifact gate can see because the artifact is the runtime's input.

    The source prints REFUSED where the confinement held and ESCAPED where it
    did not — except where it never gets to print anything.
    """

    source: str

    refused_by_dying: bool = False


FORTRAN_SPAWN = """program main
  integer :: status
  call execute_command_line("id", exitstat=status, cmdstat=status)
  if (status == 0) then
    print '(a)', 'ESCAPED'
  else
    print '(a)', 'REFUSED'
  end if
end program main
"""

COBOL_SPAWN = """IDENTIFICATION DIVISION.
PROGRAM-ID. MAIN.
DATA DIVISION.
WORKING-STORAGE SECTION.
01 RC PIC S9(9) COMP-5.
PROCEDURE DIVISION.
    MOVE 0 TO RC.
    CALL "SYSTEM" USING "id" RETURNING RC.
    IF RC = 0
        DISPLAY "ESCAPED"
    ELSE
        DISPLAY "REFUSED"
    END-IF.
    STOP RUN.
"""

PROLOG_SPAWN = """:- initialization(main).
main :-
    ( catch(shell('id', 0), _, fail) -> write('ESCAPED') ; write('REFUSED') ),
    nl, halt.
"""

# The two doors a JVM cell has: the compiler the JDK ships as a module, and
# native code through the foreign-function API. Both are closed the way the
# other JVM cells close them — a jlink'ed runtime and `--illegal-native-access`.
CLOJURE_SMUGGLE = (
    """(let [service (try (some?
                    (.invoke (.getMethod (Class/forName "javax.tools.ToolProvider")
                                         "getSystemJavaCompiler"
                                         (into-array Class []))
                             nil (object-array 0)))
                  (catch Throwable _ false))
      on-disk (try (java.nio.file.Files/exists
                     (.getPath (java.nio.file.FileSystems/getFileSystem
                                 (java.net.URI/create "jrt:/"))
                               "%s" (into-array String []))
                     (into-array java.nio.file.LinkOption []))
                (catch Throwable _ false))]
  (println (if (or service on-disk) "ESCAPED" "REFUSED")))
"""
    % JAVAC_CLASS
)

CLOJURE_FFM = """(println
  (try
    (let [linker (java.lang.foreign.Linker/nativeLinker)
          sym (.orElseThrow (.find (.defaultLookup linker) "strlen"))
          fd (java.lang.foreign.FunctionDescriptor/of
               java.lang.foreign.ValueLayout/JAVA_LONG
               (into-array java.lang.foreign.MemoryLayout
                           [java.lang.foreign.ValueLayout/ADDRESS]))
          h (.downcallHandle linker sym fd
                             (into-array java.lang.foreign.Linker$Option []))]
      (with-open [a (java.lang.foreign.Arena/ofConfined)]
        (if (= 2 (.invokeWithArguments h [(.allocateFrom a "hi")]))
          "ESCAPED" "REFUSED")))
    (catch Throwable _ "REFUSED")))
"""

MOJO_FFI = """from std.ffi import external_call
def main():
    var s = String("id")
    var rc = external_call["system", Int32](s.unsafe_ptr())
    print("ESCAPED" if rc == 0 else "REFUSED")
"""

MOJO_SPAWN = """from std.subprocess import run
def main() raises:
    _ = run("id")
    print("ESCAPED")
"""

RUNTIME_ESCAPES: dict[Language, dict[str, RuntimeEscape]] = {
    Language.JS_TS: {
        "child process": RuntimeEscape(
            'const cp = require("child_process");\n'
            'try { cp.execSync("id"); console.log("ESCAPED"); }\n'
            'catch (e) { console.log("REFUSED"); }\n'
        ),
        "webassembly": RuntimeEscape(
            "console.log(\n"
            '  typeof WebAssembly === "undefined" ? "REFUSED" : "ESCAPED"\n'
            ");\n"
        ),
    },
    Language.RUBY: {
        "subprocess": RuntimeEscape(
            "begin\n"
            '  puts system("id").nil? ? "REFUSED" : "ESCAPED"\n'
            "rescue SystemCallError\n"
            '  puts "REFUSED"\n'
            "end\n"
        ),
    },
    Language.KOTLIN: {
        "java compiler": RuntimeEscape(KOTLIN_SMUGGLE),
        "native code (FFM)": RuntimeEscape(KOTLIN_FFM),
    },
    Language.JAVA: {
        "java compiler": RuntimeEscape(JAVA_SMUGGLE),
        "native code (FFM)": RuntimeEscape(JAVA_FFM),
    },
    Language.SCALA: {
        "java compiler": RuntimeEscape(SCALA_SMUGGLE),
        "native code (FFM)": RuntimeEscape(SCALA_FFM),
    },
    Language.JULIA: {"subprocess": RuntimeEscape(JULIA_SPAWN)},
    # `external_call` is the whole of Mojo's FFI: any symbol already in the
    # process, `system` included. `std.subprocess.run` is the same door with a
    # nicer handle on it, and it does not survive being refused — the runtime
    # takes SIGSEGV rather than raising, so nothing prints.
    Language.MOJO: {
        "native call (FFI)": RuntimeEscape(MOJO_FFI),
        "subprocess": RuntimeEscape(MOJO_SPAWN, refused_by_dying=True),
    },
    Language.FORTRAN: {"subprocess": RuntimeEscape(FORTRAN_SPAWN)},
    Language.COBOL: {"subprocess": RuntimeEscape(COBOL_SPAWN)},
    Language.PROLOG: {"subprocess": RuntimeEscape(PROLOG_SPAWN)},
    Language.CLOJURE: {
        "java compiler": RuntimeEscape(CLOJURE_SMUGGLE),
        "native code (FFM)": RuntimeEscape(CLOJURE_FFM),
    },
    Language.ERLANG_ELIXIR: {
        "load_nif": RuntimeEscape(ERLANG_LOAD_NIF),
        "erl_ddll": RuntimeEscape(ERLANG_DDLL),
        "port spawn": RuntimeEscape(ERLANG_PORT_SPAWN, refused_by_dying=True),
    },
}


def check_runtime_escapes_are_refused(language: Language, program: Program) -> None:
    """The escapes an interpreted cell's runtime offers, which no artifact gate
    can see because the artifact is the interpreter's input, tried for real
    under the confinement the scored run has."""
    for name, escape in RUNTIME_ESCAPES[language].items():
        build_dir, error = _build_as_grading_would(language, program, escape.source)
        if error is not None:
            raise SystemExit(
                f"FAIL {language}: the {name} probe did not build, so nothing was "
                f"tested: {error}"
            )

        finished = _run_confined(language, build_dir, program[3])
        report = (finished.stdout + finished.stderr).strip()
        if escape.refused_by_dying:
            if "ESCAPED" in report or finished.returncode == 0:
                raise SystemExit(
                    f"FAIL {language}: {name} was expected to take the runtime down "
                    f"and did not, exited {finished.returncode}\n{report}"
                )
            continue
        if "REFUSED" not in report:
            raise SystemExit(
                f"FAIL {language}: {name} was not refused under the confinement, "
                f"exited {finished.returncode}\n{report}"
            )

    print(
        f"PASS {language}: {', '.join(RUNTIME_ESCAPES[language])} refused at run time"
    )


def check_student_never_reaches_the_toolchain(
    language: Language, programs: list[Program], workdir: Path
) -> None:
    """With the toolchain installed and open for the builder, every build
    command must still be out of the student's reach: the build MCP tool is
    the one way to compile."""
    build_tools = {
        command[0]
        for _, _, build_commands, _ in programs
        for command in build_commands
        if command[0] not in BASE_IMAGE_BUILD_UTILITIES
    }
    for tool in sorted(build_tools):
        if _as_student(["sh", "-lc", f"command -v {tool}"], workdir).returncode == 0:
            raise SystemExit(
                f"FAIL {language}: student can reach {tool} with the toolchain installed"
            )
        if _as_builder(["sh", "-lc", f"command -v {tool}"], workdir).returncode != 0:
            raise SystemExit(
                f"FAIL {language}: builder cannot reach {tool} with the toolchain installed"
            )
    if build_tools:
        print(f"PASS {language}: the toolchain is the builder's, not the student's")


def build_one(language: Language, workdir: Path, program: Program) -> None:
    """Build one of the cell's languages as the builder and run what came out
    as the student, before any sealing.

    The builder's passwd home does not exist, so its commands get `HOME` set to
    the working directory the way `build_submission` does.
    """
    filename, source, build_commands, run_command = program

    # Prepended as `env` rather than passed through: `runuser` is between this
    # and the compiler.
    build_env = [
        f"{name}={value}" for name, value in BUILD_ENV.get(filename, {}).items()
    ]

    program_path = workdir / filename
    program_path.write_text(source)
    shutil.chown(program_path, user="student", group="student")

    def run_as(
        as_user, command: list[str], env: list[str] = []
    ) -> subprocess.CompletedProcess:
        result = as_user(["env", *env, *command] if env else command, workdir)
        if result.returncode != 0:
            raise SystemExit(
                f"FAIL {language}: {' '.join(command)} exited {result.returncode}\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        return result

    for command in build_commands:
        run_as(_as_builder, command, [f"HOME={workdir}", *build_env])
    result = run_as(_as_student, run_command, [*build_env, *_run_env(language)])

    # Either stream: which one a hello-world lands on is the language's business,
    # and what is being tested is that a toolchain built something that runs.
    if OK not in result.stdout + result.stderr:
        raise SystemExit(
            f"FAIL {language}: {filename} ran but printed "
            f"{result.stdout!r} / {result.stderr!r}, expected {OK!r}"
        )


def check_c_frontend(language: Language, workdir: Path) -> None:
    """Even for the builder, gcc must refuse to compile C unless the table
    says the cell keeps the frontend: the driver stays for assembling and
    linking, but `cc1` and `cc1plus` are sealed out from under it."""
    expected = TOOLCHAINS[language].c_frontend
    checked = False
    for compiler, filename, source in (
        ("gcc", "probe.c", "int main(void) { return 0; }\n"),
        ("g++", "probe.cpp", "int main() { return 0; }\n"),
    ):
        if _as_builder(["sh", "-lc", f"command -v {compiler}"], workdir).returncode:
            continue
        checked = True
        probe = workdir / filename
        probe.write_text(source)
        probe.chmod(0o644)
        result = _as_builder([compiler, "-o", f"{filename}.out", filename], workdir)
        if (result.returncode == 0) is not expected:
            verb = "compiled C" if result.returncode == 0 else "cannot compile C"
            raise SystemExit(
                f"FAIL {language}: {compiler} {verb}, but the table says "
                f"c_frontend={expected}\nstderr:\n{result.stderr}"
            )
    if checked:
        verdict = (
            "compiles C by design" if expected else "drives builds but cannot compile C"
        )
        print(f"PASS {language}: the builder's toolchain {verdict}")


# Every way into a compiler that a sealed image must no longer have. The first
# is the language's own; the rest come with it or with the base image, and are
# what student code would reach for if it wanted to write something other than
# the language it was asked for.
SEALED_TOOLS = ("cc", "gcc", "g++", "as", "ld")

# General-purpose interpreters to seal.
SEALED_INTERPRETERS = ("perl", "awk", "gawk", "python3", "ruby", "node", "java")


def check_managed_python_is_unreachable(language: Language, workdir: Path) -> None:
    """A sealed `python3` with a readable `libpython3.*.so` beside it is still
    Python: the library exports `Py_Initialize`, and the stdlib and
    `lib-dynload` next to it are the rest of an interpreter for anything that
    can dlopen it. So nothing under the managed tree may be student-readable —
    except in the cell whose run is that interpreter."""
    if language is Language.PYTHON:
        return
    # root first — root reads the sealed tree, so a probe that prints nothing
    # there printed nothing because it is broken.
    probe = [
        "sh",
        "-c",
        f"find {UV_PYTHON_DIR} -type f -readable -print -quit 2>/dev/null",
    ]
    if not subprocess.run(probe, capture_output=True, text=True).stdout.strip():
        raise SystemExit(f"FAIL {language}: the probe read nothing even as root")

    readable = _as_student(probe, workdir).stdout.strip()
    if readable:
        raise SystemExit(
            f"FAIL {language}: {readable} is student-readable after sealing; "
            "the managed CPython can be dlopen'ed back into an interpreter"
        )
    print(f"PASS {language}: the managed CPython is unreachable, library and all")


def check_sealing(language: Language, workdir: Path, programs: list[Program]) -> None:
    """Seal the toolchain and check the two things that have to be true: nothing
    that compiles is still reachable, and the binary built a moment ago still
    runs. Sealing takes the compiler, not the libraries the artifact links."""
    seal_toolchain(language, keep_student_python=language is Language.PYTHON)

    # Every binary the build used, except the ones the base image ships and
    # this file's sealing never claimed to close: a cell whose compiler cannot
    # package its own output borrows `unzip`, which is no more a way around
    # the artifact than the `cp` beside it.
    build_tools = [
        command[0]
        for _, _, build_commands, _ in programs
        for command in build_commands
        if command[0] not in BASE_IMAGE_BUILD_UTILITIES
    ]
    reachable = [
        tool
        for tool in (*build_tools, *SEALED_TOOLS)
        if _as_student(["sh", "-lc", f"command -v {tool}"], workdir).returncode == 0
    ]
    if reachable:
        raise SystemExit(
            f"FAIL {language}: still reachable after sealing: {', '.join(reachable)}"
        )

    # The builder too: sealing ends the build phase for everyone but root.
    open_for_builder = [
        tool
        for tool in (*build_tools, *SEALED_TOOLS)
        if _as_builder(["sh", "-lc", f"command -v {tool}"], workdir).returncode == 0
    ]
    if open_for_builder:
        raise SystemExit(
            f"FAIL {language}: builder can still reach after sealing: "
            f"{', '.join(open_for_builder)}"
        )

    keep = set(TOOLCHAINS[language].keep)
    if language is Language.PYTHON:
        keep.add("python3")
    open_interpreters = [
        name
        for name in SEALED_INTERPRETERS
        if name not in keep
        and _as_student(["sh", "-lc", f"command -v {name}"], workdir).returncode == 0
    ]
    if open_interpreters:
        raise SystemExit(
            f"FAIL {language}: interpreters still reachable after sealing: "
            f"{', '.join(open_interpreters)}"
        )

    check_managed_python_is_unreachable(language, workdir)

    # Every cell, interpreted ones included: what `run` starts has to survive
    # the sealing, and for a cell with no build step that is the interpreter
    # itself — which the sealing would take away without a `keep`.
    for filename, _, _, run_command in programs:
        result = _as_student(["env", *_run_env(language), *run_command], workdir)
        if result.returncode != 0 or OK not in result.stdout + result.stderr:
            raise SystemExit(
                f"FAIL {language}: what {filename}'s run starts stopped working "
                f"once the toolchain was sealed, exited {result.returncode}\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )

    print(f"PASS {language}: sealed away, and what it built still runs")


SLOW_CELLS = (Language.ZIG, Language.SCALA, Language.KOTLIN, Language.CSHARP)


def check_order(languages: frozenset[Language] | set[Language]) -> list[Language]:
    """The slow cells first, so a parallel `just check-toolchains` does not end
    with the slowest one running alone; the rest alphabetically."""
    slow = [language for language in SLOW_CELLS if language in languages]
    rest = sorted(languages - set(SLOW_CELLS), key=lambda language: language.value)
    return slow + rest


def main() -> None:
    if sys.argv[1:] == ["--list"]:
        print(" ".join(language.value for language in check_order(enabled_languages())))
        return
    if len(sys.argv) != 2 or sys.argv[1] not in set(Language):
        raise SystemExit(f"usage: {sys.argv[0]} --list | " + "|".join(Language))
    check(Language(sys.argv[1]))


if __name__ == "__main__":
    main()
