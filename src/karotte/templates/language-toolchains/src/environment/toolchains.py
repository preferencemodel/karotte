"""The language toolchains a task can grant the builder uid, for the `build`
MCP tool to compile with on the student's behalf.

Which languages the image carries is the environment's choice, made in
`toolchain_config.py` next to this file; nothing is enabled until it names
something. The per-language knowledge here (archives, RPMs, what sealing must
leave alone) is the template's; update it via `karotte update`.

Stdlib only and free of `environment` imports: the build half runs this file as
a script with no venv and no package installed.
"""

import concurrent.futures
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
import threading
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

# Spelled out rather than taken from `paths.py`: the build half imports this file
# before the environment package is installed.
TOOLCHAINS_DIR = Path("/opt/toolchains")

RPM_SUBDIR = "rpms"

# Root-owned, already on the default `PATH`, and ahead of /usr/bin.
WRAPPER_DIR = Path("/usr/local/bin")

# Everything inside a language directory is left world-readable at build time, so
# flipping the one directory is the whole of publishing a toolchain.
SEALED_MODE = 0o700
OPEN_MODE = 0o755
READ_ONLY_MODE = 0o555

# The uid the build MCP tool demotes to. The student never runs a compiler:
# everything `install_toolchain` grants goes `root:builder` with this mode, so
# the toolchain runs for the builder and for no one the student can become.
BUILDER_UID = 900
BUILDER_MODE = 0o750

SUBORDINATE_ID_FILES = (Path("/etc/subuid"), Path("/etc/subgid"))


class Language(StrEnum):
    """The value names the subdirectory under `TOOLCHAINS_DIR` and appears in
    task ids, so it is spelled the way a directory should be: `c_cpp`."""

    PYTHON = "python"
    RUST = "rust"
    ZIG = "zig"
    HASKELL = "haskell"
    SWIFT = "swift"
    GO = "go"
    OCAML = "ocaml"
    KOTLIN = "kotlin"
    JAVA = "java"
    DART = "dart"
    C_CPP = "c_cpp"
    JS_TS = "js_ts"
    RUBY = "ruby"
    SCALA = "scala"
    ERLANG_ELIXIR = "erlang_elixir"
    JULIA = "julia"
    ASSEMBLY = "assembly"
    PASCAL = "pascal"
    CSHARP = "csharp"
    LLVM_IR = "llvm_ir"
    FORTRAN = "fortran"
    CLOJURE = "clojure"
    COBOL = "cobol"
    PROLOG = "prolog"
    MOJO = "mojo"

    @property
    def display_name(self) -> str:
        """What the student is told they have."""
        return {
            Language.C_CPP: "C/C++",
            Language.JS_TS: "JS/TS",
            Language.ERLANG_ELIXIR: "Erlang/Elixir",
            Language.CSHARP: "C#",
            Language.LLVM_IR: "LLVM IR",
            Language.OCAML: "OCaml",
            Language.COBOL: "COBOL",
        }.get(self, self.value.capitalize())

    @property
    def article(self) -> str:
        """`a` or `an`, for a prompt that puts one in front of `display_name`.

        Spelled from the first letter, plus `LLVM IR` — the one name here read
        as an initialism, so a consonant said as a vowel.
        """
        vowel = self.display_name[0] in "AEIOU" or self is Language.LLVM_IR
        return "an" if vowel else "a"


class RunConfinement(StrEnum):
    """Which of the three shims a cell's scored run is launched under.

    The order is strongest first, and a cell takes the weakest one its runtime
    actually needs: each step down is a way to run bytes that were not on disk
    when the run started.
    """

    # No page ever becomes executable, so no JIT and no dlopen of anything the
    # loader has not already mapped.
    STRICT = "strict"

    # A mapping of a file may be executable. For a runtime that maps its own
    # code out of one; anonymous memory still never becomes executable, and the
    # read-only filesystem is what leaves no file to write first and map after.
    MAPPED_CODE = "mapped_code"

    # Any page may become executable. Only for a runtime that compiles as it
    # runs and cannot be told not to — the JVM, whose interpreter-only mode
    # still JITs its own startup path. Everything else the filter refuses
    # (execve, ptrace, memfd_create, io_uring, SysV shm) is refused as before.
    JIT = "jit"


# --- Which languages this environment enables ------------------------------- #


def _load_config():
    """`toolchain_config.py`, loaded from beside this file rather than through
    `environment`: the build half runs before the package exists, and the image
    sets PYTHONSAFEPATH, so neither a package import nor a bare sibling import
    resolves in every context this file runs in."""
    path = Path(__file__).with_name("toolchain_config.py")
    spec = importlib.util.spec_from_file_location("toolchain_config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_languages(names: frozenset[str]) -> frozenset[Language]:
    """The config holds plain strings so it can stay import-free; a typo must
    fail loudly here rather than build an image with a toolchain missing."""
    values = {language.value for language in Language}
    unknown = sorted(set(names) - values)
    if unknown:
        raise ValueError(
            f"unknown language(s) in toolchain_config.ENABLED_LANGUAGES: "
            f"{', '.join(unknown)}; valid values: {', '.join(sorted(values))}"
        )
    return frozenset(Language(name) for name in names)


def enabled_languages() -> frozenset[Language]:
    return validate_languages(frozenset(_load_config().ENABLED_LANGUAGES))


@dataclass(frozen=True)
class Archive:
    """One downloadable toolchain distribution, keyed by `uname -m`."""

    urls: dict[str, str]
    sha256: dict[str, str]

    # As `tar --strip-components`; the distributions disagree about whether they
    # carry a top-level directory and what it is named.
    strip_components: int = 0

    # Run instead of unpacking in place: (unpacked directory, destination), and
    # responsible for leaving the toolchain at the destination.
    installer: str | None = None

    # Where under the language's directory this lands; empty is the directory
    # itself. The BEAM cell needs a split, or OTP's and Elixir's `lib` trees grow
    # into each other.
    subdir: str = ""

    def url(self, machine: str) -> str:
        return self.urls[machine]


@dataclass(frozen=True)
class Toolchain:
    language: Language

    # Installed from the staged RPMs at `pre_hook`; dependencies are staged too,
    # so the install needs no repository.
    rpm_packages: tuple[str, ...] = ()

    # Unpacked under the language's directory at build time, in order.
    archives: tuple[Archive, ...] = ()

    # Relative to the language's directory. Every executable in them gets a
    # wrapper in `WRAPPER_DIR`.
    bin_dirs: tuple[str, ...] = ()

    # Exported by the wrappers before exec'ing the real binary, `{prefix}` being
    # the language's directory. For toolchains that cannot find their own
    # libraries from `argv[0]`.
    wrapper_env: dict[str, str] = field(default_factory=dict)

    # Basenames the run still needs after `seal_toolchain`, which therefore
    # leaves them alone. Empty where a built artifact is standalone.
    keep: tuple[str, ...] = ()

    # Whether the build reaches `BASE_IMAGE_BUILD_TOOLS` — the assembler and
    # linker the base image carries whatever the cell. True where the pinned
    # argv names them and where a compiler drives them out of sight, which is
    # every cell that stages gcc.
    #
    # False is the interesting half. A prebuilt object needs no compiler to run,
    # only bytes and something to jump to them, so an assembler in the hands of
    # a cell whose language never compiles is a way to answer in machine code
    # while the score is calibrated against the language that was asked for.
    # `install_toolchain` leaves them closed for those cells rather than waiting
    # for `seal_toolchain`, which comes too late: the blob is built during the
    # episode and only carried into the run.
    binutils: bool = False

    # Whether the cell keeps gcc's C and C++ frontends runnable. Most cells
    # that stage gcc stage it as the driver their compiler assembles and links
    # through, and the driver compiles nothing itself — `cc1` and `cc1plus`
    # do. `install_toolchain` seals those two unless the cell says otherwise,
    # for the `binutils` reason: C in a cell whose language is not C is a way
    # to answer in the wrong language. True where the frontend is the point (C
    # itself) or the cell's compiler emits C (GnuCOBOL). Zig and Swift carry
    # their own clang inside the archive, which this cannot reach; those cells
    # accept a C compiler by design.
    c_frontend: bool = False

    # Seal the archive's executables one by one and leave the directory open
    # even though `keep` names nothing in it — for a cell whose *artifacts*
    # load shared libraries out of the archive at run time. Mojo's do, and it
    # has no static link to offer instead.
    seal_by_file: bool = False

    seal_within: tuple[str, ...] = ()

    # Which shim the scored run is launched under. Everything defaults to the
    # strict one; a cell that needs less is a cell whose runtime dies without it.
    run_confinement: RunConfinement = RunConfinement.STRICT

    # Injected after the program in the run argv, as harden_build_argv does.
    run_flags: tuple[str, ...] = ()

    # A subdirectory of the language's directory that is the run's own runtime
    # rather than part of the toolchain: sealing leaves everything under it
    # alone, and nothing in it is ever the builder's alone. The JVM cells put
    # their compiler-free `jre` here.
    run_tree: str = ""

    # Exported into the scored run's environment, `{prefix}` filled in as in
    # `wrapper_env`. For a runtime the run starts directly, which therefore has
    # no wrapper to export these for it.
    run_env: dict[str, str] = field(default_factory=dict)

    # Sonames loaded beside the shim in the scored run's `LD_PRELOAD`, for a
    # runtime that dlopens a library once the shim's filter is already
    # installed and the loader's PROT_EXEC mapping is refused. Bare names: the
    # ld cache is what knows where each architecture keeps them.
    run_preload: tuple[str, ...] = ()


# --- The table ------------------------------------------------------------- #

GO_VERSION = "1.26.5"
TYPESCRIPT_VERSION = "5.9.3"
RUST_VERSION = "1.97.1"
RUST_DIST_DATE = "2026-07-16"
ZIG_VERSION = "0.16.0"
DART_VERSION = "3.12.2"
KOTLIN_VERSION = "2.4.10"
SCALA_VERSION = "3.8.4"
OTP_VERSION = "29.0.3"
# Elixir publishes one build per OTP major; this has to be the one above.
ELIXIR_VERSION = "1.20.2"
JULIA_VERSION = "1.12.6"
FPC_VERSION = "3.2.2"
DOTNET_SDK_VERSION = "10.0.302"

# .NET names its own architectures, and neither is `uname -m`'s.
DOTNET_ARCH = {"x86_64": "x64", "aarch64": "arm64"}

# Free Pascal names its own target and compiler binary per architecture.
FPC_TARGET = {"x86_64": "x86_64-linux", "aarch64": "aarch64-linux"}
FPC_COMPILER = {"x86_64": "ppcx64", "aarch64": "ppca64"}

# The subset a single-file server can want. The other ~120 packages are bindings
# (gtk2, mysql, opengl) to libraries this image does not have.
FPC_PACKAGES = (
    "base",
    "utils-fpcmkcfg",
    "units-rtl-objpas",
    "units-rtl-extra",
    "units-rtl-generics",
    "units-rtl-unicode",
    "units-fcl-base",
)
SWIFT_VERSION = "6.3.3"
GHC_VERSION = "9.8.4"
GNUCOBOL_VERSION = "3.2"
GPROLOG_VERSION = "1.5.0"
CLOJURE_TOOLS_VERSION = "1.12.5.1664"

# GHC publishes per build platform, and neither is Amazon Linux. Both of these
# are built against a glibc older than the image's 2.34, the direction that works.
GHC_PLATFORM = {"x86_64": "x86_64-fedora33-linux", "aarch64": "aarch64-deb11-linux"}

JVM_JDK = "24"

# JDK execs this for every `ProcessBuilder`
JVM_SPAWN_HELPER = "jspawnhelper"

# Where a JVM cell's own runtime lands inside its language directory.
JRE_SUBDIR = "jre"

JVM_RUN_FLAGS = ("--illegal-native-access=deny",)

# What a scored JVM run may have, as an allowlist — which is the whole point.
# `jdk.compiler` is not on it, and neither is `jdk.jshell`: a JDK carries a full
# Java compiler as a module, so a kept `java` from the distribution's JDK lets a
# Kotlin or Scala submission compile and run Java at run time, in process,
# needing neither a writable file nor a syscall the filter sees. A runtime
# jlink'ed without those modules has the bytes nowhere on disk to be read back.
JRE_MODULES = (
    "java.base",
    "java.logging",
    "java.management",
    "java.naming",
    "java.net.http",
    "java.sql",
    "java.xml",
    "jdk.crypto.ec",
    "jdk.httpserver",
    "jdk.unsupported",
    "jdk.zipfs",
)

# scalac has no `-include-runtime`, so the build folds this into the jar by hand;
# otherwise the artifact needs a library the sealing is about to close.
SCALA_LIBRARY_JAR = (
    f"{TOOLCHAINS_DIR}/scala/maven2/org/scala-lang/scala-library/"
    f"{SCALA_VERSION}/scala-library-{SCALA_VERSION}.jar"
)

# The uberjar `clojure.main` is loaded from. Named once so `_install_clojure`
# and `clojure_argv` cannot disagree about where it landed.
CLOJURE_JAR = (
    f"{TOOLCHAINS_DIR}/clojure/libexec/clojure-tools-{CLOJURE_TOOLS_VERSION}.jar"
)

# cobc links the shared libcob by default, which lives in the directory the
# sealing closes; a graded build takes this one instead, via
# COB_LIBS="<this> -lgmp". What the artifact then needs at run time is libgmp,
# an ordinary system library.
COBOL_LIBCOB_ARCHIVE = f"{TOOLCHAINS_DIR}/cobol/lib/libcob.a"

# GHC generates and compiles a small C `main` wrapper at every link — the one
# thing a plain Haskell build needs the sealed C frontend for. The builder
# compiles that wrapper once (see `_build_haskell_main_stub`), and a build
# links `ghc -no-hs-main` with this object instead. The stub bakes in
# `-with-rtsopts=-N`, so the link that takes it must say `-threaded`.
HASKELL_MAIN_STUB = f"{TOOLCHAINS_DIR}/haskell/lib/hs_main_stub.o"

OTP_ROOT_SUBDIR = "lib/erlang"

# A stable name for OTP's versioned `erts-<version>`, linked by `_install_otp`
# so the run argv and `run_env` need not know the version.
ERTS_SUBDIR = "erts"

# Holds the `{lookup, [file]}.` that keeps the VM from spawning a resolver.
ERLANG_INETRC = "erl_inetrc"

TOOLCHAINS: dict[Language, Toolchain] = {
    # Already in the image, so this cell installs nothing. What enabling it
    # means is that sealing leaves the one managed CPython runnable.
    Language.PYTHON: Toolchain(language=Language.PYTHON),
    # Self-contained: its own assembler and linker. The `golang` RPM is not —
    # `golang-bin` requires gcc — so this takes the upstream tarball.
    Language.GO: Toolchain(
        language=Language.GO,
        archives=(
            Archive(
                urls={
                    "x86_64": f"https://go.dev/dl/go{GO_VERSION}.linux-amd64.tar.gz",
                    "aarch64": f"https://go.dev/dl/go{GO_VERSION}.linux-arm64.tar.gz",
                },
                sha256={
                    "x86_64": "5c2c3b16caefa1d968a94c1daca04a7ca301a496d9b086e17ad77bb81393f053",
                    "aarch64": "fe4789e92b1f33358680864bbe8704289e7bb5fc207d80623c308935bd696d49",
                },
                strip_components=1,
            ),
        ),
        bin_dirs=("bin",),
    ),
    # rustc drives the link through `cc`, so gcc comes with it either way; the
    # upstream tarball is only several releases newer than the RPM. No cargo
    # registry is reachable in the image, so builds are `std` and nothing else.
    Language.RUST: Toolchain(
        language=Language.RUST,
        rpm_packages=("gcc",),
        binutils=True,
        archives=(
            Archive(
                urls={
                    "x86_64": f"https://static.rust-lang.org/dist/{RUST_DIST_DATE}/rust-{RUST_VERSION}-x86_64-unknown-linux-gnu.tar.xz",
                    "aarch64": f"https://static.rust-lang.org/dist/{RUST_DIST_DATE}/rust-{RUST_VERSION}-aarch64-unknown-linux-gnu.tar.xz",
                },
                sha256={
                    "x86_64": "88f28fa9af20594179f85d6df67078dfd6fa93e2f6da5e1e9b0ac4997988ca4f",
                    "aarch64": "9a7a2c336b4787f1b72f6bab7c35d5b7af2fd03cbd39b4fc721466a70d402a7d",
                },
                strip_components=1,
                installer="rust",
            ),
        ),
        bin_dirs=("bin",),
    ),
    # Ships its own clang, so `zig cc` is a C compiler either way.
    Language.ZIG: Toolchain(
        language=Language.ZIG,
        archives=(
            Archive(
                urls={
                    "x86_64": f"https://ziglang.org/download/{ZIG_VERSION}/zig-x86_64-linux-{ZIG_VERSION}.tar.xz",
                    "aarch64": f"https://ziglang.org/download/{ZIG_VERSION}/zig-aarch64-linux-{ZIG_VERSION}.tar.xz",
                },
                sha256={
                    "x86_64": "70e49664a74374b48b51e6f3fdfbf437f6395d42509050588bd49abe52ba3d00",
                    "aarch64": "ea4b09bfb22ec6f6c6ceac57ab63efb6b46e17ab08d21f69f3a48b38e1534f17",
                },
                strip_components=1,
            ),
        ),
        # The tarball is the binary plus a `lib/` it resolves relative to
        # itself, with no bin/ of its own.
        bin_dirs=(".",),
    ),
    # AOT-compiles to a standalone executable, whose own code its runtime maps.
    Language.DART: Toolchain(
        language=Language.DART,
        run_confinement=RunConfinement.MAPPED_CODE,
        archives=(
            Archive(
                urls={
                    "x86_64": f"https://storage.googleapis.com/dart-archive/channels/stable/release/{DART_VERSION}/sdk/dartsdk-linux-x64-release.zip",
                    "aarch64": f"https://storage.googleapis.com/dart-archive/channels/stable/release/{DART_VERSION}/sdk/dartsdk-linux-arm64-release.zip",
                },
                sha256={
                    "x86_64": "28e47b44cf075f36771046c068bb0d174201cf9c7608744aed1cc23204299c2d",
                    "aarch64": "f82c83ece7d168047550dfd4a664e4071ac7c488bddb72dc43102c22d7e0b518",
                },
                strip_components=1,
            ),
        ),
        bin_dirs=("bin",),
    ),
    # A JVM application, so the toolchain is the architecture-independent
    # compiler archive plus a JDK, and no C compiler. The JDK is the builder's
    # alone: kotlinc runs on it, and nothing else may, because the distribution's
    # `java` carries a Java compiler inside it (see JRE_MODULES). What the run
    # starts is `run_tree`'s jlink'ed runtime, which does not.
    Language.KOTLIN: Toolchain(
        language=Language.KOTLIN,
        run_confinement=RunConfinement.JIT,
        run_flags=JVM_RUN_FLAGS,
        run_tree=JRE_SUBDIR,
        rpm_packages=(f"java-{JVM_JDK}-amazon-corretto-headless",),
        archives=(
            Archive(
                urls=dict.fromkeys(
                    ("x86_64", "aarch64"),
                    f"https://github.com/JetBrains/kotlin/releases/download/v{KOTLIN_VERSION}/kotlin-compiler-{KOTLIN_VERSION}.zip",
                ),
                sha256=dict.fromkeys(
                    ("x86_64", "aarch64"),
                    "473dd66c7a3ef4b182065b3da670466c1bf2773a9dbb0ed8b33a39fe9d4f876d",
                ),
                strip_components=1,
            ),
        ),
        bin_dirs=("bin",),
    ),
    # The devel Corretto, the one cell that needs javac. Kotlin's shape
    # otherwise: the JDK builds, the jlink'ed runtime runs. That the submission
    # is Java does not make the compiler the run's to keep — a scored run that
    # can compile is a scored run that can replace what was graded.
    Language.JAVA: Toolchain(
        language=Language.JAVA,
        run_confinement=RunConfinement.JIT,
        run_flags=JVM_RUN_FLAGS,
        run_tree=JRE_SUBDIR,
        rpm_packages=(f"java-{JVM_JDK}-amazon-corretto-devel",),
    ),
    # swift.org publishes an Amazon Linux 2023 build, which is this image. gcc
    # and binutils are listed prerequisites. Builds must use `-static-stdlib`:
    # the Swift runtime lives inside the toolchain directory, so a dynamically
    # linked artifact dies the moment `seal_toolchain` closes it. It is also why
    # `gcc-c++` is staged — the static stdlib links `-lstdc++`, whose
    # unversioned symlink ships there rather than in libstdc++-devel.
    Language.SWIFT: Toolchain(
        language=Language.SWIFT,
        rpm_packages=("gcc", "gcc-c++"),
        binutils=True,
        archives=(
            Archive(
                urls={
                    "x86_64": f"https://download.swift.org/swift-{SWIFT_VERSION}-release/amazonlinux2023/swift-{SWIFT_VERSION}-RELEASE/swift-{SWIFT_VERSION}-RELEASE-amazonlinux2023.tar.gz",
                    "aarch64": f"https://download.swift.org/swift-{SWIFT_VERSION}-release/amazonlinux2023-aarch64/swift-{SWIFT_VERSION}-RELEASE/swift-{SWIFT_VERSION}-RELEASE-amazonlinux2023-aarch64.tar.gz",
                },
                sha256={
                    "x86_64": "9eb2afb94bc0fa1ce4adb55886e46f4b219d6c239c1d3d6e00437b59110e1375",
                    "aarch64": "bdc050e7b7478a7614f99e5ab3ba9480ce1f1bf60c46015e1e5243d2734d343d",
                },
                strip_components=1,
            ),
        ),
        bin_dirs=("usr/bin",),
    ),
    # GHC drives gcc for assembling and for the final link, so gcc is not
    # optional, and its distribution must be configured against the machine
    # before any of it runs. Neither use needs the C frontend — the one C file
    # GHC compiles, the link-time `main` wrapper, is precompiled by the
    # builder as `HASKELL_MAIN_STUB` and linked with `-no-hs-main` instead.
    # Nothing a plain build links lives under the sealed directory; checked
    # with ldd in the image.
    Language.HASKELL: Toolchain(
        language=Language.HASKELL,
        rpm_packages=("gcc",),
        binutils=True,
        seal_within=("lib/ghc-*/lib/bin/ghc-iserv*",),
        archives=(
            Archive(
                urls={
                    arch: f"https://downloads.haskell.org/~ghc/{GHC_VERSION}/ghc-{GHC_VERSION}-{platform_name}.tar.xz"
                    for arch, platform_name in GHC_PLATFORM.items()
                },
                sha256={
                    "x86_64": "5f03d48f118abd30aee37d8bcddc1d7012193ff205b15121807ecd979c4cf947",
                    "aarch64": "310204daf2df6ad16087be94b3498ca414a0953b29e94e8ec8eb4a5c9bf603d3",
                },
                strip_components=1,
                installer="ghc",
            ),
        ),
        bin_dirs=("bin",),
    ),
    # No usable upstream binary distribution, so this is the OS package: old
    # (4.13), no opam or dune, but `ocamlopt` and `unix` are there.
    Language.OCAML: Toolchain(
        language=Language.OCAML,
        rpm_packages=("ocaml", "gcc"),
        binutils=True,
    ),
    Language.C_CPP: Toolchain(
        language=Language.C_CPP,
        rpm_packages=("gcc", "gcc-c++"),
        binutils=True,
        c_frontend=True,
    ),
    # Runtime is the distribution's Node; the compiler is the typescript npm
    # tarball, whose bin/tsc is a node script the wrapper machinery serves as-is.
    # `--jitless` because V8 compiles as it runs, and no page may become executable.
    Language.JS_TS: Toolchain(
        language=Language.JS_TS,
        run_flags=("--jitless", "--no-expose-wasm"),
        # `node-22` is the name that does the work: /usr/bin/node is a symlink
        # to it, and the sealing decides against resolved names because chmod
        # resolves too.
        keep=("node", "node-22"),
        rpm_packages=("nodejs22",),
        archives=(
            Archive(
                urls=dict.fromkeys(
                    ("x86_64", "aarch64"),
                    f"https://registry.npmjs.org/typescript/-/typescript-{TYPESCRIPT_VERSION}.tgz",
                ),
                sha256=dict.fromkeys(
                    ("x86_64", "aarch64"),
                    "10e108c9cf7d5f2879053dff18515fb405abf2ccef63eaaf017d9c571687a1d3",
                ),
                strip_components=1,
            ),
        ),
        bin_dirs=("bin",),
    ),
    # Kotlin's shape: compiler archive plus a JDK, no C compiler. scalac has no
    # `-include-runtime`, so a jar that must outlive the sealing folds
    # `SCALA_LIBRARY_JAR` in by hand.
    Language.SCALA: Toolchain(
        language=Language.SCALA,
        run_confinement=RunConfinement.JIT,
        run_flags=JVM_RUN_FLAGS,
        run_tree=JRE_SUBDIR,
        rpm_packages=(f"java-{JVM_JDK}-amazon-corretto-devel",),
        archives=(
            Archive(
                urls=dict.fromkeys(
                    ("x86_64", "aarch64"),
                    f"https://github.com/scala/scala3/releases/download/{SCALA_VERSION}/scala3-{SCALA_VERSION}.tar.gz",
                ),
                sha256=dict.fromkeys(
                    ("x86_64", "aarch64"),
                    "cb2f9a358ec449eec417d63efd9b6fc6bd66a13b1347d49c25571eca284857d3",
                ),
                strip_components=1,
            ),
        ),
        bin_dirs=("bin",),
    ),
    # Two languages in one cell, because the sealing cannot separate them: a
    # kept VM compiles Erlang whether or not the cell says so, and an Elixir
    # program runs on that VM.
    #
    # OTP has no package here and no upstream binary, so it is built from source
    # in the builder stage. Elixir is BEAM bytecode and unpacks beside it.
    Language.ERLANG_ELIXIR: Toolchain(
        language=Language.ERLANG_ELIXIR,
        run_confinement=RunConfinement.JIT,
        keep=("beam.smp",),
        seal_by_file=True,
        run_env={
            "BINDIR": "{prefix}/" + OTP_ROOT_SUBDIR + "/" + ERTS_SUBDIR + "/bin",
            "ERL_INETRC": "{prefix}/" + ERLANG_INETRC,
            "ERL_CRASH_DUMP_SECONDS": "0",
        },
        archives=(
            Archive(
                urls=dict.fromkeys(
                    ("x86_64", "aarch64"),
                    f"https://github.com/erlang/otp/releases/download/OTP-{OTP_VERSION}/otp_src_{OTP_VERSION}.tar.gz",
                ),
                sha256=dict.fromkeys(
                    ("x86_64", "aarch64"),
                    "f920c660b16794bcb7270d1cbf680f7747c719650bcd6ac449508a32c2a8972a",
                ),
                strip_components=1,
                installer="otp",
            ),
            # Under `elixir/` rather than over OTP's prefix: both carry a `lib`
            # of BEAM applications and merging them would leave no way to tell
            # whose is whose.
            Archive(
                urls=dict.fromkeys(
                    ("x86_64", "aarch64"),
                    f"https://github.com/elixir-lang/elixir/releases/download/v{ELIXIR_VERSION}/elixir-otp-{OTP_VERSION.split('.')[0]}.zip",
                ),
                sha256=dict.fromkeys(
                    ("x86_64", "aarch64"),
                    "a9e88cd41fbbba7da6f6dc237a49dd2ed4e70457121035cc7fc56ad05582f394",
                ),
                subdir="elixir",
            ),
        ),
        bin_dirs=("bin", "elixir/bin"),
    ),
    # Links through the base image's binutils, no C compiler of its own. The
    # distribution is an interactive installer, so `_install_fpc` stands in.
    Language.PASCAL: Toolchain(
        language=Language.PASCAL,
        # `PPC_CONFIG_PATH` points the compiler at the generated fpc.cfg, where
        # the unit paths are written down. Without it the compiler falls back to
        # an /etc/fpc.cfg this image does not have. Set by the wrapper, so the
        # compiler works for anyone who runs it, grader and student alike.
        wrapper_env={"PPC_CONFIG_PATH": "{prefix}/etc"},
        # For the linker, not for a compiler: this cell has no C frontend. The
        # base image ships `libc.so.6` but not the `libc.so` that `-lc` resolves
        # through, and fpc's `cthreads` — the only way to threads on Linux —
        # emits `-lc -lpthread -ldl`. Without this the RTL cannot link its own
        # threading unit, which is a zero at build time for reasons that are not
        # the agent's.
        rpm_packages=("glibc-devel",),
        # On glibc 2.34 the `cthreads` RTL dlopens the libpthread.so.0 stub at
        # thread init and glibc dlopens libgcc_s.so.1 to unwind out of
        # pthread_exit; without both mapped up front a threaded program exits
        # 216 with no diagnostic.
        run_preload=("libpthread.so.0", "libgcc_s.so.1"),
        # The linker itself, and the assembler fpc writes its object through.
        binutils=True,
        archives=(
            Archive(
                urls={
                    arch: f"https://downloads.freepascal.org/fpc/dist/{FPC_VERSION}/{target}/fpc-{FPC_VERSION}.{target}.tar"
                    for arch, target in FPC_TARGET.items()
                },
                sha256={
                    "x86_64": "5adac308a5534b6a76446d8311fc340747cbb7edeaacfe6b651493ff3fe31e83",
                    "aarch64": "b39470f9b6b5b82f50fc8680a5da37d2834f2129c65c24c5628a80894d565451",
                },
                strip_components=1,
                installer="fpc",
            ),
        ),
        bin_dirs=("bin",),
    ),
    # Backend and no frontend, by design: assembly is what a compiler emits, so
    # any language targeting it would let the student write C. binutils is the
    # whole toolchain and the base image already has it, so this installs
    # nothing. `as` and `ld` are in `BASE_IMAGE_BUILD_TOOLS`, which every other
    # cell closes *because* nothing installed them; here they are the toolchain.
    #
    # No libc — glibc-devel is staged for the Pascal cell and installed only in
    # its own runs, so a program here is freestanding.
    Language.ASSEMBLY: Toolchain(
        language=Language.ASSEMBLY,
        binutils=True,
    ),
    # An interpreted cell — `julia` compiles as it runs, so sealing cannot
    # separate the compiler from the run — with the runtime inside the archive,
    # so the sealing goes file by file.
    Language.JULIA: Toolchain(
        language=Language.JULIA,
        run_confinement=RunConfinement.JIT,
        # `lld` is part of the runtime, not a leftover compiler: Julia links its
        # own package images with the bundled linker through `Base.Linking`, so
        # any `using` that has to precompile dies without it. It links nothing a
        # student could not already emit from the kept `julia`, and the rest of
        # `libexec` — `dsymutil`, `7z` — is closed as before.
        keep=("julia", "lld"),
        archives=(
            Archive(
                urls={
                    "x86_64": f"https://julialang-s3.julialang.org/bin/linux/x64/{'.'.join(JULIA_VERSION.split('.')[:2])}/julia-{JULIA_VERSION}-linux-x86_64.tar.gz",
                    "aarch64": f"https://julialang-s3.julialang.org/bin/linux/aarch64/{'.'.join(JULIA_VERSION.split('.')[:2])}/julia-{JULIA_VERSION}-linux-aarch64.tar.gz",
                },
                sha256={
                    "x86_64": "bbabf3bef19421a9dbd24a767d807606ab85e444323b5a1c73ffe293fa3d079a",
                    "aarch64": "029b93b857bd0ffd627f9a8580d3bbaa63daf008d7b7aed02fbceb8fd57c4899",
                },
                strip_components=1,
            ),
        ),
        bin_dirs=("bin",),
    ),
    # The distribution's interpreter and nothing else; it brings no C compiler.
    # Nothing to build, and unlike Python the interpreter is one the sealing
    # would otherwise take away — hence `keep`. `ruby3.2` is the name that does
    # the work: /usr/bin/ruby is a symlink to it, and the sealing decides
    # against resolved names because chmod resolves too.
    # A stdlib `require` dlopens a .so before the submission runs, hence the shim.
    Language.RUBY: Toolchain(
        language=Language.RUBY,
        keep=("ruby", "ruby3.2"),
        rpm_packages=("ruby3.2",),
        run_confinement=RunConfinement.MAPPED_CODE,
    ),
    # One tarball, plus gcc: `dotnet publish` of a single `.cs` (a file-based
    # app, .NET 10's) is native-AOT here and hands gcc the link, so the binary
    # links only libc and libm and outlives the sealing with nothing kept.
    Language.CSHARP: Toolchain(
        language=Language.CSHARP,
        # A publish restores its packages with no network, so the installer
        # staged them inside the toolchain and this points at them; without it
        # every build ends at NU1301. Set by the wrapper, so it holds for
        # whoever runs `dotnet`.
        wrapper_env={
            "NUGET_PACKAGES": "{prefix}/nuget",
            "DOTNET_CLI_TELEMETRY_OPTOUT": "1",
            "DOTNET_NOLOGO": "1",
        },
        rpm_packages=("gcc",),
        binutils=True,
        archives=(
            Archive(
                urls={
                    arch: f"https://builds.dotnet.microsoft.com/dotnet/Sdk/{DOTNET_SDK_VERSION}/dotnet-sdk-{DOTNET_SDK_VERSION}-linux-{name}.tar.gz"
                    for arch, name in DOTNET_ARCH.items()
                },
                sha256={
                    "x86_64": "264a838d6f5d1a252489c7bb2e2946a579d6a881391d50ffd175a01e4d948c1c",
                    "aarch64": "1c56318e4099990719f6369184e08bbad0248c09c5ad7532d2516e3cdfc3ab6d",
                },
                installer="dotnet",
            ),
        ),
        # The prefix root is the `dotnet` launcher; no bin/ of its own.
        bin_dirs=(".",),
    ),
    # The Assembly cell's rule: LLVM IR is what a compiler emits, so any language
    # targeting it would let the student write C. `llvm` is the code generator
    # alone — clang is a separate package and nothing here pulls it. `lli` runs
    # IR directly, so the sealing has to close it alongside `llc`. No libc:
    # glibc-devel is absent, so a program here is freestanding.
    Language.LLVM_IR: Toolchain(
        language=Language.LLVM_IR,
        rpm_packages=("llvm",),
        binutils=True,
    ),
    # gfortran is another gcc frontend, so this is the C/C++ cell with a
    # different language in front: RPMs only, and the artifact's libgfortran
    # is a shared library sealing leaves alone. The RPM requires gcc anyway;
    # naming it keeps the C-compiler grouping visible in this table.
    Language.FORTRAN: Toolchain(
        language=Language.FORTRAN,
        rpm_packages=("gcc-gfortran", "gcc"),
        binutils=True,
    ),
    # A JVM cell like Kotlin's, except the compiler is not separable: Clojure
    # compiles source as it loads it, inside the JVM that runs it — the BEAM
    # cell's deal, on the JVM. The archive's jar is upstream's self-contained
    # uberjar (Clojure plus spec, everything `clojure.main` needs offline);
    # the installer writes a `clojure` launcher around it because the tarball's
    # own resolves classpaths against Maven Central.
    Language.CLOJURE: Toolchain(
        language=Language.CLOJURE,
        # The same three the other JVM cells take: Clojure emits bytecode and
        # defines it into the running VM, so the JIT has to stay, native access
        # has to be denied, and what the run starts is the jlink'ed runtime
        # rather than a distribution `java` carrying `jdk.compiler`.
        run_confinement=RunConfinement.JIT,
        run_flags=JVM_RUN_FLAGS,
        run_tree=JRE_SUBDIR,
        # The uberjar the run reads lives in the archive, so the sealing has to
        # go file by file to leave it readable; the `clojure` launcher it used
        # to keep is no longer what starts the run (see `clojure_argv`).
        seal_by_file=True,
        rpm_packages=(f"java-{JVM_JDK}-amazon-corretto-headless",),
        archives=(
            Archive(
                urls=dict.fromkeys(
                    ("x86_64", "aarch64"),
                    f"https://github.com/clojure/brew-install/releases/download/{CLOJURE_TOOLS_VERSION}/clojure-tools-{CLOJURE_TOOLS_VERSION}.tar.gz",
                ),
                sha256=dict.fromkeys(
                    ("x86_64", "aarch64"),
                    "77dd6868948074adcc93e83a796f8e8f15a1a92bcb1b9002d715fd2210e476f3",
                ),
                strip_components=1,
                installer="clojure",
            ),
        ),
        bin_dirs=("bin",),
    ),
    # No package and no upstream binary, so cobc is built from source in the
    # builder stage, OTP's way. It compiles COBOL by generating C and driving
    # gcc, so the C compiler is staged openly — frontend and all, the one cell
    # besides C's own that keeps it. A graded artifact links the static libcob
    # (`COBOL_LIBCOB_ARCHIVE`) precisely so the sealing can close the
    # directory the shared one lives in.
    Language.COBOL: Toolchain(
        language=Language.COBOL,
        rpm_packages=("gcc", "gmp-devel"),
        binutils=True,
        c_frontend=True,
        archives=(
            Archive(
                urls=dict.fromkeys(
                    ("x86_64", "aarch64"),
                    f"https://ftp.gnu.org/gnu/gnucobol/gnucobol-{GNUCOBOL_VERSION}.tar.gz",
                ),
                sha256=dict.fromkeys(
                    ("x86_64", "aarch64"),
                    "29f30a77176015847f0afb2e22939e39798bb4d98c7c7a26f6765930b4553c52",
                ),
                strip_components=1,
                installer="gnucobol",
            ),
        ),
        bin_dirs=("bin",),
    ),
    # gplc emits native code through the staged gcc and the base image's
    # assembler, and links the Prolog engine in statically — the artifact
    # needs nothing from the directory afterwards, so a Prolog cell still
    # gets the compile-then-seal shape.
    Language.PROLOG: Toolchain(
        language=Language.PROLOG,
        rpm_packages=("gcc",),
        binutils=True,
        archives=(
            Archive(
                urls=dict.fromkeys(
                    ("x86_64", "aarch64"),
                    f"https://ftp.gnu.org/gnu/gprolog/gprolog-{GPROLOG_VERSION}.tar.gz",
                ),
                sha256=dict.fromkeys(
                    ("x86_64", "aarch64"),
                    "670642b43c0faa27ebd68961efb17ebe707688f91b6809566ddd606139512c01",
                ),
                strip_components=1,
                installer="gprolog",
            ),
        ),
        # `make install` appends its own versioned directory to the prefix,
        # and gplc's compiled-in paths point inside it, so it stays.
        bin_dirs=(f"gprolog-{GPROLOG_VERSION}/bin",),
    ),
    # The compiler ships as pip wheels (a wheel is a zip), unpacked here so the
    # language directory is what pip would have called `site-packages/modular`.
    # `mojo build` hands the link to gcc, and what it builds loads shared
    # libraries out of this directory at run time with no static option —
    # hence `seal_by_file`: the compiler is closed, the directory is not.
    # The wheels are under Modular's own license (LicenseRef-MAX-Platform-Software-License).
    Language.MOJO: Toolchain(
        language=Language.MOJO,
        seal_by_file=True,
        # What upstream's own pip launcher exports before exec'ing the real
        # binary: the driver and stdlib paths.
        wrapper_env={
            "MODULAR_MAX_PACKAGE_ROOT": "{prefix}",
            "MODULAR_MOJO_MAX_PACKAGE_ROOT": "{prefix}",
            "MODULAR_MOJO_MAX_DRIVER_PATH": "{prefix}/bin/mojo",
            "MODULAR_MOJO_MAX_IMPORT_PATH": "{prefix}/lib/mojo",
        },
        rpm_packages=("gcc",),
        binutils=True,
        archives=(
            # The compiler: bin/{mojo,lld}, lib/*.so. Wheel URLs are
            # content-addressed on PyPI, so a version bump replaces them whole.
            Archive(
                urls={
                    "x86_64": "https://files.pythonhosted.org/packages/f0/5f/f38fefe327d1c81e28def69c4a52ae4f75e389cb6e613a2c04ca8d68d582/mojo_compiler-1.0.0-py3-none-manylinux_2_34_x86_64.whl",
                    "aarch64": "https://files.pythonhosted.org/packages/a5/f9/cbfe2bf947d0926ad57599513d898f1e02ee0c60af1253b779ecaa810235/mojo_compiler-1.0.0-py3-none-manylinux_2_34_aarch64.whl",
                },
                sha256={
                    "x86_64": "e9e60f9638e69ca0f4be7292468523fc98a143f58dbf9024f60ed68b874a867e",
                    "aarch64": "3ae3eb0c58a8956f542e324f736866df0ae2f9f875e8a279cbeea961ec4e9ee8",
                },
                # mojo_compiler-*.data/platlib/modular/; the wheel's pip shims
                # and dist-info have fewer components and fall away.
                strip_components=3,
            ),
            # The stdlib, one bytecode file the compiler wants at
            # lib/mojo/std.mojoc under its package root.
            Archive(
                urls=dict.fromkeys(
                    ("x86_64", "aarch64"),
                    "https://files.pythonhosted.org/packages/54/99/ea401ff1db56a4af8607283b95627e01b986fc67f510715b07f100118105/mojo_compiler_mojo_libs-1.0.0-py3-none-any.whl",
                ),
                sha256=dict.fromkeys(
                    ("x86_64", "aarch64"),
                    "20a92e37ecbd19e2dbb1a525612a8de4c9f266f5536def55c5e2b0786320e8b3",
                ),
                strip_components=5,
                subdir="lib/mojo",
            ),
        ),
        bin_dirs=("bin",),
    ),
}


def toolchain_dir(language: Language) -> Path:
    return TOOLCHAINS_DIR / language.value


def run_tree_dir(language: Language) -> Path | None:
    """The cell's own runtime inside its directory, where it has one."""
    subdir = TOOLCHAINS[language].run_tree
    return toolchain_dir(language) / subdir if subdir else None


def jvm_java(language: Language) -> Path:
    """The `java` a JVM cell's scored run starts: the jlink'ed runtime with no
    compiler in it, rather than the distribution's JDK, which has one whatever
    its `bin` holds. A task's run argv names this by absolute path."""
    tree = run_tree_dir(language)
    if tree is None:
        raise ValueError(f"{language.value} has no runtime of its own")
    return tree / "bin" / "java"


def julia_binary() -> Path:
    """Julia's own executable. Named directly for `beam_smp`'s reason: what is
    on `PATH` is a wrapper script, and the shim refuses the execve with which
    it would reach the real one."""
    return toolchain_dir(Language.JULIA) / "bin" / "julia"


def clojure_argv(*clojure_args: str) -> tuple[str, ...]:
    """The argv that starts a Clojure run without the launcher script.

    `clojure` is a shell script that execs `java`, and the shim refuses
    execve, so the run makes that argv itself — against the cell's own
    jlink'ed runtime, which carries no Java compiler.
    """
    return (
        str(jvm_java(Language.CLOJURE)),
        "-cp",
        CLOJURE_JAR,
        "clojure.main",
        *clojure_args,
    )


def otp_root() -> Path:
    """The `-root` of the installed distribution, which is not the prefix."""
    return toolchain_dir(Language.ERLANG_ELIXIR) / OTP_ROOT_SUBDIR


def beam_smp() -> Path:
    """The emulator a BEAM run starts, named directly because the `erl` that
    would have found it execs its way there and the shim refuses execve."""
    return otp_root() / ERTS_SUBDIR / "bin" / "beam.smp"


def beam_argv(home: str, *emulator_args: str) -> tuple[str, ...]:
    """The argv that starts the BEAM the way `erl` would have, minus the two
    execs on the way.

    `erl` is a script that execs `erlexec`, which works out these arguments and
    execs `beam.smp`; under the shim neither exec is allowed, so the run makes
    the same argv itself. The `--` are what separate the emulator's three
    argument sections, and every one of them is load-bearing.
    """
    root = otp_root()
    return (
        str(beam_smp()),
        "--",
        "-root",
        str(root),
        "-bindir",
        str(root / ERTS_SUBDIR / "bin"),
        "-progname",
        "erl",
        "--",
        "-home",
        home,
        "--",
        *emulator_args,
    )


def elixir_argv(home: str, *elixir_args: str) -> tuple[str, ...]:
    """`beam_argv` for an Elixir submission: the same emulator, started through
    the boot sequence the `elixir` script would have asked for.

    Elixir's stdlib is BEAM code like any other, but its applications have to be
    started before a submission's module runs, which is what `elixir start_cli`
    does. Everything after `-extra` is Elixir's own argv rather than the
    emulator's.
    """
    elixir_lib = toolchain_dir(Language.ERLANG_ELIXIR) / "elixir" / "lib"
    return beam_argv(
        home,
        "-noshell",
        "-elixir_root",
        str(elixir_lib),
        "-pa",
        str(elixir_lib / "elixir" / "ebin"),
        "-s",
        "elixir",
        "start_cli",
        "--",
        "--",
        "-extra",
        *elixir_args,
    )


# --- Python, which every cell has whether or not it is the Python cell ------ #

# uv installs the interpreter the student venv is built from here; the
# Containerfile hands the tree to root.
UV_PYTHON_DIR = Path("/opt/uv/python")

# Every Python *binary* the student could otherwise reach once the clock starts:
# the distribution's own (dnf is written in it), the uv-managed one, and uv
# itself, which can run either. Globs rather than a list because the version is
# in the path, and resolved before sealing since `python3` is a symlink chain.
PYTHON_GLOBS = (
    "usr/bin/python3*",
    "usr/local/bin/python3*",
    "opt/uv/python/*/bin/python*",
    "opt/uv/uv",
    "opt/uv/uvx",
)


def student_python() -> Path:
    """The interpreter a Python submission is run with.

    Not `/workdir/.venv/bin/python`, which the agent owns outright — a
    submission could replace the interpreter itself and the grader would exec
    whatever was put there. This is the root-owned CPython that venv was built
    from, and since the venv deliberately carries no dependencies, running the
    base interpreter runs the same code.
    """
    if "KAROTTE_CONTAINERIZED" not in os.environ:
        return Path(sys.executable)
    # uv installs `cpython-<major>.<minor>.<patch>-*` and symlinks
    # `cpython-<major>.<minor>-*` at it, so the glob sees one interpreter under
    # two names. Only the real install counts, and it is what gets returned:
    # resolving instead would name the same file as `/workdir/.venv/bin/python`
    # does, which the checks read as the student-owned interpreter.
    found = sorted(
        path
        for path in UV_PYTHON_DIR.glob("cpython-*/bin/python3")
        if not path.parent.parent.is_symlink()
    )
    if len(found) != 1:
        raise RuntimeError(
            f"expected exactly one managed CPython under {UV_PYTHON_DIR}, "
            f"found {[str(p) for p in found]}"
        )
    return found[0]


def _seal_uv_python() -> list[Path]:
    """Close the uv-managed CPython whole rather than binary by binary.

    A sealed `bin/python3` is not a sealed Python: the tree's
    `libpython3.*.so` exports `Py_Initialize`, and with the stdlib and
    `lib-dynload` beside it readable, anything that can dlopen the library —
    a compiled artifact in any of these languages — is a full interpreter.
    Adding the library to `PYTHON_GLOBS` would not do either, since
    `Py_Initialize` needs only the .py files it reads.

    One chmod on the top directory, the way a language directory is sealed:
    nothing under it is reachable by path once it cannot be traversed. Root
    traverses it regardless, which is what leaves the grader's own runs of the
    managed interpreter working.
    """
    if not UV_PYTHON_DIR.is_dir():
        return []
    if os.geteuid() == 0:
        os.chown(UV_PYTHON_DIR, 0, 0)
    UV_PYTHON_DIR.chmod(SEALED_MODE)
    return [UV_PYTHON_DIR]


def python_interpreters() -> list[Path]:
    """Every Python binary on the image, by the path chmod would land on.

    Resolved and de-duplicated: `python3`, `python3.12` and `python` in one
    directory are usually three names for one file, and sealing it once under
    its real name closes all three.
    """
    found: dict[Path, None] = {}
    for pattern in PYTHON_GLOBS:
        for path in sorted(Path("/").glob(pattern)):
            if path.is_file() and os.access(path, os.X_OK):
                found[path.resolve()] = None
    return list(found)


# --- The assembler and linker, which every cell has the same way ------------ #

# Base-image executables that can still turn source into something that runs.
# Nothing here installed them, so unlike the RPM half there is nothing to derive
# them from: `binutils` is in the image for `strings` and `objdump`, and brings
# an assembler and a linker with it.
BASE_IMAGE_BUILD_TOOLS = ("as", "ld", "ld.bfd", "ld.gold", "gold", "objcopy")

# Base-image utilities a cell's *build* borrows, deliberately not sealed: they
# turn no source into anything that runs, and `cp`, `tar` and `gzip` stay
# reachable beside them regardless.
BASE_IMAGE_BUILD_UTILITIES = ("unzip",)

BASE_IMAGE_INTERPRETER_GLOBS = (
    "perl*",
    "awk",
    "gawk",
    "mawk",
    "nawk",
    "busybox",
    "tclsh*",
    "lua*",
    "ruby*",
    "node*",
    "php*",
)

# gcc's C and C++ frontends, at the paths the distribution installs them. The
# `gcc` on a cell's PATH is a driver: it hands compilation to these, assembly
# to `as` and the link to `ld`, so with them sealed it still drives builds but
# cannot compile C. They arrive with the RPMs, so unlike `as` and `ld` there
# is nothing to close at image-build time — `install_toolchain` closes them
# the moment they land.
C_FRONTEND_GLOBS = ("usr/libexec/gcc/*/*/cc1", "usr/libexec/gcc/*/*/cc1plus")

# Absolute, and chmod'ed rather than read, which is why every function that puts
# a mode on them refuses to run anywhere but the image: on a dev box these are
# the host's own binaries.
SYSTEM_BIN_DIRS = ("/usr/bin", "/usr/local/bin", "/bin")


def base_image_build_tools() -> list[Path]:
    """Wherever the image put each name in `BASE_IMAGE_BUILD_TOOLS`.

    Unresolved: `ld` is usually a link to `ld.bfd`, and chmod follows it, so
    both names land on one file and closing either closes the pair.
    """
    return [
        tool
        for name in BASE_IMAGE_BUILD_TOOLS
        for directory in SYSTEM_BIN_DIRS
        if (tool := Path(directory) / name).is_file()
    ]


def _mode_base_image_build_tools(
    mode: int, keep: frozenset[str] = frozenset()
) -> list[Path]:
    """Put `mode` on the base image's assembler and linker, and say what moved."""
    changed: list[Path] = []
    for tool in base_image_build_tools():
        if tool.name in keep:
            continue
        tool.chmod(mode)
        changed.append(tool)
    return changed


def base_image_interpreters() -> list[Path]:
    """Wherever the image put a name matching `BASE_IMAGE_INTERPRETER_GLOBS`."""
    found: dict[Path, None] = {}
    for directory in SYSTEM_BIN_DIRS:
        for pattern in BASE_IMAGE_INTERPRETER_GLOBS:
            for path in sorted(Path(directory).glob(pattern)):
                if path.is_file() and os.access(path, os.X_OK):
                    found[path] = None
    return list(found)


def _seal_base_image_interpreters(keep: frozenset[str]) -> list[Path]:
    """Close every interpreter `keep` does not name, and say what moved."""
    closed: list[Path] = []
    for interpreter in base_image_interpreters():
        if interpreter.name in keep:
            continue
        interpreter.chmod(SEALED_MODE)
        closed.append(interpreter)
    return closed


def c_frontends() -> list[Path]:
    """Wherever the installed RPMs put `cc1` and `cc1plus`; empty in a cell
    whose packages staged neither."""
    return [
        path
        for pattern in C_FRONTEND_GLOBS
        for path in sorted(Path("/").glob(pattern))
        if path.is_file()
    ]


def _mode_c_frontends(mode: int) -> list[Path]:
    """Put `mode` on gcc's C and C++ frontends, and say what moved."""
    changed: list[Path] = []
    for frontend in c_frontends():
        frontend.chmod(mode)
        changed.append(frontend)
    return changed


def sealed_within(toolchain: Toolchain) -> list[Path]:
    """Wherever the cell's `seal_within` globs landed under its directory."""
    prefix = toolchain_dir(toolchain.language)
    return [
        path
        for pattern in toolchain.seal_within
        for path in sorted(prefix.glob(pattern))
        if path.is_file()
    ]


def _seal_within(toolchain: Toolchain) -> list[Path]:
    """Close what the cell forbids even the builder, and say what moved.

    Resolved before the chmod: these are usually links into a versioned
    directory beside them, and a mode on the link is a mode on the target.
    """
    changed: list[Path] = []
    for path in sealed_within(toolchain):
        path.resolve().chmod(SEALED_MODE)
        changed.append(path)
    return changed


def _grant_builder(path: Path) -> None:
    """Make `path` runnable by the builder uid alone: root-owned, group
    `builder`, closed to everyone else. Ownership only moves where this runs
    as root; on a dev box the mode is the whole of it."""
    if os.geteuid() == 0:
        os.chown(path, 0, BUILDER_UID)
    path.chmod(BUILDER_MODE)


# --- Build half: run from the Containerfile, on the bare image -------------- #

ARCHIVE_BUILD_JOBS = 4

# Each build thread's log file; `_run` and `_download` write there when it is
# set, so parallel builds don't interleave in the image build output.
_build_log = threading.local()


def _build_log_file():
    return getattr(_build_log, "file", None)


def _machine() -> str:
    """The architecture key the table is written against."""
    machine = platform.machine()
    return {"amd64": "x86_64", "arm64": "aarch64"}.get(machine, machine)


def _run(
    *args: str, cwd: Path | None = None, env: dict[str, str] | None = None
) -> None:
    log = _build_log_file()
    subprocess.run(args, cwd=cwd, env=env, check=True, stdout=log, stderr=log)


def _download(url: str, sha256: str, dest: Path) -> None:
    """Fetch `url` to `dest`, refusing anything that is not the pinned bytes."""
    import hashlib

    print(f"Fetching {url}", file=_build_log_file() or sys.stderr, flush=True)
    digest = hashlib.sha256()
    with urllib.request.urlopen(url) as response, open(dest, "wb") as f:
        while chunk := response.read(1 << 20):
            digest.update(chunk)
            f.write(chunk)

    if digest.hexdigest() != sha256:
        raise SystemExit(f"{url}: expected sha256 {sha256}, got {digest.hexdigest()}")


def _unpack(archive_path: Path, dest: Path, strip_components: int) -> None:
    """Unpack `archive_path` into `dest`, dropping `strip_components` leading
    path components the way `tar --strip-components` does.

    `tarfile` and `zipfile` have no such option, so members are rewritten on the
    way out.
    """
    dest.mkdir(parents=True, exist_ok=True)

    def strip(name: str) -> str | None:
        parts = Path(name).parts[strip_components:]
        return str(Path(*parts)) if parts else None

    # A wheel is a zip with a different name.
    if archive_path.suffix in (".zip", ".whl"):
        with zipfile.ZipFile(archive_path) as zf:
            for info in zf.infolist():
                stripped = strip(info.filename)
                if stripped is None:
                    continue
                target = dest / stripped
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out)
                # ZIP stores the mode in the top half of external_attr, which
                # Kotlin's and Dart's launcher scripts rely on. Group and
                # other write bits are dropped: Mojo's wheels record 0777, and
                # a world-writable file under root's toolchain directory is a
                # file the student may rewrite.
                mode = info.external_attr >> 16
                if mode:
                    target.chmod(mode & 0o7777 & ~0o022)
        return

    # `tar` rather than `tarfile`: these run to 2.6GB unpacked, and the C
    # implementation is several times faster at that size.
    #
    # `--no-same-owner` is not tidiness. Unpacking as root otherwise honours the
    # uids in the archive, and Julia's carries 1000 — the student — so the
    # distribution would land owned by the uid the sealing must keep out, who
    # can then put any mode back.
    _run(
        "tar",
        "-xf",
        str(archive_path),
        "-C",
        str(dest),
        "--no-same-owner",
        f"--strip-components={strip_components}",
    )


def _install_rust(unpacked: Path, dest: Path) -> None:
    """Rust ships a component tree with an installer rather than a prefix."""
    _run(
        str(unpacked / "install.sh"),
        f"--prefix={dest}",
        "--components=rustc,cargo,rust-std-" + _rust_target(),
        "--disable-ldconfig",
    )


def _rust_target() -> str:
    return f"{_machine()}-unknown-linux-gnu"


def _install_otp(unpacked: Path, dest: Path) -> None:
    """No package and no upstream binary, so the BEAM is built from source.

    The applications turned off have a window in them or a dependency the image
    would then have to carry, and each is minutes of build time.
    """
    _run(
        "./configure",
        f"--prefix={dest}",
        "--without-javac",
        "--without-wx",
        "--without-debugger",
        "--without-observer",
        "--without-et",
        "--without-jinterface",
        "--without-megaco",
        "--without-odbc",
        "--without-ssl",
        cwd=unpacked,
    )
    _run("make", f"-j{os.cpu_count() or 1}", cwd=unpacked)
    _run("make", "install", cwd=unpacked)

    root = dest / OTP_ROOT_SUBDIR
    (erts,) = sorted(root.glob("erts-*"))
    (root / ERTS_SUBDIR).symlink_to(erts)

    (dest / ERLANG_INETRC).write_text("{lookup, [file]}.\n")


def _install_fpc(unpacked: Path, dest: Path) -> None:
    """Free Pascal ships an interactive installer wrapped around the packages,
    so this does what it would.

    None of the three steps is optional: unpack the subset a server needs,
    symlink the compiler proper into `bin` (only the `fpc` driver ships there,
    and it looks for `ppcx64` on `PATH`), and generate the `fpc.cfg` that
    `PPC_CONFIG_PATH` points at.
    """
    target = FPC_TARGET[_machine()]
    inner = unpacked / "inner"
    inner.mkdir(exist_ok=True)
    _run("tar", "-xf", str(unpacked / f"binary.{target}.tar"), "-C", str(inner))

    dest.mkdir(parents=True, exist_ok=True)
    for package in FPC_PACKAGES:
        _run(
            "tar",
            "-xzf",
            str(inner / f"{package}.{target}.tar.gz"),
            "-C",
            str(dest),
            "--no-same-owner",
        )

    name = FPC_COMPILER[_machine()]
    compiler = dest / "lib" / "fpc" / FPC_VERSION / name
    (dest / "bin" / name).symlink_to(compiler)

    config_dir = dest / "etc"
    config_dir.mkdir(exist_ok=True)
    _run(
        str(dest / "bin" / "fpcmkcfg"),
        "-d",
        f"basepath={dest / 'lib' / 'fpc' / FPC_VERSION}",
        "-o",
        str(config_dir / "fpc.cfg"),
    )


def _install_ghc(unpacked: Path, dest: Path) -> None:
    """GHC records the machine's C toolchain into `settings` at configure time,
    so it must be configured where it will run rather than copied into place."""
    _run("./configure", f"--prefix={dest}", cwd=unpacked)
    _run("make", "install", cwd=unpacked)
    _build_haskell_main_stub(dest)


def _build_haskell_main_stub(dest: Path) -> None:
    """Compile the C `main` wrapper GHC otherwise generates and compiles at
    every link, so a `-no-hs-main` link against it never runs the C frontend.

    Harvested from a throwaway link rather than spelled out here, so its
    RtsConfig fields track the installed GHC. `-with-rtsopts=-N` is baked in
    to match the reference build, which is why the stub demands `-threaded`.
    """
    work = dest / "stub-work"
    tmp = work / "tmp"
    tmp.mkdir(parents=True)
    (work / "dummy.hs").write_text("main :: IO ()\nmain = pure ()\n")
    ghc = dest / "bin" / "ghc"
    # `-c` first, so the second run is link-only and its tmpdir holds exactly
    # one .c: the wrapper.
    _run(str(ghc), "-c", "dummy.hs", cwd=work)
    _run(
        str(ghc),
        "-threaded",
        "-with-rtsopts=-N",
        "-keep-tmp-files",
        "-tmpdir",
        str(tmp),
        "-o",
        "dummy",
        "dummy.o",
        cwd=work,
    )
    (wrapper_source,) = tmp.rglob("*.c")
    (rts_include,) = dest.glob("lib/ghc-*/lib/*/rts-*/include")
    stub = dest / "lib" / "hs_main_stub.o"
    # The flags GHC itself passes when it compiles the wrapper.
    _run(
        "gcc",
        "-c",
        str(wrapper_source),
        "-o",
        str(stub),
        "-fPIC",
        "-U__PIC__",
        "-D__PIC__",
        f"-I{rts_include}",
    )
    shutil.rmtree(work)


def _install_dotnet(unpacked: Path, dest: Path) -> None:
    """The SDK unpacks in place; what needs doing is filling a package cache.

    Native AOT restores its ILCompiler and runtime packs off nuget.org the first
    time, and a cell has no network — so the first time has to be here. One
    publish of a throwaway program leaves the ~300MB on disk. Without it the
    grader's build ends at NU1301 with the student's zero.
    """
    for entry in unpacked.iterdir():
        shutil.move(str(entry), dest)

    prime = dest / "prime"
    prime.mkdir(parents=True, exist_ok=True)
    (prime / "server.cs").write_text(
        'class Server { static void Main() { System.Console.WriteLine("ok"); } }\n'
    )
    _run(
        str(dest / "dotnet"),
        "publish",
        str(prime / "server.cs"),
        "-o",
        str(prime / "out"),
        "-p:NuGetAudit=false",
        env={
            "PATH": "/usr/bin:/bin",
            "HOME": str(prime),
            "NUGET_PACKAGES": str(dest / "nuget"),
            "DOTNET_CLI_TELEMETRY_OPTOUT": "1",
            "DOTNET_NOLOGO": "1",
        },
    )
    shutil.rmtree(prime)

    # NuGet extracts with the mode the package recorded, and the ILCompiler's
    # `ilc` carries 0744 — root only. The grader's build is demoted, so without
    # this the publish stops at the native step with code 126. `a+rX` not
    # `a+rx`: the execute bit only goes to directories and files that had one.
    _run("chmod", "-R", "a+rX", str(dest))


def _install_gnucobol(unpacked: Path, dest: Path) -> None:
    """No package and no upstream binary, so cobc is built from source.

    Everything optional is off: each switch would otherwise buy a library the
    image has to carry (libdb for indexed files, curses for SCREEN SECTION,
    XML/JSON). What is left needs only gmp.
    """
    _run(
        "./configure",
        f"--prefix={dest}",
        "--without-db",
        "--with-curses=no",
        "--without-xml2",
        "--with-json=no",
        "--disable-nls",
        cwd=unpacked,
    )
    _run("make", f"-j{os.cpu_count() or 1}", cwd=unpacked)
    _run("make", "install", cwd=unpacked)


def _install_gprolog(unpacked: Path, dest: Path) -> None:
    """GNU Prolog's configure lives under src/, and the tree is small enough
    that a serial make is not worth second-guessing."""
    src = unpacked / "src"
    _run("./configure", f"--prefix={dest}", cwd=src)
    _run("make", cwd=src)
    _run("make", "install", cwd=src)


def _install_clojure(unpacked: Path, dest: Path) -> None:
    """Take the self-contained uberjar and write a launcher around it.

    Not the tarball's own launcher, which resolves the user classpath against
    Maven Central on a cold cache; handing the file straight to `clojure.main`
    inside the uberjar needs no network and no cache.
    """
    jar = Path(CLOJURE_JAR)
    jar.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(unpacked / jar.name), jar)

    launcher = dest / "bin" / "clojure"
    launcher.parent.mkdir(parents=True, exist_ok=True)
    launcher.write_text(f'#!/bin/sh\nexec java -cp "{jar}" clojure.main "$@"\n')
    launcher.chmod(0o755)


INSTALLERS = {
    "rust": _install_rust,
    "ghc": _install_ghc,
    "otp": _install_otp,
    "fpc": _install_fpc,
    "dotnet": _install_dotnet,
    "gnucobol": _install_gnucobol,
    "gprolog": _install_gprolog,
    "clojure": _install_clojure,
}


def _install_archive(toolchain: Toolchain, archive: Archive, workdir: Path) -> None:
    machine = _machine()
    dest = toolchain_dir(toolchain.language) / archive.subdir

    # Named after what was fetched: `_unpack` tells a zip from a tarball by the
    # extension.
    url = archive.url(machine)
    download_path = workdir / Path(urllib.parse.urlparse(url).path).name
    _download(url, archive.sha256[machine], download_path)

    if archive.installer is None:
        _unpack(download_path, dest, archive.strip_components)
    else:
        unpacked = workdir / f"{toolchain.language.value}.unpacked"
        _unpack(download_path, unpacked, archive.strip_components)
        dest.mkdir(parents=True, exist_ok=True)
        INSTALLERS[archive.installer](unpacked, dest)
        shutil.rmtree(unpacked)

    download_path.unlink()


def _stage_rpms(toolchain: Toolchain) -> None:
    """Download a language's RPMs and their not-yet-installed dependencies.

    `dnf download --resolve` resolves against the rpm database of the image it
    runs in, so this has to run in the final image; resolving anywhere else
    stages packages that conflict with what the image already has.
    """
    if not toolchain.rpm_packages:
        return
    dest = toolchain_dir(toolchain.language) / RPM_SUBDIR
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Staging RPMs for {toolchain.language}", file=sys.stderr, flush=True)
    _run("dnf", "download", "--resolve", f"--destdir={dest}", *toolchain.rpm_packages)


def _seal() -> None:
    """Close every language directory, the parent over them, and the assembler
    and linker the base image carries.

    0711 on the parent so the student cannot list which languages exist, 0700 on
    each child so guessing a name gets them nothing. Everything below stays
    readable, which is what makes publishing one toolchain a single chmod.

    `as` and `ld` go the same way as a language directory — closed here, opened
    by `install_toolchain` for the cells that name them — rather than staying
    open until the grading seals them. A submission can carry a prebuilt object
    as bytes, so what an assembler during the episode buys a cell that never
    compiles is a run answered in machine code against a score calibrated for
    the language that was asked for.
    """
    if "KAROTTE_CONTAINERIZED" not in os.environ:
        raise RuntimeError("_seal only runs inside the environment image")

    for language in Language:
        directory = toolchain_dir(language)
        if directory.exists():
            directory.chmod(SEALED_MODE)
    TOOLCHAINS_DIR.chmod(0o711)
    _mode_base_image_build_tools(SEALED_MODE)


def _build_log_path(workdir: Path, language: Language) -> Path:
    return workdir / f"{language.value}.build.log"


def _install_language_archives(language: Language, workdir: Path) -> None:
    """One language's downloads and installers, output gathered into a log the
    parent dumps once the language finishes."""
    print(f"Building {language.value} toolchain", file=sys.stderr, flush=True)
    with open(_build_log_path(workdir, language), "w") as log:
        _build_log.file = log
        try:
            for archive in TOOLCHAINS[language].archives:
                _install_archive(TOOLCHAINS[language], archive, workdir)
        finally:
            del _build_log.file


def install_archives(workdir: Path) -> None:
    """Install every enabled toolchain that comes as an archive into
    `TOOLCHAINS_DIR`.

    Runs in the builder stage, which has the C toolchain GHC needs to configure
    itself. Only `TOOLCHAINS_DIR` is copied out, which is how the compilers that
    did the work stay out of the final image.

    Languages install concurrently — each stays inside its own directory. A
    failure is raised after the rest have finished, so one bad download does
    not hide what else would have broken.

    Every language gets a directory, enabled or not: the Containerfile copies
    them out one at a time, and a copy with nothing at the path fails the
    build.
    """
    TOOLCHAINS_DIR.mkdir(parents=True, exist_ok=True)
    for language in Language:
        toolchain_dir(language).mkdir(exist_ok=True)
    workdir.mkdir(parents=True, exist_ok=True)

    languages = [
        language
        for language in sorted(enabled_languages())
        if TOOLCHAINS[language].archives
    ]
    for language in sorted(enabled_languages()):
        if TOOLCHAINS[language].run_tree:
            _build_run_jre(language)

    failed: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(ARCHIVE_BUILD_JOBS) as pool:
        futures = {
            pool.submit(_install_language_archives, language, workdir): language
            for language in languages
        }
        for future in concurrent.futures.as_completed(futures):
            language = futures[future]
            log_path = _build_log_path(workdir, language)
            if log_path.exists():
                sys.stderr.write(log_path.read_text(errors="replace"))
                sys.stderr.flush()
            if (error := future.exception()) is not None:
                print(f"{language.value}: {error!r}", file=sys.stderr, flush=True)
                failed.append(language.value)
    if failed:
        raise SystemExit(f"toolchain archives failed: {', '.join(sorted(failed))}")


def _build_run_jre(language: Language) -> None:
    """jlink the runtime a JVM cell's scored run starts.

    Built here rather than installed at run time because jlink only ships with
    the devel JDK, which the Kotlin cell does not stage, and because a run has
    better things to do than link a runtime.
    """
    dest = toolchain_dir(language) / TOOLCHAINS[language].run_tree
    if dest.exists():
        shutil.rmtree(dest)
    print(f"Linking the {language.value} run's own JRE", file=sys.stderr, flush=True)
    _run(
        "jlink",
        f"--add-modules={','.join(JRE_MODULES)}",
        "--strip-debug",
        "--no-header-files",
        "--no-man-pages",
        f"--output={dest}",
    )


def _create_builder() -> None:
    """The account builds demote to: not root, so a submission cannot bake
    root-only data into its artifact, and not the student, so nothing granted
    to its group is the student's to run."""
    _run("groupadd", "--gid", str(BUILDER_UID), "builder")
    _run(
        "useradd",
        "--no-create-home",
        "--no-user-group",
        "--uid",
        str(BUILDER_UID),
        "--gid",
        str(BUILDER_UID),
        "--shell",
        "/sbin/nologin",
        "builder",
    )
    # useradd may allocate subordinate id ranges, with which a user namespace
    # can remap uids; the student's entries are removed for the same reason.
    for allocations in SUBORDINATE_ID_FILES:
        if allocations.exists():
            kept = [
                line
                for line in allocations.read_text().splitlines(keepends=True)
                if not line.startswith("builder:")
            ]
            allocations.write_text("".join(kept))


def stage_rpms_and_seal() -> None:
    """Create the builder account, stage every enabled language's RPMs, then
    close all the directories.

    Has to run in the final image; see `_stage_rpms`. Runs even with nothing
    enabled: the sealed, unlistable `TOOLCHAINS_DIR` and the closed base-image
    build tools are what the permission checks assert against.
    """
    _create_builder()
    TOOLCHAINS_DIR.mkdir(parents=True, exist_ok=True)
    for language in sorted(enabled_languages()):
        _stage_rpms(TOOLCHAINS[language])
    _seal()


# --- Run half: called from a suite's pre_hook ------------------------------- #


def _install_rpms(toolchain: Toolchain) -> None:
    """Install the staged RPMs with no repository and no network.

    `--disablerepo` is what makes this work inside a run: everything the packages
    need was staged beside them at build time.
    """
    rpm_dir = toolchain_dir(toolchain.language) / RPM_SUBDIR
    rpms = sorted(str(p) for p in rpm_dir.glob("*.rpm"))
    if not rpms:
        return
    _run("dnf", "install", "-y", "--quiet", "--disablerepo=*", *rpms)


def _archive_executables(toolchain: Toolchain) -> list[Path]:
    """The binaries in the toolchain's own directory, one wrapper's worth each."""
    prefix = toolchain_dir(toolchain.language)
    found: list[Path] = []
    for bin_dir in toolchain.bin_dirs:
        directory = prefix / bin_dir
        if not directory.is_dir():
            continue
        found.extend(
            entry
            for entry in sorted(directory.iterdir())
            if entry.is_file() and os.access(entry, os.X_OK)
        )
    return found


def _wrapper_paths(toolchain: Toolchain) -> list[Path]:
    """This toolchain's entries on the student's `PATH`."""
    return [WRAPPER_DIR / entry.name for entry in _archive_executables(toolchain)]


def _write_wrappers(toolchain: Toolchain) -> list[Path]:
    """Put each of the toolchain's executables on the student's `PATH`.

    Wrappers rather than symlinks: several of these resolve their own libraries
    from `argv[0]`, which a symlink would point at `/usr/local/bin`. It is also
    where `wrapper_env` gets set.
    """
    prefix = toolchain_dir(toolchain.language)
    exports = "\n".join(
        f'export {name}="{value.format(prefix=prefix)}"'
        for name, value in sorted(toolchain.wrapper_env.items())
    )

    written: list[Path] = []
    for entry in _archive_executables(toolchain):
        wrapper = WRAPPER_DIR / entry.name
        wrapper.write_text(
            f'#!/bin/sh\n{exports}\nexec "{entry}" "$@"\n'
            if exports
            else f'#!/bin/sh\nexec "{entry}" "$@"\n'
        )
        wrapper.chmod(0o755)
        written.append(wrapper)
    return written


def _rpm_executables(toolchain: Toolchain) -> list[Path]:
    """Every executable file the language's staged RPMs put on the system.

    Derived with `rpm -qlp` rather than from a list kept here: a list would drift
    from whatever `dnf download --resolve` dragged in, and forgetting one fails
    silently as a compiler still reachable while a server is timed.

    Symlinks are resolved first, because `chmod` resolves them too — the mode
    lands on the target either way. Debuginfo `.build-id` entries are why that
    matters: hex-named links straight at a binary, which `keep` and the
    shared-library exemption would both miss.

    Shared libraries are left alone despite the execute bit: they are loaded,
    not run, and sealing one starves what must survive — a kept `java` cannot
    load its own `libjli.so`.
    """
    rpm_dir = toolchain_dir(toolchain.language) / RPM_SUBDIR
    rpms = sorted(str(p) for p in rpm_dir.glob("*.rpm"))
    if not rpms:
        return []
    listed = subprocess.run(
        ["rpm", "-qlp", *rpms], check=True, capture_output=True, text=True
    )
    # Keyed by resolved path, so the several names one binary answers to close it
    # once rather than once each.
    executables: dict[Path, None] = {}
    for line in listed.stdout.splitlines():
        path = Path(line.strip())
        if not path.is_file() or not os.access(path, os.X_OK):
            continue
        real = path.resolve()
        if ".so" not in real.name:
            executables[real] = None
    return list(executables)


def _archive_tree_executables(toolchain: Toolchain) -> list[Path]:
    """Every executable file anywhere under the toolchain's own directory,
    except the run tree's.

    Wider than `_archive_executables`, which only looks in `bin_dirs`. Sealing
    cares about everything reachable by path — a BEAM distribution keeps a second
    copy of its compiler several directories down. Same two rules as the RPM
    half: resolved, and no shared libraries.

    The run tree is skipped rather than exempted by name: `keep` matches
    basenames, and the name a JVM cell would have to keep is `java`, which is
    also what the distribution's JDK calls the one binary that must not survive.
    """
    run_tree = (
        toolchain_dir(toolchain.language) / toolchain.run_tree
        if toolchain.run_tree
        else None
    )
    executables: dict[Path, None] = {}
    for path in sorted(toolchain_dir(toolchain.language).rglob("*")):
        if run_tree is not None and run_tree in path.parents:
            continue
        if not path.is_file() or not os.access(path, os.X_OK):
            continue
        real = path.resolve()
        if ".so" not in real.name:
            executables[real] = None
    return list(executables)


def _archive_holds_something_kept(toolchain: Toolchain) -> bool:
    """Does the run need something that lives inside the archive?

    Derived rather than declared, so a cell cannot disagree with its own `keep`.
    """
    keep = set(toolchain.keep)
    return any(entry.name in keep for entry in _archive_executables(toolchain))


def _seals_file_by_file(toolchain: Toolchain) -> bool:
    """Whether sealing must leave the language directory open and close its
    executables one by one: because the run keeps an executable that lives in
    the archive, because the cell's artifacts load libraries out of it
    (`seal_by_file`), or because the run's own runtime lives in it."""
    return (
        toolchain.seal_by_file
        or bool(toolchain.run_tree)
        or _archive_holds_something_kept(toolchain)
    )


def seal_toolchain(
    language: Language, *, keep_student_python: bool = False
) -> list[Path]:
    """Undo `install_toolchain`, and close the image's interpreters with it.

    Called by the grader after anything it compiles is compiled and before
    student code runs unwatched. The toolchain was only ever the builder's, so
    that half is defense in depth; what this takes away that the student did
    have is every interpreter `keep` does not name.

    `keep_student_python` is for the Python cell, whose submission the grader
    runs with `student_python()` after sealing; everywhere else every
    interpreter is closed.

    Returns what it closed.
    """
    # This chmods absolute paths under /usr. In the image those belong to the run
    # and are meant to stop working; on a dev box they are the host's.
    if "KAROTTE_CONTAINERIZED" not in os.environ:
        raise RuntimeError("seal_toolchain only runs inside the environment image")

    toolchain = TOOLCHAINS[language]
    keep = set(toolchain.keep)
    closed: list[Path] = []

    # Python is in the image whatever the cell, because dnf is written in it and
    # because the student's venv has to come from somewhere. Left open it is a
    # general-purpose runtime in every cell: a compiled artifact that execs it,
    # a `ctypes.CDLL` over something prebuilt, or a published reference script
    # run as if it were a submission. So every interpreter is closed unless the
    # caller says its run needs the one managed CPython.
    wanted = {student_python().resolve()} if keep_student_python else set()
    for interpreter in python_interpreters():
        if interpreter in wanted:
            continue
        interpreter.chmod(SEALED_MODE)
        closed.append(interpreter)

    for interpreter in sorted(wanted):
        interpreter.chmod(READ_ONLY_MODE)

    if not keep_student_python:
        closed.extend(_seal_uv_python())

    # The same argument closes the rest of the image's scripting runtimes,
    # perl and gawk among them, unless the cell's `keep` names one because it
    # is the cell's own runtime.
    closed.extend(_seal_base_image_interpreters(frozenset(keep)))

    # Before the directory is sealed: the wrappers are named after what is inside
    # it, so this has to look while it can still be listed.
    for wrapper in _wrapper_paths(toolchain):
        if wrapper.name not in keep and wrapper.exists():
            wrapper.unlink()
            closed.append(wrapper)

    for executable in _rpm_executables(toolchain):
        if executable.name in keep:
            continue
        executable.chmod(SEALED_MODE)
        closed.append(executable)

    # Closed again for the cells that had them open, and left closed for the
    # rest — `install_toolchain` never opened those.
    closed.extend(_mode_base_image_build_tools(SEALED_MODE, frozenset(keep)))

    # Closing the directory is the whole of the archive half, and it is the
    # cheap way: one chmod over a tree with a compiler in it.
    #
    # It is not available to a cell whose *runtime* lives in that tree — a
    # BEAM VM, a Julia — because closing the directory takes the run away with
    # the compiler. Those are sealed file by file instead, and the directory
    # is left open so the kept binary can still load what it needs. Which one
    # a cell gets is derived from `keep` rather than declared (plus the one
    # cell whose artifacts load libraries from the tree, which says so with
    # `seal_by_file`), and the polarity is deliberate: everything is closed
    # unless `keep` names it, so a name left out breaks the run loudly instead
    # of leaving a compiler reachable.
    directory = toolchain_dir(language)
    if not directory.exists():
        return closed

    if _seals_file_by_file(toolchain):
        for executable in _archive_tree_executables(toolchain):
            if executable.name not in keep:
                # Handed to root first, because a mode is only a restraint on
                # someone who does not own the file. Archives are unpacked
                # `--no-same-owner` so this should already hold; doing it here
                # too means the seal does not depend on that having worked.
                os.chown(executable, 0, 0)
                executable.chmod(SEALED_MODE)
                closed.append(executable)
    else:
        directory.chmod(SEALED_MODE)
        closed.append(directory)

    return closed


def install_toolchain(language: Language) -> None:
    """Make exactly `language`'s toolchain runnable by the builder uid.

    Called from a `pre_hook`, as root, before the agent gets a shell. Nothing
    opens for the student: the build MCP tool compiles on their behalf,
    demoted to the builder, and only what `keep` names — the artifacts' own
    runtime — is theirs to run.
    """
    # Same reason `seal_toolchain` has one: the modes below land on /usr/bin.
    if "KAROTTE_CONTAINERIZED" not in os.environ:
        raise RuntimeError("install_toolchain only runs inside the environment image")

    # A disabled language has nothing staged, so opening it would half-work:
    # the chmod succeeds on nothing and the RPM install silently installs
    # nothing. Refusing is the loud version of the same outcome.
    if language not in enabled_languages():
        raise RuntimeError(
            f"{language.value} is not in toolchain_config.ENABLED_LANGUAGES; "
            "enable it and rebuild the image"
        )

    toolchain = TOOLCHAINS[language]
    keep = set(toolchain.keep)

    directory = toolchain_dir(language)
    if directory.exists():
        if _seals_file_by_file(toolchain):
            # The kept runtime lives in the tree or the artifacts load
            # libraries out of it, so the directory stays open and each
            # non-kept executable becomes the builder's alone.
            directory.chmod(OPEN_MODE)
            for executable in _archive_tree_executables(toolchain):
                if executable.name not in keep:
                    _grant_builder(executable)
        else:
            _grant_builder(directory)

    # Set either way rather than only where the cell asks for them: what the
    # builder can reach is the table's to say, not something a cell inherits
    # from whatever the image happened to leave behind.
    if toolchain.binutils:
        for tool in base_image_build_tools():
            _grant_builder(tool)
    else:
        _mode_base_image_build_tools(SEALED_MODE)

    _install_rpms(toolchain)

    # The RPMs install their executables world-runnable; what `keep` names is
    # the run's own runtime and stays that way, the rest is the builder's.
    for executable in _rpm_executables(toolchain):
        if executable.name not in keep:
            _grant_builder(executable)

    # Last, so the table outranks the loop above: a cell that stages gcc only
    # to link still lists the frontends among its RPM executables, and the
    # grant there would hand the builder a C compiler the table denies it.
    if toolchain.c_frontend:
        for frontend in c_frontends():
            _grant_builder(frontend)
    else:
        _mode_c_frontends(SEALED_MODE)

    for wrapper in _write_wrappers(toolchain):
        if wrapper.name not in keep:
            _grant_builder(wrapper)

    _seal_within(toolchain)


def main() -> None:
    phases = {
        "archives": lambda: install_archives(Path("/tmp/toolchain_build")),
        "rpms": stage_rpms_and_seal,
    }
    if len(sys.argv) != 2 or sys.argv[1] not in phases:
        raise SystemExit(f"usage: {sys.argv[0]} " + "|".join(phases))
    phases[sys.argv[1]]()


if __name__ == "__main__":
    main()
