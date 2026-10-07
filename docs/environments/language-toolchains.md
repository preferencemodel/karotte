# Language toolchains

!!! warning "Experimental"

    The `language-toolchains` template is experimental.
    Its functions, files and guarantees can change in any release.

The `language-toolchains` template builds compilers, SDKs and runtimes into your image and locks them behind root-only permissions.
Use it for tasks that vary by programming language.

- In the idle image, the student can't reach any compiler.
- A task's `pre_hook` opens exactly one language, and only for a dedicated `builder` user.
  The student compiles through the `build` tool, which runs the task's pinned grading build on the student's behalf.
- The grader seals everything again before the student's code runs in a scored phase, so nothing can be compiled on the fly.
- The scored run starts under a seccomp filter on a read-only filesystem.
  Even if a submission got machine code past the compiler, it still can't exec, ptrace, or write files.
  See [Confining the scored run](#confining-the-scored-run).

## Opting in

```sh
karotte create-env my_env --template language-toolchains
```

The template builds on the `default` template.
To add it to an existing environment, run `karotte update --add-template language-toolchains` (see [Updating environments](updating.md#adding-a-template)).

## Enabling languages

No language is enabled by default.
To enable some, edit `src/environment/toolchain_config.py` and rebuild the image:

```python
ENABLED_LANGUAGES: frozenset[str] = frozenset({"rust", "c_cpp", "python"})
```

The valid names are the `Language` values in `src/environment/toolchains.py`:

`python`, `rust`, `zig`, `haskell`, `swift`, `go`, `ocaml`, `kotlin`, `java`, `dart`, `c_cpp`, `js_ts`, `ruby`, `scala`, `erlang_elixir`, `julia`, `assembly`, `pascal`, `csharp`, `llvm_ir`, `fortran`, `clojure`, `cobol`, `prolog`, `mojo`

If you list an unknown name, the build fails.
Only enable the languages your tasks use.
Each one adds its archives and staged RPMs to the image, which takes hundreds of MB to a few GB per language.
`erlang_elixir` also compiles OTP from source during the build.

!!! note

    The toolchains are third-party software, each under its own license.
    Several are GPL or LGPL.
    Amazon Corretto is GPL-2.0 with the Classpath Exception, and Clojure is EPL-1.0.
    Mojo has its own license terms; check them for the version you enable.
    If you distribute a built image, you're responsible for complying with the licenses of the software in it.

## At image build

The template's `Containerfile` does the following:

1. A builder stage downloads each enabled language's distribution into `/opt/toolchains/<language>/`.
   Each download is pinned by URL and sha256, separately for each architecture (x86_64 and aarch64).
2. The JVM cells (`java`, `kotlin`, `scala`, `clojure`) get a runtime linked with `jlink` under `<language>/jre`.
   It contains the modules in `toolchains.JRE_MODULES` and leaves out `jdk.compiler`.
3. The three confinement shims and the probe that checks them are compiled in the builder stage and installed into `/opt/grader` with mode 0555: `sandbox_shim.so`, `sandbox_shim_mapped.so`, `sandbox_shim_jit.so` and `sandbox_probe`.
4. In the final image, `dnf download --resolve` stages each language's RPMs in `<language>/rpms`, so a run can install them offline.
5. It creates the `builder` account (uid 900) and seals everything.
   Each language directory is 0700.
   `/opt/toolchains` is 0711, so the student can't list it.
   The base image's own assembler and linker (`as`, `ld` and friends) are closed too.

`karotte check` verifies the result during the build.
See [Verifying the image](#verifying-the-image).

## During a run

`install_toolchain(language)`
: Call it from your task's `pre_hook`.
It opens the language directory, the staged RPMs, and `as`/`ld` (if the toolchain needs them) to the `builder` user.
It refuses any language that isn't enabled.
The student gets nothing, except a runtime for cells that need one to run.
`node`, `ruby`, `julia` and the BEAM emulator stay runnable by name (`Toolchain.keep`).
The JVM cells instead keep their compiler-free `jre` tree (`Toolchain.run_tree`), and the run calls it by absolute path.

`seal_toolchain(language, keep_student_python=...)`
: Call it from your grader once everything is compiled, before the student's code runs in a scored phase.
It removes the builder's access.
It also closes what the student could use during the run: every Python interpreter on the image, and the general-purpose interpreters that the base image or an RPM dependency left behind (`perl`, `awk`, anything matching `BASE_IMAGE_INTERPRETER_GLOBS`).
It leaves one of those general-purpose interpreters open if the cell keeps it as its own runtime.
Shells and `sed` stay open.

With `keep_student_python=False`, the whole uv-managed CPython tree at `/opt/uv/python` becomes root-only, not just its `bin/`.
That's because its `libpython3.*.so` and the stdlib next to it make a working interpreter for anything that can `dlopen` it.
Your grader can still run everything in there as root.
But if your task reads anything from that tree as the student or the builder, it has to do that before sealing.

Both functions only run inside the image.

### The `build` tool

The student compiles with `environment/tools/build.py`.
It takes no arguments.
It picks the submission from the workdir the same way grading does.
Then it builds it as the builder, using the exact grading commands.
It publishes the artifact, plus any `keep_built` companions, to a root-owned directory.
That directory is `/opt/build` by default, and every call wipes it.
The compiler's output goes to `build.stdout` and `build.stderr` in the same directory.
The student can read and run everything there, but can't write anything.

Register it in each task that uses it:

```python
@property
def tools(self):
    return ["bash", "view_lines_in_file", "replace_in_file", "build"]

def configure_tools(self):
    from karotte import ToolConfigWriter
    from environment.tools.build import BuildConfig

    ToolConfigWriter().write("build", BuildConfig.from_submissions([RUST]))
```

For most cells, pass one `Submission`.
For the BEAM cell, pass one per language.
The tool then accepts either file, but exactly one of them.
Interpreted languages have no build commands and don't use the tool.
See [Tools](../tasks/tools.md) for how tool configuration works in general.

## Grading single-file submissions

`environment/toolchain_grading.py` has the build pipeline for a common kind of task: the student submits one source file, and the grader builds it and runs it with a pinned argv.

```python
from environment.toolchain_grading import (
    Submission, chosen_submission, make_build_dir, stage_submission,
    build_submission, resolve_argv, harden_run_argv, sandboxed_env,
    run_noexec_dirs, make_run_preexec, check_native_sandbox,
    SAFE_PATH, UTF8_LOCALE,
)
from environment.toolchains import Language, seal_toolchain
from environment.paths import STUDENT_WORKDIR

RUST = Submission(
    source="server.rs",
    build=(("rustc", "-C", "opt-level=3", "-o", "{artifact}", "{source}"),),
)

submission, why_not = chosen_submission([RUST])          # symlink-safe file pick
build_dir = make_build_dir()                             # random name under root-owned /opt/grader/build
source = stage_submission(submission, build_dir)         # copy the one file; nothing resolves beside it
result = build_submission(submission, source, build_dir) # demoted build, env from scratch, then sealed read-only
# result is a BuildResult: .artifact, .error (None on success), .stdout, .stderr
seal_toolchain(Language.RUST)                            # no compiler during the scored run

run_argv = harden_run_argv(
    resolve_argv(submission.run, source, result.artifact, build_dir), Language.RUST
)
env = sandboxed_env(
    {"PATH": SAFE_PATH, "HOME": str(STUDENT_WORKDIR), "LC_CTYPE": UTF8_LOCALE},
    Language.RUST,
)
noexec = run_noexec_dirs(Language.RUST, run_argv, result.artifact, build_dir)
preexec = make_run_preexec(
    1, 2, seal_root=True, filter_syscalls=True, noexec_dirs=noexec
)
unconfined = check_native_sandbox(preexec, Language.RUST, run_argv, noexec)
# unconfined is None, or why this machine couldn't confine the run: grade it
# unreliable rather than scoring the submission.
# launch run_argv with env and preexec_fn=preexec
```

The `build` tool runs the same pipeline during the episode.
That's what makes the tool's build the grading build.
`build_submission` is only for cells with build commands.

The build runs as the builder, so a submission can't `include_bytes!` root-only data.
Its environment is built from scratch, so a `PATH` shim owned by the student never takes effect.
Each build command gets 300 seconds.

The build also gets its own mount namespace.
That's because a compiler can be tricked into running the student's code at compile time, even when the pinned argv doesn't change.
For example, GHC's `{-# OPTIONS_GHC -F -pgmF … #-}` pragma names a program to run, and it sits inside the submitted source.
Instead of chasing every hook like that, the build gets a filesystem that keeps nothing.
`/tmp`, `/var/tmp`, `/dev/shm` and the student's workdir are fresh, empty tmpfs mounts that go away with the build (`EPHEMERAL_BUILD_DIRS`).
The build directory lives under `/opt/grader`, which is root-owned and can't be listed.
`TMPDIR` points there for compilers that need scratch space.

Once the last build command exits 0, `build_submission` does a few more things:

- **Kills builder processes.** A compiler that runs student code at compile time can leave a process behind that swaps out the artifact later.
  So every process under the builder uid is killed before the artifact checks.
  If the kill fails, the build fails.
- **Refuses a symlink at the artifact path.** The chmods that seal the directory follow links, so root would act on whatever the link points at.
- **Checks the artifact.** See [The artifact gate](#the-artifact-gate).
- **Clears the build scratch.** It deletes everything the build wrote except the artifact.
  Languages that need extra build output at run time keep it with `keep_built=("*.beam",)`.
- **Restores the staged source.** A build command can rewrite the staged copy.
  So `build_submission` reads its bytes before the first command and writes them back after the last.
  That way, a grader that treats the staged copy as ground truth always has the right data.
- **Seals the build directory.** The directory goes back to root and becomes read-only, so the artifact can't be swapped after sealing.

The image sets `LC_CTYPE=C.UTF-8`, and `build_submission` passes the same value (`UTF8_LOCALE`) to the build.
Pass it when you launch the run, too.
Put it in the environment, not in a runtime flag.
A server that re-execs itself keeps its environment but loses the flag.

The template doesn't say what each language's `Submission` looks like, because that depends on the task.
The [reference below](#per-language-submission-reference) collects values that are known to work.

## Confining the scored run

Sealing takes the compilers away, but the scored run still executes code the student wrote.
Every cell is confined end to end by a seccomp filter and a read-only filesystem.
The cells that need it also get a noexec bind over wherever the submission is staged.

The Python cell bootstraps through `environment/sandbox.py`.
`stage_sandbox` copies that module next to the staged source, and `sandboxed_argv` puts it in front of the student's file.
The module then confines the interpreter before `runpy` gets the file.
`check_sandbox(sandbox, preexec_fn)` runs the module's self-check under the launch's own preexec.
It returns the reason the machine couldn't confine a run, or `None`.

The other cells have no interpreter to bootstrap through.
For them, the filter goes in a constructor of a preloaded shared object.
`sandboxed_env(env, language)` adds that object to `LD_PRELOAD`.
The loader runs it after every library is mapped and before the artifact's own initializers.

Both approaches refuse the same syscalls: `execve`, `execveat`, `ptrace`, `memfd_create`, the `io_uring` family, `shmget`/`shmat`, and `prctl(PR_SET_DUMPABLE)`.
So `subprocess`, `system`, `child_process` and every other way of reaching a shell fail.

### The three shims

`run_confinement(language)` picks the shim from the toolchain table, not from the task.
That way `sandboxed_env` and `check_native_sandbox` can't describe different confinements.
Each cell gets the weakest shim its runtime needs.

| Shim          | Executable pages                                                           | Cells                                                          |
| ------------- | -------------------------------------------------------------------------- | -------------------------------------------------------------- |
| `STRICT`      | None, ever. No JIT, and no `dlopen` of anything that isn't already mapped. | Every cell not listed below.                                   |
| `MAPPED_CODE` | A file mapping may be executable if it isn't also writable.                | `dart`, `ruby`                                                 |
| `JIT`         | Any page may become executable.                                            | `java`, `kotlin`, `scala`, `clojure`, `erlang_elixir`, `julia` |

The JIT shim really is weaker.
If a submission can get its own machine code into the process, that code runs.
In those cells, the syscall filter and the read-only root are what still hold.

### What a task has to pass

`harden_run_argv(argv, language)`
: Adds the flags the cell's runtime needs: `--illegal-native-access=deny` on the JVM cells, and `--jitless --no-expose-wasm` on `node`.
This means **a JS submission runs with no JIT and no WebAssembly**.
Keep that in mind when you calibrate a throughput task.

`sandboxed_env(env, language)`
: Adds `LD_PRELOAD` (the shim, plus the cell's `run_preload` libraries) and the cell's `run_env`.
Anything that shows the student how its submission will be run must take the value from here.
Otherwise it won't match grading.

`run_noexec_dirs(language, run_argv, artifact, build_dir)`
: For a mapped-code cell whose program is an interpreter, it names the build directory and the workdir.
For anything else, it names nothing.
Read-only doesn't mean noexec.
Without the bind, a Ruby submission can `mmap` its own file with `PROT_EXEC` and jump into bytes it placed after `__END__`.

`make_run_preexec(*chown_fds, seal_root=, filter_syscalls=, noexec_dirs=)`
: Joins the process confinement and seals the filesystem read-only.
It also filters the syscalls that can be refused before the exec, binds `noexec_dirs`, covers the platform's own mounts, and drops to the student's user.
`filter_syscalls=True` is for a cell that execs an artifact.
An interpreted run is filtered from inside the interpreter.

`check_native_sandbox(preexec_fn, language, run_argv, noexec_dirs)`
: Asks the probe whether all of that confinement took effect.
For a cell with injected run flags (the JVM cells and `js_ts`), it refuses a launch without `run_argv` or with flags missing from it.
For a mapped-code cell, it refuses a launch without `noexec_dirs`.
That's because the injected flags are the one part of the confinement that no probe can see from inside.
If it returns anything other than `None`, the problem is the environment's, not the student's.
Report the run as unreliable.

The platform mounts that `make_run_preexec` and the build cover come from the `karotte.platform_tooling_dirs` entry point (see [Plugins](../extending/plugins.md)).

### The artifact gate

A preloaded shim only works on an artifact that the loader sets up.
So `build_submission` refuses an ELF artifact that:

- is statically linked (it has no `PT_INTERP`, so nothing reads `LD_PRELOAD`),
- names an interpreter other than the image's loader, or
- carries a `.preinit_array` hook or an ifunc resolver, since both run before the shim's constructor.

A jar or a script isn't checked.
One of the image's own runtimes loads it.

`harden_build_argv` stops the compilers from producing such an artifact in the first place.
It adds these flags:

| Compiler         | Injected                                                                                                                                    |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `go build`       | `-buildmode=pie`                                                                                                                            |
| `zig build-exe`  | `-lc`                                                                                                                                       |
| `ld`             | `-pie -dynamic-linker <loader>`. This means the assembly and LLVM IR cells need position-independent code.                                  |
| `llc`            | `-relocation-model=pic`                                                                                                                     |
| `fpc`            | `-k-lc -k--dynamic-linker=<loader>`                                                                                                         |
| `dotnet publish` | `-p:StaticExecutable=false -p:PublishAot=true`                                                                                              |
| `gplc`           | `--global-size 524288 --local-size 131072 --trail-size 131072`. GNU Prolog defaults to a 32 MB global stack that it never garbage collects. |

## Enforcing the language

`harden_build_argv` also closes each language's own escape hatch.
So if a task leaves the flag out of its pinned argv, it still doesn't end up grading an unenforced submission.
If your task is about what unsafe code can do, pass `allow_unsafe=True` on the `Submission` to skip this.

| Cell      | Injected                                           | Closes                                                    |
| --------- | -------------------------------------------------- | --------------------------------------------------------- |
| `rust`    | `-F unsafe-code`                                   | `asm!`/`global_asm!` and the unsafe blocks that reach FFI |
| `swift`   | `-strict-memory-safety -Werror StrictMemorySafety` | the `Unsafe*Pointer` family                               |
| `haskell` | `-fexternal-interpreter`                           | Template Haskell splices                                  |
| `csharp`  | `-p:AllowUnsafeBlocks=false`                       | `unsafe` blocks                                           |

For Haskell, `-fexternal-interpreter` moves splice evaluation into a separate `ghc-iserv`.
The cell makes every `ghc-iserv` unrunnable, so not even the builder can start one.

For the JVM cells, the runtime does the enforcing instead of a build flag.
A JDK ships `jdk.compiler` as a module, so a scored Kotlin or Scala run could compile Java in its own process and run it.
The cell's own `jlink`ed runtime leaves that module out.
`jvm_java(language)` names that runtime's `java`.
`--illegal-native-access=deny` makes the foreign function API refuse native calls.
Both rely on the run sealing the filesystem read-only.
Otherwise a submission could drop an ELF file and `System.load` it, which reaches native code without needing either.

**Where enforcement stops.** Some cells have no such flag to inject, because reaching machine code is a first-class feature of the language: C, C++, assembly, LLVM IR, Go's `unsafe`, Free Pascal's `asm`, Zig's `asm volatile`, Mojo's inline assembly and external calls, and Julia's `ccall` to a computed pointer.
A submission in one of those cells can carry its own machine code.
There, the only guarantee is at the syscall level.

## Per-language submission reference

`resolve_argv` fills in `{source}`, `{artifact}`, `{python}` and `{build_dir}`.
The hello-world programs in `scripts/check_toolchains.py` build and run each cell the same way, except that the Python one runs the image's `python3`.

| Language        | Source    | Build                                                                                                                                                           | Run                                                                                                                         | Notes                                                                                                                                                                                                                                                                                               |
| --------------- | --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `python`        | `x.py`    | none                                                                                                                                                            | `{python} {source}`                                                                                                         | Run it with the managed CPython, which root owns. Never use `/workdir/.venv/bin/python`, which the student owns. Seal with `keep_student_python=True`.                                                                                                                                              |
| `rust`          | `x.rs`    | `rustc -O -o {artifact} {source}`                                                                                                                               | `{artifact}`                                                                                                                | No cargo and no registry, so only `std` is available.                                                                                                                                                                                                                                               |
| `c_cpp`         | `x.cpp`   | `g++ -O3 -pthread -o {artifact} {source}`                                                                                                                       | `{artifact}`                                                                                                                | One cell covers both languages, since g++ compiles C-style code as C++. The student can't pass flags.                                                                                                                                                                                               |
| `go`            | `x.go`    | `go build -o {artifact} {source}`                                                                                                                               | `{artifact}`                                                                                                                | Building from a list of files is the only mode that works without a module.                                                                                                                                                                                                                         |
| `zig`           | `x.zig`   | `zig build-exe -O ReleaseFast -femit-bin={artifact} {source}`                                                                                                   | `{artifact}`                                                                                                                | Without `-femit-bin`, the binary is named after the source.                                                                                                                                                                                                                                         |
| `haskell`       | `x.hs`    | `ghc -O2 -c {source}`, then `ghc -O2 -threaded -no-hs-main -o {artifact} x.o <toolchains.HASKELL_MAIN_STUB>`                                                    | `{artifact}`                                                                                                                | GHC's usual link step compiles a generated C `main`, but this cell's C frontend is sealed. So link against the precompiled stub instead. The stub turns on multi-core, which is why the link keeps `-threaded`.                                                                                     |
| `swift`         | `x.swift` | `swiftc -O -static-stdlib -o {artifact} {source}`                                                                                                               | `{artifact}`                                                                                                                | `-static-stdlib` matters, because the runtime lives in the sealed directory.                                                                                                                                                                                                                        |
| `ocaml`         | `x.ml`    | `ocamlopt -I +threads unix.cmxa threads.cmxa -o {artifact} {source}`                                                                                            | `{artifact}`                                                                                                                | Link `unix` and `threads` up front, because the student can't add link arguments. This is the distribution's OCaml, without opam or dune.                                                                                                                                                           |
| `kotlin`        | `x.kt`    | `kotlinc {source} -include-runtime -d {artifact}`                                                                                                               | `<toolchains.jvm_java(KOTLIN)> -jar {artifact}`                                                                             | The artifact must end in `.jar`. `-include-runtime` is what makes it survive sealing.                                                                                                                                                                                                               |
| `java`          | `x.java`  | `javac -d classes {source}`, then `jar --create --file {artifact} --main-class=<class> -C classes .`                                                            | `<toolchains.jvm_java(JAVA)> -jar {artifact}`                                                                               | Don't use `java {source}`, because source-file mode compiles at startup. javac ties the class name to the file name.                                                                                                                                                                                |
| `scala`         | `x.scala` | `scalac -d {artifact} {source}`, then `unzip -oq <toolchains.SCALA_LIBRARY_JAR> -d library -x 'META-INF/*'`, then `jar --update --file {artifact} -C library .` | `<toolchains.jvm_java(SCALA)> -jar {artifact}`                                                                              | scalac has no `-include-runtime`. Fold the library into the jar, or it breaks at sealing.                                                                                                                                                                                                           |
| `dart`          | `x.dart`  | `dart compile exe {source} -o {artifact}`                                                                                                                       | `{artifact}`                                                                                                                |                                                                                                                                                                                                                                                                                                     |
| `js_ts`         | `x.ts`    | `tsc --noCheck --target es2023 --module commonjs --esModuleInterop --outDir . {source}`                                                                         | `node {artifact}`                                                                                                           | The artifact is `<stem>.js`, since tsc has no `-o`. `--noCheck` is there because `@types/node` needs a registry the image doesn't have. It runs with no JIT and no WebAssembly.                                                                                                                     |
| `ruby`          | `x.rb`    | none                                                                                                                                                            | `ruby {source}`                                                                                                             | This is a mapped-code cell, so pass what `run_noexec_dirs` returns.                                                                                                                                                                                                                                 |
| `julia`         | `x.jl`    | none                                                                                                                                                            | `<toolchains.julia_binary()> --startup-file=no --compiled-modules=existing -O3 {source}`                                    | Call the binary by absolute path. The one on `PATH` is a wrapper script, and the shim refuses the exec it needs.                                                                                                                                                                                    |
| `erlang_elixir` | `x.erl`   | `erlc {source}`                                                                                                                                                 | `toolchains.beam_argv("{build_dir}", "-noshell", "-pa", "{build_dir}", "-s", <module>, "main", "-s", "init", "stop", "--")` | erlc names the beam after the module, so pin the module name and the file name together. `erl` is a script that execs the emulator, so the run builds that argv itself.                                                                                                                             |
| `erlang_elixir` | `x.ex`    | `elixirc {source}` (env `ELIXIR_ERL_OPTIONS=+fnu`)                                                                                                              | `toolchains.elixir_argv("{build_dir}", "-pa", "{build_dir}", "-e", "<Module>.main()")`                                      | Elixir doesn't tie the module name to anything, so the task must pin it and tell the student. Needs `keep_built=("*.beam",)`, because there's one `.beam` per module.                                                                                                                               |
| `csharp`        | `x.cs`    | `dotnet publish {source} -o . -p:NuGetAudit=false`                                                                                                              | `{artifact}`                                                                                                                | Builds with Native AOT. The artifact is named after the source stem. It restores offline from the staged NuGet cache.                                                                                                                                                                               |
| `pascal`        | `x.pas`   | `fpc -O3 -o{artifact} {source}`                                                                                                                                 | `{artifact}`                                                                                                                | On x86_64, `uses cthreads` works, because the cell preloads `libpthread.so.0` and `libgcc_s.so.1` before the filter goes in. On aarch64, a program that uses `cthreads` fails to link, because FPC 3.2.2's `cprt0.o` needs symbols that glibc 2.34 removed. Programs without threads build on both. |
| `assembly`      | `x.s`     | `as -o x.o {source}`, then `ld -o {artifact} x.o`                                                                                                               | `{artifact}`                                                                                                                | Freestanding: no libc, `_start` instead of `main`, and raw syscalls. Tell the student.                                                                                                                                                                                                              |
| `llvm_ir`       | `x.ll`    | `llc -O3 -filetype=obj -o x.o {source}`, then `ld -o {artifact} x.o`                                                                                            | `{artifact}`                                                                                                                | Freestanding, like assembly.                                                                                                                                                                                                                                                                        |
| `fortran`       | `x.f90`   | `gfortran -O2 -o {artifact} {source}`                                                                                                                           | `{artifact}`                                                                                                                |                                                                                                                                                                                                                                                                                                     |
| `clojure`       | `x.clj`   | none                                                                                                                                                            | `toolchains.clojure_argv("{source}")`                                                                                       | It compiles at run time, so the runtime is what bounds this cell: the `jlink`ed JVM without `jdk.compiler`, plus `--illegal-native-access=deny`. Don't use the `clojure` launcher, which is a shell script that execs `java`.                                                                       |
| `cobol`         | `x.cob`   | `cobc -x -free -O2 -o {artifact} {source}` (env `COB_LIBS="<toolchains.COBOL_LIBCOB_ARCHIVE> -lgmp -lm"`)                                                       | `{artifact}`                                                                                                                | The env swaps the shared libcob for the static archive. Without it, the binary breaks at sealing. `-free` spares the student the fixed-form column rules.                                                                                                                                           |
| `prolog`        | `x.pl`    | `gplc -o {artifact} {source}`                                                                                                                                   | `{artifact}`                                                                                                                | GNU Prolog links its engine in statically. The program must end in `halt`, or the binary falls into the interactive top level.                                                                                                                                                                      |
| `mojo`          | `x.mojo`  | `mojo build -o {artifact} {source}`                                                                                                                             | `{artifact}`                                                                                                                | The artifact loads shared libraries from the toolchain directory, so this cell seals file by file and leaves the directory open. See the license note above.                                                                                                                                        |

`erlang_elixir` is one cell for two languages, because sealing can't separate them.
Accept both files, but refuse to grade if the student wrote both.
`chosen_submission` does this for you.

The student never runs these commands.
The `build` tool does.
Publish them in the instructions anyway, quoted exactly and with any env such as `CGO_ENABLED=0`.
That way the student knows how its file will be compiled.

## Verifying the image

`karotte check` runs during every image build.
Through the template's `toolchain_checks.py`, it asserts these sealing invariants:

- There's no compiler on `PATH`.
- `/opt/toolchains` can't be listed.
- The language directories are closed.
- `as` and `ld` are closed.
- The `builder` account exists, and the student's user isn't in its group.
- There's exactly one managed CPython.

It also asserts that the three shims and the probe were built, have mode 0555, and load and hold.
The `default` template's `check_permissions` runs the `check()` of every `environment.*_checks` module.
That's how it finds this one.

`just check-toolchains` builds and runs a hello-world program for each enabled language inside the built image:

```sh
just check-toolchains                 # image "karotte", every enabled language
just check-toolchains karotte rust    # one language
just check-toolchains karotte '' podman
```

For each language, it installs the toolchain and confirms that the builder can reach the build commands but the student can't.
It builds as the builder, runs the result as the student, and seals.
Then it confirms that the compilers are gone and the artifact still runs.
It also checks that a compile-time hook can't leave anything behind.
It runs the scored-run confinement for every cell except `python`.
And it runs known escapes, which must all be refused.
It uses one container per language, because installing a toolchain can't be undone.
Each container must be started with `CAP_SYS_ADMIN` and with AppArmor and seccomp unconfined, which the recipe passes.
The default AppArmor profile blocks `mount(2)` even with the capability.

## Caveats

- The build's mount namespace needs `CAP_SYS_ADMIN`, and the default AppArmor profile must not apply.
  Without it, the build fails instead of running unprotected.
  So a container started without the capability can't build.
- The confinement needs the run to seal the filesystem read-only.
  Without that, a submission can write an ELF file and load it, which needs neither a compiler nor FFI.
- The `build` tool limits how the student compiles, not what the language allows.
  Inline assembly, embedded bytes plus FFI, and arbitrary JVM bytecode can all go in a source file, and the tool compiles it faithfully.
  What limits them is [Enforcing the language](#enforcing-the-language) where the language has a flag that forbids them, and [Confining the scored run](#confining-the-scored-run) where it doesn't.
- Because gcc's C frontend is sealed, Haskell modules that use `{-# LANGUAGE CPP #-}` don't compile (GHC's CPP is `gcc -E`).
  `hsc2hs` doesn't work either.
- The assembly and LLVM IR checks in `scripts/check_toolchains.py` are written for x86_64, so run `just check-toolchains` for those cells on an x86_64 machine.
  The toolchain table itself pins both architectures.
- Sealing assumes the `default` template's permission model: the student runs as an unprivileged user, builds run as the `builder` user, and grading is root-owned.
  If you change that, you have to work out the guarantees again.
- Toolchain versions are pinned in `toolchains.py`, and `karotte update` updates them.
  If you edit the table locally, expect merge conflicts on the next update.
