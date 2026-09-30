import subprocess


def check():
    """Check that the sealed-toolchain model holds in the freshly built image.

    Named `check` because `environment.check_permissions` discovers and runs
    every `environment.*_checks` module's `check()` during `karotte check`.
    Anything reachable before a task's `pre_hook` opens a toolchain is a
    compiler the language choice does not control, so the image must hold none.
    """
    from karotte.trusted_bin import trusted_binary

    from environment.paths import STUDENT_WORKDIR
    from environment.toolchain_grading import (
        GRADER_BIN_DIR,
        JIT_SHIM,
        MAPPED_CODE_SHIM,
        SANDBOX_PROBE,
        SHIM,
    )
    from environment.toolchains import (
        BUILDER_UID,
        TOOLCHAINS_DIR,
        UV_PYTHON_DIR,
        Language,
        base_image_build_tools,
        student_python,
    )

    language_dirs = " ".join(
        str(TOOLCHAINS_DIR / language.value) for language in Language
    )
    # By path rather than by name: as root, `command -v as` finds the assembler
    # whatever mode it carries, and the mode is what this checks.
    build_tools = " ".join(str(tool) for tool in base_image_build_tools())
    # Raises if the image does not hold exactly one, which is the invariant a
    # Python run argv is built on.
    interpreter = student_python()

    test_script = f"""
set -e

echo "Running test: no compiler in the image"
for compiler in cc gcc g++ clang cargo rustc go ghc swiftc ocamlopt zig dart kotlinc javac; do
  if command -v "$compiler" > /dev/null 2>&1; then
    echo "FAIL: $compiler is on PATH in the base image ($(command -v $compiler))"
    exit 1
  fi
done
echo "PASS: no compiler in the image"

echo "Running test: {TOOLCHAINS_DIR} is not listable by the student"
toolchains_mode=$(stat -c '%a' {TOOLCHAINS_DIR})
if [ "$toolchains_mode" != "711" ]; then
  echo "FAIL: {TOOLCHAINS_DIR} mode is $toolchains_mode, expected 711"
  exit 1
fi
if runuser -u student -- ls {TOOLCHAINS_DIR}/ 2>/dev/null; then
  echo "FAIL: Student can list {TOOLCHAINS_DIR}"
  exit 1
fi
echo "PASS: {TOOLCHAINS_DIR} is not listable by the student"

echo "Running test: the builder account exists and the student is not in its group"
if [ "$(id -u builder)" != "{BUILDER_UID}" ]; then
  echo "FAIL: builder uid is $(id -u builder), expected {BUILDER_UID}"
  exit 1
fi
if id -Gn student | tr ' ' '\\n' | grep -qx builder; then
  echo "FAIL: student is in the builder group"
  exit 1
fi
echo "PASS: the builder account exists and the student is not in its group"

echo "Running test: every language directory is sealed"
for language_dir in {language_dirs}; do
  [ -d "$language_dir" ] || continue
  language_mode=$(stat -c '%a' "$language_dir")
  if [ "$language_mode" != "700" ]; then
    echo "FAIL: $language_dir mode is $language_mode, expected 700"
    exit 1
  fi
  if runuser -u student -- ls "$language_dir/" 2>/dev/null; then
    echo "FAIL: Student can reach $language_dir before any task opened it"
    exit 1
  fi
done
echo "PASS: every language directory is sealed"

# The base image's own assembler and linker. A cell that opens them is one whose
# compiler cannot build without them; the rest never see them, because a
# prebuilt object assembled during the episode runs with no compiler behind it.
echo "Running test: the assembler and linker are closed until a cell opens them"
if [ -z "{build_tools}" ]; then
  echo "FAIL: no base image build tools found to check"
  exit 1
fi
for tool in {build_tools}; do
  # -L because `ld` is a symlink to `ld.bfd`, and the mode that decides whether
  # it runs is the target's — which is also the one chmod lands on.
  tool_mode=$(stat -Lc '%a' "$tool")
  if [ "$tool_mode" != "700" ]; then
    echo "FAIL: $tool mode is $tool_mode, expected 700"
    exit 1
  fi
  if runuser -u student -- test -x "$tool"; then
    echo "FAIL: Student can run $tool before any task opened it"
    exit 1
  fi
done
echo "PASS: the assembler and linker are closed until a cell opens them"

# A Python-cell run argv uses {interpreter}, and `seal_toolchain` closes every
# other Python before student code runs unwatched. Both halves need this one to
# be root-owned, outside the workdir, and runnable by the student.
echo "Running test: the submission interpreter is outside the student workdir"
case "{interpreter}" in
  {STUDENT_WORKDIR}/*)
    echo "FAIL: {interpreter} is under {STUDENT_WORKDIR}, which the student owns"
    exit 1
    ;;
esac
echo "PASS: the submission interpreter is outside the student workdir"

echo "Running test: the submission interpreter is root-owned and student-runnable"
interpreter_owner=$(stat -c '%U' $(readlink -f {interpreter}))
if [ "$interpreter_owner" != "root" ]; then
  echo "FAIL: {interpreter} is owned by $interpreter_owner, expected root"
  exit 1
fi
if ! runuser -u student -- {interpreter} -c 'pass'; then
  echo "FAIL: student cannot run {interpreter}"
  exit 1
fi
echo "PASS: the submission interpreter is root-owned and student-runnable"

echo "Running test: exactly one managed CPython to choose between"
managed=$(readlink -f {UV_PYTHON_DIR}/cpython-*/bin/python3 2>/dev/null | sort -u | wc -l)
if [ "$managed" != "1" ]; then
  echo "FAIL: {UV_PYTHON_DIR} holds $managed cpython-*/bin/python3, expected 1"
  exit 1
fi
echo "PASS: exactly one managed CPython to choose between"

echo "Running test: the student venv's interpreter is not what runs a submission"
venv_python={STUDENT_WORKDIR}/.venv/bin/python
if [ "$(readlink -f $venv_python)" = "{interpreter}" ]; then
  echo "FAIL: {interpreter} is the student-owned venv interpreter"
  exit 1
fi
echo "PASS: the student venv's interpreter is not what runs a submission"

# Graders build submissions in random-named subdirectories of this, so the
# demoted build has to reach one without being able to list what else is there.
echo "Running test: {GRADER_BIN_DIR} is traversable but not listable"
grader_mode=$(stat -c '%a' {GRADER_BIN_DIR})
if [ "$grader_mode" != "711" ]; then
  echo "FAIL: {GRADER_BIN_DIR} mode is $grader_mode, expected 711"
  exit 1
fi
if runuser -u student -- ls {GRADER_BIN_DIR}/ 2>/dev/null; then
  echo "FAIL: Student can list {GRADER_BIN_DIR}"
  exit 1
fi
echo "PASS: {GRADER_BIN_DIR} is traversable but not listable"

# ld.so treats a preload it cannot open as a warning and carries on, so a shim
# that failed to build would leave every compiled run unconfined and exit 0.
echo "Running test: the compiled-run confinement is built and unwritable"
for artifact in {SHIM} {MAPPED_CODE_SHIM} {JIT_SHIM} {SANDBOX_PROBE}; do
  if [ ! -f "$artifact" ]; then
    echo "FAIL: $artifact is missing, so a compiled run would be unconfined"
    exit 1
  fi
  artifact_mode=$(stat -c '%a' "$artifact")
  if [ "$artifact_mode" != "555" ]; then
    echo "FAIL: $artifact mode is $artifact_mode, expected 555"
    exit 1
  fi
done
echo "PASS: the compiled-run confinement is built and unwritable"

# A subshell body, not a braced one: this file is a Jinja template before it is
# an f-string, and a brace means something to both.
confinement_holds() (
  shim="$1"
  probe_output=$(runuser -u student -- env LD_PRELOAD="$shim" {SANDBOX_PROBE} "$2" 2>&1 || true)
  if echo "$probe_output" | grep -q "cannot be preloaded"; then
    echo "FAIL: $shim does not load: $probe_output"
    exit 1
  fi
  # No mount namespace during the build, so the writable directories the run's
  # seal closes are expected here — and their absence means the probe never ran
  # to completion. Anything else the probe names is a hole.
  if ! echo "$probe_output" | grep -q "is writable"; then
    echo "FAIL: the probe did not report the build's writable /tmp, so it did not run: $probe_output"
    exit 1
  fi
  unexpected=$(echo "$probe_output" | grep -v "is writable" || true)
  if [ -n "$unexpected" ]; then
    echo "FAIL: $shim left something open:"
    echo "$unexpected"
    exit 1
  fi
)

# All three shims, each asked the questions its own rules answer: the strict
# one, the one a cell whose runtime maps its own code gets, and the one a cell
# whose runtime compiles as it runs gets.
echo "Running test: the compiled-run confinement loads and holds"
confinement_holds {SHIM} ""
confinement_holds {MAPPED_CODE_SHIM} --mapped-code
confinement_holds {JIT_SHIM} --jit
echo "PASS: the compiled-run confinement loads and holds"

echo "All toolchain permission tests passed!"
"""

    result = subprocess.run(
        [trusted_binary("bash"), "-c", test_script], capture_output=True, text=True
    )
    assert result.returncode == 0, (
        f"Toolchain permission tests failed with exit code {result.returncode}:\n"
        f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
    )
