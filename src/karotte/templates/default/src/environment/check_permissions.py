import subprocess


def check_permissions():
    """Checks that the permission model is correctly enforced in the container."""
    from karotte.trusted_bin import trusted_binary

    from environment.paths import (
        INTERMEDIATE_DATA_DIR,
        ROOT_DATA_DIR,
        SHARED_DATA_DIR,
        STUDENT_DATA_DIR,
    )

    student_workdir = SHARED_DATA_DIR.parent

    test_script = f"""
set -e

# ===== Student-writable tests ({STUDENT_DATA_DIR}) =====
# Test 1-2: Student user can read/write student data and access venv
echo "Running test 1-2: Student writable tests"
runuser -u student -- bash -c '
  echo "test" > {STUDENT_DATA_DIR}/test_write.txt
  cat {STUDENT_DATA_DIR}/test_write.txt | grep -q "test"
  rm {STUDENT_DATA_DIR}/test_write.txt
  ls -la /workdir/.venv/ > /dev/null
'
echo "PASS: Test 1-2"

# ===== Student-readable tests ({SHARED_DATA_DIR}) =====
# Test 3-4: Student user can read shared directory and files
echo "Running test 3-4: Student readable tests"
runuser -u student -- bash -c '
  ls -la {SHARED_DATA_DIR}/ > /dev/null
  if [ -f {SHARED_DATA_DIR}/test_permissions.txt ]; then
    cat {SHARED_DATA_DIR}/test_permissions.txt > /dev/null
  fi
'
echo "PASS: Test 3-4"

# Test 5: Student user CANNOT write to shared directory
echo "Running test 5: Cannot write to shared"
if runuser -u student -- touch {SHARED_DATA_DIR}/illegal.txt 2>/dev/null; then
  echo "FAIL: Test 5 - Student user was able to write to shared directory"
  exit 1
fi
echo "PASS: Test 5"

# Test 6: Student user CANNOT delete from shared directory
echo "Running test 6: Cannot delete from shared"
if runuser -u student -- rm {SHARED_DATA_DIR}/test_permissions.txt 2>/dev/null; then
  echo "FAIL: Test 6 - Student user was able to delete from shared directory"
  exit 1
fi
echo "PASS: Test 6"

# ===== Shared directory mode tests =====

echo "Running test: /workdir/shared has sticky bit (mode 1755)"
shared_mode=$(stat -c '%a' {SHARED_DATA_DIR})
if [ "$shared_mode" != "1755" ]; then
  echo "FAIL: {SHARED_DATA_DIR} mode is $shared_mode, expected 1755"
  exit 1
fi
echo "PASS: /workdir/shared has sticky bit (mode 1755)"

# `test_permissions.txt` is created by check_permissions() via Python at the
# default umask, bypassing COPY --chmod=444, so it's exempt from this check.
echo "Running test: All files under /workdir/shared have mode 0444"
bad_files=$(find {SHARED_DATA_DIR} -mindepth 1 -type f ! -name 'test_permissions.txt' ! -perm 0444)
if [ -n "$bad_files" ]; then
  echo "FAIL: files under {SHARED_DATA_DIR} with mode != 0444:"
  echo "$bad_files" | while read -r f; do echo "  $(stat -c '%a' "$f") $f"; done
  exit 1
fi
echo "PASS: All files under /workdir/shared have mode 0444"

echo "Running test: All subdirectories under /workdir/shared have mode 0555"
bad_dirs=$(find {SHARED_DATA_DIR} -mindepth 1 -type d ! -perm 0555)
if [ -n "$bad_dirs" ]; then
  echo "FAIL: subdirectories under {SHARED_DATA_DIR} with mode != 0555:"
  echo "$bad_dirs" | while read -r d; do echo "  $(stat -c '%a' "$d") $d"; done
  exit 1
fi
echo "PASS: All subdirectories under /workdir/shared have mode 0555"

# Write-denial probes that leave no state on success. The path arguments don't
# exist; the syscalls fail before anything is written.
echo "Running test: Student CANNOT create files in /workdir/shared"
if runuser -u student -- touch {SHARED_DATA_DIR}/.write_probe 2>/dev/null; then
  rm -f {SHARED_DATA_DIR}/.write_probe
  echo "FAIL: Student created a file in {SHARED_DATA_DIR}"
  exit 1
fi
echo "PASS: Student CANNOT create files in /workdir/shared"

echo "Running test: Student CANNOT create subdirectories in /workdir/shared"
if runuser -u student -- mkdir {SHARED_DATA_DIR}/.dir_probe 2>/dev/null; then
  rmdir {SHARED_DATA_DIR}/.dir_probe
  echo "FAIL: Student created a subdirectory in {SHARED_DATA_DIR}"
  exit 1
fi
echo "PASS: Student CANNOT create subdirectories in /workdir/shared"

# ===== Shared directory replacement tests =====
# The student must not be able to remove or replace the shared directory.
# On overlayfs (container builds/runtime), rename(2) fails with EXDEV for
# lower-layer dirs, so mv falls back to copy+delete. We test the actual
# attack vectors: rmdir on empty dir, and mv (which uses copy+rmdir).

# Test: Student CANNOT rmdir shared (empty case)
# This is the primary attack vector: rmdir empty shared, then mkdir a new one.
echo "Running test: Cannot rmdir empty shared directory"
shared_backup=$(mktemp -d)
mv {SHARED_DATA_DIR}/* "$shared_backup/" 2>/dev/null || true
if runuser -u student -- rmdir {SHARED_DATA_DIR} 2>/dev/null; then
  mkdir -p {SHARED_DATA_DIR}
  mv "$shared_backup"/* {SHARED_DATA_DIR}/ 2>/dev/null || true
  rm -rf "$shared_backup"
  echo "FAIL: Student was able to rmdir empty shared directory"
  exit 1
fi
mv "$shared_backup"/* {SHARED_DATA_DIR}/ 2>/dev/null || true
rm -rf "$shared_backup"
echo "PASS: Cannot rmdir empty shared directory"

# Test: Student CANNOT rmdir shared (non-empty case)
echo "Running test: Cannot rmdir non-empty shared directory"
if runuser -u student -- rmdir {SHARED_DATA_DIR} 2>/dev/null; then
  mkdir -p {SHARED_DATA_DIR}
  echo "FAIL: Student was able to rmdir non-empty shared directory"
  exit 1
fi
echo "PASS: Cannot rmdir non-empty shared directory"

# Test: Student CANNOT mv (replace) shared directory
# mv falls back to copy+rmdir on overlayfs; on regular filesystems it uses
# rename(2). Either way, the student must not be able to replace shared.
echo "Running test: Cannot mv shared directory"
if runuser -u student -- mv {SHARED_DATA_DIR} {student_workdir}/_shared_test_mv 2>/dev/null; then
  # Restore: move it back and clean up
  mv {student_workdir}/_shared_test_mv {SHARED_DATA_DIR} 2>/dev/null || true
  rm -rf {student_workdir}/_shared_test_mv 2>/dev/null || true
  echo "FAIL: Student was able to mv shared directory"
  exit 1
fi
rm -rf {student_workdir}/_shared_test_mv 2>/dev/null || true
echo "PASS: Cannot mv shared directory"

# ===== Student workdir sanity tests =====
# These ensure normal student operations in /workdir/ still work

# Test: Student can create files in workdir
echo "Running test: Student can create files in workdir"
runuser -u student -- bash -c '
  echo "hello" > {student_workdir}/test_student_file.txt
  cat {student_workdir}/test_student_file.txt | grep -q "hello"
'
echo "PASS: Student can create files in workdir"

# Test: Student can create directories in workdir
echo "Running test: Student can create directories in workdir"
runuser -u student -- mkdir {student_workdir}/test_student_dir
echo "PASS: Student can create directories in workdir"

# Test: Student-created items are owned by student
echo "Running test: Student-created items are owned by student"
file_owner=$(stat -c '%U' {student_workdir}/test_student_file.txt)
dir_owner=$(stat -c '%U' {student_workdir}/test_student_dir)
if [ "$file_owner" != "student" ] || [ "$dir_owner" != "student" ]; then
  echo "FAIL: Student-created items not owned by student (file=$file_owner, dir=$dir_owner)"
  rm -f {student_workdir}/test_student_file.txt
  rm -rf {student_workdir}/test_student_dir
  exit 1
fi
echo "PASS: Student-created items are owned by student"

# Test: Student can delete their own files and directories in workdir
echo "Running test: Student can delete own items in workdir"
runuser -u student -- rm {student_workdir}/test_student_file.txt
runuser -u student -- rmdir {student_workdir}/test_student_dir
echo "PASS: Student can delete own items in workdir"

# ===== Writable-surface invariant =====
# Everything the student can write, or owns and could chmod, has to live under
# the workdir or a temp directory.
echo "Running test: Student can only write under the workdir and temp dirs"
writable_surface=$(runuser -u student -- find / \\
  \\( -path /proc -o -path /sys -o -path /dev \\) -prune -o \\
  \\( -writable -o -user student \\) \\( -type f -o -type d \\) -print 2>/dev/null \\
  | grep -vE '^({student_workdir}|/tmp|/var/tmp|/usr/tmp)(/|$)' || true)
if [ -n "$writable_surface" ]; then
  echo "FAIL: student-writable paths outside {student_workdir} and the temp dirs:"
  echo "$writable_surface" | while read -r p; do
    echo "  $(stat -c '%a %U:%G' "$p") $p"
  done
  exit 1
fi
echo "PASS: Student can only write under the workdir and temp dirs"

# The scan above runs as the student, so it stops wherever the student cannot
# traverse: a world-writable file inside a 0700 directory is invisible to it.
# Repeat it as root so a hole held shut only by its parent's mode still fails.
echo "Running test: Nothing outside the workdir and temp dirs is world-writable"
world_writable=$(find / \\
  \\( -path /proc -o -path /sys -o -path /dev \\) -prune -o \\
  -perm -002 \\( -type f -o -type d \\) -print 2>/dev/null \\
  | grep -vE '^({student_workdir}|/tmp|/var/tmp|/usr/tmp)(/|$)' || true)
if [ -n "$world_writable" ]; then
  echo "FAIL: world-writable paths outside {student_workdir} and the temp dirs:"
  echo "$world_writable" | while read -r p; do
    echo "  $(stat -c '%a %U:%G' "$p") $p"
  done
  exit 1
fi
echo "PASS: Nothing outside the workdir and temp dirs is world-writable"

# ===== No access tests (root data) =====
echo "Running test 7-10: Root access tests"
ls -la /root/ > /dev/null
ls -la /root/.venv/ > /dev/null
ls -la {ROOT_DATA_DIR}/ > /dev/null
cat {ROOT_DATA_DIR}/test_permissions.txt | grep -q 'This is secret root data'
echo "PASS: Test 7-10"

echo "Running test 11: Cannot access /root"
if runuser -u student -- ls -la /root/ 2>/dev/null; then
  echo "FAIL: Test 11 - Student user can access /root"
  exit 1
fi
echo "PASS: Test 11"

echo "Running test 12: Cannot access /root/.venv"
if runuser -u student -- ls -la /root/.venv/ 2>/dev/null; then
  echo "FAIL: Test 12 - Student user can access /root/.venv"
  exit 1
fi
echo "PASS: Test 12"

echo "Running test 13: Cannot access {ROOT_DATA_DIR}"
if runuser -u student -- ls -la {ROOT_DATA_DIR}/ 2>/dev/null; then
  echo "FAIL: Test 13 - Student user can access {ROOT_DATA_DIR}"
  exit 1
fi
echo "PASS: Test 13"

echo "Running test 14: Cannot read root data files"
if runuser -u student -- cat {ROOT_DATA_DIR}/test_permissions.txt 2>/dev/null; then
  echo "FAIL: Test 14 - Student user can read root data files"
  exit 1
fi
echo "PASS: Test 14"

# ===== No access tests (intermediate data) =====
echo "Running test 15-16: Root intermediate data access"
ls -la {INTERMEDIATE_DATA_DIR}/ > /dev/null
cat {INTERMEDIATE_DATA_DIR}/test_permissions.txt | grep -q 'This is intermediate data'
echo "PASS: Test 15-16"

echo "Running test 17: Cannot access {INTERMEDIATE_DATA_DIR}"
if runuser -u student -- ls -la {INTERMEDIATE_DATA_DIR}/ 2>/dev/null; then
  echo "FAIL: Test 17 - Student user can access {INTERMEDIATE_DATA_DIR}"
  exit 1
fi
echo "PASS: Test 17"

echo "Running test 18: Cannot read intermediate data files"
if runuser -u student -- cat {INTERMEDIATE_DATA_DIR}/test_permissions.txt 2>/dev/null; then
  echo "FAIL: Test 18 - Student user can read intermediate data files"
  exit 1
fi
echo "PASS: Test 18"

# ===== Environment variable and home directory tests =====
# KAROTTE_WORKDIR must be set and point to the student workdir
echo "Running test: KAROTTE_WORKDIR is set correctly"
if [ "$KAROTTE_WORKDIR" != "{student_workdir}" ]; then
  echo "FAIL: KAROTTE_WORKDIR='$KAROTTE_WORKDIR', expected '{student_workdir}'"
  exit 1
fi
echo "PASS: KAROTTE_WORKDIR is set correctly"

# Student user's home in /etc/passwd must point to the student workdir
echo "Running test: Student passwd home is set correctly"
passwd_home=$(getent passwd student | cut -d: -f6)
if [ "$passwd_home" != "{student_workdir}" ]; then
  echo "FAIL: Student passwd home='$passwd_home', expected '{student_workdir}'"
  exit 1
fi
echo "PASS: Student passwd home is set correctly"

# Student's HOME env var (via runuser) must match
echo "Running test: Student HOME env var is correct"
student_home=$(runuser -u student -- bash -c 'echo $HOME')
if [ "$student_home" != "{student_workdir}" ]; then
  echo "FAIL: Student HOME='$student_home', expected '{student_workdir}'"
  exit 1
fi
echo "PASS: Student HOME env var is correct"

echo "All permission tests passed!"
"""

    # Create test files
    root_test_file = ROOT_DATA_DIR / "test_permissions.txt"
    intermediate_test_file = INTERMEDIATE_DATA_DIR / "test_permissions.txt"
    shared_test_file = SHARED_DATA_DIR / "test_permissions.txt"

    root_test_file.write_text("This is secret root data")
    intermediate_test_file.write_text("This is intermediate data")
    shared_test_file.write_text("This is shared data")

    try:
        result = subprocess.run(
            [trusted_binary("bash"), "-c", test_script],
            capture_output=True,
            text=True,
        )
    finally:
        # Clean up test files
        root_test_file.unlink(missing_ok=True)
        intermediate_test_file.unlink(missing_ok=True)
        shared_test_file.unlink(missing_ok=True)

    assert result.returncode == 0, (
        f"Permission tests failed with exit code {result.returncode}:\n"
        f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
    )

    # Add-on templates ship extra image checks as `*_checks` modules exposing a
    # `check()`.
    import importlib
    import pkgutil

    import environment

    for info in pkgutil.iter_modules(environment.__path__):
        if info.name.endswith("_checks"):
            importlib.import_module(f"environment.{info.name}").check()
