import os
import subprocess


def check():
    """Check that the managed CPython holds what the Containerfile gave it.

    Named `check` because `environment.check_permissions` discovers and runs
    every `environment.*_checks` module's `check()` during `karotte check`.
    """
    from karotte.trusted_bin import trusted_binary

    python_glob = f"{os.environ['UV_PYTHON_INSTALL_DIR']}/cpython-*/bin/python3"

    test_script = f"""
set -e

interpreters=$(readlink -f {python_glob} 2>/dev/null | sort -u)
if [ -z "$interpreters" ]; then
  echo "FAIL: no managed CPython found at {python_glob}"
  exit 1
fi

# glibc reads a missing PT_GNU_STACK header as a request for executable
# thread stacks: every thread gets an 8MB rwx mapping. The Containerfile
# clears the header with scripts/clear_execstack.py.
echo "Running test: the managed CPython asks for a non-executable stack"
for interpreter in $interpreters; do
  stack_header=$(readelf -lW "$interpreter" | grep GNU_STACK || true)
  case "$stack_header" in
    "") echo "FAIL: $interpreter has no PT_GNU_STACK header"; exit 1 ;;
    *E*) echo "FAIL: $interpreter asks for an executable stack: $stack_header"; exit 1 ;;
  esac
done
echo "PASS: the managed CPython asks for a non-executable stack"

echo "Running test: the managed CPython cannot be written over"
for interpreter in $interpreters; do
  if [ -n "$(find "$interpreter" -perm /222)" ]; then
    echo "FAIL: $interpreter mode is $(stat -c '%a' "$interpreter"), which is writable"
    exit 1
  fi
done
echo "PASS: the managed CPython cannot be written over"

echo "All interpreter tests passed!"
"""

    result = subprocess.run(
        [trusted_binary("bash"), "-c", test_script], capture_output=True, text=True
    )
    assert result.returncode == 0, (
        f"Interpreter tests failed with exit code {result.returncode}:\n"
        f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
    )
