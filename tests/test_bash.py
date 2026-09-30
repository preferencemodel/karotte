import asyncio
import json
import os
import pwd
import signal
import time
import uuid
from pathlib import Path
from textwrap import dedent

import psutil
import pytest
import pytest_asyncio
from fastmcp import Client, FastMCP
from fastmcp.client.transports.memory import FastMCPTransport
from fastmcp.exceptions import ToolError
from fastmcp.tools.tool import ToolResult
from mcp.types import TextContent

from karotte.tools.bash import (
    _MARKER_FD,  # pyright: ignore[reportPrivateUsage]
    BashConfig,
    _BashSession,  # pyright: ignore[reportPrivateUsage]
    _Marker,  # pyright: ignore[reportPrivateUsage]
    bash,
)
from tests.conftest import register_harness_secrets


def _parse_result(result: ToolResult) -> dict[str, str | int]:
    assert isinstance(result.content[0], TextContent)
    return json.loads(result.content[0].text)


@pytest.mark.asyncio
async def test_echo(bash_tool: bash):
    result = await bash_tool(command="echo hello")

    assert isinstance(result.content[0], TextContent)
    assert json.loads(result.content[0].text)["stdout"] == "hello\n"


@pytest.mark.asyncio
async def test_empty_output(bash_tool: bash):
    result = await bash_tool(command="true")

    assert isinstance(result.content[0], TextContent)
    assert json.loads(result.content[0].text)["stdout"] == ""


@pytest.mark.asyncio
async def test_error_output(bash_tool: bash):
    result = await bash_tool(command="echo 'error message' >&2")

    assert isinstance(result.content[0], TextContent)
    assert json.loads(result.content[0].text)["stdout"] == ""
    assert json.loads(result.content[0].text)["stderr"] == "error message\n"


@pytest.mark.asyncio
async def test_shell_arithmetic(bash_tool: bash):
    result = await bash_tool(command="echo $((1 + 2))")

    assert isinstance(result.content[0], TextContent)
    assert json.loads(result.content[0].text)["stdout"] == "3\n"


@pytest.mark.asyncio
async def test_persistent_variables(bash_tool: bash):
    await bash_tool(command="TEST_VAR='hello'")
    result = await bash_tool(command="echo $TEST_VAR")

    assert isinstance(result.content[0], TextContent)
    assert json.loads(result.content[0].text)["stdout"] == "hello\n"


@pytest.mark.asyncio
async def test_persistent_workdir(bash_tool: bash):
    result = await bash_tool(command="pwd")

    assert isinstance(result.content[0], TextContent)
    assert json.loads(result.content[0].text)["stdout"] != "/tmp\n"

    await bash_tool(command="cd /tmp")
    result = await bash_tool(command="pwd")

    assert isinstance(result.content[0], TextContent)
    assert json.loads(result.content[0].text)["stdout"] == "/tmp\n"


@pytest.mark.asyncio
async def test_multiline_output(bash_tool: bash):
    result = await bash_tool(command="printf 'line1\\nline2\\nline3'")

    assert isinstance(result.content[0], TextContent)
    assert json.loads(result.content[0].text)["stdout"] == "line1\nline2\nline3"


@pytest.mark.asyncio
async def test_piped_command(bash_tool: bash):
    result = await bash_tool(command="echo -e 'apple\\nbanana\\ncherry' | grep banana")

    assert isinstance(result.content[0], TextContent)
    assert json.loads(result.content[0].text)["stdout"] == "banana\n"


@pytest.mark.asyncio
async def test_background_process(bash_tool: bash):
    start_time = time.time()
    await bash_tool(command="sleep 1 &")

    # Should return without waiting for the process to finish
    assert time.time() - start_time < 1


@pytest.mark.asyncio
async def test_raises_error_if_no_command_provided(bash_tool: bash):
    with pytest.raises(ValueError, match="No command provided."):
        await bash_tool()


@pytest.mark.asyncio
async def test_raises_error_if_command_empty(bash_tool: bash):
    with pytest.raises(ValueError, match="No command provided."):
        await bash_tool(command="")


@pytest.mark.asyncio
async def test_manual_restart(bash_tool: bash):
    await bash_tool(command="export TEST_VAR='before_restart'")

    result = await bash_tool(restart=True)

    assert isinstance(result.content[0], TextContent)
    assert (
        json.loads(result.content[0].text)["system"]
        == "Tool has been manually restarted."
    )

    # Variable should be gone after restart
    result = await bash_tool(command="echo ${TEST_VAR:-empty}")

    assert isinstance(result.content[0], TextContent)
    assert json.loads(result.content[0].text)["stdout"] == "empty\n"


@pytest.mark.asyncio
async def test_command_gets_ignored_when_restarting(bash_tool: bash):
    result = await bash_tool(command="echo 'hello'", restart=True)

    assert isinstance(result.content[0], TextContent)
    assert (
        json.loads(result.content[0].text)["system"]
        == "Tool has been manually restarted."
    )
    assert json.loads(result.content[0].text)["stdout"] == ""


@pytest.mark.asyncio
async def test_command_with_nonzero_exit_code(bash_tool: bash):
    # Should not crash
    await bash_tool(command="false")


@pytest.mark.asyncio
async def test_command_with_special_characters(bash_tool: bash):
    result = await bash_tool(command="echo '!@#$%^&*()'")

    assert isinstance(result.content[0], TextContent)
    assert json.loads(result.content[0].text)["stdout"] == "!@#$%^&*()\n"


@pytest.mark.asyncio
async def test_file_operations(bash_tool: bash):
    file_content = str(uuid.uuid4())

    await bash_tool(command=f"echo -n {file_content} > /tmp/{file_content}")

    result = await bash_tool(command=f"cat /tmp/{file_content}")

    assert isinstance(result.content[0], TextContent)
    assert json.loads(result.content[0].text)["stdout"] == file_content


@pytest.mark.asyncio
async def test_env_var_operations(bash_tool: bash):
    await bash_tool(command="export TEST_VAR='hello'")

    result = await bash_tool(command="printenv TEST_VAR")

    assert isinstance(result.content[0], TextContent)
    assert json.loads(result.content[0].text)["stdout"] == "hello\n"


@pytest.mark.asyncio
async def test_student_shell_does_not_inherit_harness_secrets(
    bash_tool: bash, monkeypatch: pytest.MonkeyPatch
):
    """The session is built from this process's environment; the harness's
    credentials must not come along, or a plain `env` puts them in the
    transcript."""
    register_harness_secrets(monkeypatch, internal=("BACKEND_TOKEN",))
    monkeypatch.setenv("BACKEND_TOKEN", "eyJ-not-for-students")
    monkeypatch.setenv("KAROTTE_WORKDIR_MARKER", "visible")

    result = await bash_tool(
        command='echo "${BACKEND_TOKEN:-withheld} ${KAROTTE_WORKDIR_MARKER:-missing}"'
    )

    assert isinstance(result.content[0], TextContent)
    assert json.loads(result.content[0].text)["stdout"] == "withheld visible\n"


@pytest.mark.asyncio
async def test_shell_functions(bash_tool: bash):
    await bash_tool(command="my_func() { echo 'function called'; }")

    result = await bash_tool(command="my_func")

    assert isinstance(result.content[0], TextContent)
    assert json.loads(result.content[0].text)["stdout"] == "function called\n"


@pytest.mark.asyncio
async def test_command_substitution(bash_tool: bash):
    result = await bash_tool(command="echo $(echo hello)")

    assert isinstance(result.content[0], TextContent)
    assert json.loads(result.content[0].text)["stdout"] == "hello\n"


@pytest.mark.asyncio
async def test_for_loop(bash_tool: bash):
    result = await bash_tool(command="for i in 1 2 3; do echo $i; done")

    assert isinstance(result.content[0], TextContent)
    assert json.loads(result.content[0].text)["stdout"] == "1\n2\n3\n"


@pytest.mark.asyncio
async def test_nonexistent_command(bash_tool: bash):
    result = await bash_tool(command="nonexitent")

    assert isinstance(result.content[0], TextContent)
    assert json.loads(result.content[0].text)["stdout"] == ""
    assert json.loads(result.content[0].text)["stderr"].endswith("command not found\n")


@pytest.mark.asyncio
async def test_recovery_from_process_death(bash_tool: bash):
    await bash_tool(command="echo hello")  # Starts the session

    assert bash_tool._session  # pyright: ignore[reportPrivateUsage]
    await bash_tool._session.stop()  # pyright: ignore[reportPrivateUsage]

    result = await bash_tool(command="echo hello")

    assert isinstance(result.content[0], TextContent)
    assert json.loads(result.content[0].text)["stdout"] == "hello\n"
    assert (
        json.loads(result.content[0].text)["system"]
        == "Session was automatically restarted."
    )


@pytest.mark.asyncio
async def test_recovery_from_timeout(bash_tool: bash):
    result = await bash_tool(command="sleep 1", timeout_s=0.01)

    parsed = _parse_result(result)
    assert parsed["stdout"] == ""
    assert "[Interrupted due to timeout after 0.01s]" in str(parsed["stderr"])

    result = await bash_tool(command="echo hello")
    assert isinstance(result.content[0], TextContent)
    assert json.loads(result.content[0].text)["stdout"] == "hello\n"


@pytest.mark.asyncio
async def test_stdout_doesnt_get_stripped(bash_tool: bash, tmp_path: Path):
    file_path = tmp_path / "test_file.txt"

    file_path.write_text("test\n")

    result = await bash_tool(command=f"cat {file_path}")

    assert isinstance(result.content[0], TextContent)
    assert json.loads(result.content[0].text)["stdout"] == "test\n"


@pytest.mark.asyncio
async def test_dispose(bash_tool: bash):
    await bash_tool.dispose()

    assert bash_tool._session is None  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_multiple_commands_at_once(bash_tool: bash):
    # Wouldn't work without locking because both commands would read/write
    # to/from stdin/stdout concurrently
    await asyncio.gather(
        bash_tool(command="which python3"),
        bash_tool(command="python3 --version"),
    )


@pytest.mark.asyncio
async def test_heredoc_command_does_not_hang(bash_tool: bash, tmp_path: Path):
    await bash_tool(
        command=dedent(f"""
            cat > {tmp_path / "test_file.txt"} << 'EOF'
            blablabla
            EOF
            """)
    )


@pytest.mark.asyncio
async def test_stdout_gets_truncated(bash_tool: bash):
    max_output_length = BashConfig().max_output_length

    super_long_string = "a" * (max_output_length + 1)
    result = await bash_tool(command=f"echo {super_long_string}")

    assert isinstance(result.content[0], TextContent)

    stdout = json.loads(result.content[0].text)["stdout"]
    system = json.loads(result.content[0].text)["system"]

    assert stdout.endswith("...")
    assert len(stdout.removesuffix("...")) == max_output_length
    assert "stdout was truncated" in system


@pytest.mark.asyncio
async def test_stderr_gets_truncated(bash_tool: bash):
    max_output_length = BashConfig().max_output_length

    super_long_string = "a" * (max_output_length + 1)
    result = await bash_tool(command=f"echo {super_long_string} >&2")

    assert isinstance(result.content[0], TextContent)

    stderr = json.loads(result.content[0].text)["stderr"]
    system = json.loads(result.content[0].text)["system"]

    assert stderr.endswith("...")
    assert len(stderr.removesuffix("...")) == max_output_length
    assert "stderr was truncated" in system


# ---------------------------------------------------------------------------
# Exit code propagation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_exit_code_zero_on_success(bash_tool: bash):
    result = await bash_tool(command="true")
    assert _parse_result(result)["exit_code"] == 0


@pytest.mark.asyncio
async def test_exit_code_one_on_failure(bash_tool: bash):
    result = await bash_tool(command="false")
    assert _parse_result(result)["exit_code"] == 1


@pytest.mark.asyncio
async def test_exit_code_2_on_misuse(bash_tool: bash):
    result = await bash_tool(command="exit 2")
    assert _parse_result(result)["exit_code"] == 2


@pytest.mark.asyncio
async def test_exit_code_137_on_sigkill(bash_tool: bash):
    result = await bash_tool(command="bash -c 'kill -9 $$'")
    assert _parse_result(result)["exit_code"] == 137


@pytest.mark.asyncio
async def test_exit_code_absent_on_restart(bash_tool: bash):
    result = await bash_tool(restart=True)
    assert "exit_code" not in _parse_result(result)


@pytest.mark.asyncio
async def test_exit_code_persists_across_commands(bash_tool: bash):
    result = await bash_tool(command="false")
    assert _parse_result(result)["exit_code"] == 1

    result = await bash_tool(command="true")
    assert _parse_result(result)["exit_code"] == 0


# ---------------------------------------------------------------------------
# Dead process detection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_early_detection_of_bash_process_death(bash_tool: bash):
    """When the bash shell is killed mid-command, the tool should detect it
    quickly and return instead of blocking until the full timeout."""
    await bash_tool(command="echo warmup")

    assert bash_tool._session is not None  # pyright: ignore[reportPrivateUsage]
    assert bash_tool._session._process is not None  # pyright: ignore[reportPrivateUsage]

    bash_pid = bash_tool._session._process.pid  # pyright: ignore[reportPrivateUsage]

    async def kill_bash_after_delay():
        await asyncio.sleep(0.5)
        try:
            os.kill(bash_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    timeout = 30.0
    kill_task = asyncio.create_task(kill_bash_after_delay())

    start = time.monotonic()
    result = await bash_tool(command="sleep 60", timeout_s=timeout)
    elapsed = time.monotonic() - start

    await kill_task

    assert elapsed < 5.0, (
        f"Tool took {elapsed:.1f}s to return after bash was killed. "
        f"Expected <5s, not the full {timeout}s timeout."
    )

    assert "Process died" in str(_parse_result(result).get("stderr", ""))


@pytest.mark.asyncio
async def test_recovery_after_process_death(bash_tool: bash):
    """After the bash process dies, the next command should work via auto-restart."""
    await bash_tool(command="echo warmup")
    assert bash_tool._session is not None  # pyright: ignore[reportPrivateUsage]
    assert bash_tool._session._process is not None  # pyright: ignore[reportPrivateUsage]

    bash_pid = bash_tool._session._process.pid  # pyright: ignore[reportPrivateUsage]
    os.kill(bash_pid, signal.SIGKILL)
    await asyncio.sleep(0.2)

    result = await bash_tool(command="echo recovered")
    assert _parse_result(result)["stdout"] == "recovered\n"


# ---------------------------------------------------------------------------
# SIGKILL distinguishable from normal exit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sigkill_distinguishable_from_normal_exit(bash_tool: bash):
    parsed_normal = _parse_result(await bash_tool(command="true"))
    parsed_killed = _parse_result(await bash_tool(command="bash -c 'kill -9 $$'"))

    assert parsed_normal != parsed_killed
    assert parsed_killed.get("exit_code") == 137


# ---------------------------------------------------------------------------
# _parse_marker unit tests
# ---------------------------------------------------------------------------


def _parse(line: str, nonce: str = "n0") -> _Marker | None:
    session = _BashSession(
        max_output_length=1000, disable_networking=True, python_venv=None
    )
    session._marker_nonce = nonce  # pyright: ignore[reportPrivateUsage]
    return session._parse_marker(line)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize("code", [0, 1, 137])
def test_parse_marker_exit_codes(code: int):
    marker = _parse(f"<<exit>> n0 {code}")
    assert marker is not None
    assert marker.exit_code == code


def test_parse_marker_captures_shell_options():
    marker = _parse("<<exit>> n0 0 extglob:globstar braceexpand:xtrace")
    assert marker is not None
    assert marker.bashopts == "extglob:globstar"
    assert marker.shellopts == "braceexpand:xtrace"


def test_parse_marker_options_are_optional():
    marker = _parse("<<exit>> n0 3")
    assert marker is not None
    assert marker.bashopts is None
    assert marker.shellopts is None


@pytest.mark.parametrize(
    "line",
    [
        "<<exit>>",
        "<<exit>> n0",
        "<<exit>> n0 abc",
        "<<exit>> other 0",
        "<<exit>> 0",
    ],
)
def test_parse_marker_rejects(line: str):
    assert _parse(line) is None


def test_parse_marker_rejects_everything_before_a_command_is_sent():
    session = _BashSession(
        max_output_length=1000, disable_networking=True, python_venv=None
    )
    assert session._parse_marker("<<exit>>  0") is None  # pyright: ignore[reportPrivateUsage]


# ---------------------------------------------------------------------------
# Timeout preserves partial output
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_timeout_preserves_partial_stdout(bash_tool: bash):
    result = await bash_tool(
        command="echo partial_before_timeout; sleep 60",
        timeout_s=1.0,
    )
    parsed = _parse_result(result)
    assert "partial_before_timeout" in str(parsed["stdout"])
    assert "timeout" in str(parsed.get("stderr", "")).lower()


@pytest.mark.asyncio
async def test_timeout_preserves_partial_stderr(bash_tool: bash):
    result = await bash_tool(
        command="echo partial_err >&2; sleep 60",
        timeout_s=1.0,
    )
    parsed = _parse_result(result)
    assert "partial_err" in str(parsed.get("stderr", ""))


# ---------------------------------------------------------------------------
# Timeout kills only the hung foreground command
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_timeout_spares_background_jobs_from_earlier_calls(bash_tool: bash):
    result = await bash_tool(command="nohup sleep 60 >/dev/null 2>&1 & echo $!")
    bg_pid = int(str(_parse_result(result)["stdout"]).strip())

    try:
        result = await bash_tool(command="sleep 60", timeout_s=1.0)
        parsed = _parse_result(result)
        assert "Interrupted due to timeout" in str(parsed.get("stderr", ""))

        os.kill(bg_pid, 0)
    finally:
        try:
            os.kill(bg_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


@pytest.mark.asyncio
async def test_timeout_preserves_session_state(bash_tool: bash):
    await bash_tool(command="TEST_VAR=survives; cd /tmp")

    result = await bash_tool(command="sleep 60", timeout_s=1.0)
    parsed = _parse_result(result)
    assert "Interrupted due to timeout" in str(parsed.get("stderr", ""))
    assert "restarted" not in str(parsed.get("system", ""))

    result = await bash_tool(command="echo $TEST_VAR; pwd")
    assert _parse_result(result)["stdout"] == "survives\n/tmp\n"


@pytest.mark.asyncio
async def test_timeout_reports_the_killed_commands_exit_code(bash_tool: bash):
    result = await bash_tool(command="sleep 60", timeout_s=1.0)
    assert _parse_result(result).get("exit_code") in (143, 137)


@pytest.mark.asyncio
async def test_timeout_falls_back_to_restart_when_shell_itself_hangs(
    bash_tool: bash, monkeypatch: pytest.MonkeyPatch
):
    """A hang inside the shell (a loop respawning children) has no foreground
    process group to kill, so the old full-restart path must still trigger."""
    monkeypatch.setattr(_BashSession, "_command_kill_timeout_s", 0.5)

    result = await bash_tool(command="while true; do sleep 0.2; done", timeout_s=1.0)
    parsed = _parse_result(result)
    assert "Interrupted due to timeout" in str(parsed.get("stderr", ""))
    assert parsed.get("system") == "Session was automatically restarted."

    result = await bash_tool(command="echo recovered")
    assert _parse_result(result)["stdout"] == "recovered\n"


async def _wait_for_death(pid: int, timeout: float = 5.0) -> bool:
    """True once pid is gone or a zombie awaiting reaping."""
    for _ in range(int(timeout / 0.05)):
        try:
            if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                return True
        except psutil.NoSuchProcess:
            return True
        await asyncio.sleep(0.05)
    return False


@pytest.mark.asyncio
async def test_manual_restart_kills_background_jobs(bash_tool: bash):
    result = await bash_tool(command="nohup sleep 60 >/dev/null 2>&1 & echo $!")
    bg_pid = int(str(_parse_result(result)["stdout"]).strip())

    try:
        await bash_tool(restart=True)
        assert await _wait_for_death(bg_pid)
    finally:
        try:
            os.kill(bg_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


@pytest.mark.asyncio
async def test_timeout_fallback_restart_spares_wrapped_background_jobs(
    bash_tool: bash, monkeypatch: pytest.MonkeyPatch
):
    """The full-restart fallback must spare a job the student backgrounded in an
    earlier call."""
    monkeypatch.setattr(_BashSession, "_command_kill_timeout_s", 0.5)

    duration = _unique_sleep_duration()
    await bash_tool(command=f"nohup sleep {duration} >/dev/null 2>&1 &")
    bg_pid = await _find_sleep_pid(duration)

    try:
        result = await bash_tool(
            command="while true; do sleep 0.2; done", timeout_s=1.0
        )
        parsed = _parse_result(result)
        assert parsed.get("system") == "Session was automatically restarted."

        assert not await _wait_for_death(bg_pid, timeout=1.0)

        result = await bash_tool(command="echo recovered")
        assert _parse_result(result)["stdout"] == "recovered\n"
    finally:
        try:
            os.kill(bg_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


@pytest.mark.asyncio
async def test_spared_pgids_can_never_wedge_the_session(
    bash_tool: bash, monkeypatch: pytest.MonkeyPatch
):
    """A spare set can never keep the shell alive; the direct killpg on the
    session leader always wins."""
    await bash_tool(command="echo warmup")
    session = bash_tool._session  # pyright: ignore[reportPrivateUsage]
    assert session is not None
    monkeypatch.setattr(_BashSession, "_command_kill_timeout_s", 0.5)
    process = session._process  # pyright: ignore[reportPrivateUsage]
    assert process is not None
    session._protected_pgids = {  # pyright: ignore[reportPrivateUsage]
        os.getpgid(process.pid),
        os.getpgid(0),
        1,
    }

    result = await bash_tool(command="while true; do sleep 0.2; done", timeout_s=1.0)
    assert "Interrupted due to timeout" in str(_parse_result(result).get("stderr", ""))

    result = await bash_tool(command="echo still_works")
    assert _parse_result(result)["stdout"] == "still_works\n"


def _unique_sleep_duration() -> str:
    return str(10**9 + uuid.uuid4().int % 10**9)


async def _find_sleep_pid(duration: str, timeout: float = 5.0) -> int:
    for _ in range(int(timeout / 0.05)):
        for proc in psutil.process_iter(["cmdline"]):
            if proc.info["cmdline"] == ["sleep", duration]:
                return proc.pid
        await asyncio.sleep(0.05)
    raise AssertionError(f"sleep {duration} never appeared")


@pytest.mark.asyncio
async def test_manual_restart_kills_wrapped_background_jobs(bash_tool: bash):
    """A command ending in `&` is wrapped in a subshell, so its background job
    reparents to init and is no longer a child of the shell."""
    duration = _unique_sleep_duration()
    await bash_tool(command=f"sleep {duration} >/dev/null 2>&1 &")
    bg_pid = await _find_sleep_pid(duration)

    try:
        await bash_tool(restart=True)
        assert await _wait_for_death(bg_pid)
    finally:
        try:
            os.kill(bg_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


async def _run(bash_tool: bash, *parts: str) -> dict[str, str | int]:
    return _parse_result(await bash_tool(command="; ".join(parts), timeout_s=30))


@pytest.mark.parametrize(
    ("name", "redirect", "echo_code"),
    [
        ("closed", "exec 1>&-", 1),
        ("to a file", "exec >/dev/null 2>&1", 0),
        ("onto stderr", "exec 1>&2", 0),
        ("with a live pipe writer", "sleep 30 & exec >/dev/null", 0),
    ],
)
@pytest.mark.asyncio
async def test_redirecting_own_stdout_keeps_the_session(
    bash_tool: bash, name: str, redirect: str, echo_code: int
):
    """The marker goes to the marker fd, so a command that redirects the shell's
    stdout still reports its exit code promptly and leaves the session usable.

    The redirect itself stands — it is what the command asked for — so later
    output goes wherever fd 1 now points, possibly nowhere. Exit codes are the
    thing that has to keep flowing.
    """
    start = time.monotonic()
    parsed = _parse_result(
        await bash_tool(command=f"{redirect}; echo hi", timeout_s=20)
    )
    elapsed = time.monotonic() - start

    assert elapsed < 10, f"{name}: took {elapsed:.1f}s, should not wait for a timeout"
    assert parsed.get("exit_code") == echo_code, name
    assert "Interrupted due to timeout" not in str(parsed.get("stderr", "")), name
    assert parsed.get("system") != "Session was automatically restarted.", name
    assert "<<exit>>" not in str(parsed.get("stderr", "")), f"{name}: marker leaked"

    assert (await _run(bash_tool, "(exit 7)")).get("exit_code") == 7, name


@pytest.mark.parametrize(
    ("name", "redirect"),
    [
        ("to a file", "exec >/dev/null"),
        ("onto stderr", "exec 1>&2"),
    ],
)
@pytest.mark.asyncio
async def test_restoring_stdout_from_the_marker_fd_works(
    bash_tool: bash, name: str, redirect: str
):
    """The marker fd holds the session's original stdout, so a command that
    redirected fd 1 away can put it back."""
    assert (await _run(bash_tool, redirect, "echo lost")).get("exit_code") == 0, name

    parsed = await _run(bash_tool, f"exec 1>&{_MARKER_FD}", "echo back")
    assert parsed["stdout"] == "back\n", name


@pytest.mark.asyncio
async def test_output_before_a_redirect_survives(bash_tool: bash, tmp_path: Path):
    """Whatever reached us before the shell redirected itself away must still be
    reported."""
    parsed = await _run(
        bash_tool, "echo kept", f"exec >{tmp_path / 'hidden.log'}", "echo hidden"
    )

    assert parsed["stdout"] == "kept\n"
    assert parsed.get("exit_code") == 0


@pytest.mark.asyncio
async def test_redirect_followed_by_a_timeout_is_still_interrupted(bash_tool: bash):
    result = await bash_tool(command="exec >/dev/null; sleep 30", timeout_s=2)
    parsed = _parse_result(result)

    assert "Interrupted due to timeout" in str(parsed.get("stderr", ""))
    assert (await _run(bash_tool, "(exit 7)")).get("exit_code") == 7


@pytest.mark.asyncio
async def test_closing_every_output_descriptor_recovers_promptly(bash_tool: bash):
    """Closing fd 1 and the marker fd leaves nothing to read; our end of the pipe
    hits EOF and the session must restart rather than block until the timeout."""
    start = time.monotonic()
    result = await bash_tool(
        command=f"exec 1>&- {_MARKER_FD}>&-; echo bye", timeout_s=20
    )
    elapsed = time.monotonic() - start
    parsed = _parse_result(result)

    assert elapsed < 10, f"took {elapsed:.1f}s, should detect lost output quickly"
    assert parsed.get("system") == "Session was automatically restarted."

    result = await bash_tool(command="echo ok")
    assert _parse_result(result)["stdout"] == "ok\n"


# ---------------------------------------------------------------------------
# Redirections that only look like a lost stdout
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_group_command_redirect_keeps_session(bash_tool: bash, tmp_path: Path):
    log = tmp_path / "group.log"
    parsed = await _run(
        bash_tool,
        f"{{ sleep 1; echo inside; }} >{log}",
        "echo done",
        f"cat {log}",
    )

    assert parsed.get("system") != "Session was automatically restarted."
    assert parsed["stdout"] == "done\ninside\n"
    assert parsed.get("exit_code") == 0


@pytest.mark.asyncio
async def test_loop_redirect_keeps_session(bash_tool: bash, tmp_path: Path):
    log = tmp_path / "loop.log"
    parsed = await _run(
        bash_tool,
        f"for i in 1 2 3; do sleep 0.4; echo $i; done >{log}",
        f"cat {log}",
    )

    assert parsed.get("system") != "Session was automatically restarted."
    assert parsed["stdout"] == "1\n2\n3\n"


@pytest.mark.asyncio
async def test_sourced_script_redirect_keeps_session(bash_tool: bash, tmp_path: Path):
    script = tmp_path / "src.sh"
    parsed = await _run(
        bash_tool,
        f"printf 'sleep 1\\necho hi\\n' >{script}",
        f"source {script} >{tmp_path / 'src.log'}",
        "echo after",
    )

    assert parsed.get("system") != "Session was automatically restarted."
    assert parsed["stdout"] == "after\n"


@pytest.mark.asyncio
async def test_redirect_then_restore_keeps_session(bash_tool: bash, tmp_path: Path):
    """A command that redirects the shell and puts fd 1 back before returning
    has not lost anything."""
    parsed = await _run(
        bash_tool,
        f"exec 3>&1 >{tmp_path / 'restore.log'}",
        "sleep 1",
        "echo captured",
        "exec 1>&3 3>&-",
        "echo back",
    )

    assert parsed.get("system") != "Session was automatically restarted."
    assert parsed["stdout"] == "back\n"


# ---------------------------------------------------------------------------
# Commands that take the marker fd over
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_closing_the_marker_fd_falls_back_to_stdout(bash_tool: bash):
    """The marker fd is ours by convention only. With it closed the marker write
    fails and the epilogue falls back to fd 1, quietly."""
    parsed = await _run(bash_tool, f"exec {_MARKER_FD}>&-", "echo hi")

    assert parsed["stdout"] == "hi\n"
    assert parsed.get("exit_code") == 0
    assert parsed.get("stderr", "") == "", "the failed marker write should be silent"

    later = await _run(bash_tool, "(exit 5)")
    assert later.get("exit_code") == 5, "session still reports exit codes"


@pytest.mark.asyncio
async def test_repointing_the_marker_fd_recovers_via_the_timeout(
    bash_tool: bash, tmp_path: Path
):
    """Repointing the marker fd at a file makes the write *succeed* into that
    file, so the fallback can not fire and the timeout has to clean up. The
    student gets an error and a working session, not a wedged tool."""
    result = await bash_tool(
        command=f"exec {_MARKER_FD}>{tmp_path / 'taken.log'}; echo hi", timeout_s=2
    )
    parsed = _parse_result(result)

    assert "Interrupted due to timeout" in str(parsed.get("stderr", ""))
    assert parsed.get("system") == "Session was automatically restarted."

    assert _parse_result(await bash_tool(command="echo ok"))["stdout"] == "ok\n"


@pytest.mark.asyncio
async def test_marker_never_leaks_into_redirected_output(bash_tool: bash):
    """A marker write must not land on stderr: bash can reuse a freed fd 2 to
    stash a saved descriptor, which silently reroutes the marker there.

    `set -x` is excluded: xtrace echoes the epilogue as it runs it, which has
    always put the marker on stderr and is not what this change is about.
    """
    for command in (
        "exec 1>&2; echo hi",
        "exec 2>&1; echo hi",
        "exec 1>&2 2>&1; echo hi",
    ):
        parsed = _parse_result(await bash_tool(command=command, timeout_s=20))
        assert "<<exit>>" not in str(parsed.get("stdout", "")), command
        assert "<<exit>>" not in str(parsed.get("stderr", "")), command
        assert "Bad file descriptor" not in str(parsed.get("stderr", "")), command


@pytest.mark.asyncio
async def test_restricted_shell_produces_no_marker_noise(bash_tool: bash):
    """`set -r` rejects redirections to a filename, so the epilogue must only
    ever duplicate or close descriptors."""
    parsed = await _run(bash_tool, "set -r", "echo hi")
    assert parsed["stdout"] == "hi\n"
    assert parsed.get("exit_code") == 0
    assert parsed.get("stderr", "") == ""

    parsed = await _run(bash_tool, "echo after")
    assert parsed["stdout"] == "after\n"
    assert parsed.get("stderr", "") == ""


@pytest.mark.asyncio
async def test_marker_lookalike_in_output_is_not_taken_for_a_marker(bash_tool: bash):
    """Only a marker carrying the command's nonce counts, so output that mimics
    one can not fake an exit code or truncate the command's own output."""
    parsed = await _run(
        bash_tool,
        "echo '<<exit>> deadbeef 5 opts opts'",
        "echo '<<exit>> 0'",
        "true",
    )

    assert parsed.get("exit_code") == 0
    assert parsed.get("system") != "Session was automatically restarted."
    assert "<<exit>> deadbeef 5" in str(parsed["stdout"])

    parsed = await _run(bash_tool, "echo after")
    assert parsed["stdout"] == "after\n"


@pytest.mark.asyncio
async def test_restart_preempts_running_command(bash_tool: bash):
    """restart=True must not queue behind a long-running command."""

    async def long_command():
        return await bash_tool(command="sleep 30", timeout_s=30)

    async def restart():
        await asyncio.sleep(0.5)
        return await bash_tool(restart=True)

    start = time.monotonic()
    _, restart_result = await asyncio.gather(long_command(), restart())
    elapsed = time.monotonic() - start

    assert elapsed < 10, f"restart waited {elapsed:.1f}s behind the running command"
    assert (
        _parse_result(restart_result)["system"] == "Tool has been manually restarted."
    )

    result = await bash_tool(command="echo ok")
    assert _parse_result(result)["stdout"] == "ok\n"


@pytest.mark.asyncio
async def test_preempted_command_reports_instead_of_raising(bash_tool: bash):
    """A restart can land before the command even reaches the shell's stdin. The
    preempted call still owes the caller a result, not a transport error."""
    await bash_tool(command="echo warm")

    results = await asyncio.gather(
        bash_tool(command="echo x", timeout_s=30),
        bash_tool(restart=True),
        return_exceptions=True,
    )

    for result in results:
        assert not isinstance(result, BaseException), repr(result)

    assert _parse_result(await bash_tool(command="echo ok"))["stdout"] == "ok\n"


@pytest.mark.asyncio
async def test_unwritable_stdin_returns_a_result(
    bash_tool: bash, monkeypatch: pytest.MonkeyPatch
):
    """The same path with the shell still alive: report it and replace the
    session, rather than raising or claiming an exit code we never saw."""
    await bash_tool(command="echo warm")
    session = bash_tool._session  # pyright: ignore[reportPrivateUsage]
    assert session is not None and session._process is not None  # pyright: ignore[reportPrivateUsage]
    session._command_kill_timeout_s = 0.5  # pyright: ignore[reportPrivateUsage]

    async def refuse():
        raise ConnectionResetError("Connection lost")

    monkeypatch.setattr(session._process.stdin, "drain", refuse)  # pyright: ignore[reportPrivateUsage]

    parsed = _parse_result(await bash_tool(command="echo hi", timeout_s=10))
    assert "stopped accepting input" in str(parsed.get("stderr", ""))
    assert "exit_code" not in parsed

    assert _parse_result(await bash_tool(command="echo ok"))["stdout"] == "ok\n"


# ---------------------------------------------------------------------------
# Exit marker in user output
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_exit_marker_in_user_output(bash_tool: bash):
    result = await bash_tool(
        command="echo '<<exit>>'",
        timeout_s=5.0,
    )
    parsed = _parse_result(result)
    assert "<<exit>>" in str(parsed["stdout"])
    assert parsed.get("exit_code") == 0


# ---------------------------------------------------------------------------
# Hardening
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bash_uses_absolute_path():
    """The bash command should use an absolute path, not /usr/bin/env."""
    session = _BashSession(
        max_output_length=1000, disable_networking=True, python_venv=None
    )
    assert session._command == ["/usr/bin/bash"]  # pyright: ignore[reportPrivateUsage]


# ---------------------------------------------------------------------------
# Syntax-check must honor parse-time shell options of the live session
# ---------------------------------------------------------------------------
#
# The pre-flight `bash -n` check runs in a fresh process and does not see
# parse-time shell options (e.g. `extglob`) that an earlier call enabled in the
# persistent session. Without carrying that state over, a valid command like
# `echo !(foo)` issued after `shopt -s extglob` would be wrongly rejected as a
# syntax error. The session's enabled options must be threaded into the check.

# An extglob pattern: valid only when `extglob` is enabled.
_EXTGLOB_COMMAND = "echo !(foo)"


@pytest.mark.asyncio
async def test_extglob_command_not_rejected_when_enabled_in_prior_call(
    bash_tool: bash,
):
    """A command using extglob must run, not be flagged as a syntax error, when
    extglob was enabled in an earlier call against the persistent session."""
    await bash_tool(command="shopt -s extglob")

    result = await bash_tool(command=_EXTGLOB_COMMAND)
    parsed = _parse_result(result)

    assert "syntax error" not in str(parsed.get("stderr", "")), parsed
    assert parsed.get("exit_code") == 0, parsed


@pytest.mark.asyncio
async def test_extglob_command_rejected_when_not_enabled(bash_tool: bash):
    """Without extglob enabled, the same command is genuinely a syntax error in
    the session and must still be rejected fast (not executed, not hung)."""
    result = await bash_tool(command=_EXTGLOB_COMMAND, timeout_s=10.0)
    parsed = _parse_result(result)

    assert "Interrupted due to timeout" not in str(parsed.get("stderr", "")), parsed
    assert parsed.get("exit_code") == 2, parsed
    assert "syntax error" in str(parsed.get("stderr", "")), parsed


@pytest.mark.asyncio
async def test_extglob_state_reset_after_restart(bash_tool: bash):
    """A restart resets the shell to defaults, so a previously enabled extglob
    must no longer be assumed by the syntax check."""
    await bash_tool(command="shopt -s extglob")
    await bash_tool(restart=True)

    result = await bash_tool(command=_EXTGLOB_COMMAND, timeout_s=10.0)
    parsed = _parse_result(result)

    assert parsed.get("exit_code") == 2, parsed
    assert "syntax error" in str(parsed.get("stderr", "")), parsed


# Incomplete commands (unterminated quote, unmatched heredoc) must fail fast
# instead of leaving the shell at its continuation prompt.

# Generous per-call budget: a fast failure returns in well under a second, while
# this keeps the suite bounded if a call ever hangs.
_TRANSCRIPT_HARNESS_TIMEOUT_S = 10.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command",
    [
        # Unterminated double quote (`python3 -c "` is never closed).
        pytest.param(
            """python3 -c "
import csv
rows = list(csv.reader(open('data/input.csv')))""",
            id="unterminated-quote",
        ),
        # Heredoc with a stray `>` after the delimiter, so `EOF` never matches.
        pytest.param(
            """python3 << 'EOF'>
import csv
rows = list(csv.reader(open('data/input.csv')))
totals = {}
for name, value in rows:
    totals[name] = totals.get(name, 0) + float(value)
for name, total in sorted(totals.items()):
    print(name, total)
EOF>
""",
            id="malformed-heredoc",
        ),
        pytest.param(
            """cat > /tmp/summary.py << 'PYEOF'>
import csv
rows = list(csv.reader(open('data/input.csv')))
print(len(rows))
PYEOF>
python3 /tmp/summary.py
""",
            id="malformed-heredoc-then-command",
        ),
        pytest.param(
            """cat > /tmp/script.py << 'EOF'>
import csv
rows = list(csv.reader(open('data/input.csv')))
     if rows:
         print(rows[0])
EOF>
python3 /tmp/script.py
""",
            id="malformed-heredoc-with-indented-body",
        ),
    ],
)
async def test_transcript_bash_call_does_not_hang(bash_tool: bash, command: str):
    """Each malformed transcript command must fail fast instead of hanging until
    the timeout."""
    start = time.monotonic()
    result = await bash_tool(command=command, timeout_s=_TRANSCRIPT_HARNESS_TIMEOUT_S)
    elapsed = time.monotonic() - start
    parsed = _parse_result(result)

    assert "Interrupted due to timeout" not in str(parsed.get("stderr", "")), (
        f"command hung and was interrupted by the timeout after {elapsed:.1f}s:\n"
        f"{command!r}"
    )


@pytest.mark.asyncio
async def test_session_hands_its_own_pipes_to_the_shell(
    bash_tool: bash, monkeypatch: pytest.MonkeyPatch
):
    """The session opens all three of the shell's standard streams, so all
    three go to the student — otherwise a program it runs cannot reopen them by
    path (/dev/stdout, /proc/self/fd/N)."""
    named: list[tuple[int, ...]] = []

    def _spy(argv: list[str], *chown_fds: int, disable_networking: bool):
        _ = disable_networking
        named.append(chown_fds)
        return argv, None

    monkeypatch.setattr("karotte.tools.bash.student_session_command", _spy)

    await bash_tool(command="true")

    assert named == [(0, 1, 2)]


# ---------------------------------------------------------------------------
# The session tells the shell who it is running as
# ---------------------------------------------------------------------------
#
# setuid(2) leaves the environment alone, so a demoted shell keeps root's HOME
# and cannot write it: `echo $HOME` gives /root while `ls /root` is denied,
# breaking every tool that caches under $HOME
# (Triton, JAX, pip, uv). check_permissions.py asserts `runuser -u student`
# yields HOME=<student workdir>; the bash tool has to match it.


def _no_preexec(argv: list[str], *chown_fds: int, disable_networking: bool):
    """Stand-in for student_session_command that skips the privilege drop.

    The drop needs root, but the env the session builds does not depend on it.
    """
    _ = chown_fds, disable_networking
    return argv, None


@pytest.mark.asyncio
async def test_session_repoints_home_at_the_demoted_users_home(
    bash_tool: bash, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("HOME", "/root")
    monkeypatch.setenv("KAROTTE_DEMOTE_ID", str(os.getuid()))
    monkeypatch.setattr("karotte.tools.bash.student_session_command", _no_preexec)

    result = _parse_result(await bash_tool(command="printenv HOME"))

    assert result["stdout"] == f"{pwd.getpwuid(os.getuid()).pw_dir}\n"


@pytest.mark.asyncio
async def test_session_names_the_demoted_user(
    bash_tool: bash, monkeypatch: pytest.MonkeyPatch
):
    """Under gVisor the shell sees the overflow uid, so $USER is the only way
    for it to learn its own name."""
    monkeypatch.setenv("USER", "root")
    monkeypatch.setenv("KAROTTE_DEMOTE_ID", str(os.getuid()))
    monkeypatch.setattr("karotte.tools.bash.student_session_command", _no_preexec)

    result = _parse_result(await bash_tool(command="printenv USER; printenv LOGNAME"))

    name = pwd.getpwuid(os.getuid()).pw_name
    assert result["stdout"] == f"{name}\n{name}\n"


@pytest.mark.asyncio
async def test_session_leaves_home_alone_without_demotion(
    bash_tool: bash, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Outside a container there is no student to become, so the caller's own
    HOME is the right one."""
    monkeypatch.delenv("KAROTTE_DEMOTE_ID", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))

    result = _parse_result(await bash_tool(command="printenv HOME"))

    assert result["stdout"] == f"{tmp_path}\n"


@pytest.mark.asyncio
async def test_session_keeps_venv_on_path_alongside_the_home_fix(
    bash_tool: bash, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """The venv stays on PATH now that the session always sets HOME."""
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /usr\n")

    monkeypatch.setenv("KAROTTE_DEMOTE_ID", str(os.getuid()))
    monkeypatch.setattr("karotte.tools.bash.student_session_command", _no_preexec)
    monkeypatch.setattr(bash_tool.config, "python_venv", str(venv))

    result = _parse_result(
        await bash_tool(command="printenv VIRTUAL_ENV; printenv HOME")
    )

    assert result["stdout"] == (f"{venv}\n{pwd.getpwuid(os.getuid()).pw_dir}\n")


@pytest.mark.asyncio
async def test_session_drops_pythonsafepath(
    bash_tool: bash, monkeypatch: pytest.MonkeyPatch
):
    """The image sets PYTHONSAFEPATH to protect the judge's python. The student
    has nothing to protect from itself and needs `python -c` to see its cwd."""
    monkeypatch.setenv("PYTHONSAFEPATH", "1")

    result = _parse_result(await bash_tool(command="printenv PYTHONSAFEPATH"))

    assert result["stdout"] == ""
    assert result["exit_code"] == 1


_SPARE_UID = 60123
"""Uid the demoted shell below runs as, kept clear of anything else on the box."""


@pytest.mark.requires_root
@pytest.mark.asyncio
async def test_demoted_shell_can_reopen_its_own_stdout(
    bash_tool: bash, monkeypatch: pytest.MonkeyPatch
):
    """Redirecting to /dev/stdout is an ordinary thing to write in a shell, and
    it opens the pipe by path rather than using the inherited fd."""
    monkeypatch.setenv("KAROTTE_DEMOTE_ID", str(_SPARE_UID))

    result = _parse_result(await bash_tool(command="echo hi > /dev/stdout"))

    assert result["stdout"] == "hi\n"
    assert not result.get("stderr")


@pytest.mark.requires_root
@pytest.mark.asyncio
async def test_timeout_spares_demoted_background_jobs(
    bash_tool: bash, monkeypatch: pytest.MonkeyPatch
):
    """Same as test_timeout_spares_background_jobs_from_earlier_calls, but with
    the real production topology: the shell setsids into its own process group
    and runs demoted, so the fallback killpg would reap the background job."""
    monkeypatch.setenv("KAROTTE_DEMOTE_ID", str(_SPARE_UID))

    duration = _unique_sleep_duration()
    await bash_tool(command=f"nohup sleep {duration} >/dev/null 2>&1 & true")
    bg_pid = await _find_sleep_pid(duration)

    try:
        result = await bash_tool(command="sleep 60", timeout_s=1.0)
        parsed = _parse_result(result)
        assert "Interrupted due to timeout" in str(parsed.get("stderr", ""))
        assert "restarted" not in str(parsed.get("system", ""))

        os.kill(bg_pid, 0)  # raises ProcessLookupError if the bg job died

        result = await bash_tool(command="echo still alive")
        assert _parse_result(result)["stdout"] == "still alive\n"
    finally:
        try:
            os.kill(bg_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


@pytest.mark.requires_root
@pytest.mark.asyncio
async def test_timeout_fallback_restart_spares_demoted_wrapped_background_jobs(
    bash_tool: bash, monkeypatch: pytest.MonkeyPatch
):
    """Same as test_timeout_fallback_restart_spares_wrapped_background_jobs, but
    in the demoted (setsid) topology."""
    monkeypatch.setenv("KAROTTE_DEMOTE_ID", str(_SPARE_UID))
    monkeypatch.setattr(_BashSession, "_command_kill_timeout_s", 0.5)

    duration = _unique_sleep_duration()
    await bash_tool(command=f"sleep {duration} >/dev/null 2>&1 &")
    bg_pid = await _find_sleep_pid(duration)

    try:
        result = await bash_tool(
            command="while true; do sleep 0.2; done", timeout_s=1.0
        )
        parsed = _parse_result(result)
        assert parsed.get("system") == "Session was automatically restarted."

        assert not await _wait_for_death(bg_pid, timeout=1.0)
    finally:
        try:
            os.kill(bg_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_cap_buffer_bounds_flood_keeps_head_and_exit_marker():
    from karotte.tools.bash import _cap_buffer  # pyright: ignore[reportPrivateUsage]

    cap = 1024 * 1024  # _OUTPUT_HARD_CAP

    small = "hello world\n<<exit>> 0"
    assert _cap_buffer(small) == small

    flood = ("HEAD" + "A" * 500_000) + ("B" * 3_000_000) + ("Z" * 100 + "<<exit>> 0")
    out = _cap_buffer(flood)
    assert len(out) <= cap + 100
    assert out.startswith("HEAD")
    assert out.endswith("<<exit>> 0")
    assert "truncated" in out


@pytest.mark.asyncio
async def test_flood_is_capped_in_memory_and_exit_code_survives(bash_tool: bash):
    _ = bash_tool  # applies the platform bash-path patch to _BashSession
    # max_output_length above the hard cap so the returned stdout exposes the
    # in-memory capping instead of the return-time truncation.
    session = _BashSession(
        max_output_length=2_000_000, disable_networking=False, python_venv=None
    )
    await session.start()
    try:
        result = await session.run(
            "head -c 5000000 /dev/zero | tr '\\0' A; echo; echo DONE",
            timeout_s=60.0,
        )
    finally:
        await session.stop()

    parsed = _parse_result(result)
    stdout = str(parsed["stdout"])
    assert parsed["exit_code"] == 0
    assert len(stdout) <= 1024 * 1024 + 100
    assert stdout.startswith("AAAA")
    assert "truncated" in stdout
    assert stdout.rstrip().endswith("DONE")


@pytest.mark.asyncio
async def test_drain_caps_the_buffer_and_keeps_the_tail():
    """The drain consumes a flood to the end; the buffer keeps head and tail."""
    from karotte.tools.bash import (
        _OUTPUT_HARD_CAP,  # pyright: ignore[reportPrivateUsage]
    )

    session = _BashSession(
        max_output_length=16000, disable_networking=True, python_venv=None
    )
    reader = asyncio.StreamReader()
    reader.feed_data(b"x" * (3 * 1024 * 1024) + b"tail")
    reader.feed_eof()

    await session._drain(reader, session._append_stdout)  # pyright: ignore[reportPrivateUsage]

    buffer = session._stdout_buffer  # pyright: ignore[reportPrivateUsage]
    assert len(buffer) <= _OUTPUT_HARD_CAP + 200
    assert buffer.startswith("xxx")
    assert buffer.endswith("tail")
    assert reader.at_eof()


@pytest.mark.asyncio
async def test_drain_decodes_a_character_split_across_reads():
    session = _BashSession(
        max_output_length=16000, disable_networking=True, python_venv=None
    )
    reader = asyncio.StreamReader()
    reader.feed_data(b"a" * 65535 + "\u00e4".encode())
    reader.feed_eof()

    await session._drain(reader, session._append_stdout)  # pyright: ignore[reportPrivateUsage]

    assert session._stdout_buffer.endswith("a\u00e4")  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_drain_flushes_a_partial_character_at_eof():
    session = _BashSession(
        max_output_length=16000, disable_networking=True, python_venv=None
    )
    reader = asyncio.StreamReader()
    reader.feed_data(b"abc\xc3")
    reader.feed_eof()

    await session._drain(reader, session._append_stdout)  # pyright: ignore[reportPrivateUsage]

    assert session._stdout_buffer == "abc\ufffd"  # pyright: ignore[reportPrivateUsage]


def test_append_capped_keeps_notice_and_slides_the_tail():
    from karotte.tools.bash import (
        _OUTPUT_HARD_CAP,  # pyright: ignore[reportPrivateUsage]
        _OUTPUT_TRUNCATION_NOTICE,  # pyright: ignore[reportPrivateUsage]
        _append_capped,  # pyright: ignore[reportPrivateUsage]
    )

    buf = _append_capped("", "x" * (_OUTPUT_HARD_CAP + 100))
    capped_len = len(buf)
    assert _OUTPUT_TRUNCATION_NOTICE in buf

    buf = _append_capped(buf, "MARKER")
    assert len(buf) == capped_len
    assert buf.startswith("xxx")
    assert _OUTPUT_TRUNCATION_NOTICE in buf
    assert buf.endswith("MARKER")
    assert buf.index(_OUTPUT_TRUNCATION_NOTICE) == _OUTPUT_HARD_CAP - 8192


@pytest.mark.asyncio
async def test_a_cancelled_run_leaves_a_usable_session(bash_tool: bash):
    await bash_tool(command="echo warm")
    call = asyncio.create_task(bash_tool(command="sleep 30"))
    await asyncio.sleep(0.5)
    _ = call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call

    result = await bash_tool(command="echo after")
    assert result.structured_content is not None
    assert result.structured_content["stdout"].strip() == "after"


@pytest_asyncio.fixture
async def bash_client(bash_tool: bash):
    server = FastMCP("test")
    server.tool(bash_tool.__call__, name="bash")
    async with Client(server) as client:
        yield client


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["nan", "inf", "-inf", 0, -1.0])
async def test_invalid_timeout_is_rejected(
    bash_client: Client[FastMCPTransport], bad: float | str
):
    """NaN/inf must be refused; passed through, it would hang the event loop forever."""
    with pytest.raises(ToolError):
        await bash_client.call_tool("bash", {"command": "echo hi", "timeout_s": bad})

    # the session must still be usable afterwards
    result = await bash_client.call_tool("bash", {"command": "echo alive"})
    assert isinstance(result.content[0], TextContent)
    assert json.loads(result.content[0].text)["stdout"] == "alive\n"


@pytest.mark.asyncio
@pytest.mark.parametrize("ok", [30, "30"])
async def test_finite_timeout_still_accepted(
    bash_client: Client[FastMCPTransport], ok: float | str
):
    result = await bash_client.call_tool(
        "bash", {"command": "echo ok", "timeout_s": ok}
    )
    assert isinstance(result.content[0], TextContent)
    assert json.loads(result.content[0].text)["stdout"] == "ok\n"
