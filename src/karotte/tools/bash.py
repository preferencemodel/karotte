# Adapted from anthropics/anthropic-quickstarts (computer-use-demo), MIT License:
#
# Copyright (c) 2023 Anthropic
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import asyncio
import codecs
import os
import secrets
import signal
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Annotated, Final, final

import psutil
from fastmcp.tools.tool import ToolResult
from karotte import ToolBase
from karotte.confinement import network_needs_namespace
from karotte.subprocess import student_env, student_session_command
from pydantic import BaseModel, Field

# max_output_length only truncates at return time; a flooding command would
# otherwise grow the in-memory buffers unbounded and OOM the root MCP server.
_OUTPUT_HARD_CAP = 1024 * 1024
_OUTPUT_TAIL_KEEP = 8192
_OUTPUT_TRUNCATION_NOTICE = "\n[... output truncated: exceeded 1 MiB in memory ...]\n"


def _cap_buffer(buf: str) -> str:
    """Cap a buffer to _OUTPUT_HARD_CAP, keeping the head plus a tail large
    enough to preserve the trailing exit marker."""
    if len(buf) <= _OUTPUT_HARD_CAP:
        return buf
    head = buf[: _OUTPUT_HARD_CAP - _OUTPUT_TAIL_KEEP]
    tail = buf[-_OUTPUT_TAIL_KEEP:]
    return head + _OUTPUT_TRUNCATION_NOTICE + tail


def _append_capped(buf: str, text: str) -> str:
    """Append to a buffer; once capped, only the tail moves."""
    if len(buf) <= _OUTPUT_HARD_CAP:
        return _cap_buffer(buf + text)
    frozen = len(buf) - _OUTPUT_TAIL_KEEP
    return buf[:frozen] + (buf[frozen:] + text)[-_OUTPUT_TAIL_KEEP:]


class BashConfig(BaseModel):
    default_timeout_s: float = 3600.0
    """Per-command timeout when none is given; also the maximum a command may request."""
    max_output_length: int = 16000
    """Characters of stdout and stderr each kept in a result."""
    disable_networking: bool = Field(default_factory=network_needs_namespace)
    """Put bash and its children in their own network namespace, cutting them off from the network.
    On by default where iptables can't be used (gVisor)."""
    python_venv: str | None = "/workdir/.venv"
    """Venv activated only in the student's bash (VIRTUAL_ENV and PATH), never for root, so a
    student-planted binary in the venv can't shadow a tool the harness runs."""


def _bash_tool_result(
    stdout: str = "",
    stderr: str = "",
    system: str = "",
    exit_code: int | None = None,
) -> ToolResult:
    """Create a ToolResult from a Bash command."""
    content_dict: dict[str, str | int] = {"stdout": stdout}
    if stderr:
        content_dict["stderr"] = stderr
    if system:
        content_dict["system"] = system
    if exit_code is not None:
        content_dict["exit_code"] = exit_code
    return ToolResult(structured_content=content_dict)


_MARKER_FD: Final[int] = 231
"""Duplicate of the shell's original stdout, sharing its pipe. The exit marker
goes here rather than to fd 1, so a command that redirects the shell's own
stdout (`exec >file`) cannot swallow it. A high number keeps it clear of the
descriptors scripts hardcode, `flock 10` above all."""

_LOST_OUTPUT_NOTE: Final[str] = (
    "[The shell closed its output descriptors, so command results can no longer "
    "be read.]"
)
_BROKEN_STDIN_NOTE: Final[str] = (
    "[The shell stopped accepting input, so the command never ran. Session restarted.]"
)


def _append_note(stderr: str, note: str) -> str:
    """Append a note from the tool itself to a command's stderr."""
    return f"{stderr}\n{note}" if stderr else note


@dataclass(frozen=True)
class _Marker:
    """A parsed exit marker line."""

    exit_code: int
    bashopts: str | None
    shellopts: str | None


class _MarkerLostError(Exception):
    """The shell closed every copy of its output pipe, so the exit marker can
    never arrive."""


class _BashSession:
    """A Bash shell session."""

    _command: Final[list[str]] = ["/usr/bin/bash"]
    _output_delay_s: float = 0.1
    _exit_marker: str = "<<exit>>"
    _command_kill_timeout_s: float = 5.0  # Time to wait after process termination

    def __init__(
        self,
        max_output_length: int,
        disable_networking: bool,
        python_venv: str | None,
    ):
        self._process: asyncio.subprocess.Process | None = None
        self._shell_pid: int | None = None
        self._stdout_buffer: str = ""
        self._stderr_buffer: str = ""
        self._drains: list[asyncio.Task[None]] = []
        self._protected_pgids: set[int] = set()
        # Parse-time shell options of the live session, captured from the exit
        # marker of the last completed command. None means "not yet observed".
        self._bashopts: str | None = None
        self._shellopts: str | None = None
        self._marker_nonce: str = ""
        """Nonce the current command's marker must carry. Never matches until a
        command is sent, since a marker field can not be empty."""
        self.max_output_length: int = max_output_length
        self.disable_networking: bool = disable_networking
        self.python_venv: str | None = python_venv

    @property
    def _started(self) -> bool:
        return self._process is not None and self._process.returncode is None

    async def start(self):
        if self._process and self._process.returncode is None:
            # Already running
            return

        # All three streams below are pipes this process opens, so all three
        # are handed to the student along with the privilege drop.
        argv, preexec = student_session_command(
            self._command, 0, 1, 2, disable_networking=self.disable_networking
        )

        # The shell runs as the student, so it needs to be told so (see
        # student_identity_env; without it the session inherits root's HOME) and
        # it must not inherit the harness's credentials (scrub_harness_secrets).
        env = student_env()
        env.pop("PYTHONSAFEPATH", None)
        if self.python_venv and (Path(self.python_venv) / "pyvenv.cfg").is_file():
            env["VIRTUAL_ENV"] = self.python_venv
            venv_bin = f"{self.python_venv}/bin"
            existing_path = env.get("PATH", "")
            env["PATH"] = f"{venv_bin}:{existing_path}" if existing_path else venv_bin

        self._process = await asyncio.create_subprocess_exec(
            *argv,
            preexec_fn=preexec,
            start_new_session=preexec is None,
            bufsize=0,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        self._shell_pid = None
        self._stdout_buffer = ""
        self._stderr_buffer = ""
        self._start_drains()
        self._bashopts = None
        self._shellopts = None
        self._marker_nonce = ""
        self._protected_pgids = set()

        assert self._process.stdin
        self._process.stdin.write(f"set -m\nexec {_MARKER_FD}>&1\n".encode())
        await self._process.stdin.drain()

    async def stop(self, *, spare_background: bool = False):
        """Tear down the session. With ``spare_background``, background jobs from
        earlier commands survive."""
        if not self._started:
            return

        assert self._process
        pid = self._process.pid
        session_wide = self._leads_own_session()
        spare = frozenset(self._protected_pgids) if spare_background else None
        doomed = self._doomed_pgids(pid, session_wide, spare)

        await self._close_stdin()
        self._signal_pgids(doomed, signal.SIGTERM)
        await self._terminate_process(pid)

        try:
            await self._wait_for_process_to_die()
        except TimeoutError:
            self._signal_pgids(
                doomed | self._doomed_pgids(pid, session_wide, spare), signal.SIGKILL
            )
            await self._kill_process(pid)

        try:
            await self._wait_for_process_to_die()
        except TimeoutError:
            # Process is really stuck. No idea what to do.
            pass

        self._signal_pgids(self._doomed_pgids(pid, session_wide, spare), signal.SIGKILL)

        await self._close_stdout_and_stderr()
        self._cancel_drains()

        transport = getattr(self._process, "_transport", None)
        if transport is not None:
            transport.close()
        self._process = None

    async def _close_stdin(self):
        assert self._process
        assert self._process.stdin
        try:
            self._process.stdin.close()
            await self._process.stdin.wait_closed()
        except Exception:
            pass

    async def _terminate_process(self, pid: int):
        assert self._process
        try:
            os.killpg(pid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            try:
                self._process.terminate()
            except (ProcessLookupError, OSError):
                pass

    async def _kill_process(self, pid: int):
        assert self._process
        try:
            os.killpg(pid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            try:
                self._process.kill()
            except (ProcessLookupError, OSError):
                pass

    async def _wait_for_process_to_die(self):
        assert self._process
        await asyncio.wait_for(
            self._process.wait(), timeout=self._command_kill_timeout_s
        )

    async def _close_stdout_and_stderr(self):
        assert self._process
        assert self._process.stdout
        assert self._process.stderr
        try:
            self._process.stdout.feed_eof()
            self._process.stderr.feed_eof()
        except Exception:
            pass

    async def kill(self):
        """SIGKILL the shell and its process groups without touching the
        transports, so an in-flight run() sees the death and returns."""
        if not self._started:
            return
        assert self._process
        pid = self._process.pid
        doomed = self._doomed_pgids(pid, self._leads_own_session())
        self._signal_pgids(doomed, signal.SIGKILL)
        await self._kill_process(pid)

    async def _cleanup_dead_process(self):
        """Clean up a process that has already died.

        Closes the subprocess transport to prevent a ResourceWarning when
        the transport's __del__ fires after the event loop is closed.
        """
        assert self._process
        await self._close_stdin()
        await self._close_stdout_and_stderr()
        self._cancel_drains()
        # Close the subprocess transport so its __del__ is a no-op.
        # asyncio.subprocess.Process has no public close(), so we access
        # the internal transport directly.
        transport = getattr(self._process, "_transport", None)
        if transport is not None:
            transport.close()
        self._process = None

    async def _ensure_process_alive(self) -> bool:
        """Ensure the process is alive, auto-restart if needed.

        Returns True if the process needed to be restarted."""
        if self._started:
            return False

        await self.start()
        return True

    def _start_drains(self) -> None:
        """Consume both output pipes continuously into the capped buffers."""
        assert self._process
        assert self._process.stdout
        assert self._process.stderr
        self._cancel_drains()
        self._drains = [
            asyncio.create_task(self._drain(self._process.stdout, self._append_stdout)),
            asyncio.create_task(self._drain(self._process.stderr, self._append_stderr)),
        ]

    def _append_stdout(self, text: str) -> None:
        self._stdout_buffer = _append_capped(self._stdout_buffer, text)

    def _append_stderr(self, text: str) -> None:
        self._stderr_buffer = _append_capped(self._stderr_buffer, text)

    async def _drain(
        self, stream: asyncio.StreamReader, append: Callable[[str], None]
    ) -> None:
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        while True:
            try:
                chunk = await stream.read(65536)
            except Exception:
                return
            text = decoder.decode(chunk, final=not chunk)
            if text:
                append(text)
            if not chunk:
                return

    def _cancel_drains(self) -> None:
        for task in self._drains:
            _ = task.cancel()
        self._drains = []

    async def _clear_buffers(self):
        self._stdout_buffer = ""
        self._stderr_buffer = ""

    def _parse_marker(self, marker_line: str) -> _Marker | None:
        """Parse '<<exit>> <nonce> <code> <BASHOPTS> <SHELLOPTS>'.

        Returns None unless the line carries the current command's nonce, which
        keeps output that looks like a marker from being taken for one.
        """
        fields = marker_line[len(self._exit_marker) :].split()
        if len(fields) < 2 or fields[0] != self._marker_nonce:
            return None
        try:
            exit_code = int(fields[1])
        except ValueError:
            return None
        return _Marker(
            exit_code=exit_code,
            bashopts=fields[2] if len(fields) > 2 else None,
            shellopts=fields[3] if len(fields) > 3 else None,
        )

    def _split_at_exit_marker(self, output: str) -> tuple[str, _Marker | None]:
        """Split output at the exit marker, returning (stdout, marker).

        Lines that merely look like a marker are skipped; without one for the
        current command this returns (output, None).
        """
        search_start = 0
        while True:
            try:
                marker_idx = output.index(self._exit_marker, search_start)
            except ValueError:
                return output, None
            marker_line = output[marker_idx:].split("\n", 1)[0]
            marker = self._parse_marker(marker_line)
            if marker is not None:
                self._bashopts = marker.bashopts
                self._shellopts = marker.shellopts
                return output[:marker_idx], marker
            search_start = marker_idx + len(self._exit_marker)

    async def _handle_dead_process(self) -> tuple[str, str, int | None]:
        """Handle a process that has died mid-command."""
        assert self._process
        if self._drains:
            # Background jobs may hold the pipes open, so EOF is not guaranteed.
            _ = await asyncio.wait(self._drains, timeout=self._output_delay_s)
        stdout, marker = self._split_at_exit_marker(self._stdout_buffer)
        stderr = self._stderr_buffer
        exit_code = marker.exit_code if marker is not None else None
        if exit_code is None:
            exit_code = self._process.returncode
            stderr = _append_note(
                stderr, f"[Process died with return code {exit_code}]"
            )
        await self._cleanup_dead_process()
        await self._clear_buffers()
        return stdout, stderr, exit_code

    async def _handle_timeout(self, timeout: float) -> tuple[str, str, int | None]:
        """Handle a command that timed out."""
        stdout, marker = self._split_at_exit_marker(self._stdout_buffer)
        stderr = self._stderr_buffer
        await self.stop(spare_background=True)
        await self.start()
        stderr = _append_note(stderr, f"[Interrupted due to timeout after {timeout}s]")
        return stdout, stderr, marker.exit_code if marker is not None else None

    async def _handle_lost_stdout(self) -> tuple[str, str, bool, int | None]:
        """Handle a shell that closed every copy of its output pipe."""
        assert self._process
        try:
            await asyncio.wait_for(self._process.wait(), timeout=self._output_delay_s)
        except TimeoutError:
            pass
        if self._process.returncode is not None:
            stdout, stderr, exit_code = await self._handle_dead_process()
            return stdout, stderr, False, exit_code
        stdout, marker = self._split_at_exit_marker(self._stdout_buffer)
        stderr = self._stderr_buffer
        await self.stop(spare_background=True)
        await self.start()
        stderr = _append_note(stderr, _LOST_OUTPUT_NOTE)
        return stdout, stderr, True, marker.exit_code if marker is not None else None

    async def _handle_dead_stdin(self) -> tuple[str, str, int | None]:
        """Handle a shell we could not hand the command to."""
        assert self._process
        try:
            await asyncio.wait_for(
                self._process.wait(), timeout=self._command_kill_timeout_s
            )
        except TimeoutError:
            pass
        if self._process.returncode is not None:
            return await self._handle_dead_process()
        stdout = self._stdout_buffer
        stderr = self._stderr_buffer
        await self.stop(spare_background=True)
        await self.start()
        return stdout, _append_note(stderr, _BROKEN_STDIN_NOTE), None

    async def _read_until_exit_marker(
        self, timeout: float
    ) -> tuple[str, str, bool, int | None]:
        try:
            result = await self._poll_for_completion(timeout)
            timed_out = result is None
            if timed_out:
                result = await self._interrupt_foreground()

            if result is not None:
                stdout, stderr, exit_code = result
                if timed_out:
                    stderr = _append_note(
                        stderr, f"[Interrupted due to timeout after {timeout}s]"
                    )
                return stdout, stderr, False, exit_code

            stdout, stderr, exit_code = await self._handle_timeout(timeout)
            return stdout, stderr, True, exit_code
        except _MarkerLostError:
            return await self._handle_lost_stdout()

    async def _poll_for_completion(
        self, timeout: float
    ) -> tuple[str, str, int | None] | None:
        """Poll until the command completes or dies; None on timeout."""
        assert self._process
        assert self._process.stdout
        assert self._process.stderr

        try:
            async with asyncio.timeout(timeout):
                while True:
                    await asyncio.sleep(self._output_delay_s)

                    if self._process.returncode is not None:
                        return await self._handle_dead_process()

                    stdout, marker = self._split_at_exit_marker(self._stdout_buffer)
                    if marker is not None:
                        stderr = self._stderr_buffer
                        await self._clear_buffers()
                        return stdout, stderr, marker.exit_code

                    if self._process.stdout.at_eof():
                        raise _MarkerLostError

        except TimeoutError:
            return None

    def _resolve_shell_pid(self) -> int | None:
        """Pid of the bash process itself, as seen from this process.

        Wrappers can separate it from ``self._process.pid``: with a PID
        namespace, ``unshare --fork`` stays alive as the shell's parent.
        Descends the single-child chain until it reaches the shell."""
        if self._shell_pid is not None:
            return self._shell_pid
        assert self._process
        try:
            proc = psutil.Process(self._process.pid)
            for _ in range(5):
                if proc.cmdline() == self._command:
                    self._shell_pid = proc.pid
                    return proc.pid
                children = proc.children()
                if len(children) != 1:
                    return None
                proc = children[0]
        except psutil.Error:
            return None
        return None

    def _child_pgids(self) -> set[int]:
        """Process groups of the shell's live children."""
        shell_pid = self._resolve_shell_pid()
        if shell_pid is None:
            return set()
        try:
            children = psutil.Process(shell_pid).children()
        except psutil.Error:
            return set()
        pgids: set[int] = set()
        for child in children:
            try:
                pgids.add(os.getpgid(child.pid))
            except (ProcessLookupError, PermissionError):
                continue
        return pgids

    def _background_pgids(self) -> set[int]:
        """Pgids of background jobs still running after a command finishes.

        Session-wide, because a backgrounded job reparents to init and is no
        longer a child. Falls back to children when the shell shares our
        session, where a session scan would catch the harness too.
        """
        assert self._process
        if not self._leads_own_session():
            return self._child_pgids() - {os.getpgid(0)}
        pid = self._process.pid
        return self._session_pgids(pid) - {pid, os.getpgid(0)}

    def _leads_own_session(self) -> bool:
        assert self._process
        try:
            return os.getsid(self._process.pid) == self._process.pid
        except OSError:
            return False

    def _doomed_pgids(
        self, sid: int, session_wide: bool, spare: frozenset[int] | None = None
    ) -> set[int]:
        """Pgids to tear down with the session, minus ``spare``."""
        pgids = self._session_pgids(sid) if session_wide else self._child_pgids()
        return pgids - (spare or set()) - {os.getpgid(0)}

    @staticmethod
    def _session_pgids(sid: int) -> set[int]:
        """Pgids of live processes in session ``sid``.

        Unlike ``_child_pgids``, this also finds background jobs that were
        spawned via a subshell and reparented to init, which keep the shell's
        session id."""
        pgids: set[int] = set()
        for proc in psutil.process_iter():
            try:
                if os.getsid(proc.pid) == sid:
                    pgids.add(os.getpgid(proc.pid))
            except (OSError, psutil.Error):
                continue
        return pgids

    @staticmethod
    def _signal_pgids(pgids: set[int], sig: signal.Signals) -> None:
        for pgid in pgids:
            try:
                os.killpg(pgid, sig)
            except OSError:
                continue

    def _foreground_pgids(self) -> set[int]:
        """Process groups belonging to the currently running command."""
        shell_pid = self._resolve_shell_pid()
        if shell_pid is None:
            return set()
        exclude = self._protected_pgids | {os.getpgid(0)}
        try:
            exclude.add(os.getpgid(shell_pid))
        except ProcessLookupError:
            return set()
        return self._child_pgids() - exclude

    async def _interrupt_foreground(self) -> tuple[str, str, int | None] | None:
        """Kill the hung foreground command's process group(s), sparing the
        session and background jobs from earlier commands.

        Returns the command's output once the shell recovers, or None if it
        does not (e.g. the hang is in the shell itself)."""
        for attempt, sig in enumerate((signal.SIGTERM, signal.SIGKILL)):
            targets = self._foreground_pgids()
            if not targets and attempt == 0:
                await asyncio.sleep(self._output_delay_s)
                targets = self._foreground_pgids()
            if not targets:
                return None
            self._signal_pgids(targets, sig)
            result = await self._poll_for_completion(self._command_kill_timeout_s)
            if result is not None:
                return result
        return None

    async def _check_syntax(self, command: str) -> str | None:
        """Syntax-check a command without running it.

        Returns the error message if the command is syntactically invalid or
        incomplete, otherwise None. Uses `bash -n`, which parses the command but
        does not execute it.

        The check runs in a fresh process, which would otherwise miss parse-time
        shell options (e.g. extglob) the live session enabled in an earlier call.
        Bash enables the options listed in $BASHOPTS/$SHELLOPTS at startup, so we
        forward the session's last-observed values to keep the check in sync.
        """
        env = os.environ.copy()
        if self._bashopts is not None:
            env["BASHOPTS"] = self._bashopts
        if self._shellopts is not None:
            env["SHELLOPTS"] = self._shellopts
        proc = await asyncio.create_subprocess_exec(
            *self._command,
            "-n",
            "-c",
            command,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        _, stderr = await proc.communicate()
        if proc.returncode == 0:
            return None
        return stderr.decode("utf-8", errors="replace").strip()

    def _epilogue(self) -> str:
        """Shell lines appended after every command to emit the exit marker.

        The marker goes to the marker fd rather than fd 1, so a command that
        redirects the shell's own stdout can not swallow it, and falls back to
        fd 1 if that write fails — the marker fd is only ours by convention, and
        an environment whose open-file limit is below it never had it at all.
        `2>&-` comes first so the failed write stays quiet, and `|| :` keeps
        errexit out of it.

        The marker line also carries the session's current parse-time shell
        options ($BASHOPTS and $SHELLOPTS, both space-free, colon-separated
        lists) so the next call's syntax check can reproduce them. See
        _check_syntax.
        """
        marker = (
            f"'{self._exit_marker} {self._marker_nonce}' $__karotte_ec"
            ' "$BASHOPTS" "$SHELLOPTS"'
        )
        return (
            "__karotte_ec=$?\n"
            f"echo {marker} 2>&- >&{_MARKER_FD} || echo {marker} 2>&- || :\n"
        )

    async def run(self, command: str, timeout_s: float) -> ToolResult:
        """Execute a command in the bash shell."""
        restarted = await self._ensure_process_alive()
        if restarted:
            # Let user know we auto-restarted
            system_msg = "Session was automatically restarted."
        else:
            system_msg = ""

        assert self._process
        assert self._process.stdin

        command = command.rstrip()

        # Fail fast on syntactically incomplete input (unterminated quotes,
        # heredocs whose delimiter never matches, etc). Sending such a command to
        # the persistent shell would leave it waiting at its continuation prompt,
        # swallowing the exit marker and blocking until the timeout.
        syntax_error = await self._check_syntax(command)
        if syntax_error is not None:
            return _bash_tool_result(
                stderr=syntax_error, system=system_msg, exit_code=2
            )

        if command.endswith("&"):
            command = f"({command})"

        self._marker_nonce = secrets.token_hex(8)
        full_command = f"{command}\n{self._epilogue()}"

        try:
            self._process.stdin.write(full_command.encode())
            await self._process.stdin.drain()
        except (OSError, RuntimeError):
            stdout, stderr, exit_code = await self._handle_dead_stdin()
            return self._truncated_result(stdout, stderr, system_msg, exit_code)

        # Read output
        try:
            stdout, stderr, restarted, exit_code = await self._read_until_exit_marker(
                timeout_s
            )

            if restarted:
                system_msg = "Session was automatically restarted."
            elif self._started:
                self._protected_pgids = self._background_pgids()

            return self._truncated_result(stdout, stderr, system_msg, exit_code)

        except asyncio.CancelledError:
            # The command is still running in the shell; a fresh session keeps
            # its leftovers out of the next call.
            await self.stop(spare_background=True)
            await self.start()
            raise
        except Exception as e:
            await self.start()
            raise Exception(
                f"Unexpected error in bash session: {e}. Session restarted."
            ) from e

    def _truncated_result(
        self, stdout: str, stderr: str, system_msg: str, exit_code: int | None
    ) -> ToolResult:
        if len(stdout) > self.max_output_length:
            stdout = stdout[: self.max_output_length] + "..."
            system_msg += (
                f"stdout was truncated to {self.max_output_length} characters."
            )

        if len(stderr) > self.max_output_length:
            stderr = stderr[: self.max_output_length] + "..."
            system_msg += (
                f"stderr was truncated to {self.max_output_length} characters."
            )

        return _bash_tool_result(
            stdout=stdout,
            stderr=stderr,
            system=system_msg,
            exit_code=exit_code,
        )


@final
class bash(ToolBase[BashConfig]):
    """A tool that allows the model to run bash commands."""

    config_schema = BashConfig

    _session: _BashSession | None
    _lock: asyncio.Lock

    def __init__(self):
        super().__init__()
        self._session = None
        self._lock = asyncio.Lock()

        self._set_docstring()

    async def __call__(
        self,
        *,
        command: str | None = None,
        restart: bool = False,
        timeout_s: Annotated[float, Field(gt=0, allow_inf_nan=False)] | None = None,
    ) -> ToolResult:
        """THIS DOCSTRING IS SET DYNAMICALLY BELOW.
        f-strings dont work in docstrings, so we need to set it dynamically."""

        timeout_s = (
            min(timeout_s, self.config.default_timeout_s)
            if timeout_s is not None
            else self.config.default_timeout_s
        )

        if restart and self._session is not None and self._lock.locked():
            await self._session.kill()

        async with self._lock:
            if restart:
                if self._session:
                    try:
                        await self._session.stop()
                    except KeyboardInterrupt:
                        raise
                    except Exception:
                        pass  # Ignore errors during shutdown
                self._session = _BashSession(
                    max_output_length=self.config.max_output_length,
                    disable_networking=self.config.disable_networking,
                    python_venv=self.config.python_venv,
                )
                await self._session.start()
                return _bash_tool_result(
                    system="Tool has been manually restarted.",
                )

            if self._session is None:
                self._session = _BashSession(
                    max_output_length=self.config.max_output_length,
                    disable_networking=self.config.disable_networking,
                    python_venv=self.config.python_venv,
                )
                await self._session.start()

            if command:
                return await self._session.run(command, timeout_s)

            raise ValueError("No command provided.")

    async def dispose(self):
        """Dispose of the Bash session."""
        if self._session:
            try:
                await self._session.stop()
            except Exception:
                pass
            finally:
                self._session = None

    def _set_docstring(self):
        bash.__call__.__doc__ = dedent(f"""
            Execute a Bash `command`.

            The Bash shell session persists between calls.
            Set `restart` to True to restart the session; this also interrupts a command still running from an earlier call.

            `timeout_s` is the maximum time in seconds to wait for the command to complete.
            The timeout can not be extended beyond the default ({self.config.default_timeout_s}s).
            The maximum output length of a bash tool in characters is {self.config.max_output_length}.
            """)
