import codecs
import io
import json
import os
import select
import signal
import subprocess
import sys
import tempfile
import time
from typing import Final, TextIO, override

from loguru import logger

from karotte.judges.judge import Judge
from karotte.schemas.scoring import Scoring
from karotte.schemas.transcript import Transcript

_HEARTBEAT_INTERVAL_S: Final = 30.0
_EXIT_CHECK_INTERVAL_S: Final = 1.0
_DRAIN_AFTER_EXIT_S: Final = 10.0
_READ_SIZE: Final = 65536


class ExecutableJudge(Judge):
    """
    Runs an arbitrary executable from the model's working
    directory.

    Example use cases:
    1. Ask student to train a RL agent on CartPolev1 and save the agent to
    to 'agent.pt2'. Then you can use ExecutableJudge with a custom testing
    script to load this agent, run it on the CartPole environment, and
    evaluate performance.
    2. Ask model to curate a subset of a dataset for improved training to
    target a specific capability (i.e. subset CommonCrawl for math capabilities).
    Ask the model to save the dataset subset to a local file. Test by running
    your own optimized training script on this dataset subset, and then
    evaling however you see fit. (This way the model can 'submit' a large
    amount of data without using a lot of tokens to do so).

    Note this judge does not look at the model transcript at all.
    """

    def __init__(
        self,
        subprocess_run_args: list[str],
        continue_threshold: float = -1,
        cwd: str | None = None,
    ) -> None:
        """
        subprocess_run_args should contain all args for your executable
        (including the executable itself)
        i.e., if your testing executable is a python script inside /src/environment:

        subprocess_run_args =
        [
          sys.executable,
          '-m',
          'environment.my_testing_script',
          'my_arg1',
          'executable_output_file.json',
        ]
        IMPORTANT NOTE 1:
        Note the usage of 'sys.executable' above instead of 'python' - this ensures
        your script is with the correct python interpreter (in the correct virtual
        environment).

        IMPORTANT NOTE 2:
        Your executable should always take a filepath (executable_output_file.json) as its final
        argument, and then save the testing results to that file in the following format:
        { 'score' : <score>,
          'metadata': <metadata dictionary> }
        ExecutableJudge will in fact modify this path to point to a file
        within a temporary directory, so that it is deleted after judging is finished.

        Note that the values of the metadata dictionary must support .__str__()

        The task is continued if the score is greater than or equal to continue_threshold.
        """
        self.subprocess_run_args: Final = subprocess_run_args
        self.continue_threshold: Final = continue_threshold
        self.cwd: Final = cwd

    @override
    def evaluate(self, transcript: Transcript) -> Scoring:
        try:
            # Fails with ENOSPC if the temp filesystem is full (e.g. the
            # student filled /tmp).
            temp_dir_ctx = tempfile.TemporaryDirectory()
        except OSError as e:
            return Scoring(0, {"tempdir_error": str(e)}, False)

        with temp_dir_ctx as temp_dir:
            self.subprocess_run_args[-1] = (
                f"{temp_dir}/{os.path.basename(self.subprocess_run_args[-1])}"
            )

            try:
                process = subprocess.Popen(
                    self.subprocess_run_args,
                    cwd=self.cwd,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    process_group=0,
                )
                try:
                    stdout, stderr = _read_output(process)
                finally:
                    _kill_process_group(process.pid)
                return_code = process.wait()
                if return_code != 0:
                    raise subprocess.CalledProcessError(
                        return_code, self.subprocess_run_args, stdout, stderr
                    )
            except subprocess.CalledProcessError as e:
                score = 0
                metadata = {
                    "CalledProcessError": str(e),
                    "stderr": e.stderr if hasattr(e, "stderr") else "",
                    "stdout": e.stdout if hasattr(e, "stdout") else "",
                }
                continue_task = False
                return Scoring(score, metadata, continue_task)
            except FileNotFoundError as e:
                score = 0
                metadata = {"FileNotFoundError": str(e)}
                continue_task = False
                return Scoring(score, metadata, continue_task)

            try:
                with open(self.subprocess_run_args[-1]) as f:
                    results = json.load(f)
                    continue_task = results["score"] >= self.continue_threshold
                    metadata = {k: str(v) for k, v in results["metadata"].items()}
                    metadata["stdout"] = stdout
                    metadata["stderr"] = stderr
                    return Scoring(
                        results["score"],
                        metadata,
                        continue_task,
                    )
            except (json.JSONDecodeError, UnicodeDecodeError, OSError, KeyError) as e:
                score = 0
                metadata = {"json_read_error": str(e)}
                continue_task = False
                return Scoring(score, metadata, continue_task)


class _Pipe:
    """One of the scoring script's output pipes, echoed and collected as text."""

    def __init__(self, stream: io.TextIOWrapper, echo: TextIO) -> None:
        self.fd: Final = stream.fileno()
        os.set_blocking(self.fd, False)
        self._decoder: Final = io.IncrementalNewlineDecoder(
            codecs.getincrementaldecoder(stream.encoding)(stream.errors or "strict"),
            translate=True,
        )
        self._echo: Final = echo
        self._parts: Final[list[str]] = []
        self.at_eof: bool = False

    def read(self) -> bool:
        """Returns whether there was anything to read, counting EOF."""
        if self.at_eof:
            return False
        try:
            chunk = os.read(self.fd, _READ_SIZE)
        except BlockingIOError:
            return False
        self.at_eof = not chunk
        self._take(self._decoder.decode(chunk, final=self.at_eof))
        return True

    def text(self) -> str:
        if not self.at_eof:
            # Drop a character cut short before EOF, but keep a pending "\r".
            _, flag = self._decoder.getstate()
            self._decoder.setstate((b"", flag))
            self._take(self._decoder.decode(b"", final=True))
        return "".join(self._parts)

    def _take(self, text: str) -> None:
        if text:
            self._echo.write(text)
            self._echo.flush()
            self._parts.append(text)


def _read_output(process: subprocess.Popen[str]) -> tuple[str, str]:
    """Echo and collect the script's output until it exits, without waiting for
    EOF, which a process it started can hold off."""
    assert isinstance(process.stdout, io.TextIOWrapper)
    assert isinstance(process.stderr, io.TextIOWrapper)
    stdout = _Pipe(process.stdout, sys.stdout)
    stderr = _Pipe(process.stderr, sys.stderr)
    pipes = {stdout.fd: stdout, stderr.fd: stderr}
    started_at = time.time()
    quiet_since = time.monotonic()

    while process.poll() is None:
        fds = [fd for fd, pipe in pipes.items() if not pipe.at_eof]
        if not fds:
            break  # Both pipes are closed; only the exit is left to wait for.
        readable, _, _ = select.select(fds, [], [], _EXIT_CHECK_INTERVAL_S)
        reads = [pipes[fd].read() for fd in readable]  # every one, not any()
        if any(reads):
            quiet_since = time.monotonic()
        elif time.monotonic() - quiet_since >= _HEARTBEAT_INTERVAL_S:
            logger.info(
                "Scoring process heartbeat: PID={}, elapsed={:.0f}s",
                process.pid,
                time.time() - started_at,
            )
            quiet_since = time.monotonic()

    drain_until = time.monotonic() + _DRAIN_AFTER_EXIT_S
    while time.monotonic() < drain_until:
        reads = [pipe.read() for pipe in pipes.values()]  # every one, not any()
        if not any(reads):
            break
    return stdout.text(), stderr.text()


def _kill_process_group(pgid: int) -> None:
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
