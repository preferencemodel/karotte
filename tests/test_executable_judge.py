import errno
import os
import select
import signal
import sys
import textwrap
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

import karotte.judges.executable_judge
from karotte.schemas.scoring import Scoring
from karotte.schemas.transcript import Transcript


def test_continue(transcript: Transcript) -> None:
    test_dir_abs_path = os.path.dirname(os.path.abspath(__file__))

    subprocess_run_args = [
        "python",
        f"{test_dir_abs_path}/resources/executable_judge_testing_script.py",
        "--color",
        "green",
        "test_continue.json",
    ]

    executable_judge = karotte.judges.executable_judge.ExecutableJudge(
        subprocess_run_args,
        continue_threshold=42,
        cwd=f"{test_dir_abs_path}/resources/test_student_data",
    )

    evaluation = executable_judge.evaluate(transcript)
    assert evaluation.score == 42
    assert evaluation.metadata["color"] == "green"
    assert (
        evaluation.metadata["number"] == "123"
    )  # make sure int 42 was converted to str
    assert evaluation.continue_task
    assert not os.path.exists("test_continue.json")


def test_no_continue(transcript: Transcript) -> None:
    test_dir_abs_path = os.path.dirname(os.path.abspath(__file__))

    subprocess_run_args = [
        "python",
        f"{test_dir_abs_path}/resources/executable_judge_testing_script.py",
        "test_no_continue.json",
    ]

    executable_judge = karotte.judges.executable_judge.ExecutableJudge(
        subprocess_run_args,
        continue_threshold=43,
        cwd=f"{test_dir_abs_path}/resources/test_student_data",
    )

    evaluation = executable_judge.evaluate(transcript)
    assert evaluation.score == 42
    assert evaluation.metadata["color"] == "blue"
    assert evaluation.metadata["number"] == "123"
    assert not evaluation.continue_task
    assert not os.path.exists("test_no_continue.json")


def test_subprocess_failure(transcript: Transcript) -> None:
    subprocess_run_args = [
        "whoami",
        "jack black",
    ]

    executable_judge = karotte.judges.executable_judge.ExecutableJudge(
        subprocess_run_args, continue_threshold=1
    )

    evaluation = executable_judge.evaluate(transcript)
    assert evaluation.score == 0
    assert not evaluation.continue_task
    assert "CalledProcessError" in evaluation.metadata
    assert "stderr" in evaluation.metadata
    assert not evaluation.metadata["stderr"] == ""
    assert "stdout" in evaluation.metadata
    assert evaluation.metadata["stdout"] == ""


def test_invalid_subprocess_command(transcript: Transcript) -> None:
    subprocess_run_args = [
        "asdfjk",
        "ajskdflj",
        "asjdkf",
    ]

    executable_judge = karotte.judges.executable_judge.ExecutableJudge(
        subprocess_run_args, continue_threshold=1
    )

    evaluation = executable_judge.evaluate(transcript)
    assert evaluation.score == 0
    assert not evaluation.continue_task
    assert "FileNotFoundError" in evaluation.metadata


def test_invalid_json(transcript: Transcript) -> None:
    test_dir_abs_path = os.path.dirname(os.path.abspath(__file__))
    subprocess_run_args = [
        "python",
        f"{test_dir_abs_path}/resources/executable_judge_testing_script.py",
        "--invalid_json",
        "test_invalid_subprocess_run_args.json",
    ]

    executable_judge = karotte.judges.executable_judge.ExecutableJudge(
        subprocess_run_args,
        continue_threshold=1,
        cwd=f"{test_dir_abs_path}/resources/test_student_data",
    )

    evaluation = executable_judge.evaluate(transcript)
    assert evaluation.score == 0
    assert not evaluation.continue_task
    assert "json_read_error" in evaluation.metadata
    assert not os.path.exists("test_invalid_subprocess_run_args.json")


def test_select_timeout_does_not_hang(transcript: Transcript) -> None:
    """select() timeout (as happens under gvisor) should not prevent completion."""
    test_dir_abs_path = os.path.dirname(os.path.abspath(__file__))

    subprocess_run_args = [
        "python",
        f"{test_dir_abs_path}/resources/executable_judge_testing_script.py",
        "--print-message",
        "gvisor test",
        "test_select_timeout.json",
    ]

    executable_judge = karotte.judges.executable_judge.ExecutableJudge(
        subprocess_run_args,
        continue_threshold=0,
        cwd=f"{test_dir_abs_path}/resources/test_student_data",
    )

    original_select = select.select
    timeout_count = 0

    def mock_select(
        rlist: Any, wlist: Any, xlist: Any, timeout: Any = None
    ) -> tuple[list[Any], list[Any], list[Any]]:
        nonlocal timeout_count
        # Return empty (simulating gvisor timeout) for the first 3 calls
        if timeout_count < 3:
            timeout_count += 1
            return ([], [], [])
        return original_select(rlist, wlist, xlist, timeout)

    with patch("select.select", side_effect=mock_select):
        evaluation = executable_judge.evaluate(transcript)

    assert timeout_count == 3
    assert evaluation.score == 42
    assert "gvisor test" in evaluation.metadata["stdout"]


def test_tempdir_creation_failure_returns_score_zero(transcript: Transcript) -> None:
    """A full /tmp must yield a score-0 Scoring, not crash the run."""
    executable_judge = karotte.judges.executable_judge.ExecutableJudge(
        ["whoami", "out.json"], continue_threshold=1
    )

    with patch(
        "tempfile.TemporaryDirectory",
        side_effect=OSError(errno.ENOSPC, "No space left on device"),
    ):
        evaluation = executable_judge.evaluate(transcript)

    assert evaluation.score == 0
    assert not evaluation.continue_task
    assert "No space left on device" in evaluation.metadata["tempdir_error"]


def test_stdout_captured_on_success(transcript: Transcript) -> None:
    """Test that stdout from successful scoring script is captured in metadata."""
    test_dir_abs_path = os.path.dirname(os.path.abspath(__file__))

    subprocess_run_args = [
        "python",
        f"{test_dir_abs_path}/resources/executable_judge_testing_script.py",
        "--print-message",
        "Hello from scoring script",
        "test_stdout.json",
    ]

    executable_judge = karotte.judges.executable_judge.ExecutableJudge(
        subprocess_run_args,
        continue_threshold=0,
        cwd=f"{test_dir_abs_path}/resources/test_student_data",
    )

    evaluation = executable_judge.evaluate(transcript)
    assert evaluation.score == 42
    assert "stdout" in evaluation.metadata
    assert "Hello from scoring script" in evaluation.metadata["stdout"]


_EVALUATE_BOUND_S = 10.0


@pytest.fixture
def pid_file(tmp_path: Path) -> Iterator[Path]:
    """Pids of the scoring script and what it starts; the latter are killed afterwards."""
    path = tmp_path / "pids"
    yield path
    _kill_recorded(path, script=False)


def _kill_recorded(pid_file: Path, *, script: bool) -> None:
    """The script's own pid, the first, only with ``script``: once reaped, it can be reused."""
    pids = pid_file.read_text().split() if pid_file.exists() else []
    for line in pids if script else pids[1:]:
        try:
            os.kill(int(line), signal.SIGKILL)
        except ProcessLookupError:
            pass


def _scoring_script(tmp_path: Path, body: str) -> Path:
    """Records its pid in argv[1], runs ``body`` and writes score 1 to argv[-1]."""
    script = tmp_path / "scoring_script.py"
    script.write_text(
        "import json, os, subprocess, sys\n"
        + "pids = open(sys.argv[1], 'a')\n"
        + "print(os.getpid(), file=pids, flush=True)\n"
        + textwrap.dedent(body)
        + "\njson.dump({'score': 1, 'metadata': {}}, open(sys.argv[-1], 'w'))\n"
    )
    return script


def _evaluate_within_bound(
    judge: karotte.judges.executable_judge.ExecutableJudge,
    transcript: Transcript,
    pid_file: Path,
) -> Scoring:
    scorings: list[Scoring] = []
    errors: list[BaseException] = []

    def evaluate() -> None:
        try:
            scorings.append(judge.evaluate(transcript))
        except BaseException as e:
            errors.append(e)

    thread = threading.Thread(target=evaluate, daemon=True)
    thread.start()
    thread.join(_EVALUATE_BOUND_S)
    hung = thread.is_alive()
    if hung:
        # Free the stuck judge so it can't outlive the test.
        _kill_recorded(pid_file, script=True)
        thread.join(_EVALUATE_BOUND_S)
    assert not hung, f"evaluate() did not return within {_EVALUATE_BOUND_S}s"
    if errors:
        raise errors[0]
    return scorings[0]


def test_returns_when_the_script_exits_but_its_child_holds_the_pipes(
    transcript: Transcript, tmp_path: Path, pid_file: Path
) -> None:
    script = _scoring_script(
        tmp_path,
        """
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(3600)"])
        print(child.pid, file=pids, flush=True)
        print("tested")
        """,
    )
    judge = karotte.judges.executable_judge.ExecutableJudge(
        [sys.executable, str(script), str(pid_file), "out.json"]
    )

    evaluation = _evaluate_within_bound(judge, transcript, pid_file)

    assert evaluation.score == 1
    assert "tested" in evaluation.metadata["stdout"]


def test_returns_when_the_pipes_are_left_mid_character(
    transcript: Transcript, tmp_path: Path, pid_file: Path
) -> None:
    script = _scoring_script(
        tmp_path,
        """
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(3600)"])
        print(child.pid, file=pids, flush=True)
        sys.stdout.write("tested "); sys.stdout.flush()
        sys.stdout.buffer.write("€".encode()[:2]); sys.stdout.flush()
        """,
    )
    judge = karotte.judges.executable_judge.ExecutableJudge(
        [sys.executable, str(script), str(pid_file), "out.json"]
    )

    evaluation = _evaluate_within_bound(judge, transcript, pid_file)

    assert evaluation.score == 1
    assert evaluation.metadata["stdout"] == "tested "


@pytest.mark.parametrize(
    "partial, flooded", [("stdout", "stderr"), ("stderr", "stdout")]
)
def test_does_not_deadlock_on_a_partial_line_while_the_other_pipe_fills(
    transcript: Transcript, tmp_path: Path, pid_file: Path, partial: str, flooded: str
) -> None:
    script = _scoring_script(
        tmp_path,
        f"""
        sys.{partial}.write("working"); sys.{partial}.flush()
        sys.{flooded}.write(("x" * 99 + "\\n") * 10_000); sys.{flooded}.flush()
        sys.{partial}.write(" done\\n"); sys.{partial}.flush()
        """,
    )
    judge = karotte.judges.executable_judge.ExecutableJudge(
        [sys.executable, str(script), str(pid_file), "out.json"]
    )

    evaluation = _evaluate_within_bound(judge, transcript, pid_file)

    assert evaluation.score == 1
    assert "working done" in evaluation.metadata[partial]
    assert evaluation.metadata[flooded].count("x") == 990_000


def test_reads_what_the_script_wrote_when_select_never_reports_it(
    transcript: Transcript, tmp_path: Path, pid_file: Path
) -> None:
    script = _scoring_script(tmp_path, 'print("scored")')
    judge = karotte.judges.executable_judge.ExecutableJudge(
        [sys.executable, str(script), str(pid_file), "out.json"]
    )

    with patch("select.select", return_value=([], [], [])):
        evaluation = _evaluate_within_bound(judge, transcript, pid_file)

    assert evaluation.score == 1
    assert "scored" in evaluation.metadata["stdout"]


def test_kills_what_the_script_leaves_running(
    transcript: Transcript, tmp_path: Path, pid_file: Path
) -> None:
    script = _scoring_script(
        tmp_path,
        """
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(3600)"])
        print(child.pid, file=pids, flush=True)
        """,
    )
    judge = karotte.judges.executable_judge.ExecutableJudge(
        [sys.executable, str(script), str(pid_file), "out.json"]
    )

    evaluation = _evaluate_within_bound(judge, transcript, pid_file)

    assert evaluation.score == 1
    child = int(pid_file.read_text().split()[1])
    deadline = time.monotonic() + _EVALUATE_BOUND_S
    while _is_running(child) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _is_running(child)


def _is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_does_not_hand_the_script_its_stdin(
    transcript: Transcript, tmp_path: Path, pid_file: Path
) -> None:
    script = _scoring_script(tmp_path, "print(repr(sys.stdin.read()))")
    judge = karotte.judges.executable_judge.ExecutableJudge(
        [sys.executable, str(script), str(pid_file), "out.json"]
    )
    read_end, write_end = os.pipe()
    saved_stdin = os.dup(0)
    os.dup2(read_end, 0)
    try:
        evaluation = _evaluate_within_bound(judge, transcript, pid_file)
    finally:
        os.dup2(saved_stdin, 0)
        for fd in (saved_stdin, read_end, write_end):
            os.close(fd)

    assert evaluation.score == 1
    assert "''" in evaluation.metadata["stdout"]
