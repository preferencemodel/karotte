import errno
import os
import select
from typing import Any
from unittest.mock import patch

import karotte.judges.executable_judge
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
