import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from karotte.judges.regex_judge import RegexJudge
from karotte.schemas.chat import Message
from karotte.schemas.transcript import MessageAddedEvent, Transcript
from karotte.templates import TEMPLATES_DIR

EXAMPLE_TASK = TEMPLATES_DIR / "default/src/environment/tasks/example_task"
SCORING_SCRIPT = EXAMPLE_TASK / "scoring_script.py"


def _score(tmp_path: Path, answer: str) -> float:
    submission = tmp_path / "python_version.txt"
    submission.write_text(answer)
    output = tmp_path / "score_output.txt"
    subprocess.run(
        [sys.executable, str(SCORING_SCRIPT), str(submission), str(output)], check=True
    )
    return json.loads(output.read_text())["score"]


@pytest.mark.parametrize(
    "answer",
    ["3.12.11", "3.12.11\n", "3.12.11\r\n"],
)
def test_scores_the_python_version(tmp_path: Path, answer: str):
    assert _score(tmp_path, answer) == 1.0


@pytest.mark.parametrize(
    "answer",
    [
        "",
        "3.12.110",
        "13.12.11",
        "3.12.1",
        "3x12x11",
        "3.10 3.11 3.12.11 3.13",
        "3.12.10\n3.12.11\n3.12.12",
        "Python 3.12.11",
        "Python 3.12.11\n",
        "python 3.12.11",
        "Python 3.12.11 or Python 3.13.0",
    ],
)
def test_does_not_score_a_wrong_answer_or_a_list_of_guesses(
    tmp_path: Path, answer: str
):
    assert _score(tmp_path, answer) == 0.0


def test_docs_show_the_scoring_script():
    docs = Path(__file__).resolve().parent.parent / "docs/tasks/scoring.md"
    assert SCORING_SCRIPT.read_text(encoding="utf-8") in docs.read_text(
        encoding="utf-8"
    )


# The first step is scored by a RegexJudge. Its module imports the template's
# `environment` package, so read the step's instructions and pattern from the
# source instead of importing it.


def _get_python_path_step_argument(function: str) -> str:
    """The string literal that `GetPythonPathStep` passes to `function`."""
    tree = ast.parse((EXAMPLE_TASK / "__init__.py").read_text(encoding="utf-8"))
    [step] = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "GetPythonPathStep"
    ]
    [call] = [
        node
        for node in ast.walk(step)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == function
    ]
    [argument] = call.args
    return ast.literal_eval(argument)


def _path_score(*replies: str) -> float:
    judge = RegexJudge([re.compile(_get_python_path_step_argument("re.compile"))])
    transcript = Transcript(run_id="run_id")
    transcript.events.extend(
        MessageAddedEvent(message=Message(content=reply)) for reply in replies
    )
    return judge.evaluate(transcript).score


def test_path_step_asks_to_end_with_the_answer():
    assert "End your final message with your answer" in (
        _get_python_path_step_argument("dedent")
    )


@pytest.mark.parametrize(
    "replies",
    [
        ["path: /workdir/.venv/bin/python"],
        ["path: /workdir/.venv/bin/python3.12\n"],
        ["`which python3` prints it.\n\npath: /workdir/.venv/bin/python3."],
        ["Let me check.", "path: /workdir/.venv/bin/python"],
    ],
)
def test_path_step_scores_a_final_message_ending_with_the_path(replies: list[str]):
    assert _path_score(*replies) == 1.0


@pytest.mark.parametrize(
    "replies",
    [
        pytest.param(["path: /a /b /workdir/.venv/bin/python"], id="spaces"),
        pytest.param(
            ["path: /workdir/.venv/bin/python\npath: /usr/bin/python3"],
            id="several-lines",
        ),
        pytest.param(
            ["path: /workdir/.venv/bin/python", "Actually, path: /usr/bin/python3"],
            id="changed",
        ),
        pytest.param(["path: /usr/bin/python3"], id="wrong"),
    ],
)
def test_path_step_does_not_score_several_paths(replies: list[str]):
    assert _path_score(*replies) == 0.0
