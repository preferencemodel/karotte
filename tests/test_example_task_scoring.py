import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from karotte.templates import TEMPLATES_DIR

EXAMPLE_TASK = TEMPLATES_DIR / "default/src/environment/tasks/example_task"
SCORING_SCRIPT = EXAMPLE_TASK / "scoring_script.py"
PYTHON_PATH = r"path: /workdir/\.venv/bin/python[\d.]*\s*\Z"


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


@pytest.mark.parametrize(
    "reply",
    [
        "path: /workdir/.venv/bin/python",
        "path: /workdir/.venv/bin/python3.12\n",
        "`which python3` prints it.\n\npath: /workdir/.venv/bin/python3.",
    ],
)
def test_python_path_pattern_matches_a_reply_ending_with_the_path(reply: str):
    assert re.search(PYTHON_PATH, reply)


@pytest.mark.parametrize(
    "reply",
    [
        pytest.param("path: /a /b /workdir/.venv/bin/python", id="spaces"),
        pytest.param("path: /usr/bin/python3,/workdir/.venv/bin/python", id="commas"),
        pytest.param(
            "path: /workdir/.venv/bin/python\npath: /usr/bin/python3", id="lines"
        ),
        pytest.param(
            "path: /workdir/.venv/bin/python\n\nActually, path: /usr/bin/python3",
            id="changed",
        ),
        pytest.param("path: /workdir/.venv/bin/python-is-not/perl", id="suffix"),
        pytest.param("path: /usr/bin/python3", id="wrong"),
    ],
)
def test_python_path_pattern_does_not_match_several_or_wrong_paths(reply: str):
    assert not re.search(PYTHON_PATH, reply)


def test_example_task_uses_the_python_path_pattern():
    assert f'r"{PYTHON_PATH}"' in (EXAMPLE_TASK / "__init__.py").read_text(
        encoding="utf-8"
    )
