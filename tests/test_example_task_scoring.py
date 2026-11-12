import json
import subprocess
import sys
from pathlib import Path

import pytest

from karotte.templates import TEMPLATES_DIR

SCORING_SCRIPT = (
    TEMPLATES_DIR / "default/src/environment/tasks/example_task/scoring_script.py"
)


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
    ["3.12.11", "3.12.11\n", "3.12.11\r\n", "Python 3.12.11", "Python 3.12.11\n"],
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
