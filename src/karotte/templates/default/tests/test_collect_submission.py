"""Every task copied from `_template` collects its submission, in order.

`just create-task` copies `_template`, so whatever it does here is what every
new task does by default: `collect_submission` kills the student, frees disk
space, copies the submission somewhere root-only, and deletes the original.
"""

from pathlib import Path

import pytest
from karotte import EvaluationRunConfig

from environment import submissions
from environment.tasks._template import FirstStep, Task_


def _config() -> EvaluationRunConfig:
    return EvaluationRunConfig(
        run_id="test", task_id="", model="test-model", model_api_key="test-key"
    )


@pytest.fixture
def collected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Record the calls instead of killing and deleting for real."""
    calls: list[tuple[str, object]] = []

    def kill(uid: int) -> None:
        calls.append(("kill", uid))

    def delete(uid: int, **kwargs: object) -> None:
        calls.append(("delete", kwargs.get("extend_exclude")))

    def save(source: Path, **kwargs: object) -> Path:
        calls.append(("save", (source, kwargs)))
        return tmp_path

    monkeypatch.setattr(submissions, "kill_processes", kill)
    monkeypatch.setattr(submissions, "delete_files", delete)
    monkeypatch.setattr(submissions, "save_submission", save)

    def artifact(config: object, path: Path) -> None:
        calls.append(("artifact", path))

    monkeypatch.setattr(submissions, "save_artifact", artifact)
    return calls


def test_step_collects_in_the_documented_order(collected, tmp_path: Path):
    step = FirstStep(config=_config())

    step.pre_scoring_hook()

    assert [call[0] for call in collected] == [
        "kill",
        "delete",
        "save",
        "delete",
        "artifact",
    ]
    submission = step.submission_paths[0]
    assert collected[1][1] == (submission,)
    assert collected[2][1] == (submission, {})
    assert collected[3][1] is None
    assert collected[4][1] == tmp_path / submission.name
    assert step.saved_submissions == (tmp_path / submission.name,)


def test_a_submission_the_student_never_wrote_is_not_misbehavior(tmp_path: Path):
    """`save_submission` hands back a copy that doesn't exist rather than
    raising, so the scorer gets to say what it's worth."""
    from karotte import save_submission

    saved = save_submission(tmp_path / "never_written.txt", tmp_path / "copy.txt")

    assert not saved.exists()


def test_task_reports_the_submission_paths_of_its_steps():
    step = FirstStep(config=_config())
    assert Task_(_config()).submission_paths == step.submission_paths
