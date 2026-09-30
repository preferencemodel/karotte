import json

import pytest

from karotte.cli.tasks import list_tasks


def test_list_tasks_human_readable_output(capsys: pytest.CaptureFixture[str]):
    list_tasks(json_output=False)

    stdout, _ = capsys.readouterr()
    assert "Available tasks:" in stdout
    assert "  - " in stdout


def test_list_tasks_json_output(capsys: pytest.CaptureFixture[str]):
    list_tasks(json_output=True)

    stdout, _ = capsys.readouterr()
    tasks = json.loads(stdout)

    assert isinstance(tasks, list)
    assert len(tasks) > 0

    task = tasks[0]
    assert "id" in task
    assert "tools" in task
    assert isinstance(task["id"], str)
    assert isinstance(task["tools"], list)
