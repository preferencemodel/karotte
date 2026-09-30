import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from karotte.cli.templates import list_templates
from karotte.create_env import create_env


def test_list_templates_human_readable_output(capsys: pytest.CaptureFixture[str]):
    list_templates(json_output=False)

    stdout, _ = capsys.readouterr()
    assert "Available templates:" in stdout


def test_list_templates_json_output(capsys: pytest.CaptureFixture[str]):
    list_templates(json_output=True)

    stdout, _ = capsys.readouterr()
    templates = json.loads(stdout)

    assert isinstance(templates, list)
    assert len(templates) > 0

    template = templates[0]
    assert "id" in template
    assert "description" in template
    assert "requires" in template
    assert isinstance(template["id"], str)
    assert isinstance(template["description"], str)
    assert isinstance(template["requires"], list)


def test_list_templates_in_vendored_copy_fails_with_one_line(tmp_path: Path):
    create_env(
        tmp_path / "env", templates=["default"], vendor_karotte=True, no_lock=True
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from karotte.cli import app; app()",
            "templates",
            "list",
        ],
        env=os.environ | {"PYTHONPATH": str(tmp_path / "env" / ".karotte" / "src")},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "Traceback" not in result.stderr
    assert "vendored copies omit them" in result.stderr
    assert len(result.stderr.strip().splitlines()) == 1
