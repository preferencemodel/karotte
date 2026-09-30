import ast
from pathlib import Path

import pytest

from karotte.templates import TEMPLATES_DIR, load_templates

SKIPPED_DIRS = {"__pycache__", ".ruff_cache", ".venv"}

COLLECT_ORDER = ["kill_processes", "delete_files", "save_submission", "delete_files"]


def test_all_bundled_templates_have_valid_metadata():
    templates = load_templates(TEMPLATES_DIR)
    assert len(templates) > 0, "No templates found"


def _template_python_files():
    for path in sorted(TEMPLATES_DIR.rglob("*.py")):
        if SKIPPED_DIRS.isdisjoint(path.parts):
            yield pytest.param(path, id=str(path.relative_to(TEMPLATES_DIR)))


@pytest.mark.parametrize("path", _template_python_files())
def test_template_python_files_contain_no_jinja(path: Path):
    """`create_env` renders every template file through Jinja.

    A `.py` file cannot escape Jinja the way the justfiles do (a `{% raw %}` tag
    is a syntax error outside a string), so any `{{ ... }}` in one is silently
    replaced with the empty string on creation. Keep them Jinja-free instead.
    """
    source = path.read_text()
    for delimiter in ("{{", "{%"):
        assert delimiter not in source, (
            f"{path.name} contains the Jinja delimiter {delimiter!r}, which "
            "create_env would substitute away. Build the literal at runtime "
            "instead (see _template_suite/prompt.py)."
        )


def _collect_scopes():
    """Every template class or top-level function that saves a submission."""
    for path in sorted(TEMPLATES_DIR.rglob("*.py")):
        if not SKIPPED_DIRS.isdisjoint(path.parts) or "tests" in path.parts:
            continue
        for node in ast.parse(path.read_text()).body:
            if not isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                continue
            if "save_submission" in _called_names(node):
                yield pytest.param(
                    path, node, id=f"{path.relative_to(TEMPLATES_DIR)}::{node.name}"
                )


def _called_names(node: ast.AST) -> list[str]:
    """Names of plain function calls under `node`, in source order."""
    calls = [
        call
        for call in ast.walk(node)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    ]
    calls.sort(key=lambda call: (call.lineno, call.col_offset))
    return [call.func.id for call in calls]  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.parametrize(("path", "node"), _collect_scopes())
def test_templates_collect_submissions_in_the_documented_order(
    path: Path, node: ast.ClassDef | ast.FunctionDef
):
    """Kill the student first (nothing may rewrite the submission mid-copy),
    free disk space, copy, then delete the original. A template that ships the
    calls in another order teaches every task copied from it the wrong one.
    """
    called = [name for name in _called_names(node) if name in COLLECT_ORDER]
    assert called == COLLECT_ORDER, f"{path.name}::{node.name} saves out of order"


def test_new_task_templates_declare_a_submission():
    """`_template*` is what `just create-task` copies, so the submission wiring
    has to be in it — nobody should have to remember to add it."""
    templates = TEMPLATES_DIR.glob("*/src/environment/tasks/_template*/__init__.py")
    for path in sorted(templates):
        assert "submission_paths" in path.read_text(), (
            f"{path.relative_to(TEMPLATES_DIR)} declares no submission_paths"
        )


def test_template_tasks_name_no_hardware():
    """Hardware names belong to plugins; karotte's own templates know none."""
    tasks = TEMPLATES_DIR.glob("*/src/environment/tasks/*/__init__.py")
    for path in sorted(tasks):
        assert "required_hardware" not in path.read_text(), (
            f"{path.relative_to(TEMPLATES_DIR)} names hardware"
        )
