"""Ensures all Python projects have supply chain security config.

Every Python project must enforce a 7-day supply-chain age-delay via uv's
client-side `exclude-newer`. Workspace members inherit the root's config;
standalone projects need their own.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = REPO_ROOT / "src" / "karotte" / "templates"

EXCLUDE_NEWER = 'exclude-newer = "7 days"'

SKIP_DIRS = {".venv"}


def effective_content(pyproject_path: Path) -> str:
    """Text of the pyproject as it will be consumed.

    Template source pyprojects are Jinja files that inherit the supply-chain
    config from a base template (`{% extends "pyproject.base.toml" %}`), so the
    `exclude-newer` line lives in the base, not the child. Render them so the
    inherited config is visible; plain pyprojects are returned verbatim.
    """
    content = pyproject_path.read_text()
    is_template = TEMPLATES_DIR in pyproject_path.parents
    if not is_template or ("{%" not in content and "{{" not in content):
        return content
    from jinja2 import Environment, FileSystemLoader

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)), keep_trailing_newline=True
    )
    rel = pyproject_path.relative_to(TEMPLATES_DIR)
    return env.get_template(str(rel)).render(env_name="supply-chain-check")


def find_files(name: str) -> list[Path]:
    results = []
    for path in REPO_ROOT.rglob(name):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        results.append(path)
    return sorted(results)


def get_workspace_member_paths() -> set[Path]:
    """Get workspace member paths from uv workspace metadata."""
    # UV_FROZEN stops newer uv from re-locking; older uv (CI) ignores it.
    result = subprocess.run(
        ["uv", "workspace", "metadata", "--preview-features", "workspace-metadata"],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
        env={**os.environ, "UV_FROZEN": "1"},
    )
    metadata = json.loads(result.stdout)
    return {Path(member["path"]) for member in metadata["members"]}


def check_python_index(
    pyproject_path: Path, workspace_members: set[Path]
) -> str | None:
    """Return an error message if the pyproject.toml lacks the age-delay."""
    if pyproject_path.parent in workspace_members:
        return None

    if EXCLUDE_NEWER in effective_content(pyproject_path):
        return None
    rel = pyproject_path.relative_to(REPO_ROOT)
    return f"{rel}: no supply-chain age-delay (add `{EXCLUDE_NEWER}` to [tool.uv])"


def main() -> int:
    errors: list[str] = []

    workspace_members = get_workspace_member_paths()

    # `.jinja` template sources render to a `pyproject.toml`; check them too so
    # every generated env's supply-chain config is verified at its source.
    pyprojects = find_files("pyproject.toml") + find_files("pyproject.toml.jinja")
    for pyproject in sorted(pyprojects):
        if err := check_python_index(pyproject, workspace_members):
            errors.append(err)

    if errors:
        print("Supply chain config errors:")
        for err in errors:
            print(f"  - {err}")
        return 1

    print("All supply chain configs OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
