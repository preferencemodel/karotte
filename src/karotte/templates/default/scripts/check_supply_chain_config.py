"""Ensures all Python projects have a safe, resolvable supply chain config.

Every pyproject.toml must enforce a 7-day supply-chain age-delay
(uv `exclude-newer`), so no build takes a just-published (possibly compromised)
version of a public dependency.

It also rejects three things that make a pyproject **unresolvable**, none of
them visible to a substring scan:

1. The file must parse as TOML. Two `[tool.uv]` tables — what a "keep both sides"
   merge resolution produces — read fine in review and are invalid TOML, so uv
   refuses the file outright.
2. No two `[[tool.uv.index]]` entries may share a `name`. That is *valid* TOML (an
   array of tables permits repeats), so parsing alone passes it, but uv errors with
   `duplicate index name` and the venv cannot build at all.
3. A dependency pinned to a local version (`torch==2.11.0+cu126`, `+cpu`) needs an
   explicit index that serves that build. PyPI publishes no `+cu126`/`+cpu`
   variant, so without one the pin is unsatisfiable and `uv lock` fails with
   `No solution found`.

Deliberately stdlib-only, and run as plain `python3` rather than under `uv run`:
the breakage it detects is exactly what stops uv from building a venv, so it must
work before any venv exists.
"""

import re
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    # Runs as plain `python3`, which may be older than 3.11 (macOS ships 3.9).
    print("ERROR: check_supply_chain_config.py needs Python 3.11+ (tomllib).")
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parent.parent

EXCLUDE_NEWER_KEY = "exclude-newer"
EXCLUDE_NEWER_VALUE = "7 days"

SKIP_DIRS = {".venv", "node_modules", ".karotte"}

LOCAL_VERSION_PIN = re.compile(
    r"^\s*([A-Za-z0-9._-]+)\s*==\s*[^+\s]+\+([A-Za-z0-9._-]+)"
)

# Only *accelerator flavour* local versions need a special index: `+cpu`,
# `+cu126`, Astral's compound `+cu.13.0.torch.2.10`, `+cuda12.cudnn89`, `+rocm6.2`.
# Other local versions, like a git-describe build tag (`+gabc1234`), are served
# by ordinary indexes and must NOT be flagged.
FLAVOUR_LOCAL = re.compile(
    r"^(cpu|xpu|cu\d{2,3}|cu\.\d+\.\d+.*|cuda[\d.]+.*|rocm[\d.]*)$", re.I
)

# Accelerator wheels come from download.pytorch.org, wheels.astral.sh, data.pyg.org
# and storage.googleapis.com/jax-releases today, and elsewhere tomorrow, so any
# explicit index counts rather than an allow-list of hosts. Non-explicit indexes
# are general mirrors of PyPI and serve no flavour builds.


def find_files(name: str) -> list[Path]:
    results = []
    for path in REPO_ROOT.rglob(name):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        results.append(path)
    return sorted(results)


def _uv_table(doc: dict) -> dict:
    tool = doc.get("tool")
    if not isinstance(tool, dict):
        return {}
    uv = tool.get("uv")
    return uv if isinstance(uv, dict) else {}


def _declared_dependencies(doc: dict) -> list[str]:
    project = doc.get("project")
    project = project if isinstance(project, dict) else {}
    deps: list[str] = [
        d for d in (project.get("dependencies") or []) if isinstance(d, str)
    ]
    for extra in (project.get("optional-dependencies") or {}).values():
        if isinstance(extra, list):
            deps.extend(d for d in extra if isinstance(d, str))
    for group in (doc.get("dependency-groups") or {}).values():
        if isinstance(group, list):
            deps.extend(d for d in group if isinstance(d, str))
    return deps


def _check_local_version_pins(rel: Path, doc: dict, indexes: list[dict]) -> list[str]:
    """Flag a local-version pin that no configured index can serve."""
    pinned: dict[str, str] = {}
    for dep in _declared_dependencies(doc):
        if (m := LOCAL_VERSION_PIN.match(dep)) and FLAVOUR_LOCAL.match(m.group(2)):
            pinned[m.group(1)] = m.group(2)
    if not pinned:
        return []
    if any(i.get("explicit") for i in indexes):
        return []
    return [
        f"{rel}: {name} is pinned to a build-specific version (+{tag}) but no "
        f"explicit index is configured. PyPI publishes no +{tag} variant, so "
        f"`uv lock` cannot resolve it. Add the index that serves it with "
        f"`explicit = true` and point {name} at it via [tool.uv.sources]."
        for name, tag in sorted(pinned.items())
    ]


def check_pyproject(pyproject_path: Path) -> list[str]:
    """Return every problem found in one pyproject.toml."""
    rel = pyproject_path.relative_to(REPO_ROOT)
    try:
        doc = tomllib.loads(pyproject_path.read_text())
    except tomllib.TOMLDecodeError as e:
        # Usually a duplicate `[tool.uv]` table from a hand-resolved merge.
        return [f"{rel}: not valid TOML, so uv cannot read it at all — {e}"]

    problems: list[str] = []
    uv = _uv_table(doc)

    if uv.get(EXCLUDE_NEWER_KEY) != EXCLUDE_NEWER_VALUE:
        problems.append(
            f"{rel}: missing supply-chain age-delay "
            f'(`{EXCLUDE_NEWER_KEY} = "{EXCLUDE_NEWER_VALUE}"` in [tool.uv])'
        )

    indexes = [i for i in uv.get("index", []) if isinstance(i, dict)]
    names = [i.get("name") for i in indexes if i.get("name") is not None]
    for name in sorted({n for n in names if names.count(n) > 1}):
        problems.append(
            f"{rel}: two or more [[tool.uv.index]] entries are named {name!r} — "
            "uv rejects this with `duplicate index name`"
        )

    problems.extend(_check_local_version_pins(rel, doc, indexes))
    return problems


def main() -> int:
    errors: list[str] = []

    for pyproject in find_files("pyproject.toml"):
        errors.extend(check_pyproject(pyproject))

    if errors:
        print("Supply chain config errors:")
        for err in errors:
            print(f"  - {err}")
        return 1

    print("All supply chain configs OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
