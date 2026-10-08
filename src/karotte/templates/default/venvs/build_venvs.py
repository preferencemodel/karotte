# /// script
# requires-python = ">=3.12"
# ///
"""Build virtual environments from venvs/*/pyproject.toml definitions.

Each pyproject.toml must contain a [tool.karotte] section with:
  - access: "root" | "student:r" | "student:rw"
  - path: absolute path where the venv will be created

Lockfiles (uv.lock) are mandatory — the build always uses --frozen.
"""

import os
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

# Access levels:
#   root       - student cannot see or access (use a path under /root/ to also hide existence)
#   student:r  - student can read/execute but not modify
#   student:rw - student has full access
VALID_ACCESS_LEVELS = {"root", "student:r", "student:rw"}


@dataclass
class VenvSpec:
    """Parsed venv specification from a pyproject.toml."""

    name: str
    project_dir: Path
    path: Path
    access: str
    python_version: str


def parse_venv_spec(project_dir: Path) -> VenvSpec:
    """Parse a venv specification from a pyproject.toml file."""
    pyproject_path = project_dir / "pyproject.toml"

    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    karotte = data.get("tool", {}).get("karotte")
    if karotte is None:
        raise ValueError(f"{pyproject_path}: missing [tool.karotte] section")

    access = karotte.get("access")
    if access is None:
        raise ValueError(f"{pyproject_path}: missing 'access' in [tool.karotte]")
    if access not in VALID_ACCESS_LEVELS:
        raise ValueError(
            f"{pyproject_path}: invalid access '{access}', must be one of {VALID_ACCESS_LEVELS}"
        )

    path = karotte.get("path")
    if path is None:
        raise ValueError(f"{pyproject_path}: missing 'path' in [tool.karotte]")

    path = Path(path)
    if not path.is_absolute():
        raise ValueError(f"{pyproject_path}: 'path' must be absolute, got '{path}'")

    requires_python = data.get("project", {}).get("requires-python")
    if requires_python is None:
        raise ValueError(f"{pyproject_path}: missing 'requires-python' in [project]")
    # Require a pinned version to avoid ambiguity in which Python gets installed.
    # Accepted formats: "==3.12.*" (recommended), "==3.12", "==3.12.11"
    match = re.fullmatch(r"==(\d+\.\d+)(?:\.\d+|\.\*)?", requires_python)
    if not match:
        raise ValueError(
            f"{pyproject_path}: 'requires-python' must be a pinned version (e.g., '==3.12.*'), got '{requires_python}'"
        )
    # Always pass major.minor to uv --python (e.g., "3.12")
    python_version = match.group(1)

    return VenvSpec(
        name=project_dir.name,
        project_dir=project_dir,
        path=path,
        access=access,
        python_version=python_version,
    )


def discover_venv_specs(venvs_dir: Path) -> list[VenvSpec]:
    """Discover all venv specs under a venvs/ directory."""
    specs = []
    for pyproject in sorted(venvs_dir.glob("*/pyproject.toml")):
        specs.append(parse_venv_spec(pyproject.parent))

    # Validate no duplicate paths
    seen_paths: dict[Path, str] = {}
    for spec in specs:
        if spec.path in seen_paths:
            raise ValueError(
                f"Duplicate venv path '{spec.path}': used by both '{seen_paths[spec.path]}' and '{spec.name}'"
            )
        seen_paths[spec.path] = spec.name

    return specs


def build_venv(spec: VenvSpec, *, uv: str = "uv") -> None:
    """Build a single venv from its spec."""
    print(f"Building venv '{spec.name}' at {spec.path}")

    spec.path.parent.mkdir(parents=True, exist_ok=True)

    # Create venv and install frozen deps
    # UV_PROJECT_ENVIRONMENT controls where the venv is created.
    # Remove VIRTUAL_ENV to avoid "does not match" warnings when
    # running inside `uv run` (which sets its own VIRTUAL_ENV).
    env = {**os.environ, "UV_PROJECT_ENVIRONMENT": str(spec.path)}
    env.pop("VIRTUAL_ENV", None)
    subprocess.run(
        [
            uv,
            "sync",
            "--frozen",
            f"--python={spec.python_version}",
            f"--project={spec.project_dir}",
        ],
        check=True,
        env=env,
    )

    # Apply permissions based on access level
    _apply_permissions(spec)

    print(f"Built venv '{spec.name}' (access={spec.access})")


def _apply_permissions(spec: VenvSpec) -> None:
    """Apply filesystem permissions to a venv based on its access level."""
    if spec.access == "root":
        subprocess.run(["chown", "-R", "root:root", str(spec.path)], check=True)
        # uv leaves a mode-666 `.lock` in the venv. The 0700 below hides it from
        # the student but does not clear the write bit.
        subprocess.run(["chmod", "-R", "go-w", str(spec.path)], check=True)
        subprocess.run(["chmod", "0700", str(spec.path)], check=True)

    elif spec.access == "student:r":
        subprocess.run(["chown", "-R", "root:root", str(spec.path)], check=True)
        # Read-only, keeping the executable bit on files that had one
        subprocess.run(["chmod", "-R", "a=rX", str(spec.path)], check=True)
        subprocess.run(["chmod", "1755", str(spec.path)], check=True)

    elif spec.access == "student:rw":
        demote_id = "1000"  # KAROTTE_DEMOTE_ID default
        subprocess.run(
            ["chown", "-R", f"{demote_id}:{demote_id}", str(spec.path)],
            check=True,
        )


def build_all(venvs_dir: Path, *, uv: str = "uv") -> None:
    """Discover and build all venvs from a directory."""
    specs = discover_venv_specs(venvs_dir)

    if not specs:
        print("No venv specs found, nothing to build.")
        return

    for spec in specs:
        build_venv(spec, uv=uv)

    print(f"Built {len(specs)} venvs.")


if __name__ == "__main__":
    venvs_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./venvs")
    build_all(venvs_dir)
