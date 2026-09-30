# /// script
# requires-python = ">=3.12"
# ///
"""Post-creation hook for the environment."""

import subprocess
from pathlib import Path

VENVS_DIR = Path("venvs")


def lock_venvs() -> None:
    """Lock dependencies for all venvs."""
    if not VENVS_DIR.is_dir():
        return

    for venv_dir in sorted(VENVS_DIR.iterdir()):
        if (venv_dir / "pyproject.toml").is_file():
            print(f"Locking venv: {venv_dir.name}")
            subprocess.check_call(["uv", "lock"], cwd=venv_dir)


if __name__ == "__main__":
    lock_venvs()
