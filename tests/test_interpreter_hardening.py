"""The Containerfile must harden every interpreter `interpreter_checks` globs, and
must do it after `build_venvs.py`, which is when a second interpreter can appear."""

import re
from fnmatch import fnmatch
from pathlib import Path

import pytest

TEMPLATES = Path(__file__).resolve().parent.parent / "src/karotte/templates"
CONTAINERFILE = TEMPLATES / "default/Containerfile"
CHECKS = TEMPLATES / "default/src/environment/interpreter_checks.py"

OTHER_INTERPRETER = "cpython-3.13.14-linux-x86_64-gnu"


@pytest.fixture(scope="module")
def containerfile() -> str:
    return CONTAINERFILE.read_text()


def _env_var(containerfile: str, name: str) -> str:
    match = re.search(rf"^\s*{name}=(\S+?)\s*\\?$", containerfile, re.MULTILINE)
    assert match, f"{name} is no longer set in the Containerfile's ENV block"
    return match.group(1)


def _expand(glob: str, containerfile: str) -> str:
    for name in ("UV_PYTHON_INSTALL_DIR", "PYTHON_VERSION"):
        glob = glob.replace(f"${{{name}}}", _env_var(containerfile, name))
    return glob


def _hardening_globs(containerfile: str) -> list[tuple[int, str]]:
    return [
        (m.start(), _expand(m.group(0), containerfile))
        for m in re.finditer(r"\S*/cpython-\S+/bin/python3", containerfile)
    ]


def _check_glob(containerfile: str) -> str:
    match = re.search(r"python_glob = f\"(.+?)\"", CHECKS.read_text())
    assert match, "interpreter_checks no longer builds its glob the expected way"
    glob = match.group(1).replace(
        "{os.environ['UV_PYTHON_INSTALL_DIR']}", "${UV_PYTHON_INSTALL_DIR}"
    )
    return _expand(glob, containerfile)


@pytest.fixture(scope="module")
def other(containerfile: str) -> str:
    install_dir = _env_var(containerfile, "UV_PYTHON_INSTALL_DIR")
    return f"{install_dir}/{OTHER_INTERPRETER}/bin/python3"


def test_the_check_looks_at_interpreters_of_other_minors(
    containerfile: str, other: str
) -> None:
    assert fnmatch(other, _check_glob(containerfile))


def test_hardening_covers_every_interpreter_the_check_verifies(
    containerfile: str, other: str
) -> None:
    globs = [glob for _, glob in _hardening_globs(containerfile)]
    assert any(fnmatch(other, glob) for glob in globs), (
        f"{other} is checked but never hardened; the Containerfile only hardens {globs}"
    )


def test_hardening_runs_after_the_venvs_are_built(
    containerfile: str, other: str
) -> None:
    positions = [
        pos for pos, glob in _hardening_globs(containerfile) if fnmatch(other, glob)
    ]
    assert positions, "no hardening glob is wide enough to reach a second interpreter"
    assert max(positions) > containerfile.index("build_venvs.py"), (
        "the wide hardening glob runs before build_venvs.py, which is what "
        "downloads the second interpreter"
    )
