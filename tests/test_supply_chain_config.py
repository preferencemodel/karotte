"""Tests for the template's `scripts/check_supply_chain_config.py`, loaded by path."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "src/karotte/templates/default/scripts/check_supply_chain_config.py"
)

PROJECT = ["[project]", 'name = "x"', 'version = "0"']

GOOD_UV = [
    "[tool.uv]",
    'exclude-newer = "7 days"',
    "",
    "[[tool.uv.index]]",
    'name = "mirror"',
    'url = "https://mirror.example/simple/"',
    "",
    "[[tool.uv.index]]",
    'name = "pypi"',
    'url = "https://pypi.org/simple"',
    "default = true",
]


@pytest.fixture
def checker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """The script, loaded with REPO_ROOT pointed at a temp repo."""
    spec = importlib.util.spec_from_file_location("supply_chain_check", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["supply_chain_check"] = module
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    return module


def _write(tmp_path: Path, *lines: str) -> Path:
    path = tmp_path / "pyproject.toml"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_accepts_a_well_formed_pyproject(checker: ModuleType, tmp_path: Path) -> None:
    path = _write(tmp_path, *PROJECT, *GOOD_UV)
    assert checker.check_pyproject(path) == []


def test_flags_missing_age_delay(checker: ModuleType, tmp_path: Path) -> None:
    path = _write(tmp_path, *PROJECT, "[tool.uv]")
    (problem,) = checker.check_pyproject(path)
    assert "age-delay" in problem


def test_flags_duplicate_tool_uv_table(checker: ModuleType, tmp_path: Path) -> None:
    """A "keep both sides" merge yields two [tool.uv] tables: invalid TOML."""
    path = _write(
        tmp_path,
        *PROJECT,
        *GOOD_UV,
        "",
        "[tool.uv]",
        'index-strategy = "unsafe-best-match"',
    )
    (problem,) = checker.check_pyproject(path)
    assert "not valid TOML" in problem


def test_flags_duplicate_index_name(checker: ModuleType, tmp_path: Path) -> None:
    """Two indexes named `pypi` is valid TOML, but uv rejects it."""
    path = _write(
        tmp_path,
        *PROJECT,
        *GOOD_UV,
        "",
        "[[tool.uv.index]]",
        'name = "pypi"',
        'url = "https://pypi.org/simple"',
    )
    (problem,) = checker.check_pyproject(path)
    assert "duplicate index name" in problem


def test_repeated_exclude_newer_key_is_rejected(
    checker: ModuleType, tmp_path: Path
) -> None:
    """A repeated key is invalid TOML even though the text matches."""
    path = _write(
        tmp_path,
        *PROJECT,
        "[tool.uv]",
        'exclude-newer = "7 days"',
        'exclude-newer = "7 days"',
    )
    (problem,) = checker.check_pyproject(path)
    assert "not valid TOML" in problem


@pytest.mark.parametrize("pin", ["torch==2.11.0+cpu", "torch==2.11.0+cu126"])
def test_flags_flavour_pin_without_an_extra_index(
    checker: ModuleType, tmp_path: Path, pin: str
) -> None:
    """PyPI publishes no +cpu/+cu126 variant, so the pin cannot resolve."""
    path = _write(tmp_path, *PROJECT, f'dependencies = ["{pin}"]', *GOOD_UV)
    (problem,) = checker.check_pyproject(path)
    assert "build-specific version" in problem
    assert "tool.uv.sources" in problem


def test_flavour_pin_is_fine_with_a_serving_index(
    checker: ModuleType, tmp_path: Path
) -> None:
    path = _write(
        tmp_path,
        *PROJECT,
        'dependencies = ["torch==2.11.0+cu126"]',
        *GOOD_UV,
        "",
        "[[tool.uv.index]]",
        'name = "pytorch-cu126"',
        'url = "https://download.pytorch.org/whl/cu126"',
        "explicit = true",
        "",
        "[tool.uv.sources]",
        'torch = { index = "pytorch-cu126" }',
    )
    assert checker.check_pyproject(path) == []


def test_finds_flavour_pins_in_dependency_groups(
    checker: ModuleType, tmp_path: Path
) -> None:
    """Pins hide in `dependency-groups` and extras too, not just `dependencies`."""
    path = _write(
        tmp_path,
        *PROJECT,
        *GOOD_UV,
        "",
        "[dependency-groups]",
        'dev = ["torch==2.11.0+cpu"]',
    )
    (problem,) = checker.check_pyproject(path)
    assert "build-specific version" in problem


def test_astral_compound_local_version_is_recognised(
    checker: ModuleType, tmp_path: Path
) -> None:
    """Astral's `+cu.13.0.torch.2.10` form must count as a flavour."""
    path = _write(
        tmp_path,
        *PROJECT,
        'dependencies = ["flash-attn==2.8.3+cu.13.0.torch.2.10"]',
        *GOOD_UV,
    )
    (problem,) = checker.check_pyproject(path)
    assert "build-specific version" in problem


def test_private_package_build_tag_is_not_flagged(
    checker: ModuleType, tmp_path: Path
) -> None:
    """`examplepkg==2.3.4+gabc1234` is a git-describe tag, not a GPU flavour."""
    path = _write(
        tmp_path,
        *PROJECT,
        'dependencies = ["examplepkg==2.3.4+gabc1234"]',
        *GOOD_UV,
    )
    assert checker.check_pyproject(path) == []


def test_flavour_pin_passes_with_any_extra_index(
    checker: ModuleType, tmp_path: Path
) -> None:
    """Any explicit extra index serves a flavour pin, not just pytorch.org."""
    url = "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
    path = _write(
        tmp_path,
        *PROJECT,
        'dependencies = ["jaxlib==0.4.26+cuda12.cudnn89"]',
        *GOOD_UV,
        "",
        "[[tool.uv.index]]",
        'name = "jax-releases"',
        f'url = "{url}"',
        'format = "flat"',
        "explicit = true",
    )
    assert checker.check_pyproject(path) == []


def test_a_non_explicit_extra_index_does_not_serve_a_flavour_pin(
    checker: ModuleType, tmp_path: Path
) -> None:
    """A non-explicit index is a general mirror; a flavour build needs an explicit one."""
    path = _write(
        tmp_path,
        *PROJECT,
        'dependencies = ["torch==2.11.0+cu126"]',
        *GOOD_UV,
        "",
        "[[tool.uv.index]]",
        'name = "pytorch-cu126"',
        'url = "https://download.pytorch.org/whl/cu126"',
    )
    (problem,) = checker.check_pyproject(path)
    assert "build-specific version" in problem
