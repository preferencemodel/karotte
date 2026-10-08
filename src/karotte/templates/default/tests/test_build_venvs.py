import importlib.util
import stat
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Load build_venvs.py by file path since venvs/ is not a Python package.
_SCRIPT = Path(__file__).resolve().parent.parent / "venvs" / "build_venvs.py"
_spec = importlib.util.spec_from_file_location("build_venvs", _SCRIPT)
assert _spec and _spec.loader
build_venvs = importlib.util.module_from_spec(_spec)
sys.modules["build_venvs"] = build_venvs
_spec.loader.exec_module(build_venvs)

VALID_ACCESS_LEVELS = build_venvs.VALID_ACCESS_LEVELS
VenvSpec = build_venvs.VenvSpec
_apply_permissions = build_venvs._apply_permissions
discover_venv_specs = build_venvs.discover_venv_specs
parse_venv_spec = build_venvs.parse_venv_spec


def _write_pyproject(
    path: Path,
    *,
    access: str = "root",
    venv_path: str = "/venvs/x",
    requires_python: str = "==3.12.*",
):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""\
[project]
name = "test-env"
requires-python = "{requires_python}"
dependencies = []

[tool.karotte]
access = "{access}"
path = "{venv_path}"
"""
    )


class TestParseVenvSpec:
    def test_parses_valid_spec(self, tmp_path: Path):
        project_dir = tmp_path / "my-venv"
        _write_pyproject(
            project_dir / "pyproject.toml",
            access="student:rw",
            venv_path="/workdir/.venv",
        )

        spec = parse_venv_spec(project_dir)

        assert spec.name == "my-venv"
        assert spec.access == "student:rw"
        assert spec.path == Path("/workdir/.venv")
        assert spec.python_version == "3.12"

    def test_missing_karotte_section(self, tmp_path: Path):
        project_dir = tmp_path / "bad"
        project_dir.mkdir()
        (project_dir / "pyproject.toml").write_text(
            '[project]\nname = "x"\ndependencies = []\n'
        )

        with pytest.raises(ValueError, match="missing \\[tool.karotte\\]"):
            parse_venv_spec(project_dir)

    def test_missing_access(self, tmp_path: Path):
        project_dir = tmp_path / "bad"
        project_dir.mkdir()
        (project_dir / "pyproject.toml").write_text(
            '[project]\nname = "x"\n\n[tool.karotte]\npath = "/a"\n'
        )

        with pytest.raises(ValueError, match="missing 'access'"):
            parse_venv_spec(project_dir)

    def test_invalid_access(self, tmp_path: Path):
        project_dir = tmp_path / "bad"
        _write_pyproject(
            project_dir / "pyproject.toml",
            access="nobody:rw",
            venv_path="/a",
        )

        with pytest.raises(ValueError, match="invalid access 'nobody:rw'"):
            parse_venv_spec(project_dir)

    def test_missing_path(self, tmp_path: Path):
        project_dir = tmp_path / "bad"
        project_dir.mkdir()
        (project_dir / "pyproject.toml").write_text(
            '[project]\nname = "x"\n\n[tool.karotte]\naccess = "root"\n'
        )

        with pytest.raises(ValueError, match="missing 'path'"):
            parse_venv_spec(project_dir)

    def test_relative_path_rejected(self, tmp_path: Path):
        project_dir = tmp_path / "bad"
        _write_pyproject(
            project_dir / "pyproject.toml",
            access="root",
            venv_path="relative/.venv",
        )

        with pytest.raises(ValueError, match="must be absolute"):
            parse_venv_spec(project_dir)

    @pytest.mark.parametrize("access", VALID_ACCESS_LEVELS)
    def test_all_valid_access_levels(self, tmp_path: Path, access: str):
        project_dir = tmp_path / "venv"
        _write_pyproject(
            project_dir / "pyproject.toml",
            access=access,
            venv_path="/some/path",
        )

        spec = parse_venv_spec(project_dir)
        assert spec.access == access


class TestDiscoverVenvSpecs:
    def test_discovers_specs(self, tmp_path: Path):
        _write_pyproject(
            tmp_path / "a" / "pyproject.toml", access="root", venv_path="/venvs/a"
        )
        _write_pyproject(
            tmp_path / "b" / "pyproject.toml",
            access="student:rw",
            venv_path="/workdir/.venv",
        )

        specs = discover_venv_specs(tmp_path)

        assert len(specs) == 2
        assert specs[0].name == "a"
        assert specs[1].name == "b"

    def test_empty_directory(self, tmp_path: Path):
        specs = discover_venv_specs(tmp_path)
        assert specs == []

    def test_duplicate_paths_rejected(self, tmp_path: Path):
        _write_pyproject(
            tmp_path / "a" / "pyproject.toml", access="root", venv_path="/same/path"
        )
        _write_pyproject(
            tmp_path / "b" / "pyproject.toml",
            access="student:rw",
            venv_path="/same/path",
        )

        with pytest.raises(ValueError, match="Duplicate venv path"):
            discover_venv_specs(tmp_path)

    def test_ignores_non_directories(self, tmp_path: Path):
        _write_pyproject(
            tmp_path / "valid" / "pyproject.toml", access="root", venv_path="/venvs/a"
        )
        (tmp_path / "stray_file.txt").write_text("not a venv")

        specs = discover_venv_specs(tmp_path)
        assert len(specs) == 1


class TestRequiresPython:
    @pytest.mark.parametrize(
        ("spec", "expected"),
        [
            ("==3.12", "3.12"),
            ("==3.12.*", "3.12"),
            ("==3.12.11", "3.12"),
        ],
    )
    def test_accepted_formats(self, tmp_path: Path, spec: str, expected: str):
        project_dir = tmp_path / "v"
        _write_pyproject(
            project_dir / "pyproject.toml", requires_python=spec, venv_path="/a"
        )
        assert parse_venv_spec(project_dir).python_version == expected

    @pytest.mark.parametrize("spec", [">=3.12", ">=3.12,<4", "~=3.12", "3.12"])
    def test_specifiers_rejected(self, tmp_path: Path, spec: str):
        project_dir = tmp_path / "v"
        _write_pyproject(
            project_dir / "pyproject.toml", requires_python=spec, venv_path="/a"
        )
        with pytest.raises(ValueError, match="must be a pinned version"):
            parse_venv_spec(project_dir)

    def test_missing_requires_python(self, tmp_path: Path):
        project_dir = tmp_path / "v"
        project_dir.mkdir(parents=True)
        (project_dir / "pyproject.toml").write_text(
            '[project]\nname = "x"\ndependencies = []\n\n'
            '[tool.karotte]\naccess = "root"\npath = "/a"\n'
        )
        with pytest.raises(ValueError, match="missing 'requires-python'"):
            parse_venv_spec(project_dir)


def _make_spec(access: str, path: str = "/venvs/test") -> VenvSpec:
    return VenvSpec(
        name="test",
        project_dir=Path("/tmp/test"),
        path=Path(path),
        access=access,
        python_version="3.12",
    )


class TestApplyPermissions:
    @patch("build_venvs.subprocess.run")
    def test_root_access(self, mock_run):
        spec = _make_spec("root", "/root/venvs/scoring")
        _apply_permissions(spec)

        mock_run.assert_any_call(
            ["chown", "-R", "root:root", "/root/venvs/scoring"], check=True
        )
        mock_run.assert_any_call(["chmod", "0700", "/root/venvs/scoring"], check=True)

    @patch("build_venvs.subprocess.run")
    def test_root_access_clears_group_other_write(self, mock_run):
        """`uv sync` leaves a mode-666 `.lock` inside the venv.

        0700 on the directory hides the file but leaves the write bit set, and
        the image's `check_permissions` rejects any world-writable path outside
        the workdir whether or not the student can reach it.
        """
        spec = _make_spec("root", "/root/venvs/scoring")
        _apply_permissions(spec)

        mock_run.assert_any_call(
            ["chmod", "-R", "go-w", "/root/venvs/scoring"], check=True
        )

    @patch("build_venvs.subprocess.run")
    def test_student_read_only(self, mock_run):
        spec = _make_spec("student:r", "/workdir/venvs/tools")
        _apply_permissions(spec)

        mock_run.assert_any_call(
            ["chown", "-R", "root:root", "/workdir/venvs/tools"], check=True
        )

    def test_student_read_only_keeps_executables(self, tmp_path: Path):
        """Packages run bundled binaries (e.g. ptxas), so read-only must not strip the executable bit."""
        venv = tmp_path / "tools"
        (venv / "bin").mkdir(parents=True)
        (venv / "lib").mkdir()
        tool = venv / "bin" / "ptxas"
        tool.write_text("")
        tool.chmod(0o755)
        module = venv / "lib" / "mod.py"
        module.write_text("")
        module.chmod(0o644)

        real_run = subprocess.run

        def run_without_chown(cmd, **kwargs):
            if cmd[0] != "chown":
                real_run(cmd, **kwargs)

        with patch("build_venvs.subprocess.run", side_effect=run_without_chown):
            _apply_permissions(_make_spec("student:r", str(venv)))

        assert stat.S_IMODE(venv.stat().st_mode) == 0o1755
        assert stat.S_IMODE((venv / "bin").stat().st_mode) == 0o555
        assert stat.S_IMODE(tool.stat().st_mode) == 0o555
        assert stat.S_IMODE(module.stat().st_mode) == 0o444

    @patch("build_venvs.subprocess.run")
    def test_student_read_write(self, mock_run):
        spec = _make_spec("student:rw", "/workdir/.venv")
        _apply_permissions(spec)

        mock_run.assert_called_once_with(
            ["chown", "-R", "1000:1000", "/workdir/.venv"], check=True
        )
