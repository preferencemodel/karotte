# pyright: reportPrivateUsage=false
import json
import os
import subprocess
import tomllib
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from loguru import logger
from pydantic import ValidationError

from karotte.create_env import create_env
from karotte.update_env import (
    MergeFileError,
    UpdateMigration,
    _env_uv_flags,
    _generate_env,
    _get_latest_version,
    _index_credentials,
    _is_binary,
    _merge_do_not_recreate,
    _merge_file,
    _merge_projects,
    _migrations,
    _should_skip,
    _tool_args,
    update_env,
)


class TestMergeDoNotRecreate:
    def test_picks_up_template_additions(self):
        merged = _merge_do_not_recreate(base=["a"], theirs=["a", "b"], ours=["a"])
        assert merged == {"a", "b"}

    def test_drops_template_removals(self):
        merged = _merge_do_not_recreate(base=["a", "b"], theirs=["a"], ours=["a", "b"])
        assert merged == {"a"}

    def test_preserves_user_additions(self):
        merged = _merge_do_not_recreate(base=["a"], theirs=["a"], ours=["a", "mine"])
        assert merged == {"a", "mine"}

    def test_preserves_user_removals(self):
        merged = _merge_do_not_recreate(base=["a", "b"], theirs=["a", "b"], ours=["a"])
        assert merged == {"a"}

    def test_unseeded_env_adopts_theirs(self):
        """Manifest predating the field (ours=None) adopts the template's list
        wholesale, even when the baseline already shipped those paths."""
        merged = _merge_do_not_recreate(base=["a", "b"], theirs=["a", "b"], ours=None)
        assert merged == {"a", "b"}

    def test_unseeded_env_adopts_theirs_from_empty_baseline(self):
        merged = _merge_do_not_recreate(base=[], theirs=["a", "b"], ours=None)
        assert merged == {"a", "b"}

    def test_cleared_list_is_respected(self):
        """An explicit empty list means the user cleared it; keep it empty."""
        merged = _merge_do_not_recreate(base=["a"], theirs=["a", "b"], ours=[])
        assert merged == {"b"}


class TestShouldSkip:
    def test_skips_git_directory(self):
        assert _should_skip(Path(".git/config"))
        assert _should_skip(Path("foo/.git/objects"))

    def test_skips_pycache(self):
        assert _should_skip(Path("__pycache__/module.pyc"))
        assert _should_skip(Path("src/__pycache__/test.pyc"))

    def test_skips_manifest(self):
        assert _should_skip(Path(".manifest.json"))

    def test_skips_uv_lock(self):
        assert _should_skip(Path("uv.lock"))

    def test_does_not_skip_normal_files(self):
        assert not _should_skip(Path("pyproject.toml"))
        assert not _should_skip(Path("src/main.py"))
        assert not _should_skip(Path("README.md"))


class TestIsBinary:
    def test_text_file_is_not_binary(self, tmp_path: Path):
        text_file = tmp_path / "test.txt"
        text_file.write_text("Hello, world!")
        assert not _is_binary(text_file)

    def test_binary_file_is_binary(self, tmp_path: Path):
        binary_file = tmp_path / "test.bin"
        binary_file.write_bytes(b"Hello\x00World")
        assert _is_binary(binary_file)


class TestMergeFile:
    def test_clean_merge(self, tmp_path: Path):
        """Test merging when changes are far apart (no overlap)."""
        base = tmp_path / "base.txt"
        current = tmp_path / "current.txt"
        other = tmp_path / "other.txt"

        # Changes need to be far enough apart to not overlap
        base.write_text("line 1\nline 2\nline 3\nline 4\nline 5\nline 6\nline 7\n")
        current.write_text(
            "user modified line 1\nline 2\nline 3\nline 4\nline 5\nline 6\nline 7\n"
        )
        other.write_text(
            "line 1\nline 2\nline 3\nline 4\nline 5\nline 6\ntemplate modified line 7\n"
        )

        has_conflict = _merge_file(current, base, other)

        assert not has_conflict
        content = current.read_text()
        assert "user modified line 1" in content
        assert "template modified line 7" in content

    def test_conflicting_merge(self, tmp_path: Path):
        """Test merging when both sides modify the same line."""
        base = tmp_path / "base.txt"
        current = tmp_path / "current.txt"
        other = tmp_path / "other.txt"

        base.write_text("line 1\n")
        current.write_text("user change\n")
        other.write_text("template change\n")

        has_conflict = _merge_file(current, base, other)

        assert has_conflict
        content = current.read_text()
        # Git merge conflict markers should be present
        assert "<<<<<<<" in content or "=======" in content

    def test_git_error_leaves_the_file_alone(self, tmp_path: Path):
        base = tmp_path / "base.txt"
        current = tmp_path / "current.txt"
        other = tmp_path / "other.txt"
        base.write_text("line 1\n")
        current.write_bytes(b"user\x00data\n")
        other.write_text("template change\n")

        with pytest.raises(MergeFileError, match="Cannot merge binary files"):
            _merge_file(current, base, other)

        assert current.read_bytes() == b"user\x00data\n"


_KAROTTE_EXEMPTION = "exclude-newer-package.karotte = false\n"
_AGE_DELAY_LINE = 'exclude-newer = "7 days"\n'
_EXEMPT_PACKAGES = ("karotte", "pkg-a", "pkg-b")
# Forms older templates wrote right after the age-delay line.
_HISTORICAL_EXEMPTIONS = {
    "none": "",
    "inline-table": "exclude-newer-package = { "
    + ", ".join(f"{p} = false" for p in _EXEMPT_PACKAGES)
    + " }\n",
    "dotted": 'keyring-provider = "subprocess"\n'
    + "# This index serves no upload time.\n"
    + "".join(f"exclude-newer-package.{p} = false\n" for p in _EXEMPT_PACKAGES),
}


class TestPyprojectAgeDelayMerge:
    @pytest.fixture
    def target(self, tmp_path: Path) -> str:
        with (
            patch("karotte.create_env.subprocess.check_call"),
            patch("karotte.create_env.run_uv"),
        ):
            create_env(tmp_path / "target", templates=["default"], no_lock=True)
        return (tmp_path / "target" / "pyproject.toml").read_text()

    def _merge(self, tmp_path: Path, base: str, ours: str, target: str) -> str:
        paths = {name: tmp_path / name for name in ("base", "ours", "theirs")}
        paths["base"].write_text(base)
        paths["ours"].write_text(ours)
        paths["theirs"].write_text(target)
        assert not _merge_file(paths["ours"], paths["base"], paths["theirs"])
        return paths["ours"].read_text()

    @pytest.mark.parametrize("form", list(_HISTORICAL_EXEMPTIONS))
    def test_old_exemptions_merge_to_one_karotte_exemption(
        self, tmp_path: Path, target: str, form: str
    ):
        assert _AGE_DELAY_LINE + _KAROTTE_EXEMPTION in target
        base = target.replace(
            _AGE_DELAY_LINE + _KAROTTE_EXEMPTION,
            _AGE_DELAY_LINE + _HISTORICAL_EXEMPTIONS[form],
        )
        ours = base.replace('    "karotte",\n', '    "karotte",\n    "numpy",\n')

        merged = self._merge(tmp_path, base, ours, target)

        uv = tomllib.loads(merged)["tool"]["uv"]
        assert uv["exclude-newer-package"] == {"karotte": False}
        assert merged.count("karotte = false") == 1
        assert '"numpy",' in merged

    def test_hand_added_karotte_exemption_is_not_duplicated(
        self, tmp_path: Path, target: str
    ):
        base = target.replace(_KAROTTE_EXEMPTION, "")
        ours = base.replace(_AGE_DELAY_LINE, _AGE_DELAY_LINE + _KAROTTE_EXEMPTION)

        merged = self._merge(tmp_path, base, ours, target)

        assert tomllib.loads(merged)["tool"]["uv"]["exclude-newer-package"] == {
            "karotte": False
        }
        assert merged.count("karotte = false") == 1

    def test_kept_old_exemptions_conflict_instead_of_duplicating(
        self, tmp_path: Path, target: str
    ):
        """A silent duplicate key would leave invalid TOML with no conflict reported."""
        base = target.replace(_KAROTTE_EXEMPTION, "")
        ours = base.replace(
            _AGE_DELAY_LINE, _AGE_DELAY_LINE + _HISTORICAL_EXEMPTIONS["dotted"]
        )
        paths = {name: tmp_path / name for name in ("base", "ours", "theirs")}
        paths["base"].write_text(base)
        paths["ours"].write_text(ours)
        paths["theirs"].write_text(target)

        assert _merge_file(paths["ours"], paths["base"], paths["theirs"])


class TestMergeProjects:
    def test_adds_new_files_from_target(self, tmp_path: Path):
        """Files that exist only in target are added to project."""
        project = tmp_path / "project"
        baseline = tmp_path / "baseline"
        target = tmp_path / "target"

        project.mkdir()
        baseline.mkdir()
        target.mkdir()

        # New file only in target
        (target / "new_file.txt").write_text("new content")

        conflicts = _merge_projects(project, baseline, target)

        assert not conflicts
        assert (project / "new_file.txt").read_text() == "new content"

    def test_removes_deleted_files_from_project(self, tmp_path: Path):
        """Files removed from template are deleted from project."""
        project = tmp_path / "project"
        baseline = tmp_path / "baseline"
        target = tmp_path / "target"

        project.mkdir()
        baseline.mkdir()
        target.mkdir()

        # File exists in baseline and project, but not in target
        (baseline / "old_file.txt").write_text("old content")
        (project / "old_file.txt").write_text("old content")

        conflicts = _merge_projects(project, baseline, target)

        assert not conflicts
        assert not (project / "old_file.txt").exists()

    def test_recreates_user_deleted_files(self, tmp_path: Path):
        """Files deleted by user but still in template are recreated."""
        project = tmp_path / "project"
        baseline = tmp_path / "baseline"
        target = tmp_path / "target"

        project.mkdir()
        baseline.mkdir()
        target.mkdir()

        # File exists in both baseline and target, but user deleted from project
        (baseline / "important.txt").write_text("content")
        (target / "important.txt").write_text("updated content")

        conflicts = _merge_projects(project, baseline, target)

        assert not conflicts
        assert (project / "important.txt").read_text() == "updated content"

    def test_does_not_recreate_deleted_claude_md(self, tmp_path: Path):
        """CLAUDE.md deleted by the user is not recreated by update."""
        project = tmp_path / "project"
        baseline = tmp_path / "baseline"
        target = tmp_path / "target"

        project.mkdir()
        baseline.mkdir()
        target.mkdir()

        (baseline / "CLAUDE.md").write_text("content")
        (target / "CLAUDE.md").write_text("updated content")

        conflicts = _merge_projects(
            project, baseline, target, do_not_recreate_paths={"CLAUDE.md"}
        )

        assert not conflicts
        assert not (project / "CLAUDE.md").exists()

    def test_does_not_recreate_deleted_example_task(self, tmp_path: Path):
        """example_task files deleted by the user are not recreated by update."""
        project = tmp_path / "project"
        baseline = tmp_path / "baseline"
        target = tmp_path / "target"

        project.mkdir()
        baseline.mkdir()
        target.mkdir()

        task_rel = Path("src/environment/tasks/example_task/__init__.py")
        for root in (baseline, target):
            (root / task_rel).parent.mkdir(parents=True, exist_ok=True)
            (root / task_rel).write_text("task")

        conflicts = _merge_projects(
            project,
            baseline,
            target,
            do_not_recreate_paths={"src/environment/tasks/example_task"},
        )

        assert not conflicts
        assert not (project / task_rel).exists()

    def test_does_not_add_new_example_task(self, tmp_path: Path):
        """A newly-introduced example_task the user lacks is not added."""
        project = tmp_path / "project"
        baseline = tmp_path / "baseline"
        target = tmp_path / "target"

        project.mkdir()
        baseline.mkdir()
        target.mkdir()

        task_rel = Path("src/environment/tasks/example_task/__init__.py")
        (target / task_rel).parent.mkdir(parents=True, exist_ok=True)
        (target / task_rel).write_text("task")

        conflicts = _merge_projects(
            project,
            baseline,
            target,
            do_not_recreate_paths={"src/environment/tasks/example_task"},
        )

        assert not conflicts
        assert not (project / task_rel).exists()

    def test_still_merges_present_claude_md(self, tmp_path: Path):
        """CLAUDE.md the user kept is still merged, not preserved-as-is."""
        project = tmp_path / "project"
        baseline = tmp_path / "baseline"
        target = tmp_path / "target"

        project.mkdir()
        baseline.mkdir()
        target.mkdir()

        base_content = "\n".join([f"line{i}=original" for i in range(1, 10)]) + "\n"
        project_content = (
            "line1=user\n"
            + "\n".join([f"line{i}=original" for i in range(2, 10)])
            + "\n"
        )
        target_content = (
            "\n".join([f"line{i}=original" for i in range(1, 9)]) + "\nline9=template\n"
        )

        (baseline / "CLAUDE.md").write_text(base_content)
        (project / "CLAUDE.md").write_text(project_content)
        (target / "CLAUDE.md").write_text(target_content)

        conflicts = _merge_projects(
            project, baseline, target, do_not_recreate_paths={"CLAUDE.md"}
        )

        assert not conflicts
        content = (project / "CLAUDE.md").read_text()
        assert "line1=user" in content
        assert "line9=template" in content

    def test_merges_modified_files(self, tmp_path: Path):
        """Files modified in both project and template are merged."""
        project = tmp_path / "project"
        baseline = tmp_path / "baseline"
        target = tmp_path / "target"

        project.mkdir()
        baseline.mkdir()
        target.mkdir()

        # Changes need to be far apart to avoid git merge-file treating them as conflicts
        base_content = "\n".join([f"line{i}=original" for i in range(1, 10)]) + "\n"
        project_content = (
            "line1=user\n"
            + "\n".join([f"line{i}=original" for i in range(2, 10)])
            + "\n"
        )
        target_content = (
            "\n".join([f"line{i}=original" for i in range(1, 9)]) + "\nline9=template\n"
        )

        (baseline / "config.txt").write_text(base_content)
        (project / "config.txt").write_text(project_content)
        (target / "config.txt").write_text(target_content)

        conflicts = _merge_projects(project, baseline, target)

        assert not conflicts
        content = (project / "config.txt").read_text()
        assert "line1=user" in content
        assert "line9=template" in content

    def test_unmergeable_file_keeps_the_user_version(self, tmp_path: Path):
        project, baseline, target = (tmp_path / n for n in ("p", "b", "t"))
        for d in (project, baseline, target):
            d.mkdir()
        (baseline / "data.txt").write_bytes(b"old\x00binary\n")
        (project / "data.txt").write_bytes(b"user\x00binary\n")
        (target / "data.txt").write_text("now text\n")
        warnings: list[str] = []
        handler = logger.add(lambda m: warnings.append(str(m)), level="WARNING")
        try:
            conflicts = _merge_projects(project, baseline, target)
        finally:
            logger.remove(handler)

        assert conflicts == [Path("data.txt")]
        assert (project / "data.txt").read_bytes() == b"user\x00binary\n"
        assert any(
            "data.txt" in w and "Cannot merge binary files" in w for w in warnings
        )


class TestUpdateEnv:
    def test_raises_error_without_manifest(self, tmp_path: Path):
        """update_env fails if .manifest.json doesn't exist."""
        project = tmp_path / "project"
        project.mkdir()

        with pytest.raises(FileNotFoundError, match="No .manifest.json"):
            update_env(project)

    def test_skips_update_for_same_version(self, tmp_path: Path):
        """update_env does nothing if version is already current."""
        project = tmp_path / "project"
        project.mkdir()

        manifest = {"karotte_version": "1.0.0", "templates": ["default"]}
        (project / ".manifest.json").write_text(json.dumps(manifest))

        with patch("karotte.update_env._get_latest_version", return_value="1.0.0"):
            conflicts = update_env(project)

        assert conflicts == []
        # Manifest should be unchanged
        manifest_content = json.loads((project / ".manifest.json").read_text())
        assert manifest_content["karotte_version"] == "1.0.0"

    def test_updates_manifest_version(self, tmp_path: Path):
        """update_env updates .manifest.json with new version."""
        project = tmp_path / "project"
        project.mkdir()

        manifest = {"karotte_version": "0.9.0", "templates": ["default"]}
        (project / ".manifest.json").write_text(json.dumps(manifest))

        with (
            patch("karotte.update_env._get_latest_version", return_value="1.0.0"),
            patch("karotte.update_env._generate_env") as mock_gen,
            patch("karotte.update_env._merge_projects", return_value=[]),
            patch("karotte.update_env.subprocess.run"),
        ):
            # Mock _generate_env to create empty dirs
            mock_gen.side_effect = _mkdir_only

            update_env(project)

        manifest_content = json.loads((project / ".manifest.json").read_text())
        assert manifest_content["karotte_version"] == "1.0.0"

    def test_calls_generate_env_with_correct_versions(self, tmp_path: Path):
        """update_env generates baseline with old version, target with new."""
        project = tmp_path / "project"
        project.mkdir()

        manifest = {"karotte_version": "0.9.0", "templates": ["base", "custom"]}
        (project / ".manifest.json").write_text(json.dumps(manifest))

        with (
            patch("karotte.update_env._get_latest_version", return_value="1.0.0"),
            patch("karotte.update_env._generate_env") as mock_gen,
            patch("karotte.update_env._merge_projects", return_value=[]),
            patch("karotte.update_env.subprocess.run"),
        ):
            mock_gen.side_effect = _mkdir_only

            update_env(project)

        # Should be called twice: once for baseline, once for target
        assert mock_gen.call_count == 2
        calls = mock_gen.call_args_list

        # First call: baseline with old version
        assert calls[0][0][0] == "0.9.0"
        assert calls[0][0][1] == ["base", "custom"]

        # Second call: target with new version
        assert calls[1][0][0] == "1.0.0"
        assert calls[1][0][1] == ["base", "custom"]


class TestManifestCreation:
    """Test that create_env writes .manifest.json correctly."""

    def test_creates_manifest_with_version_and_templates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        from karotte import create_env as create_env_module
        from karotte.create_env import create_env

        # Create minimal template
        templates_dir = tmp_path / "templates" / "test"
        templates_dir.mkdir(parents=True)
        (templates_dir / "template.toml").write_text('description = "Test"\n')
        (templates_dir / "file.txt").write_text("content")

        monkeypatch.setattr(create_env_module, "TEMPLATES_DIR", tmp_path / "templates")

        with patch("karotte.create_env.pkg_version", return_value="1.2.3"):
            create_env(tmp_path / "my_env", templates=["test"])

        manifest_path = tmp_path / "my_env" / ".manifest.json"
        assert manifest_path.exists()

        manifest = json.loads(manifest_path.read_text())
        assert manifest["karotte_version"] == "1.2.3"
        assert manifest["templates"] == ["test"]

    def test_no_lock_skips_uv_lock(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        from karotte import create_env as create_env_module
        from karotte.create_env import create_env

        # Create template with pyproject.toml
        templates_dir = tmp_path / "templates" / "test"
        templates_dir.mkdir(parents=True)
        (templates_dir / "template.toml").write_text('description = "Test"\n')
        (templates_dir / "pyproject.toml").write_text("[project]\nname = 'test'\n")

        monkeypatch.setattr(create_env_module, "TEMPLATES_DIR", tmp_path / "templates")

        with patch("karotte.create_env.subprocess.check_call") as mock_check_call:
            create_env(tmp_path / "my_env", templates=["test"], no_lock=True)

        # uv lock should not have been called
        mock_check_call.assert_not_called()


class TestUvToolRunIndex:
    @staticmethod
    def _captured(call: object) -> list[str]:
        with patch("karotte.update_env.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess([], 0, stdout="1.0.0")
            call()  # pyright: ignore[reportCallIssue]
        return mock_run.call_args[0][0]

    def test_only_the_env_indexes_are_passed(self):
        flags = ["--index=private=https://private.example/simple/"]
        for call in (
            lambda: _get_latest_version(flags),
            lambda: _generate_env(
                "1.0.0", ["default"], Path("/tmp/out"), uv_flags=flags
            ),
        ):
            cmd = self._captured(call)
            assert [a for a in cmd if a.startswith("--index")] == flags

    def test_no_index_without_env_indexes(self):
        cmd = self._captured(_get_latest_version)
        assert not [a for a in cmd if a.startswith("--index")]


PRIVATE_PYPROJECT = """[project]
name = "env"

[tool.uv]
keyring-provider = "subprocess"

[[tool.uv.index]]
name = "private"
url = "https://user@registry.example/team/simple/"

[[tool.uv.index]]
name = "pypi"
url = "https://pypi.org/simple"
default = true
"""


class TestIndexCredentials:
    """Nested `uv tool run` environments put their own `keyring` first on PATH,
    so credentials are asked for once and handed down as uv's per-index vars."""

    @pytest.fixture
    def fake_keyring(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        log = tmp_path / "keyring.log"
        script = bin_dir / "keyring"
        _ = script.write_text(
            "#!/bin/sh\n"
            + f'echo "$@" >> {log}\n'
            + '[ "$1 $2 $3" = "get registry.example user" ] && echo tok && exit 0\n'
            + "exit 1\n"
        )
        script.chmod(0o755)
        monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
        monkeypatch.delenv("UV_INDEX_PRIVATE_USERNAME", raising=False)
        monkeypatch.delenv("UV_INDEX_PRIVATE_PASSWORD", raising=False)
        monkeypatch.delenv("UV_KEYRING_PROVIDER", raising=False)
        return log

    def _project(self, tmp_path: Path, pyproject: str = PRIVATE_PYPROJECT) -> Path:
        project = tmp_path / "env"
        project.mkdir()
        _ = (project / "pyproject.toml").write_text(pyproject)
        return project

    def test_asks_keyring_like_uv_does(self, tmp_path: Path, fake_keyring: Path):
        creds = _index_credentials(self._project(tmp_path))

        assert creds == {
            "UV_INDEX_PRIVATE_USERNAME": "user",
            "UV_INDEX_PRIVATE_PASSWORD": "tok",
        }
        assert fake_keyring.read_text().splitlines() == [
            "get https://registry.example/team/simple/ user",
            "get registry.example user",
        ]

    def test_nothing_without_the_subprocess_provider(
        self, tmp_path: Path, fake_keyring: Path
    ):
        pyproject = PRIVATE_PYPROJECT.replace('keyring-provider = "subprocess"\n', "")
        assert _index_credentials(self._project(tmp_path, pyproject)) == {}
        assert not fake_keyring.exists()

    @pytest.mark.usefixtures("fake_keyring")
    def test_the_provider_can_come_from_the_environment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("UV_KEYRING_PROVIDER", "subprocess")
        pyproject = PRIVATE_PYPROJECT.replace('keyring-provider = "subprocess"\n', "")
        creds = _index_credentials(self._project(tmp_path, pyproject))
        assert creds["UV_INDEX_PRIVATE_PASSWORD"] == "tok"

    def test_a_password_already_set_wins(
        self, tmp_path: Path, fake_keyring: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("UV_INDEX_PRIVATE_PASSWORD", "mine")
        assert _index_credentials(self._project(tmp_path)) == {}
        assert not fake_keyring.exists()

    @pytest.mark.usefixtures("fake_keyring")
    def test_no_answer_from_keyring_means_no_credentials(self, tmp_path: Path):
        pyproject = PRIVATE_PYPROJECT.replace("registry.example", "other.example")
        assert _index_credentials(self._project(tmp_path, pyproject)) == {}

    def test_no_keyring_on_path_means_no_credentials(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
        assert _index_credentials(self._project(tmp_path)) == {}

    @pytest.mark.usefixtures("fake_keyring")
    def test_update_hands_them_to_every_nested_uv_run(self, tmp_path: Path):
        project = self._project(tmp_path)
        _ = (project / ".manifest.json").write_text(
            json.dumps({"karotte_version": "1.0.0", "templates": ["default"]})
        )
        real_run = subprocess.run

        def fake_run(cmd: list[str], **kwargs: object):
            if cmd[0].endswith("keyring"):
                return real_run(cmd, **kwargs)  # pyright: ignore[reportCallIssue, reportArgumentType]
            return subprocess.CompletedProcess(cmd, 0, stdout="2.0.0", stderr="")

        with (
            patch.dict(os.environ),
            patch("karotte.update_env.subprocess.run", side_effect=fake_run) as run,
            patch("karotte.update_env._merge_projects", return_value=[]),
        ):
            _ = update_env(project)
        uv_calls = [c for c in run.call_args_list if c.args[0][0] == "uv"]
        assert len(uv_calls) >= 4
        for call in uv_calls:
            assert call.kwargs["env"]["UV_INDEX_PRIVATE_PASSWORD"] == "tok"


class TestRefusesIndexOverrides:
    """uv prefers these over the project's own indexes, so the relock would
    resolve somewhere the merged pyproject does not point."""

    def _project(self, tmp_path: Path) -> Path:
        project = tmp_path / "env"
        project.mkdir()
        _ = (project / ".manifest.json").write_text(
            json.dumps({"karotte_version": "1.0.0", "templates": ["default"]})
        )
        _ = (project / "pyproject.toml").write_text("[project]\n")
        return project

    @pytest.mark.parametrize(
        "var", ["UV_INDEX", "UV_DEFAULT_INDEX", "UV_INDEX_URL", "UV_EXTRA_INDEX_URL"]
    )
    def test_stops_before_touching_anything(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, var: str
    ):
        project = self._project(tmp_path)
        manifest = (project / ".manifest.json").read_text()
        monkeypatch.setenv(var, "https://mirror.example/simple/")

        with (
            patch("karotte.update_env.subprocess.run") as run,
            pytest.raises(RuntimeError, match=f"{var} is set"),
        ):
            _ = update_env(project)

        run.assert_not_called()
        assert (project / ".manifest.json").read_text() == manifest

    def test_an_empty_value_is_not_an_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("UV_INDEX", "")
        with patch("karotte.update_env._get_latest_version", return_value="1.0.0"):
            assert update_env(self._project(tmp_path)) == []


class TestOwnScriptsOffPath:
    """uv shells out to `keyring` for index credentials, and karotte's own
    dependencies install a backendless one in its venv's scripts dir."""

    SCRIPTS: str = "/tools/karotte/bin"

    @pytest.fixture
    def in_venv(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("karotte.update_env.sys.prefix", "/tools/karotte")
        monkeypatch.setattr("karotte.update_env.sys.base_prefix", "/usr")
        monkeypatch.setattr(
            "karotte.update_env.sysconfig.get_path",
            lambda _name: self.SCRIPTS,  # pyright: ignore[reportUnknownLambdaType]
        )
        monkeypatch.setenv("PATH", f"{self.SCRIPTS}:/home/u/.local/bin:/usr/bin")

    @pytest.mark.usefixtures("in_venv")
    def test_uv_tool_run_does_not_see_the_venv_scripts(self):
        with patch("karotte.update_env.subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess([], 0, stdout="1.0.0")
            _get_latest_version()
        env = run.call_args.kwargs["env"]
        assert env["PATH"] == "/home/u/.local/bin:/usr/bin"
        assert env["UV_EXCLUDE_NEWER"] == "7 days"

    @pytest.mark.usefixtures("in_venv")
    def test_relocks_do_not_see_the_venv_scripts(self, tmp_path: Path):
        project = tmp_path / "env"
        (project / "venvs/student").mkdir(parents=True)
        (project / "venvs/student/pyproject.toml").write_text("# old\n")
        (project / ".manifest.json").write_text(
            json.dumps({"karotte_version": "1.0.0", "templates": ["default"]})
        )
        (project / "pyproject.toml").write_text("[project]\n")

        def fake_merge(*_args: object) -> list[Path]:
            _ = (project / "venvs/student/pyproject.toml").write_text("# new\n")
            return []

        with (
            patch("karotte.update_env._get_latest_version", return_value="2.0.0"),
            patch("karotte.update_env._generate_env"),
            patch("karotte.update_env._merge_projects", side_effect=fake_merge),
            patch("karotte.update_env.subprocess.run") as run,
        ):
            _ = update_env(project)
        locks = [c for c in run.call_args_list if c.args[0][:2] == ["uv", "lock"]]
        assert len(locks) == 2
        for call in locks:
            assert call.kwargs["env"]["PATH"] == "/home/u/.local/bin:/usr/bin"
            assert "UV_EXCLUDE_NEWER" not in call.kwargs["env"]

    def test_outside_a_venv_path_is_left_alone(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr("karotte.update_env.sys.prefix", "/usr")
        monkeypatch.setattr("karotte.update_env.sys.base_prefix", "/usr")
        monkeypatch.setattr(
            "karotte.update_env.sysconfig.get_path",
            lambda _name: "/usr/bin",  # pyright: ignore[reportUnknownLambdaType]
        )
        monkeypatch.setenv("PATH", "/usr/bin:/bin")
        with patch("karotte.update_env.subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess([], 0, stdout="1.0.0")
            _get_latest_version()
        assert run.call_args.kwargs["env"]["PATH"] == "/usr/bin:/bin"


class TestForwardsEnvIndexes:
    PYPROJECT: str = """
[tool.uv]
keyring-provider = "subprocess"

[[tool.uv.index]]
name = "private"
url = "https://private.example/simple/"

[[tool.uv.index]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cu126"
explicit = true

[[tool.uv.index]]
name = "pypi"
url = "https://pypi.org/simple"
default = true
"""

    def test_reads_indexes_and_keyring_provider(self, tmp_path: Path):
        (tmp_path / "pyproject.toml").write_text(self.PYPROJECT)
        assert _env_uv_flags(tmp_path) == [
            "--index=private=https://private.example/simple/",
            "--default-index=https://pypi.org/simple",
            "--keyring-provider=subprocess",
        ]

    def test_no_pyproject(self, tmp_path: Path):
        assert _env_uv_flags(tmp_path) == []

    def test_unparseable_pyproject(self, tmp_path: Path):
        (tmp_path / "pyproject.toml").write_text("<<<<<<< ours\n")
        assert _env_uv_flags(tmp_path) == []

    def test_update_env_passes_them_to_every_uv_tool_run(self, tmp_path: Path):
        project = tmp_path / "project"
        project.mkdir()
        (project / "pyproject.toml").write_text(self.PYPROJECT)
        (project / ".manifest.json").write_text(
            json.dumps(
                {
                    "karotte_version": "2.13.0",
                    "templates": ["default"],
                    "extra_deps": ["pkg==1"],
                }
            )
        )

        with (
            patch(
                "karotte.update_env._get_latest_version", return_value="2.14.0"
            ) as latest,
            patch("karotte.update_env._generate_env", side_effect=_mkdir_only) as gen,
            patch("karotte.update_env._merge_projects", return_value=[]),
            patch("karotte.update_env.subprocess.run"),
        ):
            update_env(project)

        expected = _env_uv_flags(project)
        assert latest.call_args.args == (expected,)
        assert [c.kwargs["uv_flags"] for c in gen.call_args_list] == [expected] * 2


UNDATED_INDEX_STDERR = """\
warning: karotte-9.9.9-py2.py3-none-any.whl is missing an upload date, but user provided: 2000-01-01T00:00:00Z
  × No solution found when resolving tool dependencies:
  ╰─▶ Because there is no version of karotte==9.9.9 and you require
      karotte==9.9.9, we can conclude that your requirements are
      unsatisfiable.

      hint: `karotte` was filtered by `exclude-newer` to only include
      packages uploaded before 2000-01-01T00:00:00Z. Consider using
      `exclude-newer-package` to override the cutoff for this package.
"""


class TestGetLatestVersionErrors:
    def test_undated_index_gets_its_own_hint(self):
        with patch(
            "karotte.update_env.subprocess.run",
            side_effect=subprocess.CalledProcessError(
                1, "uv", stderr=UNDATED_INDEX_STDERR
            ),
        ):
            with pytest.raises(RuntimeError) as exc_info:
                _get_latest_version()
        message = str(exc_info.value)
        assert "serves no upload times" in message
        assert "authentication" not in message

    def test_raises_on_missing_uv(self):
        with patch("karotte.update_env.subprocess.run", side_effect=FileNotFoundError):
            with pytest.raises(RuntimeError, match="'uv' not found on PATH"):
                _get_latest_version()

    def test_raises_on_subprocess_failure(self):
        with patch(
            "karotte.update_env.subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "uv", stderr="network error"),
        ):
            with pytest.raises(RuntimeError, match="network error"):
                _get_latest_version()

    def test_preserves_original_exception(self):
        original = subprocess.CalledProcessError(1, "uv", stderr="boom")
        with patch("karotte.update_env.subprocess.run", side_effect=original):
            with pytest.raises(RuntimeError) as exc_info:
                _get_latest_version()
            assert exc_info.value.__cause__ is original


class TestToolArgs:
    def test_default_is_karotte_at_the_version(self):
        assert _tool_args("3.0.5") == ["karotte@3.0.5"]

    def test_a_migration_supplies_the_args_for_an_old_version(self):
        old = _Migration(tool_args={"1.4.0": ["--with", "dep<2", "old-name@1.4.0"]})
        with patch("karotte.update_env._migrations", return_value=[old]):
            assert _tool_args("1.4.0") == ["--with", "dep<2", "old-name@1.4.0"]
            assert _tool_args("3.0.5") == ["karotte@3.0.5"]

    def test_generate_env_runs_the_migration_args(self):
        old = _Migration(tool_args={"1.4.0": ["--with", "dep<2", "old-name@1.4.0"]})
        with (
            patch("karotte.update_env._migrations", return_value=[old]),
            patch("karotte.update_env.subprocess.run") as mock_run,
        ):
            mock_run.return_value = subprocess.CompletedProcess([], 0)
            _generate_env("1.4.0", ["default"], Path("/tmp/out"))
        cmd = mock_run.call_args[0][0]
        start = cmd.index("dep<2") - 1
        assert cmd[start : start + 4] == [
            "--with",
            "dep<2",
            "old-name@1.4.0",
            "create-env",
        ]


class TestGenerateEnvErrors:
    def test_undated_index_gets_its_own_hint(self):
        with patch(
            "karotte.update_env.subprocess.run",
            side_effect=subprocess.CalledProcessError(
                1, "uv", stderr=UNDATED_INDEX_STDERR
            ),
        ):
            with pytest.raises(RuntimeError) as exc_info:
                _generate_env("2.0.0", ["default"], Path("/tmp/out"))
        message = str(exc_info.value)
        assert "serves no upload times" in message
        assert "authentication" not in message

    def test_raises_on_subprocess_failure(self):
        with patch(
            "karotte.update_env.subprocess.run",
            side_effect=subprocess.CalledProcessError(
                1, "uv", stderr="version not found"
            ),
        ):
            with pytest.raises(
                RuntimeError, match="Failed to generate environment with karotte@0.9.0"
            ):
                _generate_env("0.9.0", ["default"], Path("/tmp/out"))

    def test_includes_stderr_in_message(self):
        with patch(
            "karotte.update_env.subprocess.run",
            side_effect=subprocess.CalledProcessError(
                1, "uv", stderr="No matching distribution"
            ),
        ):
            with pytest.raises(RuntimeError, match="No matching distribution"):
                _generate_env("0.0.1", ["default"], Path("/tmp/out"))

    def test_includes_version_in_message(self):
        with patch(
            "karotte.update_env.subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "uv", stderr="err"),
        ):
            with pytest.raises(RuntimeError, match="karotte@3.0.0"):
                _generate_env("3.0.0", ["default"], Path("/tmp/out"))


class TestUvLockErrors:
    def test_raises_on_uv_lock_failure(self, tmp_path: Path):
        """update_env raises RuntimeError when uv lock fails."""
        project = tmp_path / "project"
        project.mkdir()
        (project / "pyproject.toml").write_text("[project]\nname = 'test'\n")

        manifest = {"karotte_version": "0.9.0", "templates": ["default"]}
        (project / ".manifest.json").write_text(json.dumps(manifest))

        lock_error = subprocess.CalledProcessError(1, "uv", stderr="resolution failed")

        def mock_run(*args, **kwargs):  # pyright: ignore[reportMissingParameterType, reportUnusedParameter]
            raise lock_error

        with (
            patch("karotte.update_env._get_latest_version", return_value="1.0.0"),
            patch("karotte.update_env._generate_env") as mock_gen,
            patch("karotte.update_env._merge_projects", return_value=[]),
            patch("karotte.update_env.subprocess.run", side_effect=mock_run),
        ):
            mock_gen.side_effect = _mkdir_only

            with pytest.raises(RuntimeError, match="Failed to regenerate uv.lock"):
                update_env(project)


class TestRelockSkippedOnPyprojectConflict:
    """A conflicted pyproject.toml skips the relock so the conflicts are still returned."""

    def _project(self, tmp_path: Path) -> Path:
        project = tmp_path / "env"
        project.mkdir()
        (project / ".manifest.json").write_text(
            json.dumps({"karotte_version": "1.0.0", "templates": ["default"]})
        )
        (project / "pyproject.toml").write_text("[project]\nname = 'env'\n")
        return project

    def _run(self, project: Path, conflicts: list[Path]):
        """Drive update_env past env generation, with a chosen merge outcome."""
        with (
            patch("karotte.update_env._get_latest_version", return_value="2.0.0"),
            patch("karotte.update_env._generate_env"),
            patch("karotte.update_env._merge_projects", return_value=conflicts),
            patch("karotte.update_env.subprocess.run") as run,
        ):
            returned = update_env(project)
        lock_calls = [
            c for c in run.call_args_list if "lock" in (c.args[0] if c.args else [])
        ]
        return returned, lock_calls

    def test_conflicted_pyproject_skips_relock_and_returns_conflicts(
        self, tmp_path: Path
    ):
        conflicts = [Path("pyproject.toml")]
        returned, lock_calls = self._run(self._project(tmp_path), conflicts)

        # The relock must not be attempted — it would fail on invalid TOML.
        assert lock_calls == []
        # And the conflicts must survive for the caller to act on.
        assert returned == conflicts

    def test_conflict_elsewhere_still_relocks(self, tmp_path: Path):
        """Only pyproject.toml blocks the lock; other conflicts do not."""
        conflicts = [Path("Containerfile")]
        returned, lock_calls = self._run(self._project(tmp_path), conflicts)

        assert len(lock_calls) == 1
        assert returned == conflicts

    def test_clean_merge_still_relocks(self, tmp_path: Path):
        returned, lock_calls = self._run(self._project(tmp_path), [])

        assert len(lock_calls) == 1
        assert returned == []

    def test_relock_upgrades_the_template_packages_too(self, tmp_path: Path):
        """Otherwise the lock keeps the old template package the manifest no longer names."""
        project = self._project(tmp_path)
        (project / ".manifest.json").write_text(
            json.dumps(
                {
                    "karotte_version": "1.0.0",
                    "templates": ["default"],
                    "extra_deps": ["tmpl-pkg==0.1.0"],
                }
            )
        )

        _, (lock_call,) = self._run(project, [])

        assert lock_call.args[0] == [
            "uv",
            "lock",
            "--upgrade-package",
            "karotte",
            "--upgrade-package",
            "tmpl-pkg",
        ]


class TestRelocksChangedVenvs:
    """A venv lock left behind after its pyproject moved indexes breaks the image build."""

    def _project(self, tmp_path: Path) -> Path:
        project = tmp_path / "env"
        for venv in ("student", "judge"):
            (project / "venvs" / venv).mkdir(parents=True)
            (project / "venvs" / venv / "pyproject.toml").write_text(f"# {venv}\n")
            (project / "venvs" / venv / "uv.lock").write_text("")
        (project / ".manifest.json").write_text(
            json.dumps({"karotte_version": "1.0.0", "templates": ["default"]})
        )
        (project / "pyproject.toml").write_text("[project]\nname = 'env'\n")
        return project

    def _run(
        self, project: Path, merge: dict[str, str], conflicts: Sequence[str] = ()
    ) -> tuple[list[Path], list[Path]]:
        def fake_merge(*_args: object) -> list[Path]:
            for rel, text in merge.items():
                (project / rel).parent.mkdir(parents=True, exist_ok=True)
                (project / rel).write_text(text)
            return [Path(c) for c in conflicts]

        with (
            patch("karotte.update_env._get_latest_version", return_value="2.0.0"),
            patch("karotte.update_env._generate_env"),
            patch("karotte.update_env._merge_projects", side_effect=fake_merge),
            patch("karotte.update_env.subprocess.run") as run,
        ):
            returned = update_env(project)
        locked = [
            Path(c.kwargs["cwd"]).relative_to(project)
            for c in run.call_args_list
            if c.args[0][:2] == ["uv", "lock"]
        ]
        return returned, locked

    def test_a_venv_whose_pyproject_changed_is_relocked(self, tmp_path: Path):
        _, locked = self._run(
            self._project(tmp_path), {"venvs/student/pyproject.toml": "# new\n"}
        )
        assert locked == [Path("."), Path("venvs/student")]

    def test_a_venv_added_by_the_merge_is_locked(self, tmp_path: Path):
        _, locked = self._run(
            self._project(tmp_path), {"venvs/grader/pyproject.toml": "# grader\n"}
        )
        assert locked == [Path("."), Path("venvs/grader")]

    def test_a_conflicted_venv_pyproject_is_left_alone(self, tmp_path: Path):
        conflicts = ["venvs/student/pyproject.toml"]
        returned, locked = self._run(
            self._project(tmp_path),
            {"venvs/student/pyproject.toml": "<<<<<<< ours\n"},
            conflicts,
        )
        assert locked == [Path(".")]
        assert returned == [Path(c) for c in conflicts]

    def test_a_failed_venv_relock_names_the_venv(self, tmp_path: Path):
        project = self._project(tmp_path)

        def fake_run(cmd: list[str], **kwargs: object):
            if kwargs.get("cwd") == project / "venvs" / "student":
                raise subprocess.CalledProcessError(1, cmd, stderr="401")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        def fake_merge(*_args: object) -> list[Path]:
            _ = (project / "venvs/student/pyproject.toml").write_text("# new\n")
            return []

        with (
            patch("karotte.update_env._get_latest_version", return_value="2.0.0"),
            patch("karotte.update_env._generate_env"),
            patch("karotte.update_env._merge_projects", side_effect=fake_merge),
            patch("karotte.update_env.subprocess.run", side_effect=fake_run),
            pytest.raises(RuntimeError, match="venvs/student.*401"),
        ):
            update_env(project)

    def test_a_failed_venv_relock_keeps_the_conflicts(self, tmp_path: Path):
        """Raising would drop them; the caller needs them."""
        project = self._project(tmp_path)

        def fake_run(cmd: list[str], **kwargs: object):
            if kwargs.get("cwd") == project / "venvs" / "student":
                raise subprocess.CalledProcessError(1, cmd, stderr="401")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        def fake_merge(*_args: object) -> list[Path]:
            _ = (project / "venvs/student/pyproject.toml").write_text("# new\n")
            _ = (project / "pyproject.toml").write_text("<<<<<<< ours\n")
            return [Path("pyproject.toml")]

        with (
            patch("karotte.update_env._get_latest_version", return_value="2.0.0"),
            patch("karotte.update_env._generate_env"),
            patch("karotte.update_env._merge_projects", side_effect=fake_merge),
            patch("karotte.update_env.subprocess.run", side_effect=fake_run),
        ):
            assert update_env(project) == [Path("pyproject.toml")]


def _mkdir_only(
    version: str,  # pyright: ignore[reportUnusedParameter]
    templates: list[str],  # pyright: ignore[reportUnusedParameter]
    output_dir: Path,
    extra_deps: Sequence[str] = (),  # pyright: ignore[reportUnusedParameter]
    uv_flags: Sequence[str] = (),  # pyright: ignore[reportUnusedParameter]
) -> None:
    """A `_generate_env` stand-in with its signature and none of its work."""
    output_dir.mkdir(parents=True, exist_ok=True)


class TestAddTemplate:
    """`update_env(..., add_templates=...)` renders the baseline with the
    manifest's template list and the target with the additions, so a new
    template's files land through the "adding new file" branch."""

    @staticmethod
    def _project(tmp_path: Path, version: str = "1.0.0") -> Path:
        project = tmp_path / "project"
        project.mkdir()
        manifest = {"karotte_version": version, "templates": ["default"]}
        (project / ".manifest.json").write_text(json.dumps(manifest))
        return project

    def test_baseline_excludes_and_target_includes_the_addition(self, tmp_path: Path):
        project = self._project(tmp_path)

        with (
            patch("karotte.update_env._get_latest_version", return_value="2.0.0"),
            patch("karotte.update_env._generate_env") as mock_gen,
            patch("karotte.update_env._merge_projects", return_value=[]),
            patch("karotte.update_env.subprocess.run"),
        ):
            mock_gen.side_effect = _mkdir_only
            update_env(project, add_templates=["language-toolchains"])

        baseline_call, target_call = mock_gen.call_args_list
        assert baseline_call[0][1] == ["default"]
        assert target_call[0][1] == ["default", "language-toolchains"]

    def test_proceeds_when_already_at_latest_version(self, tmp_path: Path):
        """The version-equality no-op must not block a template addition."""
        project = self._project(tmp_path, version="1.0.0")

        with (
            patch("karotte.update_env._get_latest_version", return_value="1.0.0"),
            patch("karotte.update_env._generate_env") as mock_gen,
            patch("karotte.update_env._merge_projects", return_value=[]),
            patch("karotte.update_env.subprocess.run"),
        ):
            mock_gen.side_effect = _mkdir_only
            update_env(project, add_templates=["language-toolchains"])

        assert mock_gen.call_count == 2

    def test_manifest_adopts_target_resolved_templates(self, tmp_path: Path):
        """The target render resolves template dependencies; the project
        manifest adopts that resolved list rather than the raw addition."""
        project = self._project(tmp_path)

        def fake_generate(
            version: str, templates: list[str], output_dir: Path, **_: object
        ) -> None:
            output_dir.mkdir(parents=True, exist_ok=True)
            if "language-toolchains" in templates:
                (output_dir / ".manifest.json").write_text(
                    json.dumps(
                        {
                            "karotte_version": version,
                            "templates": ["default", "language-toolchains"],
                        }
                    )
                )

        with (
            patch("karotte.update_env._get_latest_version", return_value="2.0.0"),
            patch("karotte.update_env._generate_env", side_effect=fake_generate),
            patch("karotte.update_env._merge_projects", return_value=[]),
            patch("karotte.update_env.subprocess.run"),
        ):
            update_env(project, add_templates=["language-toolchains"])

        manifest = json.loads((project / ".manifest.json").read_text())
        assert manifest["templates"] == ["default", "language-toolchains"]
        assert manifest["karotte_version"] == "2.0.0"

    def test_template_already_in_manifest_is_skipped(self, tmp_path: Path):
        project = self._project(tmp_path, version="1.0.0")

        with (
            patch("karotte.update_env._get_latest_version", return_value="1.0.0"),
            patch("karotte.update_env._generate_env") as mock_gen,
        ):
            conflicts = update_env(project, add_templates=["default"])

        assert conflicts == []
        mock_gen.assert_not_called()

    def test_only_missing_templates_are_added(self, tmp_path: Path):
        project = self._project(tmp_path)

        with (
            patch("karotte.update_env._get_latest_version", return_value="2.0.0"),
            patch("karotte.update_env._generate_env") as mock_gen,
            patch("karotte.update_env._merge_projects", return_value=[]),
            patch("karotte.update_env.subprocess.run"),
        ):
            mock_gen.side_effect = _mkdir_only
            update_env(project, add_templates=["default", "language-toolchains"])

        baseline_call, target_call = mock_gen.call_args_list
        assert baseline_call[0][1] == ["default"]
        assert target_call[0][1] == ["default", "language-toolchains"]

    def test_rejects_duplicate_additions(self, tmp_path: Path):
        project = self._project(tmp_path)

        with pytest.raises(ValueError, match="listed more than once"):
            update_env(project, add_templates=["a", "a"])

    def test_without_additions_same_version_still_noops(self, tmp_path: Path):
        project = self._project(tmp_path)

        with patch("karotte.update_env._get_latest_version", return_value="1.0.0"):
            conflicts = update_env(project)

        assert conflicts == []


def test_a_migration_that_fails_to_load_is_skipped(
    monkeypatch: pytest.MonkeyPatch,
):
    def broken() -> UpdateMigration:
        raise ImportError("missing dependency")

    ok = _Migration()
    monkeypatch.setattr(
        "karotte.update_env.entry_points",
        lambda *, group: [  # pyright: ignore[reportUnknownLambdaType]
            SimpleNamespace(name="broken", load=broken),
            SimpleNamespace(name="ok", load=lambda: ok),
        ],
    )
    warnings: list[str] = []
    handler = logger.add(lambda m: warnings.append(str(m)), level="WARNING")
    try:
        assert _migrations() == [ok]
    finally:
        logger.remove(handler)
    assert any("broken" in w and "missing dependency" in w for w in warnings)


class _Migration:
    """Stands in for a `karotte.update_migrations` plugin that moves envs off an
    older naming scheme."""

    def __init__(self, tool_args: dict[str, list[str]] | None = None) -> None:
        self.tool_args: dict[str, list[str]] = tool_args or {}
        self.calls: list[str] = []

    def prepare(self, project_dir: Path) -> None:
        self.calls.append("prepare")
        path = project_dir / ".manifest.json"
        manifest = json.loads(path.read_text())
        if "old_version" in manifest:
            manifest["karotte_version"] = manifest.pop("old_version")
            path.write_text(json.dumps(manifest))

    def tool(self, version: str) -> list[str] | None:
        return self.tool_args.get(version)

    def migrate(self, baseline_dir: Path, project_dir: Path, old_version: str) -> None:
        self.calls.append(f"migrate {old_version}")
        for root in (baseline_dir, project_dir):
            for path in (root / "pyproject.toml", root / "justfile"):
                if path.is_file():
                    path.write_text(path.read_text().replace("oldname", "karotte"))


class TestUpdateMigrations:
    """A migration plugin prepares the project, then rewrites the baseline and
    the project before the merge, so a rename never conflicts."""

    BASE_PYPROJECT: str = (
        '[project]\ndependencies = [\n    "oldname",\n]\n\n# oldname locks the venvs\n'
    )
    TARGET_PYPROJECT: str = (
        '[project]\ndependencies = [\n    "karotte",\n]\n\n# karotte locks the venvs\n'
    )
    OURS_PYPROJECT: str = (
        '[project]\ndependencies = [\n    "oldname",\n    "torch",\n]\n'
    )

    @staticmethod
    def _generate(version: str, templates: list[str], out: Path, **_: object) -> None:
        out.mkdir(parents=True)
        (out / ".manifest.json").write_text(
            json.dumps({"karotte_version": version, "templates": templates})
        )
        old = version.startswith("1.")
        (out / "pyproject.toml").write_text(
            TestUpdateMigrations.BASE_PYPROJECT
            if old
            else TestUpdateMigrations.TARGET_PYPROJECT
        )
        (out / "justfile").write_text(
            "test:\n  uv run oldname check\n"
            if old
            else "test:\n  uv run karotte check\n"
        )

    def _project(self, tmp_path: Path) -> Path:
        project = tmp_path / "env"
        (project / "src" / "environment" / "tasks").mkdir(parents=True)
        (project / ".manifest.json").write_text(
            json.dumps({"old_version": "1.4.0", "templates": ["default"]})
        )
        (project / "pyproject.toml").write_text(self.OURS_PYPROJECT)
        (project / "justfile").write_text("test:\n  uv run oldname check\n")
        return project

    def _update(self, project: Path, migrations: list[_Migration]) -> list[Path]:
        real_run = subprocess.run

        def run_but_not_uv(cmd: list[str], *args: object, **kwargs: object):
            if cmd[0] == "uv":
                return subprocess.CompletedProcess(cmd, 0, "", "")
            return real_run(cmd, *args, **kwargs)  # pyright: ignore[reportArgumentType, reportCallIssue]

        with (
            patch("karotte.update_env._migrations", return_value=migrations),
            patch("karotte.update_env._get_latest_version", return_value="3.0.1"),
            patch("karotte.update_env._generate_env", side_effect=self._generate),
            patch("karotte.update_env.subprocess.run", side_effect=run_but_not_uv),
        ):
            return update_env(project)

    def test_migration_merges_cleanly_and_keeps_user_edits(self, tmp_path: Path):
        project = self._project(tmp_path)
        migration = _Migration()

        conflicts = self._update(project, [migration])

        assert conflicts == []
        assert migration.calls == ["prepare", "migrate 1.4.0"]
        pyproject = (project / "pyproject.toml").read_text()
        assert '    "karotte",\n    "torch",\n' in pyproject
        assert "locks the venvs" not in pyproject, "user deletion must survive"
        assert (project / "justfile").read_text() == "test:\n  uv run karotte check\n"
        manifest = json.loads((project / ".manifest.json").read_text())
        assert manifest["karotte_version"] == "3.0.1"
        assert "old_version" not in manifest

    def test_without_a_migration_an_old_manifest_is_rejected(self, tmp_path: Path):
        project = self._project(tmp_path)

        with pytest.raises(ValidationError, match="karotte_version"):
            self._update(project, [])
