import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from karotte.env_source import (
    GitSource,
    SourceError,
    environment_dir,
    fetch,
    image_name_for,
    parse_source,
    pre_build_scripts,
    reachable_remote,
    run_pre_build,
    single_task_id,
    split_tree_path,
)


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        [
            "git",
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "-c",
            "init.defaultBranch=main",
            *args,
        ],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A repo with an env under envs/a on main, tag v1, and branch feat/x adding envs/b."""
    path = tmp_path / "repo"
    path.mkdir()
    _git(path, "init", "-q")
    (path / "envs" / "a").mkdir(parents=True)
    (path / "envs" / "a" / "pyproject.toml").write_text("[project]\n")
    (path / "README").write_text("r")
    _git(path, "add", ".")
    _git(path, "commit", "-qm", "one")
    _git(path, "tag", "-a", "v1", "-m", "v1")
    _git(path, "checkout", "-qb", "feat/x")
    (path / "envs" / "b").mkdir()
    (path / "envs" / "b" / "pyproject.toml").write_text("[project]\n")
    _git(path, "add", ".")
    _git(path, "commit", "-qm", "two")
    _git(path, "checkout", "-q", "main")
    return path


@pytest.fixture
def upload_pack(tmp_path: Path) -> tuple[str, Path]:
    """A git option that runs a program, and the file that program creates."""
    marker = tmp_path / "upload-pack-ran"
    program = tmp_path / "upload-pack"
    program.write_text(f'#!/bin/sh\ntouch {marker}\nexec git-upload-pack "$@"\n')
    program.chmod(0o755)
    return f"--upload-pack={program}", marker


class TestParseSource:
    def test_an_existing_folder(self, tmp_path: Path):
        assert parse_source(str(tmp_path)) == tmp_path

    def test_a_missing_path_that_is_no_url(self):
        with pytest.raises(SourceError, match="no-such-env"):
            parse_source("no-such-env")

    def test_an_existing_file_is_no_environment(self, tmp_path: Path):
        file = tmp_path / "f"
        file.write_text("")
        with pytest.raises(SourceError, match="not a folder"):
            parse_source(str(file))

    @pytest.mark.parametrize(
        "url",
        [
            "https://github.com/org/repo",
            "https://github.com/org/repo/",
            "https://github.com/org/repo.git",
            "https://www.github.com/org/repo",
        ],
    )
    def test_a_github_repo(self, url: str):
        assert parse_source(url) == GitSource("https://github.com/org/repo.git")

    def test_a_github_tree_url(self):
        assert parse_source(
            "https://github.com/org/repo/tree/feat/x/envs/a"
        ) == GitSource("https://github.com/org/repo.git", tree="feat/x/envs/a")

    def test_other_github_pages_are_rejected(self):
        with pytest.raises(SourceError, match="blob"):
            parse_source("https://github.com/org/repo/blob/main/README.md")

    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            (
                "git+https://gitlab.com/org/repo.git",
                GitSource("https://gitlab.com/org/repo.git"),
            ),
            (
                "git+https://gitlab.com/org/repo.git@v1.2",
                GitSource("https://gitlab.com/org/repo.git", ref="v1.2"),
            ),
            (
                "git+ssh://git@github.com/org/repo.git@feat/x#subdirectory=envs/a",
                GitSource(
                    "ssh://git@github.com/org/repo.git",
                    ref="feat/x",
                    subdirectory="envs/a",
                ),
            ),
            (
                "git+ssh://git@github.com/org/repo.git#subdirectory=envs/a",
                GitSource("ssh://git@github.com/org/repo.git", subdirectory="envs/a"),
            ),
            (
                "git+file:///srv/repo",
                GitSource("file:///srv/repo"),
            ),
        ],
    )
    def test_a_pip_style_git_url(self, source: str, expected: GitSource):
        assert parse_source(source) == expected

    @pytest.mark.parametrize(
        "source",
        ["git+--upload-pack=x", "git+file:///srv/repo@--upload-pack=x"],
        ids=["url", "ref"],
    )
    def test_a_url_or_ref_that_looks_like_a_git_option(self, source: str):
        with pytest.raises(SourceError, match="cannot start with '-'"):
            parse_source(source)


class TestSplitTreePath:
    REFS: frozenset[str] = frozenset({"main", "feat", "feat/x", "v1"})

    @pytest.mark.parametrize(
        ("tree", "expected"),
        [
            ("main", ("main", "")),
            ("main/envs/a", ("main", "envs/a")),
            ("feat/x/envs/a", ("feat/x", "envs/a")),
            ("feat/y", ("feat", "y")),
            ("a" * 40 + "/envs/a", ("a" * 40, "envs/a")),
        ],
    )
    def test_the_longest_matching_ref_wins(self, tree: str, expected: tuple[str, str]):
        assert split_tree_path(tree, self.REFS) == expected

    def test_an_unknown_ref(self):
        with pytest.raises(SourceError, match="nope"):
            split_tree_path("nope/envs/a", self.REFS)


class TestFetch:
    def test_the_default_branch(self, repo: Path, tmp_path: Path):
        checkout = fetch(f"file://{repo}", None, "", tmp_path / "cache")

        assert (checkout / "envs" / "a" / "pyproject.toml").is_file()
        assert not (checkout / "envs" / "b").exists()
        assert checkout.name == _git(repo, "rev-parse", "main")

    def test_a_branch_with_a_slash(self, repo: Path, tmp_path: Path):
        checkout = fetch(f"file://{repo}", "feat/x", "", tmp_path / "cache")

        assert (checkout / "envs" / "b" / "pyproject.toml").is_file()

    def test_an_annotated_tag_is_cached_by_its_commit(self, repo: Path, tmp_path: Path):
        checkout = fetch(f"file://{repo}", "v1", "", tmp_path / "cache")

        assert checkout.name == _git(repo, "rev-parse", "main")

    def test_a_commit(self, repo: Path, tmp_path: Path):
        commit = _git(repo, "rev-parse", "feat/x")

        checkout = fetch(f"file://{repo}", commit, "", tmp_path / "cache")

        assert checkout.name == commit
        assert (checkout / "envs" / "b").is_dir()

    def test_a_cached_commit_is_not_fetched_again(self, repo: Path, tmp_path: Path):
        commit = _git(repo, "rev-parse", "feat/x")
        first = fetch(f"file://{repo}", commit, "", tmp_path / "cache")

        with patch("karotte.env_source.subprocess.run") as git:
            second = fetch(f"file://{repo}", commit, "", tmp_path / "cache")

        assert second == first
        git.assert_not_called()

    def test_a_subdirectory_is_checked_out_sparsely(self, repo: Path, tmp_path: Path):
        checkout = fetch(f"file://{repo}", "feat/x", "envs/a", tmp_path / "cache")

        assert (checkout / "envs" / "a" / "pyproject.toml").is_file()
        assert not (checkout / "envs" / "b").exists()

    def test_another_subdirectory_of_a_cached_commit(self, repo: Path, tmp_path: Path):
        fetch(f"file://{repo}", "feat/x", "envs/a", tmp_path / "cache")

        checkout = fetch(f"file://{repo}", "feat/x", "envs/b", tmp_path / "cache")

        assert (checkout / "envs" / "b" / "pyproject.toml").is_file()

    def test_a_missing_ref(self, repo: Path, tmp_path: Path):
        with pytest.raises(SourceError, match="nope"):
            fetch(f"file://{repo}", "nope", "", tmp_path / "cache")

    def test_a_ref_that_looks_like_a_git_option_runs_nothing(
        self, repo: Path, tmp_path: Path, upload_pack: tuple[str, Path]
    ):
        option, ran = upload_pack
        with pytest.raises(SourceError):
            fetch(f"file://{repo}", option, "", tmp_path / "cache")

        assert not ran.exists()

    def test_a_url_that_looks_like_a_git_option_runs_nothing(
        self, tmp_path: Path, upload_pack: tuple[str, Path]
    ):
        option, ran = upload_pack
        with pytest.raises(SourceError):
            fetch(option, None, "", tmp_path / "cache")

        assert not ran.exists()

    def test_leaves_no_partial_checkouts(self, repo: Path, tmp_path: Path):
        cache = tmp_path / "cache"
        with pytest.raises(SourceError):
            fetch(f"file://{repo}", "nope", "", cache)

        assert not [p for p in cache.rglob("*") if p.name.startswith(".fetch")]


class TestEnvironmentDir:
    def test_a_folder(self, tmp_path: Path):
        (tmp_path / "pyproject.toml").write_text("")

        assert environment_dir(str(tmp_path), tmp_path / "cache") == (
            tmp_path.resolve(),
            tmp_path.name,
        )

    def test_a_folder_without_pyproject(self, tmp_path: Path):
        with pytest.raises(SourceError, match="pyproject.toml"):
            environment_dir(str(tmp_path), tmp_path / "cache")

    def test_a_subdirectory_of_a_git_repo(self, repo: Path, tmp_path: Path):
        path, name = environment_dir(
            f"git+file://{repo}@feat/x#subdirectory=envs/b", tmp_path / "cache"
        )

        assert (path / "pyproject.toml").is_file()
        assert path.parts[-2:] == ("envs", "b")
        assert name == "b"

    def test_a_whole_git_repo_is_named_after_it(self, repo: Path, tmp_path: Path):
        (repo / "pyproject.toml").write_text("")
        _git(repo, "add", ".")
        _git(repo, "commit", "-qm", "root env")

        _, name = environment_dir(f"git+file://{repo}", tmp_path / "cache")

        assert name == "repo"

    def test_a_missing_subdirectory(self, repo: Path, tmp_path: Path):
        with pytest.raises(SourceError, match="envs/nope"):
            environment_dir(
                f"git+file://{repo}#subdirectory=envs/nope", tmp_path / "cache"
            )


class TestImageName:
    def test_is_the_folder_name_and_a_hash_of_its_path(self, tmp_path: Path):
        name = image_name_for(tmp_path / "x", "HTML js_filter.v2")

        assert name.startswith("karotte-html-js_filter.v2-")
        assert len(name.rsplit("-", 1)[1]) == 8

    def test_differs_between_folders_of_the_same_name(self, tmp_path: Path):
        assert image_name_for(tmp_path / "a" / "env", "env") != image_name_for(
            tmp_path / "b" / "env", "env"
        )


class TestPreBuild:
    def test_defaults_to_setup_data(self, tmp_path: Path):
        (tmp_path / "setup_data.py").write_text("")

        assert pre_build_scripts(tmp_path, None) == ["setup_data.py"]

    def test_defaults_to_nothing_without_setup_data(self, tmp_path: Path):
        assert pre_build_scripts(tmp_path, None) == []

    def test_explicit_scripts_must_exist(self, tmp_path: Path):
        (tmp_path / "a.py").write_text("")

        with pytest.raises(SourceError, match="b.py"):
            pre_build_scripts(tmp_path, ["a.py", "b.py"])

    def test_runs_each_script_with_uv_in_the_environment(self, tmp_path: Path):
        with patch("karotte.env_source.subprocess.run") as run:
            run.return_value.returncode = 0
            run_pre_build(tmp_path, ["a.py", "b.py"])

        assert [c.args[0] for c in run.call_args_list] == [
            ["uv", "run", "a.py"],
            ["uv", "run", "b.py"],
        ]
        assert {c.kwargs["cwd"] for c in run.call_args_list} == {tmp_path}

    def test_a_failing_script_stops(self, tmp_path: Path):
        (tmp_path / "fail.py").write_text("raise SystemExit(3)")

        with (
            patch(
                "karotte.env_source.subprocess.run",
                side_effect=lambda *a, **k: subprocess.CompletedProcess(a, 3),  # pyright: ignore[reportUnknownLambdaType]
            ),
            pytest.raises(SourceError, match=r"fail\.py.*3"),
        ):
            run_pre_build(tmp_path, ["fail.py", "next.py"])


class TestSingleTaskId:
    def _tasks(self, *ids: str) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            [], 0, stdout=json.dumps([{"id": i} for i in ids])
        )

    def test_asks_the_environments_karotte(self, tmp_path: Path):
        with patch(
            "karotte.env_source.subprocess.run", return_value=self._tasks("only")
        ) as run:
            assert single_task_id(tmp_path) == "only"

        assert run.call_args.args[0] == [
            "uv",
            "run",
            "--frozen",
            "--directory",
            str(tmp_path),
            "karotte",
            "tasks",
            "list",
            "--json",
        ]

    def test_several_tasks_need_task(self, tmp_path: Path):
        with (
            patch(
                "karotte.env_source.subprocess.run", return_value=self._tasks("a", "b")
            ),
            pytest.raises(SourceError, match="--task.*a, b"),
        ):
            single_task_id(tmp_path)

    def test_no_tasks(self, tmp_path: Path):
        with (
            patch("karotte.env_source.subprocess.run", return_value=self._tasks()),
            pytest.raises(SourceError, match="no tasks"),
        ):
            single_task_id(tmp_path)

    def test_a_failing_listing(self, tmp_path: Path):
        with (
            patch(
                "karotte.env_source.subprocess.run",
                return_value=subprocess.CompletedProcess([], 1, stdout=""),
            ),
            pytest.raises(SourceError, match="list"),
        ):
            single_task_id(tmp_path)


class TestReachableRemote:
    def test_lists_the_refs_of_a_reachable_url(self, repo: Path):
        url, refs = reachable_remote(f"file://{repo}")

        assert url == f"file://{repo}"
        assert refs == {"main", "feat/x", "v1"}

    def test_a_url_that_looks_like_a_git_option_runs_nothing(
        self,
        repo: Path,
        tmp_path: Path,
        upload_pack: tuple[str, Path],
        monkeypatch: pytest.MonkeyPatch,
    ):
        # Inside a clone, a bare `git ls-remote` lists the clone's origin.
        _git(tmp_path, "clone", "-q", str(repo), "clone")
        monkeypatch.chdir(tmp_path / "clone")
        option, ran = upload_pack
        with pytest.raises(SourceError):
            reachable_remote(option)

        assert not ran.exists()

    def test_github_falls_back_to_ssh(self):
        def refs(url: str) -> set[str]:
            if url.startswith("https://"):
                raise SourceError("could not read Username")
            return {"main"}

        with patch("karotte.env_source._remote_refs", side_effect=refs):
            assert reachable_remote("https://github.com/org/repo.git") == (
                "git@github.com:org/repo.git",
                {"main"},
            )

    def test_other_hosts_have_no_fallback(self):
        with (
            patch(
                "karotte.env_source._remote_refs",
                side_effect=SourceError("could not read Username"),
            ) as refs,
            pytest.raises(SourceError, match="Username"),
        ):
            reachable_remote("https://gitlab.com/org/repo.git")

        refs.assert_called_once()

    def test_reports_every_failure(self):
        with (
            patch(
                "karotte.env_source._remote_refs",
                side_effect=[SourceError("https broke"), SourceError("ssh broke")],
            ),
            pytest.raises(SourceError, match="(?s)https broke.*ssh broke"),
        ):
            reachable_remote("https://github.com/org/repo.git")


def test_nested_uv_calls_ignore_the_callers_venv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("VIRTUAL_ENV", "/somewhere/.venv")
    with patch(
        "karotte.env_source.subprocess.run",
        return_value=subprocess.CompletedProcess(
            [], 0, stdout=json.dumps([{"id": "t"}])
        ),
    ) as run:
        single_task_id(tmp_path)
        run_pre_build(tmp_path, ["a.py"])

    assert all("VIRTUAL_ENV" not in c.kwargs["env"] for c in run.call_args_list)
    assert all("PATH" in c.kwargs["env"] for c in run.call_args_list)
