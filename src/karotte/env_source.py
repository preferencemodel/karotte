"""Resolve `karotte run`'s SOURCE argument to a local environment folder."""

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from loguru import logger

SOURCES_CACHE = (
    Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    / "karotte"
    / "sources"
)

DEFAULT_PRE_BUILD_SCRIPT = "setup_data.py"

_COMMIT = re.compile(r"[0-9a-f]{40}")
_GITHUB_HOSTS = ("github.com", "www.github.com")


class SourceError(Exception):
    pass


@dataclass(frozen=True)
class GitSource:
    url: str
    ref: str | None = None
    subdirectory: str = ""
    tree: str | None = None
    """A GitHub tree URL's `<ref>/<path>`, split once the remote's refs are known."""


def parse_source(source: str) -> Path | GitSource:
    path = Path(source)
    if path.exists():
        if not path.is_dir():
            raise SourceError(f"{source} is not a folder.")
        return path
    if source.startswith("git+"):
        return _parse_pip_url(source.removeprefix("git+"))
    parts = urlsplit(source)
    if parts.scheme in ("http", "https") and parts.netloc in _GITHUB_HOSTS:
        return _parse_github_url(source, parts.path)
    raise SourceError(
        f"{source} is no folder, GitHub URL or git+<url>[@<ref>][#subdirectory=<path>]."
    )


def _parse_pip_url(url: str) -> GitSource:
    parts = urlsplit(url)
    path, at, ref = parts.path.rpartition("@")
    if not at:
        path, ref = parts.path, None
    subdirectory = parse_qs(parts.fragment).get("subdirectory", [""])[0].strip("/")
    url = parts._replace(path=path, fragment="").geturl()
    for value in (url, ref):
        if value and value.startswith("-"):
            raise SourceError(f"A Git URL or ref cannot start with '-': {value}")
    return GitSource(url, ref or None, subdirectory)


def _parse_github_url(source: str, path: str) -> GitSource:
    segments = [s for s in path.split("/") if s]
    if len(segments) < 2 or (len(segments) > 2 and segments[2] != "tree"):
        raise SourceError(
            f"{source} is no GitHub repo or folder URL (.../<repo>/tree/<ref>/<path>)."
        )
    owner, repo = segments[0], segments[1].removesuffix(".git")
    url = f"https://github.com/{owner}/{repo}.git"
    tree = "/".join(segments[3:])
    return GitSource(url, tree=tree or None)


def split_tree_path(tree: str, refs: Collection[str]) -> tuple[str, str]:
    """Split `<ref>/<path>` at the longest of ``refs`` it starts with."""
    matches = [r for r in refs if tree == r or tree.startswith(r + "/")]
    if matches:
        ref = max(matches, key=len)
    elif _COMMIT.fullmatch(first := tree.split("/")[0]):
        ref = first
    else:
        raise SourceError(f"No branch, tag or commit starts {tree}.")
    return ref, tree[len(ref) :].strip("/")


def reachable_remote(url: str) -> tuple[str, set[str]]:
    """``url``, or for GitHub its SSH twin if HTTPS has no credentials, with its refs."""
    candidates = [url]
    if url.startswith("https://github.com/"):
        candidates.append("git@github.com:" + url.removeprefix("https://github.com/"))
    errors: list[str] = []
    for candidate in candidates:
        try:
            return candidate, _remote_refs(candidate)
        except SourceError as e:
            errors.append(str(e))
    raise SourceError("\n".join(errors))


def _remote_refs(url: str) -> set[str]:
    out = _git(
        None,
        "ls-remote",
        "--heads",
        "--tags",
        "--",
        url,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )
    names = (line.split("\t", 1)[1] for line in out.splitlines() if "\t" in line)
    return {
        re.sub(r"^refs/(heads|tags)/", "", name).removesuffix("^{}") for name in names
    }


def _git(cwd: Path | None, *args: str, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, env=env, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise SourceError(f"git {args[0]} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def fetch(url: str, ref: str | None, subdirectory: str, cache: Path) -> Path:
    """Check out ``ref`` (default: the remote's HEAD) of ``url`` into ``cache``,
    keyed by commit, and return the checkout."""
    repo_cache = cache / re.sub(r"[^A-Za-z0-9._-]+", "_", url).strip("_")
    repo_cache.mkdir(parents=True, exist_ok=True)
    if ref and _COMMIT.fullmatch(ref) and (repo_cache / ref).is_dir():
        return _with_subdirectory(repo_cache / ref, subdirectory)

    tmp = Path(tempfile.mkdtemp(prefix=".fetch-", dir=repo_cache))
    try:
        _git(tmp, "init", "-q")
        _git(tmp, "remote", "add", "--", "origin", url)
        _git(
            tmp,
            "fetch",
            "-q",
            "--depth",
            "1",
            "--filter=blob:none",
            "origin",
            "--",
            ref or "HEAD",
        )
        commit = _git(tmp, "rev-parse", "FETCH_HEAD^{commit}")
        checkout = repo_cache / commit
        if not checkout.is_dir():
            if subdirectory:
                _git(tmp, "sparse-checkout", "set", "--", subdirectory)
            _git(tmp, "checkout", "-q", "--detach", commit)
            try:
                tmp.rename(checkout)
            except OSError:
                if not checkout.is_dir():
                    raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return _with_subdirectory(checkout, subdirectory)


def _with_subdirectory(checkout: Path, subdirectory: str) -> Path:
    if subdirectory and not (checkout / subdirectory).exists():
        if (checkout / ".git" / "info" / "sparse-checkout").is_file():
            _git(checkout, "sparse-checkout", "add", "--", subdirectory)
        if not (checkout / subdirectory).is_dir():
            raise SourceError(f"The repo has no folder {subdirectory}.")
    return checkout


def environment_dir(source: str, cache: Path = SOURCES_CACHE) -> tuple[Path, str]:
    """The environment folder ``source`` names, fetched if remote, and its name."""
    parsed = parse_source(source)
    if isinstance(parsed, Path):
        path, name = parsed.resolve(), parsed.resolve().name
    else:
        url, refs = reachable_remote(parsed.url)
        if parsed.tree is not None:
            ref, subdirectory = split_tree_path(parsed.tree, refs)
            parsed = GitSource(url, ref, subdirectory)
        else:
            parsed = GitSource(url, parsed.ref, parsed.subdirectory)
        checkout = fetch(parsed.url, parsed.ref, parsed.subdirectory, cache)
        path = checkout / parsed.subdirectory if parsed.subdirectory else checkout
        repo = urlsplit(parsed.url).path.rstrip("/").rsplit("/", 1)[-1]
        name = Path(parsed.subdirectory).name or repo.removesuffix(".git")
    if not (path / "pyproject.toml").is_file():
        raise SourceError(
            f"{source} is no karotte environment: it has no pyproject.toml."
        )
    return path, name


def image_name_for(path: Path, name: str) -> str:
    slug = re.sub(r"[^a-z0-9._-]+", "-", name.lower()).strip("-._") or "env"
    digest = hashlib.sha256(str(path).encode()).hexdigest()[:8]
    return f"karotte-{slug}-{digest}"


def uv_env() -> dict[str, str]:
    """This process's environment, minus the venv uv would otherwise warn it ignores."""
    return {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}


def pre_build_scripts(path: Path, scripts: list[str] | None) -> list[str]:
    if scripts is None:
        default = path / DEFAULT_PRE_BUILD_SCRIPT
        return [DEFAULT_PRE_BUILD_SCRIPT] if default.is_file() else []
    missing = [s for s in scripts if not (path / s).is_file()]
    if missing:
        raise SourceError(f"The environment has no {', '.join(missing)}.")
    return scripts


def run_pre_build(path: Path, scripts: list[str]) -> None:
    for script in scripts:
        logger.info(f"Running {script}")
        result = subprocess.run(
            ["uv", "run", script], cwd=path, env=uv_env(), check=False
        )
        if result.returncode != 0:
            raise SourceError(f"{script} failed (exit code {result.returncode}).")


def single_task_id(path: Path) -> str:
    """The environment's only task, asked of the karotte it pins."""
    result = subprocess.run(
        [
            "uv",
            "run",
            "--frozen",
            "--directory",
            str(path),
            "karotte",
            "tasks",
            "list",
            "--json",
        ],
        stdout=subprocess.PIPE,
        env=uv_env(),
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise SourceError("Could not list the environment's tasks.")
    ids = [task["id"] for task in json.loads(result.stdout)]
    if not ids:
        raise SourceError("The environment has no tasks.")
    if len(ids) > 1:
        raise SourceError(
            f"The environment has several tasks; pass --task: {', '.join(ids)}."
        )
    return ids[0]
