import json
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import tomllib
from collections.abc import Collection, Mapping, Sequence
from importlib.metadata import entry_points
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Protocol
from urllib.parse import urlsplit

from loguru import logger

from karotte.create_env import EnvManifest
from karotte.uv_supply_chain import AGE_DELAY, env_with_age_delay, run_uv

# Files/directories to skip during merge
SKIP_PATTERNS = {
    ".git",
    "__pycache__",
    ".ruff_cache",
    ".venv",
    ".manifest.json",
    "uv.lock",
}

INDEX_OVERRIDE_VARS = (
    "UV_INDEX",
    "UV_DEFAULT_INDEX",
    "UV_INDEX_URL",
    "UV_EXTRA_INDEX_URL",
)
"""uv prefers these over a project's own indexes."""

MIGRATION_ENTRY_POINT_GROUP = "karotte.update_migrations"


class MergeFileError(RuntimeError):
    pass


class UpdateMigration(Protocol):
    """Moves an env made by an older release to the current names; registered
    under `karotte.update_migrations`."""

    def prepare(self, project_dir: Path) -> None:
        """Runs before the project's manifest is read."""
        ...

    def tool(self, version: str) -> Sequence[str] | None:
        """`uv tool run` args that render `version`, or None for `karotte@version`."""
        ...

    def migrate(self, baseline_dir: Path, project_dir: Path, old_version: str) -> None:
        """Runs on the old-version render and the project before the merge."""
        ...


def _migrations() -> list[UpdateMigration]:
    migrations: list[UpdateMigration] = []
    for ep in entry_points(group=MIGRATION_ENTRY_POINT_GROUP):
        try:
            migrations.append(ep.load())
        except Exception as e:  # noqa: BLE001 - a broken plugin must not break karotte
            logger.warning("Ignoring update migration {!r}: {}", ep.name, e)
    return migrations


def update_env(
    project_dir: Path,
    add_templates: Collection[str] = (),
    extra_with: Collection[str] = (),
) -> list[Path]:
    """
    Update a project to the latest karotte templates using 3-way merge.

    `add_templates` are template names to add to the project: the baseline is
    rendered with the manifest's list and the target with the additions, so a
    new template's files arrive as plain additions rather than as merges.

    `extra_with` are packages made available to the target render on top of the
    manifest's `extra_deps`.

    Returns a list of files with merge conflicts.
    """
    for var in INDEX_OVERRIDE_VARS:
        if os.environ.get(var):
            raise RuntimeError(
                f"{var} is set in your environment. uv prefers it over the indexes "
                + "in this project's pyproject.toml, so the update would lock "
                + "against the wrong index. Unset it and run again."
            )

    migrations = _migrations()
    for migration in migrations:
        migration.prepare(project_dir)

    manifest_path = project_dir / ".manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No .manifest.json found in {project_dir}. "
            + "This project may not have been created with karotte, "
            + "or was created before manifest support was added."
        )

    manifest = EnvManifest.model_validate_json(manifest_path.read_text())
    old_version = manifest.karotte_version
    templates = manifest.templates

    add_templates = list(add_templates)
    duplicates = {t for t in add_templates if add_templates.count(t) > 1}
    if duplicates:
        raise ValueError(
            f"Template(s) listed more than once: {', '.join(sorted(duplicates))}"
        )
    present = [t for t in add_templates if t in templates]
    if present:
        logger.info(f"Template(s) already in the manifest: {', '.join(present)}")
        add_templates = [t for t in add_templates if t not in templates]
    target_templates = [*templates, *add_templates]

    os.environ.update(_index_credentials(project_dir))
    uv_flags = _env_uv_flags(project_dir)
    current_version = _get_latest_version(uv_flags)
    target_extra_deps = _target_extra_deps(manifest.extra_deps, extra_with)

    if old_version == current_version and not add_templates:
        latest_extra_deps = (
            _resolve_extra_deps(current_version, target_extra_deps, uv_flags)
            if target_extra_deps
            else []
        )
        if sorted(latest_extra_deps) == sorted(manifest.extra_deps):
            logger.info(
                f"Project is already at version {current_version}, nothing to do."
            )
            return []
        logger.info(
            f"karotte {current_version} is current; template packages "
            + f"{manifest.extra_deps} -> {latest_extra_deps}"
        )
    else:
        logger.info(f"Updating from karotte {old_version} to {current_version}")
    if add_templates:
        logger.info(f"Adding template(s): {', '.join(add_templates)}")

    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        project_name = project_dir.name
        baseline_dir = tmp_path / "baseline" / project_name
        target_dir = tmp_path / "target" / project_name

        # Generate baseline (old version)
        logger.info(f"Generating baseline with {' '.join(_tool_args(old_version))}")
        _generate_env(
            old_version,
            templates,
            baseline_dir,
            extra_deps=manifest.extra_deps,
            uv_flags=uv_flags,
        )

        # Generate target (current version, plus any templates being added)
        logger.info(f"Generating target with {' '.join(_tool_args(current_version))}")
        _generate_env(
            current_version,
            target_templates,
            target_dir,
            extra_deps=target_extra_deps,
            uv_flags=uv_flags,
        )

        for migration in migrations:
            migration.migrate(baseline_dir, project_dir, old_version)

        # 3-way merge the do-not-recreate list: apply the template's delta
        # (old -> new version) to the user-edited project list, so newly
        # shipped opt-out paths are picked up and removed ones dropped, while
        # the user's own edits survive.
        do_not_recreate = _merge_do_not_recreate(
            base=_read_do_not_recreate(baseline_dir),
            theirs=_read_do_not_recreate(target_dir),
            ours=manifest.do_not_recreate_if_deleted,
        )

        venvs_before = _venv_pyprojects(project_dir)

        # Perform 3-way merge
        conflicts = _merge_projects(
            project_dir, baseline_dir, target_dir, do_not_recreate
        )

        # The target render resolved template dependencies; its manifest holds
        # the ordered result, which is what the project should record.
        resolved_templates = _read_templates(target_dir) or target_templates
        extra_deps = _read_extra_deps(target_dir)
        if extra_deps is None:
            extra_deps = target_extra_deps

    # Update manifest with new version and merged do-not-recreate list
    manifest.karotte_version = current_version
    manifest.templates = resolved_templates
    manifest.extra_deps = extra_deps
    manifest.do_not_recreate_if_deleted = sorted(do_not_recreate)
    manifest_path.write_text(manifest.model_dump_json(indent=2))

    # Conflict markers make pyproject.toml unparseable, so skip the relock
    # rather than raise and lose the conflicts we are about to return.
    pyproject = project_dir / "pyproject.toml"
    pyproject_conflicted = Path("pyproject.toml") in conflicts
    if pyproject.exists() and pyproject_conflicted:
        logger.warning(
            "Skipping uv.lock regeneration: pyproject.toml has unresolved "
            + "merge conflicts, so it is not parseable TOML. Once they are "
            + "resolved, run `uv lock --upgrade-package karotte` — plain "
            + "`uv lock` keeps the karotte already in the lock, so the manifest "
            + "(just advanced above) and the lock would stay out of step, and "
            + "`uv run karotte update` would keep running the older karotte."
        )
    elif pyproject.exists():
        logger.info("Regenerating uv.lock")
        try:
            subprocess.run(
                ["uv", "lock", "--upgrade-package", "karotte"],
                cwd=project_dir,
                capture_output=True,
                text=True,
                check=True,
                env=_uv_env(os.environ),
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to regenerate uv.lock: {e.stderr.strip()}"
            ) from e

    _relock_changed_venvs(project_dir, venvs_before, conflicts)

    if conflicts:
        logger.warning(f"Merge conflicts in {len(conflicts)} file(s):")
        for path in conflicts:
            logger.warning(f"  - {path}")
        logger.warning("Please resolve conflicts manually.")
    else:
        logger.info("Update completed successfully.")

    return conflicts


def _venv_pyprojects(project_dir: Path) -> dict[Path, str]:
    return {
        p.relative_to(project_dir): p.read_text()
        for p in project_dir.glob("venvs/*/pyproject.toml")
    }


def _relock_changed_venvs(
    project_dir: Path, before: dict[Path, str], conflicts: Collection[Path]
) -> None:
    """Lock each venv whose pyproject the merge added or changed; its old lock
    may point at indexes the pyproject no longer declares. With conflicts, a
    failure is only logged so the conflicts still reach the caller."""
    for rel, text in sorted(_venv_pyprojects(project_dir).items()):
        if before.get(rel) == text or rel in conflicts:
            continue
        venv = rel.parent
        logger.info(f"Regenerating {venv}/uv.lock")
        try:
            subprocess.run(
                ["uv", "lock"],
                cwd=project_dir / venv,
                capture_output=True,
                text=True,
                check=True,
                env=_uv_env(os.environ),
            )
        except subprocess.CalledProcessError as e:
            message = f"Failed to regenerate {venv}/uv.lock: {e.stderr.strip()}"
            if not conflicts:
                raise RuntimeError(message) from e
            logger.warning(
                f"{message}\nRun `uv lock` in {venv} once the conflicts are resolved."
            )


def _merge_do_not_recreate(
    base: Collection[str],
    theirs: Collection[str],
    ours: Collection[str] | None,
) -> set[str]:
    """3-way merge a do-not-recreate list.

    When `ours` is None the manifest predates the field and was never seeded, so
    adopt the current template's list (`theirs`) wholesale. Otherwise apply the
    template's delta (`base` -> `theirs`) to the user-edited list (`ours`): paths
    the new template added are picked up, paths it removed are dropped, and the
    user's own additions/removals are otherwise preserved.
    """
    if ours is None:
        return set(theirs)
    base, theirs, ours = set(base), set(theirs), set(ours)
    return (ours | (theirs - base)) - (base - theirs)


_REQUIREMENT_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)(\[[^\]]*\])?")


def _unpinned(requirement: str) -> str:
    """`name[extras]` without the version specifier."""
    match = _REQUIREMENT_NAME.match(requirement)
    return match.group(1) + (match.group(2) or "") if match else requirement


def _package_name(requirement: str) -> str:
    match = _REQUIREMENT_NAME.match(requirement)
    return match.group(1) if match else requirement


def _target_extra_deps(
    recorded: Sequence[str], extra_with: Collection[str]
) -> list[str]:
    """Packages for the target render: the manifest's, unpinned so the newest
    release is used, then `extra_with`."""
    return list(dict.fromkeys([_unpinned(r) for r in recorded] + list(extra_with)))


def _read_templates(env_dir: Path) -> list[str]:
    """Read the resolved template list from a generated env's manifest."""
    manifest_path = env_dir / ".manifest.json"
    if not manifest_path.exists():
        return []
    return EnvManifest.model_validate_json(manifest_path.read_text()).templates


def _read_extra_deps(env_dir: Path) -> list[str] | None:
    manifest_path = env_dir / ".manifest.json"
    if not manifest_path.exists():
        return None
    return EnvManifest.model_validate_json(manifest_path.read_text()).extra_deps


def _read_do_not_recreate(env_dir: Path) -> list[str]:
    """Read the do-not-recreate list from a generated env's manifest.

    Returns an empty list for envs generated by a karotte version predating the
    field (it parses as None).
    """
    manifest_path = env_dir / ".manifest.json"
    if not manifest_path.exists():
        return []
    return (
        EnvManifest.model_validate_json(
            manifest_path.read_text()
        ).do_not_recreate_if_deleted
        or []
    )


_INDEX_HINT = (
    "If that reads as an authentication failure, check access to the indexes "
    + "in this project's pyproject.toml."
)
_UNDATED_INDEX_HINT = (
    f"An index in this project's pyproject.toml serves no upload times, so the {AGE_DELAY} "
    + "age delay drops every package from it. Point pyproject.toml at an index that "
    + "serves upload times."
)


def _index_hint(stderr: str) -> str:
    return _UNDATED_INDEX_HINT if "missing an upload date" in stderr else _INDEX_HINT


def _env_uv_flags(project_dir: Path) -> list[str]:
    """The project's non-explicit indexes and keyring provider as uv flags, which
    `uv tool run` would otherwise ignore."""
    try:
        doc = tomllib.loads((project_dir / "pyproject.toml").read_text())
    except (FileNotFoundError, tomllib.TOMLDecodeError):
        return []
    uv = doc.get("tool", {}).get("uv", {})
    flags: list[str] = []
    for index in uv.get("index", []):
        if index.get("explicit") or "url" not in index:
            continue
        if index.get("default"):
            flags.append(f"--default-index={index['url']}")
        elif "name" in index:
            flags.append(f"--index={index['name']}={index['url']}")
    if provider := uv.get("keyring-provider"):
        flags.append(f"--keyring-provider={provider}")
    return flags


def _index_credentials(project_dir: Path) -> dict[str, str]:
    """uv's per-index credential vars for the project's indexes that uv would ask
    keyring about, asked for once here: the `uv tool run` environments below put
    a `keyring` without the needed backend first on PATH."""
    try:
        doc = tomllib.loads((project_dir / "pyproject.toml").read_text())
    except (FileNotFoundError, tomllib.TOMLDecodeError):
        return {}
    uv = doc.get("tool", {}).get("uv", {})
    provider = os.environ.get("UV_KEYRING_PROVIDER", uv.get("keyring-provider"))
    env = _uv_env(os.environ)
    keyring = shutil.which("keyring", path=env.get("PATH"))
    if provider != "subprocess" or keyring is None:
        return {}
    creds: dict[str, str] = {}
    for index in uv.get("index", []):
        url = urlsplit(str(index.get("url", "")))
        if "name" not in index or not url.username or url.password:
            continue
        var = "UV_INDEX_" + re.sub(r"[^A-Z0-9]", "_", str(index["name"]).upper())
        if f"{var}_PASSWORD" in os.environ:
            continue
        host = url.netloc.rpartition("@")[2]
        for service in (url._replace(netloc=host).geturl(), host):
            result = subprocess.run(
                [keyring, "get", service, url.username],
                capture_output=True,
                text=True,
                env=env,
            )
            if result.returncode == 0 and result.stdout.strip():
                creds[f"{var}_USERNAME"] = url.username
                creds[f"{var}_PASSWORD"] = result.stdout.strip()
                break
    return creds


def _run_uv_tool(
    *args: str, uv_flags: Sequence[str] = (), **kwargs: Any
) -> subprocess.CompletedProcess[str]:
    """Run `uv tool run <args>` with the project's index flags and the age-delay,
    neither of which an isolated tool environment reads from the project."""
    return run_uv(
        ("tool", "run"),
        *uv_flags,
        *args,
        capture_output=True,
        text=True,
        check=True,
        env=_uv_env(env_with_age_delay()),
        **kwargs,
    )


def _uv_env(base: Mapping[str, str]) -> dict[str, str]:
    """`base` without this venv's scripts dir on PATH, where karotte's dependencies
    put a `keyring` that would shadow the one uv needs for index credentials."""
    env = dict(base)
    if sys.prefix == sys.base_prefix:
        return env
    scripts = os.path.normpath(sysconfig.get_path("scripts"))
    env["PATH"] = os.pathsep.join(
        entry
        for entry in env.get("PATH", "").split(os.pathsep)
        if os.path.normpath(entry) != scripts
    )
    return env


def _get_latest_version(uv_flags: Sequence[str] = ()) -> str:
    """Get the latest available version of karotte using uv tool run."""
    try:
        result = _run_uv_tool("karotte@latest", "--version", uv_flags=uv_flags)
    except FileNotFoundError:
        raise RuntimeError(
            "Failed to check latest karotte version: 'uv' not found on PATH"
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to check latest karotte version: {e.stderr.strip()}\n{_index_hint(e.stderr)}"
        ) from e
    return result.stdout.strip()


def _tool_args(version: str) -> list[str]:
    """The `uv tool run` args that run the release `version`."""
    for migration in _migrations():
        args = migration.tool(version)
        if args is not None:
            return list(args)
    return [f"karotte@{version}"]


def _with_flags(extra_deps: Sequence[str]) -> list[str]:
    """`--with` for each package, exempt from the age delay: template packages
    are first-party by construction and a fresh release must be visible."""
    return [arg for dep in extra_deps for arg in ("--with", dep)] + [
        f"--exclude-newer-package={_package_name(dep)}=false" for dep in extra_deps
    ]


def _resolve_extra_deps(
    version: str, extra_deps: Sequence[str], uv_flags: Sequence[str] = ()
) -> list[str]:
    """The `name==version` each of `extra_deps` resolves to next to `karotte@version`,
    as recorded by its templates; packages providing none are dropped."""
    result = _run_uv_tool(
        *_with_flags(extra_deps),
        *_tool_args(version),
        "templates",
        "list",
        "--json",
        uv_flags=uv_flags,
    )
    listed = {t["requirement"] for t in json.loads(result.stdout) if t["requirement"]}
    wanted = {_package_name(d) for d in extra_deps}
    return sorted(r for r in listed if _package_name(r) in wanted)


def _generate_env(
    version: str,
    templates: list[str],
    output_dir: Path,
    *,
    extra_deps: Sequence[str] = (),
    uv_flags: Sequence[str] = (),
) -> None:
    """Generate an environment using a specific karotte version."""
    try:
        _run_uv_tool(
            *_with_flags(extra_deps),
            *_tool_args(version),
            "create-env",
            str(output_dir),
            *[f"--template={t}" for t in templates],
            "--no-lock",
            uv_flags=uv_flags,
        )
    except subprocess.CalledProcessError as e:
        hint = _index_hint(e.stderr)
        if "not found" in e.stderr and "emplate" in e.stderr:
            hint = (
                "If the template comes from another package, pass it with "
                + "`karotte update --with <package>`."
            )
        raise RuntimeError(
            f"Failed to generate environment with {' '.join(_tool_args(version))}: "
            + f"{e.stderr.strip()}\n{hint}"
        ) from e


def _should_skip(rel_path: Path) -> bool:
    """Check if a file should be skipped during merge."""
    parts = rel_path.parts
    for pattern in SKIP_PATTERNS:
        if pattern in parts or str(rel_path) == pattern:
            return True
    return False


def _do_not_recreate(rel_path: Path, do_not_recreate_paths: Collection[str]) -> bool:
    """Whether an absent path should be left absent rather than (re)created."""
    rel_str = rel_path.as_posix()
    for pattern in do_not_recreate_paths:
        if rel_str == pattern or rel_str.startswith(pattern + "/"):
            return True
    return False


def _is_binary(file_path: Path) -> bool:
    """Check if a file is binary by reading first chunk."""
    try:
        with open(file_path, "rb") as f:
            chunk = f.read(8192)
            return b"\x00" in chunk
    except OSError:
        return False


def _merge_projects(
    project: Path,
    baseline: Path,
    target: Path,
    do_not_recreate_paths: Collection[str] = (),
) -> list[Path]:
    """
    3-way merge between baseline, target, and current project.

    `do_not_recreate_paths` are repo-relative paths (files or directories) that
    must not be (re)created in the project if they are absent there.

    Returns list of files with conflicts.
    """
    conflicts: list[Path] = []

    baseline_files = {
        f.relative_to(baseline)
        for f in baseline.rglob("*")
        if f.is_file() and not _should_skip(f.relative_to(baseline))
    }
    target_files = {
        f.relative_to(target)
        for f in target.rglob("*")
        if f.is_file() and not _should_skip(f.relative_to(target))
    }
    all_files = baseline_files | target_files

    for relative_path in sorted(all_files):
        baseline_file = baseline / relative_path
        target_file = target / relative_path
        project_file = project / relative_path

        in_baseline = baseline_file.exists()
        in_target = target_file.exists()

        if in_baseline and in_target:
            # File exists in both versions
            if not project_file.exists():
                if _do_not_recreate(relative_path, do_not_recreate_paths):
                    # User deleted it and the env opts out of recreating it.
                    logger.info(f"Preserving user deletion: {relative_path}")
                    continue
                # User deleted it, recreate from target
                logger.info(f"Recreating deleted file: {relative_path}")
                project_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(target_file, project_file)
            elif _is_binary(target_file):
                # Binary file: copy from target
                logger.info(f"Updating binary file: {relative_path}")
                shutil.copy(target_file, project_file)
            else:
                # Text file: 3-way merge
                try:
                    has_conflict = _merge_file(project_file, baseline_file, target_file)
                except MergeFileError as e:
                    logger.warning(
                        f"Could not merge {relative_path}, kept your version; "
                        + f"apply the template's changes by hand: {e}"
                    )
                    has_conflict = True
                if has_conflict:
                    conflicts.append(relative_path)

        elif in_target and not in_baseline:
            if not project_file.exists() and _do_not_recreate(
                relative_path, do_not_recreate_paths
            ):
                # Opt-out path the project doesn't have; don't add it.
                logger.info(f"Skipping opt-out path not in project: {relative_path}")
                continue
            # New file in template: add to project
            logger.info(f"Adding new file: {relative_path}")
            project_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(target_file, project_file)

        elif in_baseline and not in_target:
            # File removed from template: delete from project
            if project_file.exists():
                logger.info(f"Removing deleted file: {relative_path}")
                project_file.unlink()

    return conflicts


def _merge_file(current: Path, base: Path, other: Path) -> bool:
    """3-way merge `other`'s changes since `base` into `current`; True on conflicts.

    Raises MergeFileError, leaving `current` untouched, if git cannot merge.
    """
    result = subprocess.run(
        ["git", "merge-file", "-p", str(current), str(base), str(other)],
        capture_output=True,
        text=True,
    )
    # Exit status is the conflict count, capped at 127; errors exit negative (255).
    if not 0 <= result.returncode <= 127:
        raise MergeFileError(result.stderr.strip())
    current.write_text(result.stdout)
    return result.returncode > 0
