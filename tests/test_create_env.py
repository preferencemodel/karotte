import contextlib
import re
import shutil
import subprocess
import tomllib
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

from karotte import create_env as create_env_module
from karotte.create_env import (
    EnvManifest,
    create_env,
    resolve_template_deps,
)
from karotte.schemas.environment_template import EnvironmentTemplate
from karotte.templates import TEMPLATES_DIR


def _read_manifest(env_dir: Path) -> EnvManifest:
    return EnvManifest.model_validate_json((env_dir / ".manifest.json").read_text())


def test_manifest_records_agents(tmp_path: Path):
    with patch("karotte.create_env.subprocess.check_call"):
        create_env(tmp_path / "my_env", templates=["default"], agents=["mistral-vibe"])
    assert _read_manifest(tmp_path / "my_env").agents == ["mistral-vibe"]


def test_manifest_agents_default_empty(tmp_path: Path):
    with patch("karotte.create_env.subprocess.check_call"):
        create_env(tmp_path / "my_env", templates=["default"])
    assert _read_manifest(tmp_path / "my_env").agents == []


def test_create_env_rejects_unknown_agent(tmp_path: Path):
    with pytest.raises(ValueError, match="Unknown agent"):
        create_env(tmp_path / "my_env", templates=["default"], agents=["nope"])


def test_manifest_seeds_do_not_recreate_from_templates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The do-not-recreate list is baked into the env from its templates."""
    templates = tmp_path / "templates"
    for name, toml in {
        "base": 'description = "b"\ndo_not_recreate_if_deleted = ["a", "shared"]\n',
        "overlay": 'description = "o"\nrequires = ["base"]\n'
        + 'do_not_recreate_if_deleted = ["b", "shared"]\n',
    }.items():
        (templates / name).mkdir(parents=True)
        (templates / name / "template.toml").write_text(toml)
        (templates / name / f"{name}.txt").write_text(name)
    monkeypatch.setattr(create_env_module, "TEMPLATES_DIR", templates)

    create_env(tmp_path / "my_env", templates=["overlay"], no_lock=True)

    manifest = _read_manifest(tmp_path / "my_env")
    assert manifest.do_not_recreate_if_deleted == ["a", "b", "shared"]


def test_post_create_runs_from_a_relative_output_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    with (
        patch("karotte.create_env.subprocess.check_call"),
        patch("karotte.uv_supply_chain.subprocess.run") as run,
    ):
        create_env(Path("my_env"), templates=["default"])

    (call,) = [c for c in run.call_args_list if "run" in c.args[0]]
    script = Path(call.kwargs["cwd"], call.args[0][-1])
    assert script.is_file(), call.args[0]


def test_manifest_without_agents_field_parses():
    """Manifests written before agent support still load (additive field)."""
    manifest = EnvManifest.model_validate_json(
        '{"karotte_version": "0.0.0", "templates": ["default"]}'
    )
    assert manifest.agents == []


def test_creates_output_directory_with_template_files(tmp_path: Path):
    with patch("karotte.create_env.subprocess.check_call"):  # Patch "uv lock" call
        create_env(tmp_path / "my_env", templates=["default"], vendor_karotte=True)

    target_dir = tmp_path / "my_env"
    assert target_dir.exists()
    assert (target_dir / "Containerfile").exists()
    assert (target_dir / "venvs" / "student" / "pyproject.toml").exists()
    assert (target_dir / "justfile").exists()
    assert (target_dir / "pyproject.toml").exists()


def test_vendored_copy_builds_a_wheel(tmp_path: Path):
    with patch("karotte.create_env.subprocess.check_call"):
        create_env(tmp_path / "my_env", templates=["default"], vendor_karotte=True)
    vendored = tmp_path / "my_env" / ".karotte"
    assert (vendored / "LICENSE").read_text().startswith("MIT License")
    subprocess.run(
        ["uv", "build", "--wheel", "-o", str(tmp_path / "dist"), str(vendored)],
        check=True,
        capture_output=True,
    )
    [wheel] = (tmp_path / "dist").glob("karotte-*.whl")
    with zipfile.ZipFile(wheel) as zf:
        [metadata] = [n for n in zf.namelist() if n.endswith(".dist-info/METADATA")]
        assert "License-Expression: MIT" in zf.read(metadata).decode().splitlines()


def test_containerfile_removes_student_subordinate_uid_ranges(tmp_path: Path):
    """Student must not have subordinate UID/GID ranges, otherwise they can
    use user namespaces to remap to a different host UID and bypass iptables."""
    with patch("karotte.create_env.subprocess.check_call"):
        create_env(tmp_path / "my_env", templates=["default"], vendor_karotte=True)

    containerfile = (tmp_path / "my_env" / "Containerfile").read_text()
    assert "sed -i '/^student:/d' /etc/subuid /etc/subgid" in containerfile


def test_containerfile_hardens_the_root_venv_lock(tmp_path: Path):
    """`uv` creates `<venv>/.lock` mode 0666 and never lowers an existing mode.

    The root venv is built by `uv sync` and touched again by `uv pip install`,
    so without an explicit chmod the image ships a world-writable file outside
    the workdir. Deleting it is not enough: the next uv run against the venv
    recreates it at 0666, whereas a 0644 lock is left alone.
    """
    with patch("karotte.create_env.subprocess.check_call"):
        create_env(tmp_path / "my_env", templates=["default"], vendor_karotte=True)

    containerfile = (tmp_path / "my_env" / "Containerfile").read_text()
    assert "chmod 0644 ${ROOT_WORKDIR}/.venv/.lock" in containerfile


def test_check_permissions_scans_for_other_writable_paths_as_root(tmp_path: Path):
    """The student-run writable-surface scan cannot see inside 0700 dirs.

    `find` running as the student stops at /root, so a world-writable file in
    there is invisible to it. The same scan has to run as root to catch a hole
    that is currently closed only by its parent directory's mode.
    """
    with patch("karotte.create_env.subprocess.check_call"):
        create_env(tmp_path / "my_env", templates=["default"], vendor_karotte=True)

    script = (
        tmp_path / "my_env" / "src" / "environment" / "check_permissions.py"
    ).read_text()
    scan = next(
        (block for block in script.split("\n\n") if "-perm -002" in block),
        None,
    )
    assert scan is not None, "no other-writable scan in check_permissions"
    assert "runuser" not in scan, "the scan must run as root, not as the student"


def test_raises_error_if_target_directory_exists(tmp_path: Path):
    (tmp_path / "my_env").mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        create_env(tmp_path, templates=["default"])


def test_raises_error_if_template_does_not_exist(tmp_path: Path):
    with pytest.raises(ValueError, match="not found"):
        create_env(tmp_path / "my_env", templates=["nonexistent_template"])


def test_renders_jinja_variables_in_templates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Test that Jinja variables are rendered in the template files."""
    from karotte import create_env as create_env_module

    # Create a temporary template directory with a file containing Jinja variables
    templates_dir = tmp_path / "templates" / "test"
    templates_dir.mkdir(parents=True)
    (templates_dir / "template.toml").write_text('description = "Test template"\n')
    (templates_dir / "config.txt").write_text("Hello, {{ name }}!")

    monkeypatch.setattr(create_env_module, "TEMPLATES_DIR", tmp_path / "templates")

    # This test verifies the current behavior (no variables passed)
    # The template will render with empty/undefined variables
    create_env(tmp_path / "my_env", templates=["test"])

    assert (tmp_path / "my_env" / "config.txt").exists()


def test_preserves_subdirectory_structure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Test that nested directories in templates are preserved."""
    from karotte import create_env as create_env_module

    # Create a template with nested directories
    templates_dir = tmp_path / "templates" / "nested"
    (templates_dir / "subdir" / "deep").mkdir(parents=True)
    (templates_dir / "template.toml").write_text('description = "Test template"\n')
    (templates_dir / "root.txt").write_text("root file")
    (templates_dir / "subdir" / "mid.txt").write_text("mid file")
    (templates_dir / "subdir" / "deep" / "leaf.txt").write_text("leaf file")

    monkeypatch.setattr(create_env_module, "TEMPLATES_DIR", tmp_path / "templates")

    create_env(tmp_path / "my_env", templates=["nested"])

    target_dir = tmp_path / "my_env"
    assert (target_dir / "root.txt").read_text() == "root file"
    assert (target_dir / "subdir" / "mid.txt").read_text() == "mid file"
    assert (target_dir / "subdir" / "deep" / "leaf.txt").read_text() == "leaf file"


def test_template_toml_is_not_copied_to_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from karotte import create_env as create_env_module

    templates_dir = tmp_path / "templates" / "mytemplate"
    templates_dir.mkdir(parents=True)
    (templates_dir / "file.txt").write_text("content")
    (templates_dir / "template.toml").write_text(
        'description = "desc"\nrequires = []\n'
    )

    monkeypatch.setattr(create_env_module, "TEMPLATES_DIR", tmp_path / "templates")

    create_env(tmp_path / "my_env", templates=["mytemplate"])

    assert not (tmp_path / "my_env" / "template.toml").exists()
    assert (tmp_path / "my_env" / "file.txt").exists()


def test_templates_installed_under_a_venv_are_copied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    templates_dir = tmp_path / ".venv" / "site-packages" / "templates"
    (templates_dir / "t" / ".venv").mkdir(parents=True)
    (templates_dir / "t" / "template.toml").write_text('description = "t"\n')
    (templates_dir / "t" / "file.txt").write_text("content")
    (templates_dir / "t" / ".venv" / "junk").write_text("skip me")

    monkeypatch.setattr(create_env_module, "TEMPLATES_DIR", templates_dir)

    create_env(tmp_path / "my_env", templates=["t"])

    assert (tmp_path / "my_env" / "file.txt").read_text() == "content"
    assert not (tmp_path / "my_env" / ".venv").exists()


def test_raises_error_if_required_template_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from karotte import create_env as create_env_module

    templates_dir = tmp_path / "templates"
    (templates_dir / "overlay").mkdir(parents=True)
    (templates_dir / "overlay" / "template.toml").write_text(
        'description = "desc"\nrequires = ["base"]\n'
    )

    monkeypatch.setattr(create_env_module, "TEMPLATES_DIR", templates_dir)

    with pytest.raises(ValueError, match="'base' not found"):
        create_env(tmp_path / "my_env", templates=["overlay"])


def test_later_templates_override_earlier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Later templates override files from earlier templates."""
    from karotte import create_env as create_env_module

    templates_dir = tmp_path / "templates"

    # Base template with two files
    (templates_dir / "base").mkdir(parents=True)
    (templates_dir / "base" / "template.toml").write_text(
        'description = "Base template"\n'
    )
    (templates_dir / "base" / "shared.txt").write_text("from base")
    (templates_dir / "base" / "base_only.txt").write_text("only in base")

    # Overlay template overrides one file
    (templates_dir / "overlay").mkdir(parents=True)
    (templates_dir / "overlay" / "template.toml").write_text(
        'description = "Overlay template"\nrequires = ["base"]\n'
    )
    (templates_dir / "overlay" / "shared.txt").write_text("from overlay")

    monkeypatch.setattr(create_env_module, "TEMPLATES_DIR", templates_dir)

    create_env(tmp_path / "my_env", templates=["base", "overlay"])

    target = tmp_path / "my_env"
    assert (target / "shared.txt").read_text() == "from overlay"  # Overridden
    assert (target / "base_only.txt").read_text() == "only in base"  # Preserved


def _write_template(
    templates_dir: Path, id_: str, requires: tuple[str, ...] = ()
) -> Path:
    d = templates_dir / id_
    d.mkdir(parents=True)
    d.joinpath("template.toml").write_text(
        f'description = "{id_}"\nrequires = {list(requires)!r}\n'
    )
    return d


def test_base_file_includes_fragments_from_resolved_templates_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A file may pull in `partials/` fragments from every template the env
    resolves, so add-ons extend it without replacing another add-on's copy."""
    templates_dir = tmp_path / "templates"
    base = _write_template(templates_dir, "base")
    base.joinpath("config.toml").write_text(
        '[tool]\n{% for t in templates %}{% include t ~ "/partials/config.toml" ignore missing %}{% endfor %}'
    )
    addon = _write_template(templates_dir, "addon", ("base",))
    addon.joinpath("partials").mkdir()
    addon.joinpath("partials", "config.toml").write_text("addon = true\n")
    other = _write_template(templates_dir, "other", ("base",))
    other.joinpath("partials").mkdir()
    other.joinpath("partials", "config.toml").write_text("other = true\n")

    monkeypatch.setattr(create_env_module, "TEMPLATES_DIR", templates_dir)
    create_env(tmp_path / "my_env", templates=["addon"])

    target = tmp_path / "my_env"
    assert (target / "config.toml").read_text() == "[tool]\naddon = true\n"
    assert not (target / "partials").exists()


def _render_default(tmp_path: Path) -> Path:
    with (
        patch("karotte.create_env.subprocess.check_call"),
        patch("karotte.create_env.run_uv"),
    ):
        create_env(tmp_path / "my_env", templates=["default"], no_lock=True)
    return tmp_path / "my_env"


def test_templates_license_is_not_copied_into_the_env(tmp_path: Path):
    assert (TEMPLATES_DIR / "LICENSE").read_text().startswith("MIT No Attribution")
    env = _render_default(tmp_path)
    assert not list(env.rglob("LICENSE*"))


def test_default_pyprojects_resolve_from_pypi_only(tmp_path: Path):
    env = _render_default(tmp_path)
    assert not (env / "partials").exists()
    for rel in ("pyproject.toml", "venvs/student/pyproject.toml"):
        uv = tomllib.loads((env / rel).read_text())["tool"]["uv"]
        assert uv["exclude-newer"] == "7 days"
        assert "keyring-provider" not in uv
        assert uv["exclude-newer-package"] == {"karotte": False}
        assert [i["name"] for i in uv["index"]] == ["pypi"]
    root = tomllib.loads((env / "pyproject.toml").read_text())
    assert not any(
        "keyring" in d for d in root["project"]["optional-dependencies"]["dev"]
    )


def test_default_depends_on_karotte_only(tmp_path: Path):
    text = (_render_default(tmp_path) / "pyproject.toml").read_text()
    assert '\ndependencies = [\n    "karotte",\n]\n' in text


def test_addon_adds_dependencies_ahead_of_karotte(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Ahead of `karotte`, so envs that appended their own deps after it merge
    cleanly on update."""
    templates_dir = tmp_path / "templates"
    shutil.copytree(TEMPLATES_DIR, templates_dir)
    addon = _write_template(templates_dir, "addon", ("default",))
    addon.joinpath("partials").mkdir()
    addon.joinpath("partials", "deps.toml.jinja").write_text('    "addon-pkg",\n')
    monkeypatch.setattr(create_env_module, "TEMPLATES_DIR", templates_dir)

    with (
        patch("karotte.create_env.subprocess.check_call"),
        patch("karotte.create_env.run_uv"),
    ):
        create_env(tmp_path / "my_env", templates=["addon"], no_lock=True)

    env = tmp_path / "my_env"
    root = tomllib.loads((env / "pyproject.toml").read_text())
    assert root["project"]["dependencies"] == ["addon-pkg", "karotte"]
    student = tomllib.loads((env / "venvs/student/pyproject.toml").read_text())
    assert "addon-pkg" not in str(student)


def test_addon_extends_the_age_delay_exemptions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    templates_dir = tmp_path / "templates"
    shutil.copytree(TEMPLATES_DIR, templates_dir)
    addon = _write_template(templates_dir, "addon", ("default",))
    addon.joinpath("partials").mkdir()
    addon.joinpath("partials", "uv_extra.toml.jinja").write_text(
        "exclude-newer-package.addon-pkg = false\n"
    )
    monkeypatch.setattr(create_env_module, "TEMPLATES_DIR", templates_dir)

    with (
        patch("karotte.create_env.subprocess.check_call"),
        patch("karotte.create_env.run_uv"),
    ):
        create_env(tmp_path / "my_env", templates=["addon"], no_lock=True)

    for rel in ("pyproject.toml", "venvs/student/pyproject.toml"):
        doc = tomllib.loads((tmp_path / "my_env" / rel).read_text())
        assert doc["tool"]["uv"]["exclude-newer-package"] == {
            "karotte": False,
            "addon-pkg": False,
        }


def _run_steps(containerfile: str) -> list[str]:
    lines = [ln for ln in containerfile.splitlines() if not ln.lstrip().startswith("#")]
    joined = "\n".join(lines).replace("\\\n", " ")
    return [ln for ln in joined.splitlines() if ln.startswith("RUN ")]


def test_uv_credential_store_is_dropped_in_the_step_that_writes_it(tmp_path: Path):
    """A later layer's `rm` leaves the file in the earlier layer and hides it from
    `karotte check`."""
    steps = _run_steps((_render_default(tmp_path) / "Containerfile").read_text())
    uv_steps = [s for s in steps if "id=uv_env" in s]
    assert len(uv_steps) == 3
    for step in uv_steps:
        assert re.search(r"rm -rf [^&]*/root/\.local/share/uv/credentials", step), step
    (check,) = [s for s in steps if "karotte check" in s]
    assert "rm " not in check


def test_default_justfile_imports_optional_recipe_files(tmp_path: Path):
    env = _render_default(tmp_path)
    assert "import? 'internal.just'" in (env / "justfile").read_text()


# --- resolve_template_deps tests ---


def _make_template(id: str, requires: list[str] | None = None) -> EnvironmentTemplate:
    return EnvironmentTemplate(id=id, description="", requires=requires or [])


def test_resolve_deps_no_deps():
    available = {t.id: t for t in [_make_template("a"), _make_template("b")]}
    assert resolve_template_deps(["a", "b"], available) == ["a", "b"]


def test_resolve_deps_single_with_dep():
    available = {
        t.id: t for t in [_make_template("base"), _make_template("overlay", ["base"])]
    }
    assert resolve_template_deps(["overlay"], available) == ["base", "overlay"]


def test_resolve_deps_transitive():
    available = {
        t.id: t
        for t in [
            _make_template("a"),
            _make_template("b", ["a"]),
            _make_template("c", ["b"]),
        ]
    }
    assert resolve_template_deps(["c"], available) == ["a", "b", "c"]


def test_resolve_deps_deduplicates():
    """If a dep is already explicitly listed, it should not appear twice."""
    available = {
        t.id: t for t in [_make_template("base"), _make_template("overlay", ["base"])]
    }
    assert resolve_template_deps(["base", "overlay"], available) == ["base", "overlay"]


def test_resolve_deps_shared_dep():
    """Two templates sharing a dependency should resolve it once."""
    available = {
        t.id: t
        for t in [
            _make_template("base"),
            _make_template("x", ["base"]),
            _make_template("y", ["base"]),
        ]
    }
    assert resolve_template_deps(["x", "y"], available) == ["base", "x", "y"]


def test_resolve_deps_circular():
    available = {
        t.id: t for t in [_make_template("a", ["b"]), _make_template("b", ["a"])]
    }
    with pytest.raises(ValueError, match="Circular dependency"):
        resolve_template_deps(["a"], available)


def test_resolve_deps_self_referencing():
    available = {t.id: t for t in [_make_template("a", ["a"])]}
    with pytest.raises(ValueError, match="Circular dependency"):
        resolve_template_deps(["a"], available)


def test_resolve_deps_missing_dep():
    available = {t.id: t for t in [_make_template("overlay", ["missing"])]}
    with pytest.raises(ValueError, match="'missing' not found"):
        resolve_template_deps(["overlay"], available)


def test_resolve_deps_missing_template():
    available = {t.id: t for t in [_make_template("a")]}
    with pytest.raises(ValueError, match="'nope' not found"):
        resolve_template_deps(["nope"], available)


def test_auto_resolves_deps_in_create_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """create_env should auto-include required templates without explicit listing."""
    from karotte import create_env as create_env_module

    templates_dir = tmp_path / "templates"

    (templates_dir / "base").mkdir(parents=True)
    (templates_dir / "base" / "template.toml").write_text('description = "Base"\n')
    (templates_dir / "base" / "base.txt").write_text("from base")

    (templates_dir / "overlay").mkdir(parents=True)
    (templates_dir / "overlay" / "template.toml").write_text(
        'description = "Overlay"\nrequires = ["base"]\n'
    )
    (templates_dir / "overlay" / "overlay.txt").write_text("from overlay")

    monkeypatch.setattr(create_env_module, "TEMPLATES_DIR", templates_dir)

    create_env(tmp_path / "my_env", templates=["overlay"])

    target = tmp_path / "my_env"
    assert (target / "base.txt").read_text() == "from base"
    assert (target / "overlay.txt").read_text() == "from overlay"


def test_pyproject_description_names_the_env(tmp_path: Path):
    create_env(tmp_path / "my_env", templates=["default"], no_lock=True)
    project = tomllib.loads((tmp_path / "my_env" / "pyproject.toml").read_text())
    assert project["project"]["description"] == "The my_env environment."


_FAILURES: dict[str, tuple[dict[str, BaseException | None], bool]] = {
    "uv lock": (
        {"subprocess.check_call": subprocess.CalledProcessError(1, "uv")},
        False,
    ),
    "uv lock interrupted": ({"subprocess.check_call": KeyboardInterrupt()}, False),
    "post_create": (
        {
            "subprocess.check_call": None,
            "run_uv": subprocess.CalledProcessError(1, "uv"),
        },
        False,
    ),
    "vendoring": (
        {"subprocess.check_call": None, "_vendor_karotte": OSError("disk full")},
        True,
    ),
}


@pytest.mark.parametrize("stage", _FAILURES)
def test_failure_removes_the_half_created_env(tmp_path: Path, stage: str):
    patches, vendor = _FAILURES[stage]
    output_dir = tmp_path / "parent" / "my_env"
    with contextlib.ExitStack() as stack:
        for target, error in patches.items():
            stack.enter_context(
                patch(f"karotte.create_env.{target}", side_effect=error)
            )
        with pytest.raises((Exception, KeyboardInterrupt)):
            create_env(output_dir, templates=["default"], vendor_karotte=vendor)
    assert list(tmp_path.iterdir()) == []


def test_render_failure_removes_the_half_created_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    for name, toml, content in [
        ("good", 'description = "g"\n', "fine"),
        (
            "broken",
            'description = "b"\nrequires = ["good"]\n',
            "{{ x | no_such_filter }}",
        ),
    ]:
        (tmp_path / "templates" / name).mkdir(parents=True)
        (tmp_path / "templates" / name / "template.toml").write_text(toml)
        (tmp_path / "templates" / name / f"{name}.txt").write_text(content)
    monkeypatch.setattr(create_env_module, "TEMPLATES_DIR", tmp_path / "templates")

    with pytest.raises(Exception, match="no_such_filter"):
        create_env(tmp_path / "my_env", templates=["broken"], no_lock=True)
    assert not (tmp_path / "my_env").exists()


def test_existing_dir_is_left_alone(tmp_path: Path):
    output_dir = tmp_path / "my_env"
    output_dir.mkdir()
    (output_dir / "keep.txt").write_text("mine")

    with pytest.raises(FileExistsError):
        create_env(output_dir, templates=["default"])
    assert (output_dir / "keep.txt").read_text() == "mine"


def test_cli_reports_a_failed_uv_lock_in_one_line(tmp_path: Path):
    from typer.testing import CliRunner

    from karotte.cli import app

    with patch(
        "karotte.create_env.subprocess.check_call",
        side_effect=subprocess.CalledProcessError(1, ["uv", "lock"]),
    ):
        result = CliRunner().invoke(app, ["create-env", str(tmp_path / "my_env")])
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert "`uv lock` failed with exit code 1" in result.output
    assert not (tmp_path / "my_env").exists()
