# pyright: reportPrivateUsage=false
"""Templates provided by other packages through the `karotte.templates` entry point."""

import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from unittest.mock import patch

import pytest
from loguru import logger

from karotte.create_env import EnvManifest, create_env
from karotte.templates import (
    TEMPLATES_DIR,
    InstalledTemplate,
    discover_templates,
    load_templates,
)
from karotte.update_env import (
    _generate_env,
    _resolve_extra_deps,
    update_env,
)


def _install_template_package(
    site: Path,
    dist: str,
    version: str,
    templates: dict[str, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Write an importable package plus its dist-info into `site` and put it on `sys.path`."""
    module = dist.replace("-", "_")
    pkg = site / module
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(
        "from pathlib import Path\nTEMPLATES_DIR = Path(__file__).parent / 'templates'\n"
    )
    for template_id, files in templates.items():
        for name, content in files.items():
            path = pkg / "templates" / template_id / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    info = site / f"{module}-{version}.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {dist}\nVersion: {version}\n"
    )
    (info / "entry_points.txt").write_text(
        f"[karotte.templates]\n{module} = {module}:TEMPLATES_DIR\n"
    )
    monkeypatch.syspath_prepend(str(site))
    monkeypatch.setitem(sys.modules, module, None)
    del sys.modules[module]


@pytest.fixture
def extra_site(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    site = tmp_path / "site"
    _install_template_package(
        site,
        "extra-templates-pkg",
        "1.2.3",
        {
            "extra-overlay": {
                "template.toml": 'description = "Overlay"\nrequires = ["default"]\n',
                "extra.txt": "from extra {{ env_name }}\n",
                "pyproject.toml.jinja": (
                    '{% extends "default/pyproject.toml.jinja" %}\n'
                    "{% block indexes %}{{ super() }}\n[[tool.uv.index]]\n"
                    'name = "extra"\nurl = "https://extra.example/simple"\n{% endblock %}\n'
                    "{% block tail %}{{ super() }}\n[tool.extra]\nx = 1\n{% endblock %}\n"
                ),
            }
        },
        monkeypatch,
    )
    return site


def test_karotte_ships_two_templates() -> None:
    assert {t.id for t in load_templates(TEMPLATES_DIR)} == {
        "default",
        "language-toolchains",
    }


def test_builtin_templates_have_no_requirement() -> None:
    found = discover_templates()
    assert found["default"].dir == TEMPLATES_DIR / "default"
    assert found["default"].requirement is None


def test_entry_point_templates_are_discovered(extra_site: Path) -> None:
    found = discover_templates()
    extra = found["extra-overlay"]
    assert extra.template.requires == ["default"]
    assert (
        extra.dir == extra_site / "extra_templates_pkg" / "templates" / "extra-overlay"
    )
    assert extra.requirement == "extra-templates-pkg==1.2.3"
    assert found["default"].requirement is None


@pytest.mark.usefixtures("extra_site")
def test_list_names_the_providing_package(capsys: pytest.CaptureFixture[str]) -> None:
    from karotte.cli.templates import list_templates

    list_templates(json_output=False)
    out = capsys.readouterr().out
    assert "extra-overlay" in out
    assert "(from extra-templates-pkg==1.2.3)" in out
    assert "default — " in out


@pytest.mark.usefixtures("extra_site")
def test_list_json_carries_the_requirement(capsys: pytest.CaptureFixture[str]) -> None:
    from karotte.cli.templates import list_templates

    list_templates(json_output=True)
    by_id = {t["id"]: t for t in json.loads(capsys.readouterr().out)}
    assert by_id["extra-overlay"]["requirement"] == "extra-templates-pkg==1.2.3"
    assert by_id["default"]["requirement"] is None


def test_entry_point_template_shadows_a_builtin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_template_package(
        tmp_path / "dup",
        "dup-pkg",
        "0.1",
        {"default": {"template.toml": 'description = "shadow"\n'}},
        monkeypatch,
    )
    found = discover_templates()["default"]
    assert found.requirement == "dup-pkg==0.1"
    assert found.template.description == "shadow"


def test_two_entry_point_packages_with_the_same_id_is_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("pkg-a", "pkg-b"):
        _install_template_package(
            tmp_path / name,
            name,
            "0.1",
            {"clash": {"template.toml": 'description = "x"\n'}},
            monkeypatch,
        )
    with pytest.raises(ValueError, match="'clash'") as exc_info:
        discover_templates()
    assert "pkg-a" in str(exc_info.value) and "pkg-b" in str(exc_info.value)


def _discover_logging_warnings() -> tuple[dict[str, InstalledTemplate], list[str]]:
    warnings: list[str] = []
    handler = logger.add(lambda m: warnings.append(str(m)), level="WARNING")
    try:
        return discover_templates(), warnings
    finally:
        logger.remove(handler)


@pytest.mark.usefixtures("extra_site")
def test_a_plugin_that_fails_to_import_is_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_template_package(
        tmp_path / "broken",
        "broken-pkg",
        "0.1",
        {"broken": {"template.toml": 'description = "x"\n'}},
        monkeypatch,
    )
    (tmp_path / "broken" / "broken_pkg" / "__init__.py").write_text(
        "raise ImportError('missing dependency')\n"
    )

    found, warnings = _discover_logging_warnings()

    assert "broken" not in found
    assert "extra-overlay" in found
    assert any("broken_pkg" in w and "missing dependency" in w for w in warnings)


@pytest.mark.usefixtures("extra_site")
def test_a_plugin_with_an_invalid_template_is_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_template_package(
        tmp_path / "invalid",
        "invalid-pkg",
        "0.1",
        {"invalid": {"README.md": "no template.toml\n"}},
        monkeypatch,
    )

    found, warnings = _discover_logging_warnings()

    assert "invalid" not in found
    assert "extra-overlay" in found
    assert any("invalid_pkg" in w for w in warnings)


@pytest.mark.usefixtures("extra_site")
def test_create_env_renders_cross_package_overlay(tmp_path: Path) -> None:
    with patch("karotte.create_env.subprocess.check_call"):
        create_env(tmp_path / "my_env", templates=["extra-overlay"])
    env = tmp_path / "my_env"
    assert (env / "extra.txt").read_text() == "from extra my_env\n"
    pyproject = (env / "pyproject.toml").read_text()
    assert 'name = "environment"' in pyproject
    assert "[tool.extra]\nx = 1" in pyproject
    assert pyproject.index('name = "pypi"') < pyproject.index('name = "extra"')
    manifest = EnvManifest.model_validate_json((env / ".manifest.json").read_text())
    assert manifest.templates == ["default", "extra-overlay"]
    assert manifest.extra_deps == ["extra-templates-pkg==1.2.3"]


def test_create_env_builtin_only_records_no_extra_deps(tmp_path: Path) -> None:
    with patch("karotte.create_env.subprocess.check_call"):
        create_env(tmp_path / "my_env", templates=["default"])
    manifest = EnvManifest.model_validate_json(
        (tmp_path / "my_env" / ".manifest.json").read_text()
    )
    assert manifest.extra_deps == []


def test_manifest_without_extra_deps_parses() -> None:
    manifest = EnvManifest.model_validate_json(
        '{"karotte_version": "0.0.0", "templates": ["default"]}'
    )
    assert manifest.extra_deps == []


def test_generate_env_passes_extra_deps_before_the_tool() -> None:
    with patch("karotte.update_env.subprocess.run") as run:
        run.return_value = subprocess.CompletedProcess([], 0)
        _generate_env("3.0.0", ["default"], Path("/tmp/out"), extra_deps=["a==1", "b"])
    cmd: list[str] = run.call_args[0][0]
    withs = [cmd[i + 1] for i, a in enumerate(cmd) if a == "--with"]
    assert withs == ["a==1", "b"]
    assert cmd.index("b") < cmd.index("karotte@3.0.0")


def test_generate_env_exempts_extra_deps_from_the_age_delay() -> None:
    with patch("karotte.update_env.subprocess.run") as run:
        run.return_value = subprocess.CompletedProcess([], 0)
        _generate_env(
            "2.13.0", ["default"], Path("/tmp/out"), extra_deps=["a==1", "b[x]"]
        )
    cmd: list[str] = run.call_args[0][0]
    assert "--exclude-newer-package=a=false" in cmd
    assert "--exclude-newer-package=b=false" in cmd
    assert cmd.index("--exclude-newer-package=b=false") < cmd.index("karotte@2.13.0")


def test_resolve_extra_deps_asks_the_target_karotte_for_versions() -> None:
    listing = json.dumps(
        [
            {"id": "default", "requirement": None},
            {"id": "extra-overlay", "requirement": "extra-templates-pkg==1.3.0"},
            {"id": "other", "requirement": "extra-templates-pkg==1.3.0"},
        ]
    )
    with patch("karotte.update_env.subprocess.run") as run:
        run.return_value = subprocess.CompletedProcess([], 0, stdout=listing)
        resolved = _resolve_extra_deps("2.13.0", ["extra-templates-pkg", "unused"])
    cmd: list[str] = run.call_args[0][0]
    assert cmd[cmd.index("--with") + 1] == "extra-templates-pkg"
    assert "--exclude-newer-package=extra-templates-pkg=false" in cmd
    assert cmd[-4:] == ["karotte@2.13.0", "templates", "list", "--json"]
    assert resolved == ["extra-templates-pkg==1.3.0"]


def _project(tmp_path: Path, manifest: dict[str, object]) -> Path:
    project = tmp_path / "project"
    project.mkdir()
    (project / ".manifest.json").write_text(json.dumps(manifest))
    return project


def _fake_generate(
    version: str,
    templates: list[str],
    output_dir: Path,
    extra_deps: list[str],
    uv_flags: Sequence[str] = (),  # pyright: ignore[reportUnusedParameter]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved = [f"{d}==9.9.9" if "==" not in d else d for d in extra_deps]
    (output_dir / ".manifest.json").write_text(
        json.dumps(
            {"karotte_version": version, "templates": templates, "extra_deps": resolved}
        )
    )


def _run_update(project: Path, **kwargs: object) -> tuple[list[str], list[str]]:
    with (
        patch("karotte.update_env._get_latest_version", return_value="2.13.0"),
        patch("karotte.update_env._generate_env", side_effect=_fake_generate) as gen,
        patch("karotte.update_env._merge_projects", return_value=[]),
        patch("karotte.update_env.subprocess.run"),
    ):
        update_env(project, **kwargs)  # pyright: ignore[reportArgumentType]
    baseline, target = gen.call_args_list
    return baseline.kwargs["extra_deps"], target.kwargs["extra_deps"]


def test_update_pins_baseline_deps_and_unpins_target_deps(tmp_path: Path) -> None:
    project = _project(
        tmp_path,
        {
            "karotte_version": "1.0.0",
            "templates": ["default", "extra-overlay"],
            "extra_deps": ["extra-templates-pkg==1.0.0", "other[x]>=2"],
        },
    )
    baseline, target = _run_update(project)
    assert baseline == ["extra-templates-pkg==1.0.0", "other[x]>=2"]
    assert target == ["extra-templates-pkg", "other[x]"]
    manifest = json.loads((project / ".manifest.json").read_text())
    assert manifest["extra_deps"] == ["extra-templates-pkg==9.9.9", "other[x]==9.9.9"]


def test_update_old_manifest_without_extra_deps_uses_none(tmp_path: Path) -> None:
    project = _project(tmp_path, {"karotte_version": "1.0.0", "templates": ["default"]})
    assert _run_update(project) == ([], [])


def _run_update_at_latest_karotte(
    project: Path, resolved: list[str], **kwargs: object
) -> list[str] | None:
    """The target render's deps with karotte already current, or None if nothing rendered."""
    with (
        patch("karotte.update_env._get_latest_version", return_value="1.0.0"),
        patch("karotte.update_env._resolve_extra_deps", return_value=resolved),
        patch("karotte.update_env._generate_env", side_effect=_fake_generate) as gen,
        patch("karotte.update_env._merge_projects", return_value=[]),
        patch("karotte.update_env.subprocess.run"),
    ):
        update_env(project, **kwargs)  # pyright: ignore[reportArgumentType]
    if not gen.call_args_list:
        return None
    return gen.call_args_list[1].kwargs["extra_deps"]


def test_update_renders_when_only_a_template_package_is_newer(tmp_path: Path) -> None:
    project = _project(
        tmp_path,
        {
            "karotte_version": "1.0.0",
            "templates": ["default", "extra-overlay"],
            "extra_deps": ["extra-templates-pkg==1.0.0"],
        },
    )
    target = _run_update_at_latest_karotte(project, ["extra-templates-pkg==1.3.0"])
    assert target == ["extra-templates-pkg"]
    manifest = json.loads((project / ".manifest.json").read_text())
    assert manifest["extra_deps"] == ["extra-templates-pkg==9.9.9"]


def test_update_stops_when_karotte_and_template_packages_are_current(
    tmp_path: Path,
) -> None:
    project = _project(
        tmp_path,
        {
            "karotte_version": "1.0.0",
            "templates": ["default", "extra-overlay"],
            "extra_deps": ["extra-templates-pkg==1.0.0"],
        },
    )
    assert (
        _run_update_at_latest_karotte(project, ["extra-templates-pkg==1.0.0"]) is None
    )


def test_update_with_renders_at_the_current_karotte(tmp_path: Path) -> None:
    project = _project(tmp_path, {"karotte_version": "1.0.0", "templates": ["default"]})
    target = _run_update_at_latest_karotte(
        project, ["from-cli==0.1"], extra_with=["from-cli"]
    )
    assert target == ["from-cli"]


def test_update_adds_with_packages_to_the_target_only(tmp_path: Path) -> None:
    project = _project(tmp_path, {"karotte_version": "1.0.0", "templates": ["default"]})
    baseline, target = _run_update(project, extra_with=["from-cli"])
    assert baseline == []
    assert target == ["from-cli"]


def test_update_records_only_what_the_target_render_used(tmp_path: Path) -> None:
    project = _project(tmp_path, {"karotte_version": "1.0.0", "templates": ["default"]})

    def generate_without_providers(
        version: str,
        templates: list[str],
        output_dir: Path,
        extra_deps: list[str],  # pyright: ignore[reportUnusedParameter]
        uv_flags: Sequence[str] = (),  # pyright: ignore[reportUnusedParameter]
    ) -> None:
        _fake_generate(version, templates, output_dir, [])

    with (
        patch("karotte.update_env._get_latest_version", return_value="2.13.0"),
        patch(
            "karotte.update_env._generate_env", side_effect=generate_without_providers
        ),
        patch("karotte.update_env._merge_projects", return_value=[]),
        patch("karotte.update_env.subprocess.run"),
    ):
        update_env(project, extra_with=["unused-pkg"])
    manifest = json.loads((project / ".manifest.json").read_text())
    assert manifest["extra_deps"] == []


def test_generate_env_hints_at_with_when_a_template_is_missing() -> None:
    err = subprocess.CalledProcessError(1, "uv", stderr="Template 'vllm' not found.")
    with (
        patch("karotte.update_env.subprocess.run", side_effect=err),
        pytest.raises(RuntimeError, match=r"Template 'vllm' not found(?s:.*)--with"),
    ):
        _generate_env("2.13.0", ["vllm"], Path("/tmp/out"))
