import importlib.resources
import re
import shutil
import subprocess
import tomllib
from collections.abc import Sequence
from importlib.metadata import version as pkg_version
from pathlib import Path

from jinja2 import ChoiceLoader, Environment, FileSystemLoader, PrefixLoader
from loguru import logger
from pydantic import BaseModel

from karotte.schemas.environment_template import EnvironmentTemplate
from karotte.templates import TEMPLATES_DIR, InstalledTemplate, discover_templates
from karotte.uv_supply_chain import run_uv


class EnvManifest(BaseModel):
    karotte_version: str
    templates: list[str]
    agents: list[str] = []
    """CLI agents to bake into the image. Names which agents; the installed
    karotte pins their versions. Empty means builtin/external only."""
    do_not_recreate_if_deleted: list[str] | None = None
    """Repo-relative paths that `update` must not recreate once deleted. Seeded
    from the templates at create time; edit freely to add or remove paths. None
    marks a manifest predating the field, so `update` seeds it from the current
    templates; an explicit empty list means the user cleared it and is kept."""
    extra_deps: list[str] = []
    """Requirements (`name==version`) of the packages that provided the
    templates, passed to `uv tool run --with` when `update` re-renders them."""


def validate_agents(agents: Sequence[str]) -> None:
    from karotte.agents import cli_agent_types

    known = cli_agent_types()
    unknown = [a for a in agents if a not in known]
    if unknown:
        available = ", ".join(sorted(known)) or "(none)"
        raise ValueError(
            f"Unknown agent(s): {', '.join(unknown)}. Available: {available}."
        )


def resolve_template_deps(
    templates: list[str],
    available: dict[str, EnvironmentTemplate],
) -> list[str]:
    """Resolve template dependencies into an ordered list (dependencies first).

    Raises ValueError if a template or any of its dependencies don't exist,
    or if a circular dependency is detected.
    """
    resolved: list[str] = []
    seen: set[str] = set()
    visiting: set[str] = set()

    def visit(template_id: str) -> None:
        if template_id in seen:
            return
        if template_id not in available:
            raise ValueError(f"Template '{template_id}' not found.")
        if template_id in visiting:
            raise ValueError(f"Circular dependency detected: '{template_id}'.")
        visiting.add(template_id)
        for dep in available[template_id].requires:
            visit(dep)
        visiting.discard(template_id)
        seen.add(template_id)
        resolved.append(template_id)

    for t in templates:
        visit(t)

    return resolved


def create_env(
    output_dir: Path,
    templates: Sequence[str],
    *,
    agents: Sequence[str] = (),
    vendor_karotte: bool = False,
    no_lock: bool = False,
) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Directory {output_dir} already exists.")

    validate_agents(agents)

    installed = discover_templates(TEMPLATES_DIR)
    existing_templates = {id_: t.template for id_, t in installed.items()}

    resolved = resolve_template_deps(list(templates), existing_templates)

    created = next(
        p
        for p in (output_dir.absolute(), *output_dir.absolute().parents)
        if p.parent.exists()
    )
    output_dir.mkdir(parents=True)
    try:
        _populate(
            output_dir,
            resolved,
            installed,
            agents=agents,
            vendor_karotte=vendor_karotte,
            no_lock=no_lock,
        )
    except BaseException:
        shutil.rmtree(created, ignore_errors=True)
        if not created.exists():
            logger.warning(f"create-env failed; removed {created}")
        raise


def _populate(
    output_dir: Path,
    resolved: list[str],
    installed: dict[str, InstalledTemplate],
    *,
    agents: Sequence[str],
    vendor_karotte: bool,
    no_lock: bool,
) -> None:
    env_name = output_dir.name
    existing_templates = {id_: t.template for id_, t in installed.items()}

    jinja_env = Environment(
        loader=ChoiceLoader(
            [
                PrefixLoader(
                    {id_: FileSystemLoader(t.dir) for id_, t in installed.items()}
                ),
                FileSystemLoader(TEMPLATES_DIR),
            ]
        ),
        keep_trailing_newline=True,
    )

    for template_name in resolved:
        template_dir = installed[template_name].dir
        logger.info(f"Applying template: {template_name}")

        # Iterate over all files in the template directory
        for template_file in template_dir.rglob("*"):
            # Get relative path from template directory
            relative_path = template_file.relative_to(template_dir)

            if (
                not template_file.is_file()
                or template_file.name == "template.toml"
                or "__pycache__" in relative_path.parts
                or ".ruff_cache" in relative_path.parts
                or ".venv" in relative_path.parts
                # Fragments other templates include; never files of the env.
                or relative_path.parts[0] == "partials"
            ):
                continue

            # Load and render the template
            template_ = jinja_env.get_template(
                f"{template_name}/{str(relative_path)}",
            )
            rendered = template_.render(env_name=env_name, templates=resolved)

            # A ``.jinja`` suffix marks a source that is Jinja, not valid
            # standalone content (e.g. ``pyproject.toml.jinja`` uses
            # ``{% extends %}`` for template inheritance, which would break
            # tools that parse ``pyproject.toml``). Strip it for the output.
            out_relative_path = relative_path
            if out_relative_path.suffix == ".jinja":
                out_relative_path = out_relative_path.with_suffix("")

            if (
                vendor_karotte
                and out_relative_path.name == "pyproject.toml"
                and "karotte"
                in tomllib.loads(rendered).get("project", {}).get("dependencies", [])
            ):
                rendered = point_karotte_at_vendored_copy(rendered)

            # Write to output directory — later templates override earlier ones
            output_path = output_dir / out_relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(rendered)

    if vendor_karotte:
        _vendor_karotte(output_dir)

    # Write manifest for future updates. Seed the do-not-recreate list from the
    # resolved templates so it lives in the env and stays user-editable.
    do_not_recreate = sorted(
        {
            path
            for template_id in resolved
            for path in existing_templates[template_id].do_not_recreate_if_deleted
        }
    )
    manifest = EnvManifest(
        karotte_version=pkg_version("karotte"),
        templates=resolved,
        agents=list(agents),
        do_not_recreate_if_deleted=do_not_recreate,
        extra_deps=sorted({r for t in resolved if (r := installed[t].requirement)}),
    )

    (output_dir / ".manifest.json").write_text(manifest.model_dump_json(indent=2))

    if not no_lock and (output_dir / "pyproject.toml").exists():
        subprocess.check_call(["uv", "lock"], cwd=output_dir)

    # Run post-create hook if present
    post_create = output_dir / "post_create.py"
    if post_create.is_file():
        logger.info("Running post_create.py")
        # A PEP 723 script: uv resolves its inline dependencies with the
        # project — and the `exclude-newer` just written into it — ignored.
        run_uv(("run",), "--no-project", "post_create.py", cwd=output_dir, check=True)


def point_karotte_at_vendored_copy(text: str) -> str:
    """Set the karotte source to ``.karotte`` the way ``just vendor-karotte`` does,
    commenting out any previous one so ``just unvendor-karotte`` can restore it."""
    entry = 'karotte = { path = ".karotte" }'
    header = re.search(r"^\[tool\.uv\.sources\][ \t]*\n", text, re.M)
    if header is None:
        return text.rstrip("\n") + f"\n\n[tool.uv.sources]\n{entry}\n"
    next_table = re.search(r"^\[", text[header.end() :], re.M)
    end = header.end() + next_table.start() if next_table else len(text)
    body = re.sub(
        r"^(karotte[ \t]*=)", r"# vendored: \1", text[header.end() : end], flags=re.M
    )
    return text[: header.end()] + entry + "\n" + body + text[end:]


def _vendor_karotte(output_dir: Path) -> None:
    """Copy karotte source into the environment's .karotte/ directory."""
    vendor_dir = output_dir / ".karotte"
    pkg_root = Path(str(importlib.resources.files("karotte")))

    # Symlinked into the package, so available when installed.
    for name in ("LICENSE", "README.md"):
        shutil.copy(pkg_root / name, vendor_dir / name)
    # templates/ isn't vendored, and neither is its MIT-0 license.
    pyproject = (pkg_root / "pyproject.toml").read_text()
    (vendor_dir / "pyproject.toml").write_text(
        pyproject.replace('"MIT AND MIT-0"', '"MIT"').replace(
            ', "src/karotte/templates/LICENSE"', ""
        )
    )

    # Copy karotte package into src/karotte/
    skip_dirs = {"__pycache__", ".ruff_cache", ".venv", "templates"}
    shutil.copytree(
        pkg_root,
        vendor_dir / "src" / "karotte",
        ignore=shutil.ignore_patterns(*skip_dirs),
    )
