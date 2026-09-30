import tomllib
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

from loguru import logger

from karotte.schemas.environment_template import EnvironmentTemplate

TEMPLATES_DIR = Path(__file__).parent / "templates"
ENTRY_POINT_GROUP = "karotte.templates"


class TemplatesMissingError(FileNotFoundError):
    pass


@dataclass(frozen=True)
class InstalledTemplate:
    template: EnvironmentTemplate
    dir: Path
    requirement: str | None
    """`name==version` of the distribution providing the template; None for
    karotte's own templates."""


def load_templates(templates_dir: Path = TEMPLATES_DIR) -> list[EnvironmentTemplate]:
    """Load all templates from a directory and validate their `template.toml`.

    Raises FileNotFoundError if a template directory is missing template.toml.
    """
    templates = []
    for template_dir in sorted(d for d in templates_dir.iterdir() if d.is_dir()):
        toml_file = template_dir / "template.toml"
        if not toml_file.exists():
            raise FileNotFoundError(
                f"Template '{template_dir.name}' is missing template.toml"
            )
        with open(toml_file, "rb") as f:
            data = tomllib.load(f)
        templates.append(EnvironmentTemplate(id=template_dir.name, **data))
    return templates


def discover_templates(
    builtin_dir: Path = TEMPLATES_DIR,
) -> dict[str, InstalledTemplate]:
    """Templates from `builtin_dir` plus every `karotte.templates` entry point.

    An entry point shadows a built-in template of the same id; two entry points
    sharing an id is an error.
    """
    if not builtin_dir.is_dir():
        raise TemplatesMissingError(
            "This karotte has no templates; vendored copies omit them. "
            + "Use a karotte tool install (`uv tool install karotte`) instead."
        )
    found: dict[str, InstalledTemplate] = {}
    for template in load_templates(builtin_dir):
        found[template.id] = InstalledTemplate(
            template, builtin_dir / template.id, None
        )

    providers: dict[str, str] = {}
    for ep in metadata.entry_points(group=ENTRY_POINT_GROUP):
        dist = ep.dist
        if dist is None:
            raise ValueError(f"Entry point {ep.name!r} has no distribution.")
        try:
            templates_dir = Path(ep.load())
            templates = load_templates(templates_dir)
        except Exception as e:  # noqa: BLE001 - a broken plugin must not break karotte
            logger.warning("Ignoring templates from {!r}: {}", ep.name, e)
            continue
        for template in templates:
            if template.id in providers:
                raise ValueError(
                    f"Template '{template.id}' is provided by both "
                    + f"{providers[template.id]} and {dist.name}."
                )
            if template.id in found:
                logger.warning(
                    f"Template '{template.id}' from {dist.name} shadows karotte's own."
                )
            found[template.id] = InstalledTemplate(
                template, templates_dir / template.id, f"{dist.name}=={dist.version}"
            )
            providers[template.id] = dist.name
    return found
