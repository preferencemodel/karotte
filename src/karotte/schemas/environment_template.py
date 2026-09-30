from typing import Annotated

from pydantic import BaseModel, Field

TemplateId = Annotated[
    str,
    Field(
        pattern=r"^[a-z0-9][a-z0-9-]*[a-z0-9]$|^[a-z0-9]$",
        description="Lowercase letters, digits, and hyphens (not at start or end).",
    ),
]


class EnvironmentTemplate(BaseModel):
    id: TemplateId
    """Slug identifier matching the template's directory name."""
    description: str
    requires: list[TemplateId] = []
    """IDs of other templates required to use this template."""
    do_not_recreate_if_deleted: list[str] = []
    """Repo-relative paths this template ships that `update` must not recreate
    once the user has deleted them. A directory entry also matches everything
    under it."""
