from __future__ import annotations

import importlib
import os
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

# `karotte` is imported lazily inside the functions below (not at module top) so
# this package is importable by interpreters that don't have `karotte` installed
# — e.g. a separate scoring venv that only needs pure-stdlib helpers.
# `from __future__ import annotations` keeps the `Task` annotations lazy; the
# TYPE_CHECKING import gives type-checkers/linters the name without importing it
# at runtime.
if TYPE_CHECKING:
    from karotte import Task

STUDENT_UID = int(os.environ.get("KAROTTE_DEMOTE_ID", "1000"))

# Add task IDs that you want to include or exclude here. If INCLUDE_TASKS
# is non-empty, only tasks with IDs in that list will be included.
# If EXCLUDE_TASKS is non-empty, tasks with IDs in that list will be excluded.
INCLUDE_TASKS: set[str] = set()
EXCLUDE_TASKS: set[str] = set()


def get_tasks() -> list[type[Task]]:
    import environment.tasks

    tasks: list[type[Task]] = []

    for candidate in Path(environment.tasks.__path__[0]).glob("*"):
        if not candidate.is_dir():
            continue

        if candidate.name.startswith("_"):
            continue

        module = importlib.import_module(f"environment.tasks.{candidate.name}")

        tasks.extend(_get_tasks_from_module(module))

    return tasks


def _get_tasks_from_module(module: ModuleType) -> list[type[Task]]:
    from karotte import Task

    tasks: list[type[Task]] = []

    for member in dir(module):
        cls = getattr(module, member)

        if not isinstance(cls, type) or not issubclass(cls, Task) or cls is Task:
            continue

        id_ = getattr(cls, "id", None)

        if (
            not id_
            or id_ in EXCLUDE_TASKS
            or (INCLUDE_TASKS and id_ not in INCLUDE_TASKS)
        ):
            continue

        tasks.append(cls)

    return tasks
