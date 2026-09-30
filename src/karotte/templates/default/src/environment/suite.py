"""Shared machinery for JSON-driven task suites.

A "suite" is a family of related tasks generated from a few axes of variation
instead of being hand-written one at a time: one scoring rig and one set of
instructions scale to as many tasks as there are rows in `task_configs.json`.
The two pieces below are identical for every suite, so they live here rather
than being copy-pasted into each suite's `__init__.py`. Both are
schema-agnostic — a suite passes its own pydantic models.

Run `just create-task-suite <name>` to scaffold a suite.
"""

import importlib
import json
import sys
from pathlib import Path

from karotte import ProtectedStore
from pydantic import BaseModel, ValidationError


def register_suite(
    *,
    namespace: dict,
    base_task_cls: type,
    configs_dir: Path,
    config_model: type,
    calibration_model: type,
    suite_name: str,
) -> None:
    """Register one `base_task_cls` subclass per row of the suite's config JSON.

    `namespace` is the suite `__init__` module's `globals()`; each generated
    subclass is injected there with `id` / `task_config` / `task_calibration`
    set as class attributes, so `environment.get_tasks()` finds it.

    An unusable config file registers nothing and warns, rather than raising and
    breaking discovery for every other task.
    """
    try:
        with open(configs_dir / "task_configs.json") as f:
            task_configs = [config_model.model_validate(d) for d in json.load(f)]
        with open(configs_dir / "calibration_configs.json") as f:
            task_calibrations = {
                d["task_id"]: calibration_model.model_validate(d) for d in json.load(f)
            }
        missing = [
            c.task_id for c in task_configs if c.task_id not in task_calibrations
        ]
        if missing:
            raise KeyError(f"no calibration for task_id(s): {missing[:5]}")
    except (FileNotFoundError, json.JSONDecodeError, ValidationError, KeyError) as e:
        print(
            f"WARNING: {suite_name} configs unusable ({type(e).__name__}: {e}); "
            f"run `just gen-configs {suite_name}`. No tasks registered.",
            file=sys.stderr,
        )
        return

    for cfg in task_configs:
        name = f"Task_{cfg.task_id.replace('-', '_').replace('.', '_')}"
        namespace[name] = type(
            name,
            (base_task_cls,),
            {
                "id": cfg.task_id,
                "task_config": cfg,
                "task_calibration": task_calibrations[cfg.task_id],
            },
        )


def write_configs(path: Path, configs) -> None:
    """Serialize a list of pydantic config models to indented JSON at `path`."""
    with open(path, "w") as f:
        json.dump([c.model_dump(mode="json") for c in configs], f, indent=4)


def _store_key(suite_name: str, kind: str) -> str:
    return f"{suite_name}.{kind}"


def stash_task_configs(suite_name: str, task_config, task_calibration) -> None:
    """Write the active task's config + calibration into the `ProtectedStore`.

    Called from a suite's `pre_hook`; the grader reads them back with the two
    loaders below. This keeps every per-task parameter in one place and the
    grader fixed as the suite grows.
    """
    store = ProtectedStore()
    store.write(_store_key(suite_name, "task_config"), task_config)
    store.write(_store_key(suite_name, "calibration_config"), task_calibration)


def load_task_config[T: BaseModel](suite_name: str, config_model: type[T]) -> T:
    """Read back the config `stash_task_configs` wrote. Called from the grader."""
    return ProtectedStore().read(_store_key(suite_name, "task_config"), config_model)


def load_task_calibration[T: BaseModel](
    suite_name: str, calibration_model: type[T]
) -> T:
    """Read back the calibration `stash_task_configs` wrote."""
    return ProtectedStore().read(
        _store_key(suite_name, "calibration_config"), calibration_model
    )


def main() -> None:
    """Entry point for `just gen-configs <suite>`.

    Both generators run in one process, so the suite package is imported once —
    while its two JSON files still agree — rather than once per generator with a
    half-regenerated pair in between.
    """
    suite = sys.argv[1]
    for generator in ("create_task_configs", "create_calibration_configs"):
        module = importlib.import_module(
            f"environment.tasks.{suite}.task_configs.{generator}"
        )
        module.main()


if __name__ == "__main__":
    main()
