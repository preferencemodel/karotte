"""Generate `task_configs.json` — the suite's task definitions.

Run this whenever you change the axes below or the instruction text:

    just gen-configs _template_suite

The generated JSON is committed and is the single source of truth: code reads
the JSON, never this module, at task time.
"""

from pathlib import Path

from environment.suite import write_configs
from environment.tasks._template_suite.prompt import (
    WORKDIR_PLACEHOLDER,
    build_instructions,
)
from environment.tasks._template_suite.task_configs.task_config import TaskConfig

SUITE_ID = "template-suite"

# Axes of variation. The suite is the cartesian product of these. Add an axis by
# adding a list here, a loop in `loop_over_tasks`, and a field on `TaskConfig`.
VARIANTS: list[str] = ["one", "two"]

SUBMISSION = f"{WORKDIR_PLACEHOLDER}/solution.txt"


def get_task_id(variant: str) -> str:
    return f"{SUITE_ID}-{variant}"


def loop_over_tasks():
    """Yield every cell of the suite's axis product."""
    yield from VARIANTS


def create_task_config(variant: str) -> TaskConfig:
    return TaskConfig(
        task_id=get_task_id(variant),
        instructions=build_instructions(variant, SUBMISSION),
        submission=SUBMISSION,
        variant=variant,
    )


def main() -> None:
    configs = [create_task_config(variant) for variant in loop_over_tasks()]
    write_configs(Path(__file__).parent / "task_configs.json", configs)
    print(f"Wrote {len(configs)} task configs")


if __name__ == "__main__":
    main()
