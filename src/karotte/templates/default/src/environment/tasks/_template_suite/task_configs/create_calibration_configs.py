"""Generate `calibration_configs.json` — per-task scoring parameters.

Run alongside `create_task_configs.py`:

    just gen-configs _template_suite

The values below are placeholders; replace them with numbers from real
calibration runs.
"""

from pathlib import Path

from environment.suite import write_configs
from environment.tasks._template_suite.task_configs.create_task_configs import (
    get_task_id,
    loop_over_tasks,
)
from environment.tasks._template_suite.task_configs.task_config import TaskCalibration


def create_calibration_config(variant: str) -> TaskCalibration:
    return TaskCalibration(task_id=get_task_id(variant), max_score=1.0)


def main() -> None:
    configs = [create_calibration_config(variant) for variant in loop_over_tasks()]
    write_configs(Path(__file__).parent / "calibration_configs.json", configs)
    print(f"Wrote {len(configs)} calibration configs")


if __name__ == "__main__":
    main()
