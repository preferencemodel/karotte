"""Schemas for the suite. The schema is the contract.

Every field the suite's behaviour, instructions, or grader depends on lives
here, so `task_configs.json` is the single source of truth at runtime.
"""

from pydantic import BaseModel


class TaskConfig(BaseModel):
    """One fully-specified task — one row of `task_configs.json`."""

    task_id: str
    instructions: str
    submission: str
    # One field per axis of variation. Replace `variant` with your own axes.
    variant: str


class TaskCalibration(BaseModel):
    """Per-task scoring parameters — one row of `calibration_configs.json`.

    Kept separate from `TaskConfig` so the reward curve can be re-tuned from
    real run data without regenerating the task definitions.
    """

    task_id: str
    # Placeholder. Replace with parameters measured from calibration runs, e.g.
    # the score a typical attempt reaches, so partial work earns partial reward.
    max_score: float
