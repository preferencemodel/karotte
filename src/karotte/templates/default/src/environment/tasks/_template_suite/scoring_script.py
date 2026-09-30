"""Grader for the suite. Runs as a subprocess via `ExecutableJudge`.

Contract:
  * invoked as `python -m ...scoring_script <submission_path> <output_path>`
  * reads the active task's state from the `ProtectedStore` and ROOT_DATA_DIR
    (written by `pre_hook`), never trusting anything under the student workdir
  * reads the submission from the root-owned copy the step saved, not from
    where the student wrote it
  * writes `{"score": <float in [0, 1]>, "metadata": {...}}` to <output_path>
"""

import json
import sys
from pathlib import Path

from environment.suite import load_task_calibration, load_task_config
from environment.tasks._template_suite.task_configs.task_config import (
    TaskCalibration,
    TaskConfig,
)


def main() -> None:
    submission_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    task_config = load_task_config("_template_suite", TaskConfig)
    calibration = load_task_calibration("_template_suite", TaskCalibration)

    # Placeholder. Measure the submission and map it to a score in [0, 1], using
    # the axis values in `task_config` and the parameters in `calibration`.
    answer = submission_path.read_text()
    score = calibration.max_score

    metadata = {
        "task_id": task_config.task_id,
        "variant": task_config.variant,
        "answer_chars": len(answer),
    }
    output_path.write_text(json.dumps({"score": score, "metadata": metadata}))


if __name__ == "__main__":
    main()
