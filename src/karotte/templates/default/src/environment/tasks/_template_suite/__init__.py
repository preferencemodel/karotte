"""Template for a JSON-driven task suite.

At import time this module loads `task_configs.json`, validates each row against
`TaskConfig`, and creates one `Task` subclass per row. Adding, removing, or
editing a task is therefore a data change (edit the generators, rerun
`just gen-configs`) — the shared behaviour below is written exactly once.

The leading underscore in `_template_suite` tells `environment.get_tasks()` to
skip it, so the template registers no live tasks. Copy it with
`just create-task-suite <name>`.
"""

import sys
from pathlib import Path

from karotte import Step, Task
from karotte.judges import ExecutableJudge

from environment.paths import STUDENT_WORKDIR
from environment.submissions import collect_submission
from environment.suite import register_suite, stash_task_configs
from environment.system_prompts import get_system_prompt
from environment.tasks._template_suite.task_configs.task_config import (
    TaskCalibration,
    TaskConfig,
)


class SuiteTask(Task):
    """Shared behaviour for every task in the suite.

    Per-task differences come from `self.task_config`, set as a class attribute
    on each generated subclass by `register_suite` below.
    """

    task_config: TaskConfig
    task_calibration: TaskCalibration

    @property
    def system_prompt(self) -> str:
        return get_system_prompt(self.config.model, self.config.extra_config)

    def pre_hook(self):
        # Materialize this task's input data here, using self.task_config.
        # Anything the agent may see goes under STUDENT_WORKDIR; anything only
        # the grader may see goes under ROOT_DATA_DIR (see environment.paths).
        stash_task_configs("_template_suite", self.task_config, self.task_calibration)
        return {}

    @property
    def steps(self):
        return [SuiteStep(self.task_config, config=self.config)]

    @property
    def tools(self):
        return ["bash"]


class SuiteStep(Step):
    saved_submissions: tuple[Path, ...] = ()

    def __init__(self, task_config: TaskConfig, config):
        super().__init__(config)
        self.task_config = task_config

    @property
    def submission_paths(self) -> tuple[Path, ...]:
        submission = self.task_config.submission
        return (Path(submission.format(STUDENT_WORKDIR=STUDENT_WORKDIR)),)

    @property
    def instructions(self) -> str:
        # The stored instructions keep {STUDENT_WORKDIR} as a literal
        # placeholder, so the same JSON works locally and containerized.
        return self.task_config.instructions.format(STUDENT_WORKDIR=STUDENT_WORKDIR)

    @property
    def judge(self):
        assert self.saved_submissions, "pre_scoring_hook has not run"
        return ExecutableJudge(
            [
                sys.executable,
                "-m",
                "environment.tasks._template_suite.scoring_script",
                str(self.saved_submissions[0]),
                "/tmp/score_output.txt",
            ]
        )

    def pre_scoring_hook(self):
        self.saved_submissions = collect_submission(self.config, self.submission_paths)


register_suite(
    namespace=globals(),
    base_task_cls=SuiteTask,
    configs_dir=Path(__file__).resolve().parent / "task_configs",
    config_model=TaskConfig,
    calibration_model=TaskCalibration,
    suite_name="_template_suite",
)
