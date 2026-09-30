from pathlib import Path
from textwrap import dedent
from typing import final

from karotte import Step, Task
from karotte.judges import AlwaysPassJudge

from environment.paths import STUDENT_DATA_DIR
from environment.submissions import collect_submission
from environment.system_prompts import get_system_prompt


@final
class Task_(Task):
    id = ""

    @property
    def system_prompt(self) -> str:
        return get_system_prompt(self.config.model, self.config.extra_config)

    def pre_hook(self):
        # Gets called before the task gets executed.
        # Use this to do any dynamic setup for the task.
        return {}

    @property
    def steps(self):
        return [
            FirstStep(config=self.config),
        ]

    @property
    def tools(self):
        return ["bash", "view_lines_in_file", "replace_in_file"]


class FirstStep(Step):
    saved_submissions: tuple[Path, ...] = ()

    @property
    def submission_paths(self) -> tuple[Path, ...]:
        return (STUDENT_DATA_DIR / "submission.txt",)

    @property
    def instructions(self) -> str:
        prompt = dedent(f"""
            Do something cool and write your answer to {self.submission_paths[0]}.
            """)

        if self.config.use_hints:
            prompt += ""

        return prompt

    @property
    def judge(self):
        return AlwaysPassJudge()

    def pre_scoring_hook(self):
        self.saved_submissions = collect_submission(self.config, self.submission_paths)
