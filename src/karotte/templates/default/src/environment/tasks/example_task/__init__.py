import re
import sys
from pathlib import Path
from textwrap import dedent
from typing import final

from karotte import Step, Task, kill_processes
from karotte.judges import ExecutableJudge, RegexJudge

from environment import STUDENT_UID
from environment.paths import STUDENT_DATA_DIR
from environment.submissions import collect_submission
from environment.system_prompts import get_system_prompt


@final
class ExampleTask(Task):
    id = "example-task"

    @property
    def system_prompt(self) -> str:
        return get_system_prompt(self.config.model, self.config.extra_config)

    def pre_hook(self):
        return {}

    @property
    def steps(self):
        return [
            GetPythonPathStep(config=self.config),
            GetPythonVersionStep(config=self.config),
        ]

    @property
    def tools(self):
        return ["bash", "view_lines_in_file", "replace_in_file"]


class GetPythonPathStep(Step):
    @property
    def instructions(self) -> str:
        prompt = dedent("""
            Use the `bash` tool to determine the path to your Python executable.

            State your answer in your final message in the following format:
            path: /path/to/python3
            """)

        if self.config.use_hints:
            prompt += "\nA single bash tool call will suffice."

        return prompt

    @property
    def judge(self):
        return RegexJudge([re.compile(r"path: .*/workdir/\.venv/bin/python.*")])

    def pre_scoring_hook(self):
        kill_processes(STUDENT_UID)


class GetPythonVersionStep(Step):
    saved_submissions: tuple[Path, ...] = ()

    @property
    def submission_paths(self) -> tuple[Path, ...]:
        return (STUDENT_DATA_DIR / "python_version.txt",)

    @property
    def instructions(self) -> str:
        prompt = dedent(f"""
            Use the `bash` tool to determine the version of Python you are using.

            Write your answer to {self.submission_paths[0]}
            """)

        if self.config.use_hints:
            prompt += "\nUse the python executable from your previous answer."

        return prompt

    @property
    def judge(self):
        assert self.saved_submissions, "pre_scoring_hook has not run"
        return ExecutableJudge(
            [
                sys.executable,
                "-m",
                "environment.tasks.example_task.scoring_script",
                str(self.saved_submissions[0]),
                "/tmp/score_output.txt",
            ]
        )

    def pre_scoring_hook(self):
        self.saved_submissions = collect_submission(self.config, self.submission_paths)
