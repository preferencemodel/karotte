"""Instruction text for the suite, composed from a task's axis values.

`create_task_configs.py` calls this once per task and stores the *rendered*
string in `task_configs.json`; nothing calls it at task time. Keeping it here
lets the instructions grow without cluttering the generator's loop.
"""

from textwrap import dedent

# Interpolated as-is, so the generated JSON keeps a literal {STUDENT_WORKDIR}
# that `SuiteStep.instructions` fills in at run time. That keeps the same JSON
# working whether the environment runs locally or containerized.
WORKDIR_PLACEHOLDER = "{STUDENT_WORKDIR}"


def build_instructions(variant: str, submission: str) -> str:
    return dedent(f"""
        Do something cool. This is variant {variant}.

        Write your answer to {submission}.
    """)
