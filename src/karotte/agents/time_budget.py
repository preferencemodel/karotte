"""The time-budget paragraph for agents that karotte can't remind mid-step.

The builtin loop appends "Time remaining: N seconds" after every turn. A CLI
agent runs a whole step as one opaque subprocess, so it gets no such counter;
instead its instructions end with this paragraph, stating the budget once and
giving the agent a command that prints the seconds it has used.
"""

MARKER = "Time budget:"
"""How the paragraph starts; instructions that already contain it (an
environment that states the budget itself) don't get a second one."""


def _duration(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds} s"
    hours, rest = divmod(seconds, 3600)
    minutes = rest // 60
    parts = [f"{hours} h"] if hours else []
    if minutes or not hours:
        parts.append(f"{minutes} min")
    return " ".join(parts)


def time_budget_note(
    seconds: float, *, started_at: int, multi_step: bool = False
) -> str:
    """The paragraph for a step of ``seconds`` that started at Unix time
    ``started_at``; the start is a literal in the command, so it works on any
    host without a file to read it from."""
    total = int(seconds)
    scope = "this step" if multi_step else "this task"
    return (
        f"{MARKER} you have {total} seconds ({_duration(total)}) for {scope}. "
        + "At the limit your session ends and your work is tested as it stands. "
        + "Nothing reports the time remaining, so track it yourself: "
        + f"`echo $(( $(date +%s) - {started_at} ))` prints the seconds used so far."
    )


def with_time_budget(
    instructions: str, seconds: float, *, started_at: int, multi_step: bool = False
) -> str:
    """``instructions`` with the paragraph appended, unless they state a budget."""
    if MARKER in instructions:
        return instructions
    note = time_budget_note(seconds, started_at=started_at, multi_step=multi_step)
    return f"{instructions.rstrip()}\n\n{note}\n"
