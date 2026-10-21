"""The time-budget paragraph CLI agents get in place of a remaining-time counter."""

import pytest

from karotte.agents.time_budget import time_budget_note, with_time_budget


@pytest.mark.parametrize(
    ("seconds", "duration"),
    [
        (45, "45 s"),
        (1800, "30 min"),
        (7200, "2 h"),
        (5400, "1 h 30 min"),
        (21600, "6 h"),
    ],
)
def test_states_the_budget_in_seconds_and_plain_units(seconds: int, duration: str):
    assert (
        f"you have {seconds} seconds ({duration}) for this task."
        in time_budget_note(seconds, started_at=0)
    )


def test_a_multi_step_run_budgets_the_step():
    assert "for this step." in time_budget_note(60, started_at=0, multi_step=True)


def test_the_elapsed_command_needs_no_file_on_the_host():
    note = time_budget_note(60, started_at=1792630000)
    assert "`echo $(( $(date +%s) - 1792630000 ))`" in note
    assert "/run/" not in note
    assert "graded" not in note


def test_appends_once():
    once = with_time_budget("do it", 60, started_at=0)
    assert once.startswith("do it\n\nTime budget:")
    assert with_time_budget(once, 60, started_at=1) == once
