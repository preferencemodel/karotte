from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import os
import shutil
import signal
import subprocess
import sys
import textwrap
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from fastmcp import Client
from loguru import logger
from mcp.types import TextContent
from pydantic import ValidationError

from karotte.mcp_servers.mcp_server import McpServer
from karotte.tools.validate import ValidateConfig, validate
from karotte.validation import load_state

if TYPE_CHECKING:
    # Only the fixtures' type: `tests` is not importable at run time until
    # something has imported `karotte.agents`, which this module does not.
    from tests.conftest import ValidateEnv


class FakeClock:
    """Stands in for the `time` module the validate tool reads the time from."""

    def __init__(self) -> None:
        self.now: float = 1000.0

    def time(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    clock = FakeClock()
    monkeypatch.setattr("karotte.tools.validate.time", clock)
    return clock


@pytest.fixture
def logged_errors() -> Iterator[list[str]]:
    messages: list[str] = []
    handler = logger.add(messages.append, level="ERROR")
    yield messages
    logger.remove(handler)


async def call(tool: validate) -> dict[str, Any]:
    result = await tool()
    return json.loads(result.content[0].text)  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.asyncio
async def test_valid_submission_is_graded_on_the_dev_scorer(validate_env: ValidateEnv):
    tool = validate_env.configure()
    validate_env.write("0.82312")

    payload = await call(tool)

    assert payload["tier"] == "valid"
    assert payload["dev_metric"] == 0.823
    assert payload["metric"] == "accuracy, higher is better"
    assert payload["reason"] is None
    assert payload["delta_vs_prev_valid"] is None
    assert payload["best_dev_metric"] == 0.823
    assert payload["calls_left"] == 4
    assert isinstance(payload["elapsed_s"], float)
    assert validate_env.scorer_runs == 1


@pytest.mark.asyncio
async def test_missing_submission_is_no_valid_output_and_is_never_scored(
    validate_env: ValidateEnv,
):
    tool = validate_env.configure()

    payload = await call(tool)

    assert payload["tier"] == "no_valid_output"
    assert payload["dev_metric"] is None
    assert str(validate_env.submission) in payload["reason"]
    assert payload["calls_left"] == 4
    assert validate_env.scorer_runs == 0


@pytest.mark.parametrize(
    ("text", "tier", "reason"),
    [
        pytest.param(
            "reject need at least 10 rows",
            "no_valid_output",
            "need at least 10 rows",
            id="scorer-rejects-the-submission",
        ),
        pytest.param(
            "fail model did not load",
            "crash",
            "model did not load",
            id="scorer-reports-a-crash",
        ),
    ],
)
@pytest.mark.asyncio
async def test_scorer_outcome_picks_the_tier(
    validate_env: ValidateEnv, text: str, tier: str, reason: str
):
    tool = validate_env.configure()
    validate_env.write(text)

    payload = await call(tool)

    assert payload["tier"] == tier
    assert payload["reason"] == reason
    assert payload["dev_metric"] is None
    assert payload["best_dev_metric"] is None


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("crash", id="scorer-exits-with-an-error"),
        pytest.param("garbage", id="scorer-writes-no-result"),
    ],
)
@pytest.mark.asyncio
async def test_a_dev_scorer_that_fails_is_unavailable_and_costs_nothing(
    validate_env: ValidateEnv, text: str
):
    tool = validate_env.configure()
    validate_env.write("0.5")
    _ = await call(tool)
    validate_env.write(text)

    payload = await call(tool)

    assert payload["reason"] == "validation is unavailable"
    assert payload["tier"] == "valid"
    assert payload["dev_metric"] is None
    assert payload["delta_vs_prev_valid"] is None
    assert payload["best_dev_metric"] == 0.5
    assert payload["calls_left"] == 4


@pytest.mark.asyncio
async def test_a_dev_scorer_that_cannot_start_is_unavailable(
    validate_env: ValidateEnv,
):
    tool = validate_env.configure(dev_command=["/nonexistent/dev_scorer"])
    validate_env.write("0.5")

    payload = await call(tool)

    assert payload["tier"] == "no_valid_output"
    assert payload["reason"] == "validation is unavailable"
    assert payload["calls_left"] == 5


@pytest.mark.asyncio
async def test_a_dev_scorer_failure_is_logged_as_an_error_and_not_shown(
    validate_env: ValidateEnv, logged_errors: list[str]
):
    tool = validate_env.configure()
    validate_env.write("crash")

    result = await tool()

    assert "exited with code 3" in "".join(logged_errors)
    assert "exited with code 3" not in result.content[0].text  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.asyncio
async def test_an_unavailable_call_still_has_to_keep_to_the_interval(
    validate_env: ValidateEnv, clock: FakeClock
):
    tool = validate_env.configure(min_interval_s=60)
    validate_env.write("crash")
    _ = await call(tool)
    clock.advance(20)

    too_soon = await call(tool)

    assert too_soon["reason"] == "too soon: wait 40 s before calling again"
    assert validate_env.scorer_runs == 1


@pytest.mark.asyncio
async def test_scorer_that_overruns_its_timeout_is_a_crash(validate_env: ValidateEnv):
    tool = validate_env.configure(timeout_s=0.5)
    validate_env.write("hang")

    payload = await call(tool)

    assert payload["tier"] == "crash"
    assert payload["reason"] == "the dev scorer did not finish within 0.5 s"
    # The scorer sleeps for a minute, so it was stopped rather than waited for.
    assert payload["elapsed_s"] < 10


@pytest.mark.asyncio
async def test_calls_stop_at_the_cap_and_keep_the_last_tier(validate_env: ValidateEnv):
    tool = validate_env.configure(max_calls=2)
    validate_env.write("0.5")

    _ = await call(tool)
    last_allowed = await call(tool)
    refused = await call(tool)

    assert last_allowed["calls_left"] == 0
    assert refused["tier"] == "valid"
    assert refused["dev_metric"] is None
    assert refused["reason"] == "budget spent"
    assert refused["best_dev_metric"] == 0.5
    assert refused["calls_left"] == 0
    assert validate_env.scorer_runs == 2


@pytest.mark.asyncio
async def test_a_call_made_before_the_interval_is_up_is_refused(
    validate_env: ValidateEnv, clock: FakeClock
):
    tool = validate_env.configure(min_interval_s=60)
    validate_env.write("0.5")
    _ = await call(tool)
    clock.advance(20)

    too_soon = await call(tool)

    assert too_soon["tier"] == "valid"
    assert too_soon["dev_metric"] is None
    assert too_soon["reason"] == "too soon: wait 40 s before calling again"
    assert validate_env.scorer_runs == 1


@pytest.mark.asyncio
async def test_a_call_refused_as_too_soon_does_not_use_up_a_call(
    validate_env: ValidateEnv, clock: FakeClock
):
    tool = validate_env.configure(min_interval_s=60)
    validate_env.write("0.5")
    _ = await call(tool)
    clock.advance(20)

    too_soon = await call(tool)

    assert too_soon["calls_left"] == 4


@pytest.mark.asyncio
async def test_a_refused_call_does_not_restart_the_interval(
    validate_env: ValidateEnv, clock: FakeClock
):
    tool = validate_env.configure(min_interval_s=60)
    validate_env.write("0.5")
    _ = await call(tool)
    clock.advance(20)
    _ = await call(tool)
    clock.advance(40)

    in_time = await call(tool)

    assert in_time["dev_metric"] == 0.5
    assert in_time["calls_left"] == 3
    assert validate_env.scorer_runs == 2


@pytest.mark.parametrize(
    "between",
    [
        pytest.param("reject not a number", id="submission-rejected"),
        pytest.param("crash", id="dev-scorer-unavailable"),
    ],
)
@pytest.mark.asyncio
async def test_delta_is_against_the_previous_valid_call(
    validate_env: ValidateEnv, between: str
):
    tool = validate_env.configure()

    validate_env.write("0.5")
    _ = await call(tool)
    validate_env.write("0.62")
    second = await call(tool)
    validate_env.write("0.55")
    worse = await call(tool)
    validate_env.write(between)
    in_between = await call(tool)
    validate_env.write("0.58")
    after_between = await call(tool)

    assert second["delta_vs_prev_valid"] == 0.12
    assert worse["delta_vs_prev_valid"] == -0.07
    assert in_between["delta_vs_prev_valid"] is None
    assert after_between["delta_vs_prev_valid"] == 0.03


@pytest.mark.asyncio
async def test_a_delta_never_shows_more_than_the_rounded_metrics_do(
    validate_env: ValidateEnv,
):
    tool = validate_env.configure()

    validate_env.write("0.82312")
    _ = await call(tool)
    validate_env.write("0.82349")
    barely_better = await call(tool)
    validate_env.write("0.82351")
    over_the_rounding_edge = await call(tool)

    assert barely_better["dev_metric"] == 0.823
    assert barely_better["delta_vs_prev_valid"] == 0.0
    assert over_the_rounding_edge["dev_metric"] == 0.824
    assert over_the_rounding_edge["delta_vs_prev_valid"] == 0.001


@pytest.mark.asyncio
async def test_best_is_the_highest_when_higher_is_better(validate_env: ValidateEnv):
    tool = validate_env.configure()

    validate_env.write("0.5")
    _ = await call(tool)
    validate_env.write("0.62")
    improved = await call(tool)
    validate_env.write("0.55")
    worse = await call(tool)

    assert improved["best_dev_metric"] == 0.62
    assert worse["best_dev_metric"] == 0.62


@pytest.mark.asyncio
async def test_best_is_the_lowest_when_lower_is_better(validate_env: ValidateEnv):
    tool = validate_env.configure(metric="loss", higher_is_better=False)

    validate_env.write("0.5")
    _ = await call(tool)
    validate_env.write("0.4")
    improved = await call(tool)
    validate_env.write("0.45")
    worse = await call(tool)

    assert improved["best_dev_metric"] == 0.4
    assert worse["best_dev_metric"] == 0.4


@pytest.mark.asyncio
async def test_the_metric_says_when_lower_is_better(validate_env: ValidateEnv):
    tool = validate_env.configure(metric="loss", higher_is_better=False)
    validate_env.write("0.5")

    payload = await call(tool)

    assert payload["metric"] == "loss, lower is better"


@pytest.mark.asyncio
async def test_the_result_is_one_json_block_with_only_the_documented_fields(
    validate_env: ValidateEnv,
):
    tool = validate_env.configure()
    validate_env.write("0.5")

    result = await tool()

    assert len(result.content) == 1
    text = result.content[0].text  # pyright: ignore[reportAttributeAccessIssue]
    assert set(json.loads(text)) == {
        "tier",
        "dev_metric",
        "metric",
        "reason",
        "delta_vs_prev_valid",
        "best_dev_metric",
        "elapsed_s",
        "calls_left",
    }
    # The scorer's result also carries `metadata` with this label and an extra
    # field. Neither reaches the student.
    assert "SECRET-LABEL-7" not in text
    assert "extra" not in text


@pytest.mark.asyncio
async def test_a_long_reason_is_cut(validate_env: ValidateEnv):
    tool = validate_env.configure()
    validate_env.write("reject " + "x" * 1000)

    payload = await call(tool)

    assert payload["reason"] == "x" * 300


@pytest.mark.asyncio
async def test_control_characters_are_stripped_from_a_reason(
    validate_env: ValidateEnv,
):
    tool = validate_env.configure()
    validate_env.write("reject bad \x1b[31mred\x00 text\x7f")

    payload = await call(tool)

    assert payload["tier"] == "no_valid_output"
    assert payload["reason"] == "bad [31mred text"


@pytest.mark.asyncio
async def test_line_breaks_and_tabs_in_a_reason_become_one_space(
    validate_env: ValidateEnv,
):
    tool = validate_env.configure()
    validate_env.write("reject line one\r\nline two\ttab")

    payload = await call(tool)

    assert payload["reason"] == "line one line two tab"


@pytest.mark.asyncio
async def test_a_reason_that_is_only_control_characters_falls_back_to_the_default(
    validate_env: ValidateEnv,
):
    tool = validate_env.configure()
    validate_env.write("reject \x00\x1b\n")

    payload = await call(tool)

    assert payload["reason"] == "submission rejected"


@pytest.mark.asyncio
async def test_a_symlinked_submission_is_rejected_without_being_followed(
    validate_env: ValidateEnv, tmp_path: Path
):
    secret = tmp_path / "secret.txt"
    secret.write_text("0.99")
    validate_env.submission.symlink_to(secret)
    tool = validate_env.configure()

    payload = await call(tool)

    assert payload["tier"] == "no_valid_output"
    assert "symlink" in payload["reason"]
    assert validate_env.scorer_runs == 0


@pytest.mark.asyncio
async def test_a_symlinked_parent_directory_is_rejected(
    validate_env: ValidateEnv, tmp_path: Path
):
    real = tmp_path / "elsewhere"
    real.mkdir()
    (real / "answer.txt").write_text("0.99")
    shutil.rmtree(validate_env.submission.parent)
    validate_env.submission.parent.symlink_to(real)
    tool = validate_env.configure()

    payload = await call(tool)

    assert payload["tier"] == "no_valid_output"
    assert "symlink" in payload["reason"]
    assert validate_env.scorer_runs == 0


@pytest.mark.asyncio
async def test_a_fifo_submission_is_rejected_instead_of_blocking(
    validate_env: ValidateEnv,
):
    os.mkfifo(validate_env.submission)
    tool = validate_env.configure()

    payload = await asyncio.wait_for(call(tool), timeout=10)

    assert payload["tier"] == "no_valid_output"
    assert "not a regular file" in payload["reason"]
    assert validate_env.scorer_runs == 0


@pytest.mark.asyncio
async def test_a_submission_changed_during_scoring_is_graded_as_it_was_copied(
    validate_env: ValidateEnv,
):
    tool = validate_env.configure()
    validate_env.write("race 0.9")

    payload = await call(tool)

    # The scorer overwrote the live file while it ran.
    assert validate_env.submission.read_text() == "0.1"
    assert payload["dev_metric"] == 0.9


@pytest.mark.asyncio
async def test_a_submission_that_changes_while_it_is_copied_is_copied_again(
    validate_env: ValidateEnv,
    after_copying: Callable[[Path, Callable[[], object]], None],
):
    tool = validate_env.configure()
    validate_env.write("0.5")
    after_copying(validate_env.submission, lambda: validate_env.write("0.1"))

    payload = await call(tool)

    assert payload["tier"] == "valid"
    assert payload["dev_metric"] == 0.1


@pytest.mark.asyncio
async def test_a_file_added_to_a_submission_directory_while_it_is_copied_is_noticed(
    validate_env: ValidateEnv,
    after_copying: Callable[[Path, Callable[[], object]], None],
    tmp_path: Path,
):
    extras = tmp_path / "extras"
    extras.mkdir()
    (extras / "a.txt").write_text("a")
    tool = validate_env.configure(submission_paths=(validate_env.submission, extras))
    validate_env.write("0.5")
    after_copying(extras, lambda: (extras / "late.txt").write_text("x"))

    _ = await call(tool)

    best = load_state().best_snapshot
    assert best is not None
    assert (best / "extras" / "late.txt").read_text() == "x"


@pytest.mark.asyncio
async def test_a_file_swapped_for_a_fifo_after_the_copy_does_not_block_the_check(
    validate_env: ValidateEnv,
    after_copying: Callable[[Path, Callable[[], object]], None],
):
    tool = validate_env.configure()
    validate_env.write("0.5")

    def swap_for_a_fifo() -> None:
        validate_env.submission.unlink()
        os.mkfifo(validate_env.submission)

    after_copying(validate_env.submission, swap_for_a_fifo)

    payload = await asyncio.wait_for(call(tool), timeout=10)

    assert payload["tier"] == "no_valid_output"
    assert "not a regular file" in payload["reason"]


@pytest.mark.asyncio
async def test_a_submission_that_keeps_changing_is_a_crash_and_is_not_kept(
    validate_env: ValidateEnv,
    after_copying: Callable[[Path, Callable[[], object]], None],
    karotte_state: Path,
):
    tool = validate_env.configure()
    validate_env.write("0.5")
    versions = itertools.count(1)
    after_copying(
        validate_env.submission, lambda: validate_env.write(f"0.{next(versions)}")
    )

    payload = await call(tool)

    assert payload["tier"] == "crash"
    assert "changing while it was copied" in payload["reason"]
    assert payload["dev_metric"] is None
    assert payload["best_dev_metric"] is None
    assert payload["calls_left"] == 4
    assert validate_env.scorer_runs == 0
    assert load_state().best_snapshot is None
    assert list((karotte_state / "validate_snapshots").iterdir()) == []


@pytest.mark.asyncio
async def test_a_scorer_timeout_does_not_stop_the_students_processes(
    validate_env: ValidateEnv,
):
    student = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        tool = validate_env.configure(timeout_s=0.5)
        validate_env.write("hang")

        _ = await call(tool)

        assert student.poll() is None
    finally:
        student.kill()
        _ = student.wait()


@pytest.mark.asyncio
async def test_a_scorer_child_that_holds_the_pipes_open_does_not_hold_up_the_call(
    validate_env: ValidateEnv, tmp_path: Path
):
    stray_pid_file = tmp_path / "stray.pid"
    detaching = tmp_path / "detaching_scorer.py"
    detaching.write_text(
        textwrap.dedent(
            """
            import json, subprocess, sys
            from pathlib import Path

            stray = subprocess.Popen(["sleep", "30"], start_new_session=True)
            Path(sys.argv[1]).write_text(str(stray.pid))
            Path(sys.argv[-1]).write_text(json.dumps({"score": 0.5}))
            """
        )
    )
    tool = validate_env.configure(
        dev_command=[sys.executable, str(detaching), str(stray_pid_file)]
    )
    validate_env.write("0.5")

    try:
        payload = await asyncio.wait_for(call(tool), timeout=20)
    finally:
        with contextlib.suppress(FileNotFoundError, ProcessLookupError):
            os.kill(int(stray_pid_file.read_text()), signal.SIGKILL)

    assert payload["tier"] == "valid"


@pytest.mark.asyncio
async def test_concurrent_calls_share_one_budget(validate_env: ValidateEnv):
    tool = validate_env.configure(max_calls=1)
    validate_env.write("0.5")

    first, second = await asyncio.gather(call(tool), call(tool))

    assert {first["reason"], second["reason"]} == {None, "budget spent"}
    assert validate_env.scorer_runs == 1


@pytest.mark.asyncio
async def test_snapshot_limits_come_from_the_config(validate_env: ValidateEnv):
    tool = validate_env.configure(snapshot_limits={"max_file_bytes": 4})
    validate_env.write("0.123456")

    payload = await call(tool)

    assert payload["tier"] == "no_valid_output"
    assert "too large" in payload["reason"]
    assert validate_env.scorer_runs == 0


def make_config(**overrides: Any) -> ValidateConfig:
    settings: dict[str, Any] = {
        "dev_command": ["dev_scorer"],
        "metric": "accuracy",
        "submission_paths": (Path("/work/answer.txt"),),
    } | overrides
    return ValidateConfig(**settings)


class TestValidateConfig:
    def test_submission_paths_must_be_absolute(self):
        with pytest.raises(ValidationError, match="absolute"):
            _ = make_config(submission_paths=(Path("answer.txt"),))

    def test_submission_paths_need_distinct_names(self):
        with pytest.raises(ValidationError, match="distinct file names"):
            _ = make_config(
                submission_paths=(Path("/a/x.txt"), Path("/b/x.txt")),
            )

    @pytest.mark.parametrize(
        "overrides",
        [
            pytest.param({"dev_command": []}, id="no-dev-command"),
            pytest.param({"submission_paths": ()}, id="no-submission-paths"),
            pytest.param({"max_calls": -1}, id="negative-max-calls"),
            pytest.param({"min_interval_s": -1}, id="negative-interval"),
            pytest.param({"timeout_s": 0}, id="zero-timeout"),
            pytest.param(
                {"snapshot_limits": {"allow_symlinks": True}},
                id="unknown-snapshot-limit",
            ),
        ],
    )
    def test_invalid_settings_are_rejected(self, overrides: dict[str, Any]):
        with pytest.raises(ValidationError):
            _ = make_config(**overrides)


@pytest.mark.asyncio
async def test_over_mcp_the_tool_takes_no_arguments_and_answers_in_one_json_block(
    validate_env: ValidateEnv,
):
    _ = validate_env.configure()
    validate_env.write("0.5")
    mcp_server = McpServer("")
    mcp_server.register_tools(["validate"])

    async with Client(mcp_server.server) as client:
        listed = {tool.name: tool for tool in await client.list_tools()}
        result = await client.call_tool_mcp("validate", {})

    assert listed["validate"].inputSchema.get("properties", {}) == {}
    assert not result.isError
    assert len(result.content) == 1
    block = result.content[0]
    assert isinstance(block, TextContent)
    assert json.loads(block.text)["dev_metric"] == 0.5
