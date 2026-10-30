import asyncio
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, final

import pytest
from fastmcp import Client
from mcp.types import TextContent

from karotte import ToolConfigWriter
from karotte.mcp_servers.mcp_server import McpServer
from karotte.save_submission import save_submission
from karotte.tools.validate import ValidateConfig, validate
from karotte.validation import load_state


@final
class Env:
    def __init__(self, tmp_path: Path, scorer: Path) -> None:
        self.submission: Path = tmp_path / "work" / "answer.txt"
        self.submission.parent.mkdir()
        self.counter: Path = tmp_path / "counter"
        self.scorer: Path = scorer

    @property
    def scorer_runs(self) -> int:
        if not self.counter.exists():
            return 0
        return len(self.counter.read_text().splitlines())

    def configure(self, **overrides: Any) -> validate:
        settings: dict[str, Any] = {
            "dev_command": [
                sys.executable,
                str(self.scorer),
                str(self.counter),
                str(self.submission),
            ],
            "metric": "accuracy",
            "submission_paths": (self.submission,),
            "min_interval_s": 0,
        } | overrides
        _ = ToolConfigWriter().write("validate", ValidateConfig(**settings))
        return validate()

    def write(self, text: str) -> None:
        self.submission.write_text(text)


@pytest.fixture
def env(tmp_path: Path, karotte_state: Path, dev_scorer: Path) -> Env:  # pyright: ignore[reportUnusedParameter]
    return Env(tmp_path, dev_scorer)


async def call(tool: validate) -> dict[str, Any]:
    result = await tool()
    return json.loads(result.content[0].text)  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.asyncio
async def test_valid_submission_is_graded_on_the_dev_scorer(env: Env):
    tool = env.configure()
    env.write("0.82312")

    payload = await call(tool)

    assert payload["tier"] == "valid"
    assert payload["dev_metric"] == 0.823
    assert payload["metric"] == "accuracy, higher is better"
    assert payload["reason"] is None
    assert payload["delta_vs_prev_valid"] is None
    assert payload["best_dev_metric"] == 0.823
    assert payload["calls_left"] == 4
    assert isinstance(payload["elapsed_s"], float)
    assert env.scorer_runs == 1


@pytest.mark.asyncio
async def test_missing_submission_is_no_valid_output_and_is_never_scored(env: Env):
    tool = env.configure()

    payload = await call(tool)

    assert payload["tier"] == "no_valid_output"
    assert payload["dev_metric"] is None
    assert str(env.submission) in payload["reason"]
    assert payload["calls_left"] == 4
    assert env.scorer_runs == 0


@pytest.mark.parametrize(
    ("text", "tier", "reason"),
    [
        ("reject need at least 10 rows", "no_valid_output", "need at least 10 rows"),
        ("fail model did not load", "crash", "model did not load"),
        ("crash", "crash", "the dev scorer failed (exit code 3)"),
        ("garbage", "crash", "the dev scorer produced no result"),
    ],
)
@pytest.mark.asyncio
async def test_scorer_outcome_picks_the_tier(
    env: Env, text: str, tier: str, reason: str
):
    tool = env.configure()
    env.write(text)

    payload = await call(tool)

    assert payload["tier"] == tier
    assert payload["reason"] == reason
    assert payload["dev_metric"] is None
    assert payload["best_dev_metric"] is None


@pytest.mark.asyncio
async def test_scorer_that_overruns_its_timeout_is_a_crash(env: Env):
    tool = env.configure(timeout_s=0.5)
    env.write("hang")

    payload = await call(tool)

    assert payload["tier"] == "crash"
    assert payload["reason"] == "the dev scorer did not finish within 0.5 s"
    assert payload["elapsed_s"] < 10


@pytest.mark.asyncio
async def test_calls_stop_at_the_cap_and_keep_the_last_tier(env: Env):
    tool = env.configure(max_calls=2)
    env.write("0.5")

    _ = await call(tool)
    last_allowed = await call(tool)
    refused = await call(tool)

    assert last_allowed["calls_left"] == 0
    assert refused["tier"] == "valid"
    assert refused["dev_metric"] is None
    assert refused["reason"] == "budget spent"
    assert refused["best_dev_metric"] == 0.5
    assert refused["calls_left"] == 0
    assert env.scorer_runs == 2


@pytest.mark.asyncio
async def test_calls_must_be_spaced_apart_and_a_refused_call_costs_nothing(
    env: Env, monkeypatch: pytest.MonkeyPatch
):
    now = [1000.0]
    monkeypatch.setattr("karotte.tools.validate._now", lambda: now[0])
    tool = env.configure(min_interval_s=60)
    env.write("0.5")

    _ = await call(tool)
    now[0] += 20
    too_soon = await call(tool)
    now[0] += 40
    in_time = await call(tool)

    assert too_soon["tier"] == "valid"
    assert too_soon["dev_metric"] is None
    assert too_soon["reason"] == "too soon: wait 40 s before calling again"
    assert too_soon["calls_left"] == 4
    assert in_time["dev_metric"] == 0.5
    assert in_time["calls_left"] == 3
    assert env.scorer_runs == 2


@pytest.mark.asyncio
async def test_delta_is_against_the_previous_valid_call(env: Env):
    tool = env.configure()

    env.write("0.5")
    first = await call(tool)
    env.write("0.62")
    second = await call(tool)
    env.write("crash")
    broken = await call(tool)
    env.write("0.55")
    after_break = await call(tool)

    assert first["delta_vs_prev_valid"] is None
    assert second["delta_vs_prev_valid"] == 0.12
    assert broken["delta_vs_prev_valid"] is None
    assert broken["best_dev_metric"] == 0.62
    assert after_break["delta_vs_prev_valid"] == -0.07
    assert after_break["best_dev_metric"] == 0.62


@pytest.mark.asyncio
async def test_a_delta_never_shows_more_than_the_rounded_metrics_do(env: Env):
    tool = env.configure()

    env.write("0.82312")
    _ = await call(tool)
    env.write("0.82349")
    barely_better = await call(tool)
    env.write("0.82351")
    over_the_rounding_edge = await call(tool)

    assert barely_better["dev_metric"] == 0.823
    assert barely_better["delta_vs_prev_valid"] == 0.0
    assert over_the_rounding_edge["dev_metric"] == 0.824
    assert over_the_rounding_edge["delta_vs_prev_valid"] == 0.001


@pytest.mark.asyncio
async def test_best_is_the_lowest_when_lower_is_better(env: Env):
    tool = env.configure(metric="loss", higher_is_better=False)

    env.write("0.5")
    _ = await call(tool)
    env.write("0.4")
    improved = await call(tool)
    env.write("0.45")
    worse = await call(tool)

    assert improved["metric"] == "loss, lower is better"
    assert improved["best_dev_metric"] == 0.4
    assert worse["best_dev_metric"] == 0.4


@pytest.mark.asyncio
async def test_the_result_is_one_json_block_with_only_the_documented_fields(env: Env):
    tool = env.configure()
    env.write("0.5")

    result = await tool()

    assert len(result.content) == 1
    payload = json.loads(result.content[0].text)  # pyright: ignore[reportAttributeAccessIssue]
    assert set(payload) == {
        "tier",
        "dev_metric",
        "metric",
        "reason",
        "delta_vs_prev_valid",
        "best_dev_metric",
        "elapsed_s",
        "calls_left",
    }
    assert "SECRET-LABEL-7" not in result.content[0].text  # pyright: ignore[reportAttributeAccessIssue]
    assert "extra" not in result.content[0].text  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.asyncio
async def test_a_long_reason_is_cut(env: Env):
    tool = env.configure()
    env.write("reject " + "x" * 1000)

    payload = await call(tool)

    assert payload["reason"] == "x" * 300


@pytest.mark.asyncio
async def test_control_characters_are_stripped_from_a_reason(env: Env):
    tool = env.configure()
    env.write("reject bad \x1b[31mred\x00 line one\r\nline two\ttab\x7f")

    payload = await call(tool)

    assert payload["tier"] == "no_valid_output"
    assert payload["reason"] == "bad [31mred line one line two tab"


@pytest.mark.asyncio
async def test_a_reason_that_is_only_control_characters_falls_back_to_the_default(
    env: Env,
):
    tool = env.configure()
    env.write("reject \x00\x1b\n")

    payload = await call(tool)

    assert payload["reason"] == "submission rejected"


@pytest.mark.asyncio
async def test_a_symlinked_submission_is_rejected_without_being_followed(
    env: Env, tmp_path: Path
):
    secret = tmp_path / "secret.txt"
    secret.write_text("0.99")
    env.submission.symlink_to(secret)
    tool = env.configure()

    payload = await call(tool)

    assert payload["tier"] == "no_valid_output"
    assert "symlink" in payload["reason"]
    assert env.scorer_runs == 0


@pytest.mark.asyncio
async def test_a_symlinked_parent_directory_is_rejected(env: Env, tmp_path: Path):
    real = tmp_path / "elsewhere"
    real.mkdir()
    (real / "answer.txt").write_text("0.99")
    shutil.rmtree(env.submission.parent)
    env.submission.parent.symlink_to(real)
    tool = env.configure()

    payload = await call(tool)

    assert payload["tier"] == "no_valid_output"
    assert "symlink" in payload["reason"]
    assert env.scorer_runs == 0


@pytest.mark.asyncio
async def test_a_fifo_submission_is_rejected_instead_of_blocking(env: Env):
    os.mkfifo(env.submission)
    tool = env.configure()

    payload = await asyncio.wait_for(call(tool), timeout=10)

    assert payload["tier"] == "no_valid_output"
    assert "not a regular file" in payload["reason"]
    assert env.scorer_runs == 0


@pytest.mark.asyncio
async def test_a_submission_changed_during_scoring_is_graded_as_it_was_copied(
    env: Env,
):
    tool = env.configure()
    env.write("race 0.9")

    payload = await call(tool)

    assert env.submission.read_text() == "0.1"
    assert payload["dev_metric"] == 0.9


def after_each_copy(
    monkeypatch: pytest.MonkeyPatch, change: Callable[[int], None]
) -> None:
    """Call `change(n)` right after the n-th copy of a submission path (counted
    from 1), as a student process writing in the middle of a copy would."""
    copies: list[Path | str] = []

    def copy_then_change(
        source: Path | str, dest: Path | str | None = None, **limits: Any
    ):
        saved = save_submission(source, dest, **limits)
        copies.append(source)
        change(len(copies))
        return saved

    monkeypatch.setattr("karotte.validation.save_submission", copy_then_change)


@pytest.mark.asyncio
async def test_a_submission_that_changes_while_it_is_copied_is_copied_again(
    env: Env, monkeypatch: pytest.MonkeyPatch
):
    tool = env.configure()
    env.write("0.5")

    def rewrite(n: int) -> None:
        if n == 1:
            env.write("0.1")

    after_each_copy(monkeypatch, rewrite)

    payload = await call(tool)

    assert payload["tier"] == "valid"
    assert payload["dev_metric"] == 0.1


@pytest.mark.asyncio
async def test_a_file_added_to_a_submission_directory_while_it_is_copied_is_noticed(
    env: Env, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    extras = tmp_path / "extras"
    extras.mkdir()
    (extras / "a.txt").write_text("a")
    tool = env.configure(submission_paths=(env.submission, extras))
    env.write("0.5")

    def add_file(n: int) -> None:
        if n == 2:
            _ = (extras / "late.txt").write_text("x")

    after_each_copy(monkeypatch, add_file)

    _ = await call(tool)

    best = load_state().best_snapshot
    assert best is not None
    assert (best / "extras" / "late.txt").read_text() == "x"


@pytest.mark.asyncio
async def test_a_file_swapped_for_a_fifo_after_the_copy_does_not_block_the_check(
    env: Env, monkeypatch: pytest.MonkeyPatch
):
    tool = env.configure()
    env.write("0.5")

    def swap(n: int) -> None:
        if n == 1:
            env.submission.unlink()
            os.mkfifo(env.submission)

    after_each_copy(monkeypatch, swap)

    payload = await asyncio.wait_for(call(tool), timeout=10)

    assert payload["tier"] == "no_valid_output"
    assert "not a regular file" in payload["reason"]


@pytest.mark.asyncio
async def test_a_submission_that_keeps_changing_is_a_crash_and_is_not_kept(
    env: Env, monkeypatch: pytest.MonkeyPatch, karotte_state: Path
):
    tool = env.configure()
    env.write("0.5")

    def rewrite_every_time(n: int) -> None:
        env.write(f"0.{n}")

    after_each_copy(monkeypatch, rewrite_every_time)

    payload = await call(tool)

    assert payload["tier"] == "crash"
    assert "changing while it was copied" in payload["reason"]
    assert payload["dev_metric"] is None
    assert payload["best_dev_metric"] is None
    assert payload["calls_left"] == 4
    assert env.scorer_runs == 0
    assert load_state().best_snapshot is None
    assert list((karotte_state / "validate_snapshots").iterdir()) == []


@pytest.mark.asyncio
async def test_student_processes_are_left_running(env: Env):
    student = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        tool = env.configure(timeout_s=0.5)
        env.write("hang")

        _ = await call(tool)

        assert student.poll() is None
    finally:
        student.kill()
        _ = student.wait()


@pytest.mark.asyncio
async def test_a_scorer_child_that_holds_the_pipes_open_does_not_hold_up_the_call(
    env: Env, tmp_path: Path
):
    detaching = tmp_path / "detaching_scorer.py"
    detaching.write_text(
        "\n".join(
            [
                "import json, subprocess, sys",
                "subprocess.Popen(['sleep', '30'], start_new_session=True)",
                "open(sys.argv[-1], 'w').write(json.dumps({'score': 0.5}))",
            ]
        )
    )
    tool = env.configure(dev_command=[sys.executable, str(detaching)])
    env.write("0.5")

    payload = await asyncio.wait_for(call(tool), timeout=20)

    assert payload["tier"] == "valid"


@pytest.mark.asyncio
async def test_concurrent_calls_share_one_budget(env: Env):
    tool = env.configure(max_calls=1)
    env.write("0.5")

    payloads = await asyncio.gather(call(tool), call(tool))

    assert sorted(p["reason"] or "" for p in payloads) == ["", "budget spent"]
    assert env.scorer_runs == 1


@pytest.mark.asyncio
async def test_snapshot_limits_come_from_the_config(env: Env):
    tool = env.configure(snapshot_limits={"max_file_bytes": 4})
    env.write("0.123456")

    payload = await call(tool)

    assert payload["tier"] == "no_valid_output"
    assert "too large" in payload["reason"]
    assert env.scorer_runs == 0


class TestValidateConfig:
    def test_submission_paths_must_be_absolute(self, env: Env):
        with pytest.raises(ValueError, match="absolute"):
            _ = env.configure(submission_paths=(Path("answer.txt"),))

    def test_submission_paths_need_distinct_names(self, env: Env, tmp_path: Path):
        with pytest.raises(ValueError, match="distinct file names"):
            _ = env.configure(
                submission_paths=(tmp_path / "a" / "x.txt", tmp_path / "b" / "x.txt")
            )

    @pytest.mark.parametrize(
        "overrides",
        [
            {"dev_command": []},
            {"submission_paths": ()},
            {"max_calls": -1},
            {"min_interval_s": -1},
            {"timeout_s": 0},
            {"snapshot_limits": {"allow_symlinks": True}},
        ],
    )
    def test_nonsense_is_rejected(self, env: Env, overrides: dict[str, Any]):
        with pytest.raises(ValueError):
            _ = env.configure(**overrides)


@pytest.mark.asyncio
async def test_a_scorer_that_cannot_start_is_a_crash(env: Env):
    tool = env.configure(dev_command=["/nonexistent/dev_scorer"])
    env.write("0.5")

    payload = await call(tool)

    assert payload["tier"] == "crash"
    assert payload["reason"] == "the dev scorer could not be started"


@pytest.mark.asyncio
async def test_over_mcp_the_tool_takes_no_arguments_and_answers_in_one_json_block(
    env: Env,
):
    _ = env.configure()
    env.write("0.5")
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
