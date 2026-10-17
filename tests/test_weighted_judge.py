import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast, override

import pytest

from karotte import StepConfig, create_task
from karotte.agents import FakeSource, MessageLoopAgent
from karotte.evaluation_runner import EvaluationRunner
from karotte.judges import AlwaysPassJudge, Judge, WeightedJudge
from karotte.mcp_servers.http_mcp_server import HttpMcpServer
from karotte.schemas import EvaluationRunConfig
from karotte.schemas.chat import Message
from karotte.schemas.scoring import Metadata, Scoring
from karotte.schemas.transcript import (
    ErrorEvent,
    ScoringEvent,
    TaskCompletedEvent,
    Transcript,
)
from karotte.student_misbehavior import StudentMisbehaviorError
from karotte.transcript_markdown import convert_scoring_event_to_markdown


@dataclass
class StubJudge(Judge):
    result: Scoring
    error: Exception | None = None
    transcripts: list[Transcript] = field(default_factory=list)
    order: list[str] | None = None
    name: str = ""

    @override
    def evaluate(self, transcript: Transcript) -> Scoring:
        self.transcripts.append(transcript)
        if self.order is not None:
            self.order.append(self.name)
        if self.error is not None:
            raise self.error
        return self.result


def judge(
    score: float = 1.0, *, passed: bool = True, metadata: Metadata | None = None
) -> StubJudge:
    return StubJudge(Scoring(score, metadata or {}, passed))


def test_weighted_reward_and_breakdown(transcript: Transcript):
    correctness = judge(0.9, metadata={"checked": True})
    speed = judge(0.8, passed=False, metadata={"seconds": 2.0})
    memory = judge(0.2)
    scoring = WeightedJudge(
        {"speed": (speed, 3.0), "memory": (memory, 1.0)},
        gates={"correctness": correctness},
        continue_threshold=0.6,
    ).evaluate(transcript)

    assert scoring.score == pytest.approx(0.65)
    assert scoring.continue_task is True
    assert scoring.metadata["gates"] == {
        "correctness": {
            "evaluated": True,
            "score": 0.9,
            "continue_task": True,
            "metadata": {"checked": True},
        }
    }
    components = scoring.metadata["components"]
    assert components["speed"]["weight"] == 3.0
    assert components["speed"]["normalized_weight"] == 0.75
    assert components["speed"]["contribution"] == pytest.approx(0.6)
    assert components["speed"]["continue_task"] is False
    assert components["speed"]["metadata"] == {"seconds": 2.0}
    assert components["memory"]["contribution"] == 0.05
    assert scoring.metadata["continue_threshold"] == 0.6
    assert all(j.transcripts == [transcript] for j in (correctness, speed, memory))


@pytest.mark.parametrize("gate_score", [0.0, 0.8, 1.0])
def test_failed_gate_blocks_all_reward(transcript: Transcript, gate_score: float):
    first, failed, later, component = (
        judge(),
        judge(gate_score, passed=False),
        judge(),
        judge(),
    )
    scoring = WeightedJudge(
        {"performance": (component, 2.0)},
        gates={"first": first, "failed": failed, "later": later},
        continue_threshold=-1.0,
    ).evaluate(transcript)

    assert scoring.score == 0.0
    assert scoring.continue_task is False
    assert scoring.metadata["gates"]["failed"]["score"] == gate_score
    assert scoring.metadata["gates"]["later"] == {"evaluated": False}
    assert scoring.metadata["components"]["performance"] == {
        "evaluated": False,
        "weight": 2.0,
        "normalized_weight": 1.0,
    }
    assert first.transcripts == failed.transcripts == [transcript]
    assert later.transcripts == component.transcripts == []


def test_all_gates_precede_components_in_insertion_order(transcript: Transcript):
    order: list[str] = []
    checks = {
        name: StubJudge(Scoring(1.0, {}, True), order=order, name=name)
        for name in ("valid", "correct", "speed", "memory")
    }
    WeightedJudge(
        {"speed": (checks["speed"], 1.0), "memory": (checks["memory"], 1.0)},
        gates={"valid": checks["valid"], "correct": checks["correct"]},
        continue_threshold=1.0,
    ).evaluate(transcript)

    assert order == ["valid", "correct", "speed", "memory"]


@pytest.mark.parametrize(
    "threshold,continues", [(0.5, True), (0.500001, False), (-2.0, True), (2.0, False)]
)
def test_threshold_and_component_flags(
    transcript: Transcript, threshold: float, continues: bool
):
    scoring = WeightedJudge(
        {"partial": (judge(0.5, passed=False), 7.0)},
        continue_threshold=threshold,
    ).evaluate(transcript)

    assert scoring.score == 0.5
    assert scoring.continue_task is continues
    assert scoring.metadata["gates"] == {}


@pytest.mark.parametrize("score", [-4.0, 2.0])
def test_scores_are_not_clamped(transcript: Transcript, score: float):
    scoring = WeightedJudge(
        {"metric": (judge(score), 1.0)}, continue_threshold=score
    ).evaluate(transcript)

    assert scoring.score == score
    assert scoring.continue_task is True


def test_large_finite_values_do_not_overflow(transcript: Transcript):
    scoring = WeightedJudge(
        {"a": (judge(1e200), 1e150), "b": (judge(1e200), 1e150)},
        continue_threshold=0.0,
    ).evaluate(transcript)

    assert math.isfinite(scoring.score)
    assert scoring.score == pytest.approx(1e200)


def test_empty_components_are_rejected():
    with pytest.raises(ValueError, match="at least one component"):
        WeightedJudge({}, gates={"gate": judge()}, continue_threshold=0.0)


@pytest.mark.parametrize("weight", [0.0, -1.0, math.nan, math.inf, -math.inf])
def test_invalid_weights_are_rejected(weight: float):
    with pytest.raises(ValueError, match="finite and positive"):
        WeightedJudge({"metric": (judge(), weight)}, continue_threshold=0.0)


def test_overflowing_total_weight_is_rejected():
    with pytest.raises(ValueError, match="sum of component weights"):
        WeightedJudge(
            {"a": (judge(), 1e308), "b": (judge(), 1e308)},
            continue_threshold=0.0,
        )


@pytest.mark.parametrize("threshold", [math.nan, math.inf, -math.inf])
def test_nonfinite_thresholds_are_rejected(threshold: float):
    with pytest.raises(ValueError, match="continue_threshold must be finite"):
        WeightedJudge({"metric": (judge(), 1.0)}, continue_threshold=threshold)


@pytest.mark.parametrize("name", ["", " ", 42])
@pytest.mark.parametrize("in_gate", [False, True])
def test_names_must_be_nonempty_strings(name: object, in_gate: bool):
    name = cast(str, name)
    components = {"metric": (judge(), 1.0)} if in_gate else {name: (judge(), 1.0)}
    gates = {name: judge()} if in_gate else None
    with pytest.raises(ValueError, match="non-empty strings"):
        WeightedJudge(components, gates=gates, continue_threshold=0.0)


@pytest.mark.parametrize("score", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("in_gate", [False, True])
def test_nonfinite_child_scores_are_rejected(
    transcript: Transcript, score: float, in_gate: bool
):
    invalid = judge(score)
    component = judge() if in_gate else invalid
    weighted = WeightedJudge(
        {"metric": (component, 1.0)},
        gates={"gate": invalid} if in_gate else None,
        continue_threshold=0.0,
    )

    with pytest.raises(ValueError, match="non-finite score"):
        weighted.evaluate(transcript)
    if in_gate:
        assert component.transcripts == []


@pytest.mark.parametrize(
    "error", [RuntimeError("grader failed"), StudentMisbehaviorError("tampering")]
)
@pytest.mark.parametrize("in_gate", [False, True])
def test_errors_propagate_without_redistributing_reward(
    transcript: Transcript, error: Exception, in_gate: bool
):
    broken = judge()
    broken.error = error
    later = judge()
    weighted = WeightedJudge(
        {"first": (judge(0.5), 1.0), "broken": (broken, 1.0), "later": (later, 1.0)},
        gates={"gate": broken} if in_gate else None,
        continue_threshold=0.0,
    )

    with pytest.raises(type(error), match=str(error)):
        weighted.evaluate(transcript)
    assert later.transcripts == []


def test_input_mappings_are_copied_and_runs_have_fresh_metadata():
    component, gate = judge(0.6), judge()
    components = {"metric": (component, 1.0)}
    gates = {"valid": gate}
    weighted = WeightedJudge(components, gates=gates, continue_threshold=0.5)
    components["metric"] = (judge(0.0), 100.0)
    components["other"] = (judge(), 1.0)
    gates.clear()
    first, second = Transcript(run_id="first"), Transcript(run_id="second")

    result1 = weighted.evaluate(first)
    result2 = weighted.evaluate(second)

    assert result1.score == result2.score == 0.6
    assert component.transcripts == gate.transcripts == [first, second]
    assert result1.metadata == result2.metadata
    assert result1.metadata is not result2.metadata
    assert result1.metadata["components"]["metric"]["weight"] == 1.0
    assert list(result1.metadata["components"]) == ["metric"]
    assert list(result1.metadata["gates"]) == ["valid"]


def test_nested_weighted_judges_and_existing_composition(transcript: Transcript):
    inner = WeightedJudge(
        {"a": (judge(0.4), 1.0), "b": (judge(0.8), 1.0)},
        continue_threshold=0.7,
    )
    outer = WeightedJudge(
        {"inner": (inner, 1.0), "other": (judge(1.0), 1.0)},
        continue_threshold=0.5,
    )

    scoring = outer.evaluate(transcript)

    assert scoring.score == pytest.approx(0.8)
    assert scoring.continue_task is True
    assert scoring.metadata["components"]["inner"]["continue_task"] is False
    assert "a" in scoring.metadata["components"]["inner"]["metadata"]["components"]
    assert (AlwaysPassJudge() & outer).evaluate(transcript).score == pytest.approx(0.8)
    assert (inner | AlwaysPassJudge()).evaluate(transcript).score == 1.0


def test_breakdown_serializes_and_renders(transcript: Transcript):
    scoring = WeightedJudge(
        {"speed": (judge(0.8, metadata={"samples": [1, 2]}), 1.0)},
        gates={"correctness": judge()},
        continue_threshold=0.5,
    ).evaluate(transcript)
    event = ScoringEvent(scoring=scoring)
    transcript.events.append(event)

    restored = Transcript.model_validate_json(transcript.model_dump_json())
    restored_event = restored.events[0]
    assert isinstance(restored_event, ScoringEvent)
    assert restored_event.scoring == scoring
    rendered = convert_scoring_event_to_markdown(restored_event)
    assert "correctness" in rendered
    assert "speed" in rendered
    assert "contribution" in rendered


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case,expected_score,expected_status",
    [
        ("valid", 0.62, "passed"),
        ("failed_gate", 0.0, "failed"),
        ("misbehavior", 0.0, "failed"),
        ("grader_error", None, "error"),
        ("nonfinite", None, "error"),
    ],
)
async def test_runner_handles_weighted_rewards(
    mcp_server: HttpMcpServer,
    sample_config: EvaluationRunConfig,
    tmp_path: Path,
    case: str,
    expected_score: float | None,
    expected_status: str,
):
    speed = judge(math.nan if case == "nonfinite" else 0.8)
    if case == "misbehavior":
        speed.error = StudentMisbehaviorError("tampered submission")
    elif case == "grader_error":
        speed.error = RuntimeError("grader failed")
    weighted = WeightedJudge(
        {"speed": (speed, 0.7), "memory": (judge(0.2), 0.3)},
        gates={"correctness": judge(0.9, passed=case != "failed_gate")},
        continue_threshold=0.5,
    )
    config = sample_config.model_copy(
        update={
            "mcp_server_config": mcp_server.config,
            "transcript_file": str(tmp_path / "transcript.json"),
            "use_fake_model": True,
        }
    )
    task_cls = create_task(
        id=config.task_id,
        tools=[],
        steps=[StepConfig(instructions="Optimize the solution.", judge=weighted)],
        system_prompt=None,
    )
    runner = EvaluationRunner(config, task_cls(config))
    runner._agent = MessageLoopAgent(FakeSource([Message(content="Done.")]))  # pyright: ignore[reportPrivateUsage]

    events = [event async for event in runner.run()]
    assert isinstance(events[-1], TaskCompletedEvent)
    assert events[-1].status == expected_status
    scored = [event for event in events if isinstance(event, ScoringEvent)]
    if expected_score is None:
        assert scored == []
        assert any(isinstance(event, ErrorEvent) for event in events)
    else:
        assert len(scored) == 1
        assert scored[0].scoring.score == pytest.approx(expected_score)
        if case == "failed_gate":
            assert speed.transcripts == []
        elif case == "valid":
            assert scored[0].scoring.metadata["components"]["speed"][
                "contribution"
            ] == pytest.approx(0.56)
    restored = Transcript.model_validate_json(
        (tmp_path / "transcript.json").read_text()
    )
    assert [
        event for event in restored.events if isinstance(event, ScoringEvent)
    ] == scored
    assert isinstance(restored.events[-1], TaskCompletedEvent)
    assert restored.events[-1].status == expected_status
