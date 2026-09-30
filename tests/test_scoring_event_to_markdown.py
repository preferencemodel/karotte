from karotte.schemas.scoring import Scoring
from karotte.schemas.transcript import ResourceMetrics, ResourceSample, ScoringEvent
from karotte.transcript_markdown import convert_scoring_event_to_markdown


class TestBasicScoring:
    def test_passing_score_shows_checkmark(self) -> None:
        event = ScoringEvent(
            scoring=Scoring(score=1.0, metadata={}, continue_task=True)
        )
        result = convert_scoring_event_to_markdown(event)
        assert "✔️" in result
        assert "❌" not in result

    def test_failing_score_shows_x(self) -> None:
        event = ScoringEvent(
            scoring=Scoring(score=0.0, metadata={}, continue_task=False)
        )
        result = convert_scoring_event_to_markdown(event)
        assert "❌" in result
        assert "✔️" not in result

    def test_score_value_is_displayed(self) -> None:
        event = ScoringEvent(
            scoring=Scoring(score=0.75, metadata={}, continue_task=True)
        )
        result = convert_scoring_event_to_markdown(event)
        assert "0.75" in result

    def test_continue_task_yes(self) -> None:
        event = ScoringEvent(
            scoring=Scoring(score=1.0, metadata={}, continue_task=True)
        )
        result = convert_scoring_event_to_markdown(event)
        assert "Continue Task: Yes" in result

    def test_continue_task_no(self) -> None:
        event = ScoringEvent(
            scoring=Scoring(score=0.0, metadata={}, continue_task=False)
        )
        result = convert_scoring_event_to_markdown(event)
        assert "Continue Task: No" in result


class TestEmptyMetadata:
    def test_empty_dict(self) -> None:
        event = ScoringEvent(
            scoring=Scoring(score=1.0, metadata={}, continue_task=True)
        )
        result = convert_scoring_event_to_markdown(event)
        assert "Score: 1.0" in result


class TestDictMetadata:
    def test_single_key(self) -> None:
        event = ScoringEvent(
            scoring=Scoring(
                score=0.0,
                metadata={"error": "Expected 42, got 43"},
                continue_task=False,
            )
        )
        result = convert_scoring_event_to_markdown(event)
        assert "**error**" in result
        assert "Expected 42, got 43" in result

    def test_multiple_keys(self) -> None:
        event = ScoringEvent(
            scoring=Scoring(
                score=0.0,
                metadata={"a": "error in a", "b": "error in b"},
                continue_task=False,
            )
        )
        result = convert_scoring_event_to_markdown(event)
        assert "**a**" in result
        assert "error in a" in result
        assert "**b**" in result
        assert "error in b" in result

    def test_nested_dict_value(self) -> None:
        event = ScoringEvent(
            scoring=Scoring(
                score=1.0,
                metadata={"result": {"nested": "value", "count": 42}},
                continue_task=True,
            )
        )
        result = convert_scoring_event_to_markdown(event)
        assert "**result**" in result
        assert "nested" in result

    def test_list_value(self) -> None:
        event = ScoringEvent(
            scoring=Scoring(
                score=1.0,
                metadata={"items": [1, 2, 3]},
                continue_task=True,
            )
        )
        result = convert_scoring_event_to_markdown(event)
        assert "**items**" in result
        # List is pretty-printed with newlines
        assert "1," in result
        assert "2," in result
        assert "3" in result


class TestCompositeJudgeMetadata:
    def test_and_judge_all_pass(self) -> None:
        event = ScoringEvent(
            scoring=Scoring(
                score=1.0,
                metadata={
                    "type": "AndJudge",
                    "judges": [
                        {
                            "type": "RegexJudge",
                            "evaluated": True,
                            "score": 1.0,
                            "metadata": {},
                            "continue_task": True,
                        },
                        {
                            "type": "RegexJudge",
                            "evaluated": True,
                            "score": 1.0,
                            "metadata": {},
                            "continue_task": True,
                        },
                    ],
                },
                continue_task=True,
            )
        )
        result = convert_scoring_event_to_markdown(event)
        assert "**type**" in result
        assert "AndJudge" in result
        assert "**judges**" in result

    def test_and_judge_short_circuit(self) -> None:
        event = ScoringEvent(
            scoring=Scoring(
                score=0.0,
                metadata={
                    "type": "AndJudge",
                    "judges": [
                        {
                            "type": "RegexJudge",
                            "evaluated": True,
                            "score": 0.0,
                            "metadata": {"x": "Expected 1, got 2"},
                            "continue_task": False,
                        },
                        {"type": "RegexJudge", "evaluated": False},
                    ],
                },
                continue_task=False,
            )
        )
        result = convert_scoring_event_to_markdown(event)
        assert "❌" in result
        assert "AndJudge" in result
        assert "evaluated" in result

    def test_or_judge(self) -> None:
        event = ScoringEvent(
            scoring=Scoring(
                score=1.0,
                metadata={
                    "type": "OrJudge",
                    "judges": [
                        {
                            "type": "RegexJudge",
                            "evaluated": True,
                            "score": 1.0,
                            "metadata": {},
                            "continue_task": True,
                        },
                        {"type": "RegexJudge", "evaluated": False},
                    ],
                },
                continue_task=True,
            )
        )
        result = convert_scoring_event_to_markdown(event)
        assert "✔️" in result
        assert "OrJudge" in result


class TestNonDictMetadata:
    def test_string_metadata(self) -> None:
        event = ScoringEvent(
            scoring=Scoring(
                score=0.5,
                metadata="Just a string",  # pyright: ignore[reportArgumentType]
                continue_task=True,
            )
        )
        result = convert_scoring_event_to_markdown(event)
        assert "**Metadata:**" in result
        assert "Just a string" in result
        assert "```" in result

    def test_list_metadata(self) -> None:
        event = ScoringEvent(
            scoring=Scoring(
                score=0.5,
                metadata=["item1", "item2"],  # pyright: ignore[reportArgumentType]
                continue_task=True,
            )
        )
        result = convert_scoring_event_to_markdown(event)
        assert "**Metadata:**" in result
        assert "item1" in result
        assert "item2" in result

    def test_int_metadata(self) -> None:
        event = ScoringEvent(
            scoring=Scoring(
                score=1.0,
                metadata=42,  # pyright: ignore[reportArgumentType]
                continue_task=True,
            )
        )
        result = convert_scoring_event_to_markdown(event)
        assert "**Metadata:**" in result
        assert "42" in result


class TestResourceMetrics:
    def test_resource_metrics_displayed_when_present(self) -> None:
        metrics = ResourceMetrics(
            samples=[ResourceSample(timestamp_ms=0, cpu_percent=10.5, memory_mb=400.0)],
            peak_cpu_percent=15.0,
            avg_cpu_percent=10.5,
            peak_memory_mb=420.0,
            avg_memory_mb=400.0,
        )
        event = ScoringEvent(
            scoring=Scoring(score=1.0, metadata={}, continue_task=True),
            resource_metrics=metrics,
        )
        result = convert_scoring_event_to_markdown(event)
        assert "Container Resources" in result
        assert "10.5%" in result
        assert "15.0%" in result
        assert "400 MB" in result
        assert "420 MB" in result

    def test_resource_metrics_not_displayed_when_none(self) -> None:
        event = ScoringEvent(
            scoring=Scoring(score=1.0, metadata={}, continue_task=True),
        )
        result = convert_scoring_event_to_markdown(event)
        assert "Container Resources" not in result

    def test_resource_metrics_not_displayed_when_no_samples(self) -> None:
        metrics = ResourceMetrics(samples=[])
        event = ScoringEvent(
            scoring=Scoring(score=1.0, metadata={}, continue_task=True),
            resource_metrics=metrics,
        )
        result = convert_scoring_event_to_markdown(event)
        assert "Container Resources" not in result
