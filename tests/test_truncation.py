import pytest

from karotte.evaluation_runner import EvaluationRunner
from karotte.schemas.evaluation_run_config import EvaluationRunConfig
from karotte.schemas.scoring import Scoring
from karotte.schemas.transcript import ScoringEvent, TaskPreHookCompletedEvent
from karotte.truncation import (
    MAX_METADATA_TOTAL_CHARS,
    MAX_METADATA_VALUE_CHARS,
    head_within_json_bytes,
    json_encoded_len,
    sanitize_metadata,
    truncate_middle,
)
from tests.conftest import TestTask


def test_truncate_middle_short_text_unchanged():
    assert truncate_middle("hello", 100) == "hello"


def test_truncate_middle_exact_length_unchanged():
    text = "x" * 100
    assert truncate_middle(text, 100) == text


def test_truncate_middle_keeps_head_and_tail():
    text = "HEAD" + "x" * 1_000_000 + "TAIL"
    result = truncate_middle(text, 10_000)
    assert result.startswith("HEAD")
    assert result.endswith("TAIL")
    assert "truncated" in result
    assert len(result) < 20_000


def test_truncate_middle_tiny_cap_does_not_crash():
    result = truncate_middle("x" * 100, 0)
    assert "truncated" in result


def test_json_encoded_len_counts_escapes_and_quotes():
    assert json_encoded_len("ab") == 4
    assert json_encoded_len("\x00") == 8
    assert json_encoded_len("\u00e9") == len('"\u00e9"'.encode())


@pytest.mark.parametrize(
    "text", ["x" * 1000, "\x00" * 1000, 'a"b\\c\n' * 300, "\U0001f600" * 500]
)
@pytest.mark.parametrize("budget", [2, 100, 999, 5000, 100_000])
def test_head_within_json_bytes_fits_and_is_a_prefix(text: str, budget: int):
    head = head_within_json_bytes(text, budget)
    assert text.startswith(head)
    assert json_encoded_len(head) <= budget
    if json_encoded_len(text) <= budget:
        assert head == text
    else:
        assert json_encoded_len(head) > budget // 2


def test_sanitize_metadata_leaves_small_values_untouched():
    metadata = {"score_detail": "all good", "attempts": 3, "ratio": 0.5}
    sanitize_metadata(metadata)
    assert metadata == {"score_detail": "all good", "attempts": 3, "ratio": 0.5}


def test_sanitize_metadata_truncates_oversized_string():
    metadata = {"stdout": "HEAD" + "x" * (2 * MAX_METADATA_VALUE_CHARS) + "TAIL"}
    sanitize_metadata(metadata)
    assert len(metadata["stdout"]) < MAX_METADATA_VALUE_CHARS + 10_000
    assert metadata["stdout"].startswith("HEAD")
    assert metadata["stdout"].endswith("TAIL")
    assert "truncated" in metadata["stdout"]


def test_sanitize_metadata_truncates_oversized_non_string():
    metadata = {"samples": list(range(MAX_METADATA_VALUE_CHARS))}
    sanitize_metadata(metadata)
    assert isinstance(metadata["samples"], str)
    assert len(metadata["samples"]) < MAX_METADATA_VALUE_CHARS + 10_000


def test_sanitize_metadata_enforces_total_cap():
    n_values = MAX_METADATA_TOTAL_CHARS // MAX_METADATA_VALUE_CHARS + 2
    metadata = {
        f"key{i}": "x" * (MAX_METADATA_VALUE_CHARS + 1) for i in range(n_values)
    }
    sanitize_metadata(metadata)
    total = sum(len(v) for v in metadata.values())
    assert total < MAX_METADATA_TOTAL_CHARS + n_values * 10_000
    assert "truncated" in metadata[f"key{n_values - 1}"]


@pytest.fixture
def runner(sample_config: EvaluationRunConfig) -> EvaluationRunner:
    return EvaluationRunner(sample_config, TestTask(sample_config))


def test_process_event_sanitizes_scoring_metadata(runner: EvaluationRunner):
    scoring = Scoring(
        score=1.0,
        metadata={"stdout": "x" * (2 * MAX_METADATA_VALUE_CHARS)},
        continue_task=True,
    )
    event = runner._process_event(ScoringEvent(scoring=scoring))  # pyright: ignore[reportPrivateUsage]
    assert isinstance(event, ScoringEvent)
    assert len(event.scoring.metadata["stdout"]) < MAX_METADATA_VALUE_CHARS + 10_000


def test_process_event_sanitizes_pre_hook_metadata(runner: EvaluationRunner):
    event = runner._process_event(  # pyright: ignore[reportPrivateUsage]
        TaskPreHookCompletedEvent(
            metadata={"setup_log": "x" * (2 * MAX_METADATA_VALUE_CHARS)}
        )
    )
    assert isinstance(event, TaskPreHookCompletedEvent)
    assert len(event.metadata["setup_log"]) < MAX_METADATA_VALUE_CHARS + 10_000
