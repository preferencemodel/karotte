import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, override
from unittest.mock import MagicMock, patch

import litellm
import litellm.exceptions
import pytest

from karotte.judges.rubric_context import (
    AnswersContext,
    FileContext,
    TranscriptContext,
)
from karotte.judges.rubric_judge import RubricJudge, RubricJudgeError
from karotte.schemas.chat import ChatCompletionMessageToolCall, Function, Message
from karotte.schemas.transcript import (
    AnswersSubmittedEvent,
    MessageAddedEvent,
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
    Transcript,
)

try:
    from mcp.types import CallToolResult, TextContent
except ImportError:
    pytest.skip("mcp not available", allow_module_level=True)


def _response(text: str | None) -> litellm.ModelResponse:
    return litellm.ModelResponse(
        choices=[{"message": {"role": "assistant", "content": text}}]
    )


@pytest.fixture(autouse=True)
def reset_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(RubricJudge, "default_api_key", None)
    monkeypatch.setattr(RubricJudge, "default_model", None)


@pytest.fixture
def mock_completion() -> Iterator[MagicMock]:
    with patch("litellm.completion") as completion:
        completion.return_value = _response("YES\nCriterion is met.")
        yield completion


def test_without_a_model_evaluate_fails(
    transcript: Transcript, mock_completion: MagicMock
) -> None:
    judge = RubricJudge(rubric=[{"criterion": "c", "weight": 1.0}])
    transcript.events.append(AnswersSubmittedEvent(answers={"response": "Hello"}))

    with pytest.raises(ValueError, match="rubric_judge_model"):
        judge.evaluate(transcript)
    mock_completion.assert_not_called()


def test_default_model_can_be_set_on_the_class() -> None:
    RubricJudge.default_model = "claude-opus-5"

    assert RubricJudge(rubric=[]).model == "claude-opus-5"
    assert RubricJudge(rubric=[], model="claude-sonnet-5").model == "claude-sonnet-5"


def test_api_key_precedence() -> None:
    RubricJudge.default_model = "claude-fable-5"
    RubricJudge.default_api_key = "default-key"

    assert RubricJudge(rubric=[], api_key="explicit").api_key == "explicit"
    assert RubricJudge(rubric=[]).api_key == "default-key"


def test_the_default_key_only_goes_to_the_default_models_provider() -> None:
    RubricJudge.default_model = "claude-fable-5"
    RubricJudge.default_api_key = "sk-ant"

    assert RubricJudge(rubric=[], model="claude-sonnet-5").api_key == "sk-ant"
    assert RubricJudge(rubric=[], model="openai/gpt-5.5").api_key is None


def test_the_default_key_follows_the_default_model() -> None:
    RubricJudge.default_model = "openai/gpt-5.5"
    RubricJudge.default_api_key = "sk-openai"

    assert RubricJudge(rubric=[]).api_key == "sk-openai"
    assert RubricJudge(rubric=[], model="claude-sonnet-5").api_key is None


def test_without_a_default_model_the_default_key_is_unused() -> None:
    RubricJudge.default_api_key = "sk-ant"

    assert RubricJudge(rubric=[], model="claude-sonnet-5").api_key is None


def test_no_api_key_leaves_it_to_litellm(
    transcript: Transcript, mock_completion: MagicMock
) -> None:
    judge = RubricJudge(
        rubric=[{"criterion": "c", "weight": 1.0}], model="claude-sonnet-5"
    )

    transcript.events.append(AnswersSubmittedEvent(answers={"response": "Hello"}))
    judge.evaluate(transcript)

    assert mock_completion.call_args.kwargs["api_key"] is None


def test_bare_claude_id_is_routed_to_anthropic(
    transcript: Transcript, mock_completion: MagicMock
) -> None:
    judge = RubricJudge(
        rubric=[{"criterion": "c", "weight": 1.0}], model="claude-sonnet-5"
    )

    transcript.events.append(AnswersSubmittedEvent(answers={"response": "Hello"}))
    judge.evaluate(transcript)

    assert mock_completion.call_args.kwargs["model"] == "anthropic/claude-sonnet-5"


@pytest.mark.parametrize(
    ("model", "api_base"),
    [
        ("openai/gpt-5.5", "https://proxy.example/v1"),
        ("claude-sonnet-5", None),
        ("vertex_ai/gemini-3-pro-preview", None),
    ],
)
def test_judge_is_routed_like_the_student(
    transcript: Transcript,
    mock_completion: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    api_base: str | None,
) -> None:
    monkeypatch.setenv("KAROTTE_PROXY_URL", "https://proxy.example/")
    judge = RubricJudge(rubric=[{"criterion": "c", "weight": 1.0}], model=model)

    transcript.events.append(AnswersSubmittedEvent(answers={"response": "Hello"}))
    judge.evaluate(transcript)

    assert mock_completion.call_args.kwargs.get("api_base") == api_base


def test_judge_has_no_api_base_without_a_proxy(
    transcript: Transcript, mock_completion: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("KAROTTE_PROXY_URL", raising=False)
    judge = RubricJudge(
        rubric=[{"criterion": "c", "weight": 1.0}], model="openai/gpt-5.5"
    )

    transcript.events.append(AnswersSubmittedEvent(answers={"response": "Hello"}))
    judge.evaluate(transcript)

    assert "api_base" not in mock_completion.call_args.kwargs


class _FakeAnthropic(BaseHTTPRequestHandler):
    requests: list[tuple[str, dict[str, str], dict[str, Any]]] = []

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _FakeAnthropic.requests.append((self.path, dict(self.headers), body))
        payload = json.dumps(
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": body["model"],
                "content": [{"type": "text", "text": "YES\nlooks right"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        _ = self.wfile.write(payload)

    @override
    def log_message(self, format: str, *args: Any) -> None:
        pass


def test_anthropic_base_url_and_key_come_from_the_environment(
    transcript: Transcript, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without an explicit key, the judge reaches whatever Anthropic endpoint the
    environment names, with the environment's key."""
    server = HTTPServer(("127.0.0.1", 0), _FakeAnthropic)
    threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    ).start()
    _FakeAnthropic.requests = []
    monkeypatch.setenv("ANTHROPIC_BASE_URL", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
    monkeypatch.delenv("ANTHROPIC_API_BASE", raising=False)
    try:
        judge = RubricJudge(
            rubric=[{"criterion": "c", "weight": 1.0}], model="claude-sonnet-5"
        )
        transcript.events.append(AnswersSubmittedEvent(answers={"response": "Hi"}))
        scoring = judge.evaluate(transcript)
    finally:
        server.shutdown()

    assert scoring.score == 1.0
    [(path, headers, body)] = _FakeAnthropic.requests
    assert path == "/v1/messages"
    assert headers["x-api-key"] == "env-key"
    assert body["model"] == "claude-sonnet-5"


def test_temperature_omitted_for_fable(
    transcript: Transcript, mock_completion: MagicMock
) -> None:
    """Fable models reject sampling parameters, so temperature must not be sent."""
    judge = RubricJudge(
        rubric=[{"criterion": "Is the answer helpful?", "weight": 1.0}],
        model="claude-fable-5",
        api_key="test-key",
        temperature=0.7,
    )

    transcript.events.append(AnswersSubmittedEvent(answers={"response": "Hello world"}))
    judge.evaluate(transcript)

    mock_completion.assert_called_once()
    assert "temperature" not in mock_completion.call_args.kwargs


def test_temperature_omitted_for_opus_4_8(
    transcript: Transcript, mock_completion: MagicMock
) -> None:
    """Opus 4.7+ also rejects sampling parameters."""
    judge = RubricJudge(
        rubric=[{"criterion": "Is the answer helpful?", "weight": 1.0}],
        model="claude-opus-4-8",
        api_key="test-key",
    )

    transcript.events.append(AnswersSubmittedEvent(answers={"response": "Hello world"}))
    judge.evaluate(transcript)

    mock_completion.assert_called_once()
    assert "temperature" not in mock_completion.call_args.kwargs


def test_temperature_omitted_for_opus_5(
    transcript: Transcript, mock_completion: MagicMock
) -> None:
    """Opus 5 also rejects sampling parameters."""
    judge = RubricJudge(
        rubric=[{"criterion": "Is the answer helpful?", "weight": 1.0}],
        model="claude-opus-5",
        api_key="test-key",
    )

    transcript.events.append(AnswersSubmittedEvent(answers={"response": "Hello world"}))
    judge.evaluate(transcript)

    mock_completion.assert_called_once()
    assert "temperature" not in mock_completion.call_args.kwargs


def test_temperature_omitted_for_sonnet_5(
    transcript: Transcript, mock_completion: MagicMock
) -> None:
    """Sonnet 5 also rejects sampling parameters."""
    judge = RubricJudge(
        rubric=[{"criterion": "Is the answer helpful?", "weight": 1.0}],
        model="claude-sonnet-5",
        api_key="test-key",
    )

    transcript.events.append(AnswersSubmittedEvent(answers={"response": "Hello world"}))
    judge.evaluate(transcript)

    mock_completion.assert_called_once()
    assert "temperature" not in mock_completion.call_args.kwargs


def test_default_temperature() -> None:
    """Test that the default temperature is 0.3."""
    judge = RubricJudge(
        rubric=[{"criterion": "test", "weight": 1.0}],
        model="claude-sonnet-4-20250514",
        api_key="test-key",
    )

    assert judge.temperature == 0.3


def test_custom_temperature_can_be_set() -> None:
    """Test that a custom temperature can be set."""
    judge = RubricJudge(
        rubric=[{"criterion": "test", "weight": 1.0}],
        model="claude-sonnet-4-20250514",
        api_key="test-key",
        temperature=0.5,
    )

    assert judge.temperature == 0.5


def test_temperature_zero() -> None:
    """Test that temperature can be set to 0."""
    judge = RubricJudge(
        rubric=[{"criterion": "test", "weight": 1.0}],
        model="claude-sonnet-4-20250514",
        api_key="test-key",
        temperature=0.0,
    )

    assert judge.temperature == 0.0


def test_temperature_passed_to_api_call(
    transcript: Transcript, mock_completion: MagicMock
) -> None:
    """Test that temperature is passed to the API call."""
    judge = RubricJudge(
        rubric=[{"criterion": "Is the answer helpful?", "weight": 1.0}],
        model="claude-sonnet-4-20250514",
        api_key="test-key",
        temperature=0.7,
    )

    transcript.events.append(AnswersSubmittedEvent(answers={"response": "Hello world"}))
    judge.evaluate(transcript)

    # Verify the API was called with the correct temperature
    mock_completion.assert_called_once()
    call_kwargs = mock_completion.call_args.kwargs
    assert call_kwargs["temperature"] == 0.7


def test_default_temperature_passed_to_api_call(
    transcript: Transcript, mock_completion: MagicMock
) -> None:
    """Test that default temperature (0.3) is passed to the API call."""
    judge = RubricJudge(
        rubric=[{"criterion": "Is the answer helpful?", "weight": 1.0}],
        model="claude-sonnet-4-20250514",
        api_key="test-key",
    )

    transcript.events.append(AnswersSubmittedEvent(answers={"response": "Hello world"}))
    judge.evaluate(transcript)

    # Verify the API was called with the default temperature
    mock_completion.assert_called_once()
    call_kwargs = mock_completion.call_args.kwargs
    assert call_kwargs["temperature"] == 0.3


def test_evaluate_criterion_met(
    transcript: Transcript,
    mock_completion: MagicMock,  # pyright: ignore[reportUnusedParameter]
) -> None:
    """Test that a met criterion contributes to the score."""
    judge = RubricJudge(
        rubric=[{"criterion": "Is the answer helpful?", "weight": 0.5}],
        model="claude-sonnet-4-20250514",
        api_key="test-key",
    )

    transcript.events.append(AnswersSubmittedEvent(answers={"response": "Hello world"}))
    scoring = judge.evaluate(transcript)

    assert scoring.score == 0.5
    assert scoring.continue_task is True


def test_evaluate_criterion_not_met(
    transcript: Transcript, mock_completion: MagicMock
) -> None:
    """Test that an unmet criterion does not contribute to the score."""
    # Change mock response to NO
    mock_completion.return_value = _response("NO\nCriterion is not met.")

    judge = RubricJudge(
        rubric=[{"criterion": "Is the answer helpful?", "weight": 0.5}],
        model="claude-sonnet-4-20250514",
        api_key="test-key",
    )

    transcript.events.append(AnswersSubmittedEvent(answers={"response": "Hello world"}))
    scoring = judge.evaluate(transcript)

    assert scoring.score == 0.0
    assert scoring.continue_task is False


def test_evaluate_criterion_with_reasoning(
    transcript: Transcript, mock_completion: MagicMock
) -> None:
    """Thinking output rides next to the text and must not be parsed as it."""
    response = _response("YES\nCriterion is met.")
    response.choices[0].message.reasoning_content = "NO, let me consider this..."
    mock_completion.return_value = response

    judge = RubricJudge(
        rubric=[{"criterion": "Is the answer helpful?", "weight": 0.5}],
        model="claude-fable-5",
        api_key="test-key",
    )

    transcript.events.append(AnswersSubmittedEvent(answers={"response": "Hello world"}))
    scoring = judge.evaluate(transcript)

    assert scoring.score == 0.5
    assert scoring.continue_task is True


def _judge(model: str = "claude-sonnet-5", **kwargs: Any) -> RubricJudge:
    return RubricJudge(
        rubric=[{"criterion": "c", "weight": 1.0}], model=model, **kwargs
    )


def _answered(transcript: Transcript) -> Transcript:
    transcript.events.append(AnswersSubmittedEvent(answers={"response": "Hello"}))
    return transcript


def test_an_empty_reply_fails_the_evaluation(
    transcript: Transcript, mock_completion: MagicMock
) -> None:
    mock_completion.return_value = _response(None)

    with pytest.raises(RubricJudgeError, match="no text"):
        _judge().evaluate(_answered(transcript))


def test_a_failed_call_fails_the_evaluation(
    transcript: Transcript, mock_completion: MagicMock
) -> None:
    mock_completion.side_effect = litellm.exceptions.BadRequestError(
        "bad", model="claude-sonnet-5", llm_provider="anthropic"
    )

    with pytest.raises(litellm.exceptions.BadRequestError):
        _judge().evaluate(_answered(transcript))
    assert mock_completion.call_count == 1


def test_a_temporary_error_is_retried(
    transcript: Transcript, mock_completion: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("karotte.judges.rubric_judge.time.sleep", sleeps.append)
    mock_completion.side_effect = [
        litellm.exceptions.RateLimitError(
            "slow down", llm_provider="anthropic", model="claude-sonnet-5"
        ),
        _response("YES\nfine"),
    ]

    scoring = _judge().evaluate(_answered(transcript))

    assert scoring.score == 1.0
    assert len(sleeps) == 1


def test_the_judge_has_room_to_reason(
    transcript: Transcript, mock_completion: MagicMock
) -> None:
    _judge().evaluate(_answered(transcript))

    assert mock_completion.call_args.kwargs["max_tokens"] == 32_000


class TestProxyKey:
    @pytest.fixture(autouse=True)
    def _proxy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KAROTTE_PROXY_URL", "https://proxy.example/")
        for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "TOGETHERAI_API_KEY"):
            monkeypatch.delenv(var, raising=False)

    @pytest.mark.parametrize(
        "model", ["claude-sonnet-5", "openai/gpt-5.5", "together_ai/zai-org/GLM-5.2"]
    )
    def test_a_missing_key_gets_a_placeholder(
        self, transcript: Transcript, mock_completion: MagicMock, model: str
    ) -> None:
        _judge(model).evaluate(_answered(transcript))

        assert mock_completion.call_args.kwargs["api_key"] == "model_api_key"

    def test_a_key_in_the_environment_is_left_to_litellm(
        self,
        transcript: Transcript,
        mock_completion: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("TOGETHERAI_API_KEY", "sk-together")

        _judge("together_ai/zai-org/GLM-5.2").evaluate(_answered(transcript))

        assert mock_completion.call_args.kwargs["api_key"] is None

    def test_an_explicit_key_wins(
        self, transcript: Transcript, mock_completion: MagicMock
    ) -> None:
        _judge(api_key="sk-mine").evaluate(_answered(transcript))

        assert mock_completion.call_args.kwargs["api_key"] == "sk-mine"

    def test_a_keyless_model_gets_none(
        self, transcript: Transcript, mock_completion: MagicMock
    ) -> None:
        _judge("vertex_ai/gemini-3-pro-preview").evaluate(_answered(transcript))

        assert mock_completion.call_args.kwargs["api_key"] is None


def test_without_a_proxy_a_missing_key_is_left_to_litellm(
    transcript: Transcript, mock_completion: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("KAROTTE_PROXY_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    _judge().evaluate(_answered(transcript))

    assert mock_completion.call_args.kwargs["api_key"] is None


def test_evaluate_multiple_criteria(
    transcript: Transcript, mock_completion: MagicMock
) -> None:
    """Test evaluation of multiple criteria."""
    # First call returns YES, second returns NO
    mock_completion.side_effect = [
        _response("YES\nFirst criterion met."),
        _response("NO\nSecond criterion not met."),
    ]

    judge = RubricJudge(
        rubric=[
            {"criterion": "First criterion", "weight": 0.6},
            {"criterion": "Second criterion", "weight": 0.4},
        ],
        model="claude-sonnet-4-20250514",
        api_key="test-key",
    )

    transcript.events.append(AnswersSubmittedEvent(answers={"response": "Hello world"}))
    scoring = judge.evaluate(transcript)

    assert scoring.score == 0.6
    assert mock_completion.call_count == 2


def test_no_answers_in_transcript(transcript: Transcript) -> None:
    """Test behavior when no answers are in transcript."""
    judge = RubricJudge(
        rubric=[{"criterion": "test", "weight": 1.0}],
        model="claude-sonnet-4-20250514",
        api_key="test-key",
    )

    scoring = judge.evaluate(transcript)

    assert scoring.score == 0.0
    assert scoring.continue_task is False
    assert "error" in scoring.metadata


def test_answer_key_not_found(
    transcript: Transcript,
    mock_completion: MagicMock,  # pyright: ignore[reportUnusedParameter]
) -> None:
    """Test behavior when specified answer key is not found."""
    judge = RubricJudge(
        rubric=[{"criterion": "test", "weight": 1.0}],
        model="claude-sonnet-4-20250514",
        api_key="test-key",
        context=[AnswersContext(key="missing_key")],
    )

    transcript.events.append(
        AnswersSubmittedEvent(answers={"other_key": "Hello world"})
    )
    scoring = judge.evaluate(transcript)

    assert scoring.score == 0.0
    assert scoring.continue_task is False
    assert "error" in scoring.metadata


def test_continue_threshold(
    transcript: Transcript,
    mock_completion: MagicMock,  # pyright: ignore[reportUnusedParameter]
) -> None:
    """Test that continue_task respects the continue_threshold."""
    judge = RubricJudge(
        rubric=[{"criterion": "test", "weight": 0.5}],
        model="claude-sonnet-4-20250514",
        api_key="test-key",
        continue_threshold=0.6,
    )

    transcript.events.append(AnswersSubmittedEvent(answers={"response": "Hello world"}))
    scoring = judge.evaluate(transcript)

    # Score is 0.5, threshold is 0.6, so should not continue
    assert scoring.score == 0.5
    assert scoring.continue_task is False


# --- Context parameter tests ---


def test_answers_context(
    transcript: Transcript,
    mock_completion: MagicMock,
) -> None:
    judge = RubricJudge(
        rubric=[{"criterion": "Is the answer correct?", "weight": 1.0}],
        model="claude-sonnet-4-20250514",
        api_key="test-key",
        context=[AnswersContext()],
    )
    transcript.events.append(AnswersSubmittedEvent(answers={"response": "42"}))
    scoring = judge.evaluate(transcript)

    assert scoring.score == 1.0
    call_kwargs = mock_completion.call_args.kwargs
    prompt = call_kwargs["messages"][0]["content"]
    assert "42" in prompt


def test_file_context(
    transcript: Transcript,
    mock_completion: MagicMock,
    tmp_path: Path,
) -> None:
    f = tmp_path / "solution.py"
    f.write_text("print('hello world')")

    judge = RubricJudge(
        rubric=[{"criterion": "Does it print hello?", "weight": 1.0}],
        model="claude-sonnet-4-20250514",
        api_key="test-key",
        context=[FileContext(paths=[str(f)])],
    )
    scoring = judge.evaluate(transcript)

    assert scoring.score == 1.0
    call_kwargs = mock_completion.call_args.kwargs
    prompt = call_kwargs["messages"][0]["content"]
    assert "print('hello world')" in prompt


def test_transcript_context(
    transcript: Transcript,
    mock_completion: MagicMock,
) -> None:
    transcript.events.append(
        MessageAddedEvent(
            message=Message(content="Running training script", role="assistant")
        )
    )
    judge = RubricJudge(
        rubric=[{"criterion": "Did the student run training?", "weight": 1.0}],
        model="claude-sonnet-4-20250514",
        api_key="test-key",
        context=[TranscriptContext()],
    )
    scoring = judge.evaluate(transcript)

    assert scoring.score == 1.0
    call_kwargs = mock_completion.call_args.kwargs
    prompt = call_kwargs["messages"][0]["content"]
    assert "Running training script" in prompt


def test_transcript_context_filtered_by_tool(
    transcript: Transcript,
    mock_completion: MagicMock,
) -> None:
    # Add a bash tool call
    transcript.events.append(
        ToolCallStartedEvent(
            tool_call=ChatCompletionMessageToolCall(
                id="tc1",
                function=Function(
                    name="bash", arguments='{"command": "python train.py"}'
                ),
                type="function",
            )
        )
    )
    transcript.events.append(
        ToolCallCompletedEvent(
            tool_call_id="tc1",
            result=CallToolResult(
                content=[TextContent(type="text", text="Training complete")],
                isError=False,
            ),
        )
    )
    # Add a non-bash tool call
    transcript.events.append(
        ToolCallStartedEvent(
            tool_call=ChatCompletionMessageToolCall(
                id="tc2",
                function=Function(name="submit_answers", arguments='{"answers": {}}'),
                type="function",
            )
        )
    )
    transcript.events.append(
        ToolCallCompletedEvent(
            tool_call_id="tc2",
            result=CallToolResult(
                content=[TextContent(type="text", text="Submitted")],
                isError=False,
            ),
        )
    )

    judge = RubricJudge(
        rubric=[{"criterion": "Did training run?", "weight": 1.0}],
        model="claude-sonnet-4-20250514",
        api_key="test-key",
        context=[TranscriptContext(tool="bash")],
    )
    judge.evaluate(transcript)

    call_kwargs = mock_completion.call_args.kwargs
    prompt = call_kwargs["messages"][0]["content"]
    assert "python train.py" in prompt
    assert "submit_answers" not in prompt


def test_multiple_contexts_combined(
    transcript: Transcript,
    mock_completion: MagicMock,
    tmp_path: Path,
) -> None:
    """Multiple context sources are concatenated."""
    f = tmp_path / "code.py"
    f.write_text("import jax")

    transcript.events.append(AnswersSubmittedEvent(answers={"summary": "I used JAX"}))

    judge = RubricJudge(
        rubric=[{"criterion": "Does code use JAX?", "weight": 1.0}],
        model="claude-sonnet-4-20250514",
        api_key="test-key",
        context=[
            FileContext(paths=[str(f)]),
            AnswersContext(),
        ],
    )
    scoring = judge.evaluate(transcript)

    assert scoring.score == 1.0
    call_kwargs = mock_completion.call_args.kwargs
    prompt = call_kwargs["messages"][0]["content"]
    assert "import jax" in prompt
    assert "I used JAX" in prompt


def test_context_included_in_metadata(
    transcript: Transcript,
    mock_completion: MagicMock,  # pyright: ignore[reportUnusedParameter]
) -> None:
    """The assembled context text is stored in scoring metadata."""
    judge = RubricJudge(
        rubric=[{"criterion": "Is it correct?", "weight": 1.0}],
        model="claude-sonnet-4-20250514",
        api_key="test-key",
        context=[AnswersContext()],
    )
    transcript.events.append(AnswersSubmittedEvent(answers={"response": "42"}))
    scoring = judge.evaluate(transcript)

    assert "context" in scoring.metadata
    assert "42" in scoring.metadata["context"]


def test_empty_context_returns_error(
    transcript: Transcript,
) -> None:
    """If all context sources produce empty strings, return an error."""
    judge = RubricJudge(
        rubric=[{"criterion": "test", "weight": 1.0}],
        model="claude-sonnet-4-20250514",
        api_key="test-key",
        context=[AnswersContext()],
    )
    # No answers submitted
    scoring = judge.evaluate(transcript)

    assert scoring.score == 0.0
    assert "error" in scoring.metadata


def test_render_context_joins_providers(transcript: Transcript) -> None:
    """render_context is what evaluate judges against, usable without a model."""
    judge = RubricJudge(
        rubric=[{"criterion": "c", "weight": 1.0}],
        context=[AnswersContext(), AnswersContext()],
    )
    transcript.events.append(AnswersSubmittedEvent(answers={"response": "Hello"}))

    rendered = judge.render_context(transcript)

    single = AnswersContext().render(transcript)
    assert rendered == f"{single}\n\n{single}"


def test_criterion_prompt_is_what_the_model_receives(
    transcript: Transcript, mock_completion: MagicMock
) -> None:
    judge = RubricJudge(
        rubric=[{"criterion": "Is it polite?", "weight": 1.0}], model="claude-sonnet-5"
    )
    transcript.events.append(AnswersSubmittedEvent(answers={"response": "Hello"}))

    judge.evaluate(transcript)

    sent = mock_completion.call_args.kwargs["messages"][0]["content"]
    assert sent == RubricJudge.criterion_prompt(
        judge.render_context(transcript), "Is it polite?"
    )


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("YES\nit is", (True, "it is")),
        ("  yes, mostly\n  because  ", (True, "because")),
        ("NO\nit is not", (False, "it is not")),
        ("YES", (True, "No explanation provided")),
        ("Maybe\nunsure", (False, "unsure")),
    ],
)
def test_parse_reply(reply: str, expected: tuple[bool, str]) -> None:
    assert RubricJudge.parse_reply(reply) == expected
