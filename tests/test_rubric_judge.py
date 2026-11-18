import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, override
from unittest.mock import MagicMock, patch

import httpx
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


@pytest.mark.parametrize(
    ("model", "extra_body"),
    [
        ("openai/gpt-6.1-sol", {"store": False}),
        ("xai/grok-4.7", {"store": False}),
        ("meta/muse-spark-1.3", {"store": False}),
        ("claude-sonnet-5", None),
    ],
)
def test_judge_asks_the_provider_not_to_store_the_conversation(
    transcript: Transcript,
    mock_completion: MagicMock,
    model: str,
    extra_body: dict[str, Any] | None,
) -> None:
    judge = RubricJudge(rubric=[{"criterion": "c", "weight": 1.0}], model=model)

    transcript.events.append(AnswersSubmittedEvent(answers={"response": "Hello"}))
    judge.evaluate(transcript)

    assert mock_completion.call_args.kwargs.get("extra_body") == extra_body


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
                is_error=False,
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
                is_error=False,
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


# A decision model, like Gemini Decision 1, answers typed questions about one
# input with calibrated probabilities. It writes no text, so these tests drive the
# real request and response shapes and check what the judge does with
# probabilities: thresholds, confidence, and one shared input.

_DECISION_MODEL = "gemini/gemini-decision-1"
_DECISIONS_URL = "https://generativelanguage.googleapis.com/v1beta/decisions"

# A graded answer from an agent that was asked why a test fails.
_BUG_REPORT = (
    "The failing test is tests/test_parse.py::test_parse_dates. Root cause: "
    "parse_dates returns timezone-naive datetimes, and the test compares them "
    "with timezone-aware ones. I only ran that one test file."
)
_BUG_RUBRIC: list[Any] = [
    {
        "criterion": "The answer names tests/test_parse.py::test_parse_dates as the failing test.",
        "weight": 2.0,
    },
    {
        "criterion": "The answer attributes the failure to naive datetimes compared with aware ones.",
        "weight": 2.0,
    },
    {
        "criterion": "The answer does not claim that the whole test suite was run.",
        "weight": 1.0,
    },
]


def _decisions_reply(*probabilities: float, tokens: int = 212) -> httpx.Response:
    """What the Decisions API returns: one yes/no answer per question, no text, and
    a usage block that bills input only."""
    return httpx.Response(
        200,
        json={
            "answers": {
                f"q{i}": {
                    "type": "noul",
                    "noul": p,
                    "confidence": max(p, 1 - p),
                }
                for i, p in enumerate(probabilities)
            },
            "usage": {
                "total_input_tokens": tokens,
                "input_tokens_by_modality": [{"modality": "text", "tokens": tokens}],
                "total_output_tokens": 0,
                "total_tokens": tokens,
            },
        },
        request=httpx.Request("POST", _DECISIONS_URL),
    )


def _decision_judge(**kwargs: Any) -> RubricJudge:
    kwargs.setdefault("rubric", _BUG_RUBRIC)
    return RubricJudge(model=_DECISION_MODEL, api_key="test-key", **kwargs)


def _graded_transcript(transcript: Transcript, answer: str = _BUG_REPORT) -> Transcript:
    transcript.events.append(AnswersSubmittedEvent(answers={"response": answer}))
    return transcript


def test_a_whole_rubric_is_graded_in_one_request_that_carries_the_answer_once(
    transcript: Transcript, mock_completion: MagicMock
) -> None:
    with patch("httpx.post", return_value=_decisions_reply(0.97, 0.88, 0.91)) as post:
        _decision_judge().evaluate(_graded_transcript(transcript))

    mock_completion.assert_not_called()
    post.assert_called_once()
    body = post.call_args.kwargs["json"]
    assert json.dumps(body).count(_BUG_REPORT) == 1
    assert list(body["questions"]) == ["q0", "q1", "q2"]
    for question, item in zip(body["questions"].values(), _BUG_RUBRIC, strict=True):
        assert question["type"] == "noul"
        assert item["criterion"] in question["instructions"]


def test_the_request_asks_for_a_decision_and_for_nothing_a_chat_model_takes(
    transcript: Transcript,
) -> None:
    judge = _decision_judge(temperature=0.9)
    with patch("httpx.post", return_value=_decisions_reply(0.9, 0.9, 0.9)) as post:
        judge.evaluate(_graded_transcript(transcript))

    assert post.call_args.args == (_DECISIONS_URL,)
    assert set(post.call_args.kwargs["json"]) == {"model", "input", "questions"}
    assert post.call_args.kwargs["json"]["model"] == "gemini-decision-1"


def test_the_key_is_sent_as_a_header_and_nowhere_else(transcript: Transcript) -> None:
    with patch("httpx.post", return_value=_decisions_reply(0.9, 0.9, 0.9)) as post:
        _decision_judge().evaluate(_graded_transcript(transcript))

    assert post.call_args.kwargs["headers"] == {"x-goog-api-key": "test-key"}
    assert "test-key" not in json.dumps(post.call_args.kwargs["json"])
    assert "test-key" not in post.call_args.args[0]


def test_probabilities_are_cut_at_a_threshold_and_the_weights_of_met_criteria_add_up(
    transcript: Transcript,
) -> None:
    # Clearly yes, leaning yes, clearly no.
    with patch("httpx.post", return_value=_decisions_reply(0.97, 0.62, 0.03)):
        scoring = _decision_judge().evaluate(_graded_transcript(transcript))

    assert scoring.score == 4.0
    assert scoring.continue_task
    keys = [k for k in scoring.metadata if k.startswith("criterion_")]
    assert [scoring.metadata[k][0] for k in keys] == ["✓", "✓", "✗"]
    assert scoring.metadata[keys[0]] == "✓: P(yes)=0.97, confidence 0.97"
    assert scoring.metadata[keys[2]] == "✗: P(yes)=0.03, confidence 0.97"


@pytest.mark.parametrize(
    ("threshold", "score"),
    [
        (0.5, 4.0),  # the default: 0.62 is a yes
        (0.7, 2.0),  # a stricter grader: 0.62 is no longer enough
        (0.97, 2.0),  # exactly at the threshold is met
        (0.99, 0.0),  # nothing short of the ceiling passes
    ],
)
def test_the_threshold_decides_what_a_probability_is_worth(
    transcript: Transcript, threshold: float, score: float
) -> None:
    with patch("httpx.post", return_value=_decisions_reply(0.97, 0.62, 0.03)):
        scoring = _decision_judge(decision_threshold=threshold).evaluate(
            _graded_transcript(transcript)
        )

    assert scoring.score == score


def test_the_top_of_the_models_range_meets_the_loosest_threshold_never_the_bottom(
    transcript: Transcript,
) -> None:
    with patch("httpx.post", return_value=_decisions_reply(0.99, 0.01, 0.01)):
        scoring = _decision_judge(decision_threshold=0.01).evaluate(
            _graded_transcript(transcript)
        )

    # 0.01 is the lowest probability the model gives, so it meets a 0.01 threshold.
    assert scoring.score == 5.0


@pytest.mark.parametrize("threshold", [0.0, 0.005, 0.995, 1.0, -0.5, 2.0])
def test_a_threshold_the_model_cannot_answer_in_is_refused(threshold: float) -> None:
    # Yes/no probabilities stay within 0.01 to 0.99, so a threshold outside that
    # range would match every answer or none.
    with pytest.raises(ValueError, match="decision_threshold"):
        _decision_judge(decision_threshold=threshold)


@pytest.mark.parametrize("threshold", [0.01, 0.5, 0.99])
def test_a_threshold_inside_the_range_is_accepted(threshold: float) -> None:
    assert _decision_judge(decision_threshold=threshold).decision_threshold == threshold


def test_the_models_confidence_flags_a_borderline_grade(
    transcript: Transcript,
) -> None:
    # 0.52 is a pass at 0.5 but the model is barely more sure of yes than of no.
    with patch("httpx.post", return_value=_decisions_reply(0.97, 0.52, 0.03)):
        scoring = _decision_judge().evaluate(_graded_transcript(transcript))

    assert scoring.metadata["decision_min_confidence"] == "0.52"
    borderline = scoring.metadata[f"criterion_1_{_BUG_RUBRIC[1]['criterion'][:30]}"]
    assert borderline == "✓: P(yes)=0.52, confidence 0.52"


def test_a_clear_grade_reports_high_confidence_and_what_was_billed(
    transcript: Transcript,
) -> None:
    with patch(
        "httpx.post", return_value=_decisions_reply(0.97, 0.99, 0.01, tokens=212)
    ):
        scoring = _decision_judge().evaluate(_graded_transcript(transcript))

    assert scoring.metadata["decision_min_confidence"] == "0.97"
    assert scoring.metadata["decision_input_tokens"] == "212"


def test_answers_are_matched_to_criteria_by_question_not_by_position(
    transcript: Transcript,
) -> None:
    reply = _decisions_reply(0.97, 0.62, 0.03)
    answers = reply.json()["answers"]
    shuffled = httpx.Response(
        200,
        json={"answers": dict(reversed(list(answers.items())))},
        request=reply.request,
    )
    with patch("httpx.post", return_value=shuffled):
        scoring = _decision_judge().evaluate(_graded_transcript(transcript))

    keys = [k for k in scoring.metadata if k.startswith("criterion_")]
    assert [scoring.metadata[k][0] for k in keys] == ["✓", "✓", "✗"]


def test_questions_are_independent_so_a_rubric_grades_the_same_in_any_order(
    transcript: Transcript,
) -> None:
    probabilities = {"first": 0.97, "second": 0.62, "third": 0.03}
    rubric = [
        {"criterion": name, "weight": float(w)}
        for w, name in enumerate(probabilities, start=1)
    ]

    def grade(items: list[Any]) -> float:
        reply = _decisions_reply(*(probabilities[i["criterion"]] for i in items))
        with patch("httpx.post", return_value=reply):
            return (
                _decision_judge(rubric=items)
                .evaluate(_graded_transcript(transcript))
                .score
            )

    assert grade(rubric) == grade(list(reversed(rubric))) == 3.0


def test_a_criterion_is_sent_verbatim_whatever_it_contains(
    transcript: Transcript,
) -> None:
    criterion = (
        'The answer quotes "retry-after: 7" and the key path items[0].id\nin {braces}.'
    )
    with patch("httpx.post", return_value=_decisions_reply(0.9)) as post:
        _decision_judge(rubric=[{"criterion": criterion, "weight": 1.0}]).evaluate(
            _graded_transcript(transcript)
        )

    assert criterion in post.call_args.kwargs["json"]["questions"]["q0"]["instructions"]


def test_structured_answers_go_in_as_text_for_the_model_to_read(
    transcript: Transcript,
) -> None:
    answer = json.dumps({"verdict": "flaky", "evidence": [{"run": 3, "passed": False}]})
    with patch("httpx.post", return_value=_decisions_reply(0.9, 0.9, 0.9)) as post:
        _decision_judge().evaluate(_graded_transcript(transcript, answer))

    sent = post.call_args.kwargs["json"]["input"]
    assert isinstance(sent, str)
    assert answer in sent


def test_an_empty_rubric_makes_no_request(transcript: Transcript) -> None:
    with patch("httpx.post") as post:
        scoring = _decision_judge(rubric=[]).evaluate(_graded_transcript(transcript))

    post.assert_not_called()
    assert scoring.score == 0.0
    assert "decision_min_confidence" not in scoring.metadata


def test_nothing_to_grade_makes_no_request(transcript: Transcript) -> None:
    with patch("httpx.post") as post:
        scoring = _decision_judge().evaluate(transcript)

    post.assert_not_called()
    assert scoring.metadata == {"error": "No context available for evaluation"}


def test_a_decision_model_takes_its_key_from_the_gemini_variable(
    transcript: Transcript, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "from-env")
    judge = RubricJudge(rubric=_BUG_RUBRIC, model=_DECISION_MODEL)
    with patch("httpx.post", return_value=_decisions_reply(0.9, 0.9, 0.9)) as post:
        judge.evaluate(_graded_transcript(transcript))

    assert post.call_args.kwargs["headers"] == {"x-goog-api-key": "from-env"}


def test_a_key_for_another_provider_is_not_sent_to_google(
    transcript: Transcript, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    RubricJudge.default_model = "claude-opus-5"
    RubricJudge.default_api_key = "sk-ant-secret"
    judge = RubricJudge(rubric=_BUG_RUBRIC, model=_DECISION_MODEL)

    with (
        patch("httpx.post") as post,
        pytest.raises(RubricJudgeError, match="GEMINI_API_KEY"),
    ):
        judge.evaluate(_graded_transcript(transcript))

    post.assert_not_called()


def test_a_rejected_request_is_a_judge_error_not_a_zero_score(
    transcript: Transcript,
) -> None:
    refused = httpx.Response(
        400,
        json={"error": {"message": "Input can't be empty."}},
        request=httpx.Request("POST", _DECISIONS_URL),
    )
    with (
        patch("httpx.post", return_value=refused),
        pytest.raises(RubricJudgeError, match="gemini-decision-1 gave no verdict"),
    ):
        _decision_judge().evaluate(_graded_transcript(transcript))


def test_a_reply_missing_an_answer_is_a_judge_error_not_a_guess(
    transcript: Transcript,
) -> None:
    with (
        patch("httpx.post", return_value=_decisions_reply(0.97, 0.88)),
        pytest.raises(RubricJudgeError, match="q2"),
    ):
        _decision_judge().evaluate(_graded_transcript(transcript))
