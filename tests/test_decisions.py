from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from karotte.decisions import DecisionsError, YesNoAnswer, ask_yes_no
from karotte.model_spec import api_key_env_var, is_decision_model

_URL = "https://generativelanguage.googleapis.com/v1beta/decisions"


def _response(
    payload: Any = None, status: int = 200, headers: dict[str, str] | None = None
) -> httpx.Response:
    return httpx.Response(
        status,
        json=payload,
        headers=headers,
        request=httpx.Request("POST", _URL),
    )


def _answers(*probabilities: float) -> dict[str, Any]:
    return {
        "answers": {
            f"q{i}": {
                "type": "noul",
                "noul": p,
                "confidence": max(p, 1 - p),
            }
            for i, p in enumerate(probabilities)
        },
        "usage": {
            "total_input_tokens": 123,
            "input_tokens_by_modality": [{"modality": "text", "tokens": 123}],
            "total_output_tokens": 0,
            "total_tokens": 123,
        },
    }


@pytest.fixture(autouse=True)
def no_sleep() -> Iterator[None]:
    with patch("karotte.decisions.time.sleep"):
        yield


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gemini/gemini-decision-1", True),
        ("gemini/gemini-4-argon", False),
        ("vertex_ai/gemini-decision-1", False),
        ("gemini-decision-1", False),
        ("claude-opus-5", False),
    ],
)
def test_is_decision_model(model: str, expected: bool) -> None:
    assert is_decision_model(model) is expected


def test_a_decision_model_reads_the_gemini_key() -> None:
    assert api_key_env_var("gemini/gemini-decision-1") == "GEMINI_API_KEY"


def test_ask_yes_no_sends_one_request_for_every_question() -> None:
    with patch("httpx.post", return_value=_response(_answers(0.93, 0.02))) as post:
        result = ask_yes_no(
            "gemini/gemini-decision-1", "key", "the content", ["first?", "second?"]
        )

    post.assert_called_once()
    assert post.call_args.args == (_URL,)
    assert post.call_args.kwargs["headers"] == {"x-goog-api-key": "key"}
    assert post.call_args.kwargs["json"] == {
        "model": "gemini-decision-1",
        "input": "the content",
        "questions": {
            "q0": {"type": "noul", "instructions": "first?"},
            "q1": {"type": "noul", "instructions": "second?"},
        },
    }
    assert result.answers == (
        YesNoAnswer(probability=0.93, confidence=0.93),
        YesNoAnswer(probability=0.02, confidence=0.98),
    )
    assert result.input_tokens == 123


def test_a_missing_confidence_is_derived_from_the_probability() -> None:
    payload = {"answers": {"q0": {"type": "noul", "noul": 0.25}}}
    with patch("httpx.post", return_value=_response(payload)):
        result = ask_yes_no("gemini/gemini-decision-1", "key", "c", ["q?"])

    assert result.answers == (YesNoAnswer(probability=0.25, confidence=0.75),)
    assert result.input_tokens == 0


def test_no_questions_sends_nothing() -> None:
    with patch("httpx.post") as post:
        result = ask_yes_no("gemini/gemini-decision-1", "key", "c", [])

    post.assert_not_called()
    assert result.answers == ()


def test_too_many_questions_are_refused_before_sending() -> None:
    with patch("httpx.post") as post, pytest.raises(DecisionsError, match="1001"):
        ask_yes_no("gemini/gemini-decision-1", "key", "c", ["q?"] * 1001)

    post.assert_not_called()


def test_a_server_error_is_retried() -> None:
    with patch(
        "httpx.post",
        side_effect=[_response({}, 503), _response(_answers(0.9))],
    ) as post:
        result = ask_yes_no("gemini/gemini-decision-1", "key", "c", ["q?"])

    assert post.call_count == 2
    assert result.answers[0].probability == 0.9


def test_a_rate_limit_is_retried_after_the_requested_delay() -> None:
    sleeps: list[float] = []
    with (
        patch("karotte.decisions.time.sleep", sleeps.append),
        patch(
            "httpx.post",
            side_effect=[
                _response({}, 429, {"retry-after": "7"}),
                _response(_answers(0.9)),
            ],
        ),
    ):
        ask_yes_no("gemini/gemini-decision-1", "key", "c", ["q?"])

    assert sleeps == [7.0]


def test_a_dropped_connection_is_retried() -> None:
    with patch(
        "httpx.post",
        side_effect=[httpx.ConnectError("boom"), _response(_answers(0.9))],
    ) as post:
        ask_yes_no("gemini/gemini-decision-1", "key", "c", ["q?"])

    assert post.call_count == 2


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_a_client_error_is_not_retried(status: int) -> None:
    with (
        patch("httpx.post", return_value=_response({"error": "no"}, status)) as post,
        pytest.raises(DecisionsError, match=f"HTTP {status}"),
    ):
        ask_yes_no("gemini/gemini-decision-1", "key", "c", ["q?"])

    post.assert_called_once()


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"answers": []},
        {"answers": {}},
        {"answers": {"q0": {"type": "choice", "choice": "a"}}},
        {"answers": {"q0": {"type": "noul"}}},
    ],
)
def test_an_answer_that_isnt_a_yes_no_is_an_error(payload: dict[str, Any]) -> None:
    with (
        patch("httpx.post", return_value=_response(payload)),
        pytest.raises(DecisionsError),
    ):
        ask_yes_no("gemini/gemini-decision-1", "key", "c", ["q?"])


def test_a_body_that_isnt_json_is_an_error() -> None:
    bad = MagicMock(spec=httpx.Response, status_code=200, headers={})
    bad.json.side_effect = ValueError("nope")
    with patch("httpx.post", return_value=bad), pytest.raises(DecisionsError):
        ask_yes_no("gemini/gemini-decision-1", "key", "c", ["q?"])


def test_answers_are_found_by_question_id_whatever_order_they_come_back_in() -> None:
    reply = _answers(0.9, 0.5, 0.1)
    reply["answers"] = dict(reversed(list(reply["answers"].items())))
    with patch("httpx.post", return_value=_response(reply)):
        result = ask_yes_no("gemini/gemini-decision-1", "key", "c", ["a?", "b?", "c?"])

    assert [a.probability for a in result.answers] == [0.9, 0.5, 0.1]


def test_an_answer_to_a_question_nobody_asked_is_ignored() -> None:
    reply = _answers(0.9, 0.1)
    with patch("httpx.post", return_value=_response(reply)):
        result = ask_yes_no("gemini/gemini-decision-1", "key", "c", ["only?"])

    assert [a.probability for a in result.answers] == [0.9]


def test_the_content_is_sent_once_however_many_questions_follow() -> None:
    content = "the transcript " * 50
    questions = [f"criterion {i}?" for i in range(40)]
    with patch("httpx.post", return_value=_response(_answers(*[0.5] * 40))) as post:
        ask_yes_no("gemini/gemini-decision-1", "key", content, questions)

    sent = post.call_args.kwargs["json"]
    assert post.call_count == 1
    assert len(sent["questions"]) == 40
    assert str(sent).count(content) == 1


def test_a_request_carries_no_generation_parameters() -> None:
    with patch("httpx.post", return_value=_response(_answers(0.5))) as post:
        ask_yes_no("gemini/gemini-decision-1", "key", "c", ["q?"])

    assert set(post.call_args.kwargs["json"]) == {"model", "input", "questions"}


def test_the_endpoint_can_be_pointed_elsewhere() -> None:
    with patch("httpx.post", return_value=_response(_answers(0.5))) as post:
        ask_yes_no(
            "gemini/gemini-decision-1",
            "key",
            "c",
            ["q?"],
            api_base="https://gateway.example/",
        )

    assert post.call_args.args == ("https://gateway.example/v1beta/decisions",)


def test_a_failed_request_does_not_leak_the_key() -> None:
    with (
        patch("httpx.post", return_value=_response({"error": "bad"}, 403)),
        pytest.raises(DecisionsError) as raised,
    ):
        ask_yes_no("gemini/gemini-decision-1", "secret-key", "c", ["q?"])

    assert "secret-key" not in str(raised.value)


def test_decision_models_are_not_chat_models_so_the_catalog_omits_them() -> None:
    from karotte.model_catalog import CATALOG_MODEL_IDS

    assert not any(is_decision_model(model) for model in CATALOG_MODEL_IDS)
