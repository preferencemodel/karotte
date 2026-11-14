"""A client for decision models: models that answer typed questions about an
input with calibrated probabilities and generate no text.

They have no chat endpoint, so they can grade (see ``RubricJudge``) but cannot
drive a run. Gemini Decision 1 is the one karotte knows.
"""

import time
from dataclasses import dataclass
from typing import Any, Final

import httpx
from loguru import logger

GEMINI_API_BASE: Final = "https://generativelanguage.googleapis.com"

MIN_PROBABILITY: Final = 0.01
MAX_PROBABILITY: Final = 0.99
"""The range a yes/no answer falls in. A model never says 0 or 1, so a threshold
outside it either always or never matches."""

MAX_QUESTIONS_PER_REQUEST: Final = 1000
"""How many questions one request may carry."""

_REQUEST_TIMEOUT_S: Final = 120.0


class DecisionsError(Exception):
    """A decisions request failed, or its answer wasn't the one asked for.

    ``status_code`` is the HTTP status, or 503 when the request never got a
    response, so the shared retry policy treats it like a server error.
    """

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        retry_after: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code: int | None = status_code
        self.litellm_response_headers: dict[str, str] | None = (
            {"retry-after": retry_after} if retry_after is not None else None
        )


@dataclass(frozen=True)
class YesNoAnswer:
    """One yes/no question's answer."""

    probability: float
    """Calibrated probability of yes, between 0.01 and 0.99."""

    confidence: float
    """Probability that the more likely side is right: ``max(p, 1 - p)``."""


@dataclass(frozen=True)
class YesNoResult:
    """The answers to a batch of yes/no questions, in the order they were asked."""

    answers: tuple[YesNoAnswer, ...]
    input_tokens: int
    """Tokens billed for the request. Decision models generate none."""


def ask_yes_no(
    model: str,
    api_key: str,
    content: str,
    questions: list[str],
    *,
    api_base: str = GEMINI_API_BASE,
) -> YesNoResult:
    """Ask every yes/no question in ``questions`` about ``content`` in one request.

    ``content`` is billed once however many questions follow it, and each
    question is answered independently of the others.
    """
    if not questions:
        return YesNoResult(answers=(), input_tokens=0)
    if len(questions) > MAX_QUESTIONS_PER_REQUEST:
        raise DecisionsError(
            f"{len(questions)} questions in one request; the limit is {MAX_QUESTIONS_PER_REQUEST}."
        )
    ids = [f"q{i}" for i in range(len(questions))]
    body = {
        "model": model.partition("/")[2],
        "input": content,
        "questions": {
            question_id: {"type": "noul", "instructions": question}
            for question_id, question in zip(ids, questions, strict=True)
        },
    }
    payload = _post_with_retries(
        f"{api_base.rstrip('/')}/v1beta/decisions", api_key, body
    )
    return _parse_yes_no(payload, ids)


def _post_with_retries(url: str, api_key: str, body: dict[str, Any]) -> dict[str, Any]:
    # Imported here: the module pulls in litellm, which a run that never grades
    # with a decision model shouldn't pay for.
    from karotte.agents.builtin_source import LLM_RETRY_MAX_ATTEMPTS, llm_retry_wait

    attempt = 1
    while True:
        try:
            return _post(url, api_key, body)
        except DecisionsError as e:
            wait = llm_retry_wait(attempt, e)
            if wait is None or attempt >= LLM_RETRY_MAX_ATTEMPTS:
                raise
            logger.warning(
                "Decisions request failed: {cause}; retrying after {wait:.1f}s...",
                cause=e,
                wait=wait,
            )
            time.sleep(wait)
            attempt += 1


def _post(url: str, api_key: str, body: dict[str, Any]) -> dict[str, Any]:
    try:
        response = httpx.post(
            url,
            json=body,
            headers={"x-goog-api-key": api_key},
            timeout=_REQUEST_TIMEOUT_S,
        )
    except httpx.TransportError as e:
        raise DecisionsError(
            f"No response from {url}: {type(e).__name__}", status_code=503
        ) from e
    if response.status_code != 200:
        # The key travels only in a header, so the body can't echo it.
        raise DecisionsError(
            f"HTTP {response.status_code} from {url}: {response.text[:500]}",
            status_code=response.status_code,
            retry_after=response.headers.get("retry-after"),
        )
    try:
        payload = response.json()
    except ValueError as e:
        raise DecisionsError(f"{url} returned a body that isn't JSON") from e
    if not isinstance(payload, dict):
        raise DecisionsError(f"{url} returned {type(payload).__name__}, not an object")
    return payload


def _parse_yes_no(payload: dict[str, Any], ids: list[str]) -> YesNoResult:
    answers = payload.get("answers")
    if not isinstance(answers, dict):
        raise DecisionsError("The response has no answers.")
    parsed: list[YesNoAnswer] = []
    for question_id in ids:
        answer = answers.get(question_id)
        if not isinstance(answer, dict) or answer.get("type") != "noul":
            raise DecisionsError(f"No yes/no answer for question {question_id}.")
        try:
            probability = float(answer["noul"])
        except (KeyError, TypeError, ValueError) as e:
            raise DecisionsError(f"Question {question_id} has no probability.") from e
        confidence = answer.get("confidence")
        parsed.append(
            YesNoAnswer(
                probability=probability,
                confidence=(
                    float(confidence)
                    if isinstance(confidence, int | float)
                    else max(probability, 1 - probability)
                ),
            )
        )
    usage = payload.get("usage")
    tokens = usage.get("total_input_tokens") if isinstance(usage, dict) else None
    return YesNoResult(
        answers=tuple(parsed),
        input_tokens=tokens if isinstance(tokens, int) else 0,
    )
