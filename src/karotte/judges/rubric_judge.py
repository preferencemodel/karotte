import os
import time
from textwrap import dedent
from typing import Any, Final, TypedDict, cast, override

from loguru import logger

from karotte.judges.judge import Judge
from karotte.judges.rubric_context import AnswersContext, RubricContext
from karotte.model_spec import ModelSpec, spec_for
from karotte.providers import PROXY_PLACEHOLDER_KEY, provider_for, proxy_api_base
from karotte.schemas.scoring import Metadata, Scoring
from karotte.schemas.transcript import Transcript

_MAX_TOKENS = 32_000
"""Room for reasoning models to think before they answer."""


class RubricJudgeError(Exception):
    """The judge model gave no verdict."""


class RubricCriterion(TypedDict):
    """A single criterion in the rubric."""

    criterion: str
    weight: str | float


class RubricJudge(Judge):
    """Evaluates transcript answers against a rubric using an LLM.

    Each criterion in the rubric is evaluated by an LLM, which determines
    whether the criterion is met (binary yes/no). The final score is the
    sum of weights for met criteria.

    The task continues if the final score is greater than the continue_threshold.

    Args:
        rubric: List of criteria, each with a "criterion" (str) and "weight" (float/str)
        model: Model id, e.g. "anthropic/claude-sonnet-5" or any litellm model name.
            Defaults to `RubricJudge.default_model`; `evaluate` raises without one.
        api_key: API key for the model. Defaults to `RubricJudge.default_api_key`
            when the model has the default model's provider, else to litellm
            reading the provider's env var (e.g. OPENAI_API_KEY), else to a
            placeholder when a proxy is in use.
        context: List of context providers that determine what the LLM sees.
                 Defaults to [AnswersContext()].
        continue_threshold: Score threshold for continuing the task. Defaults to 0.0.
        temperature: Temperature for LLM generation. Defaults to 0.3.
    """

    default_api_key: str | None = None
    default_model: str | None = None

    def __init__(
        self,
        rubric: list[RubricCriterion],
        model: str | None = None,
        api_key: str | None = None,
        context: list[RubricContext] | None = None,
        continue_threshold: float = 0.0,
        temperature: float = 0.3,
    ) -> None:
        self.rubric: Final = rubric
        default_model = RubricJudge.default_model
        self.model: Final = model or default_model
        same_provider = default_model is not None and (
            spec_for(model or default_model).provider
            == spec_for(default_model).provider
        )
        self.api_key: Final = api_key or (
            RubricJudge.default_api_key if same_provider else None
        )
        self.context: Final[list[RubricContext]] = context or [AnswersContext()]
        self.continue_threshold: Final = continue_threshold
        self.temperature: Final = temperature

    def render_context(self, transcript: Transcript) -> str:
        """The content the criteria are judged against: every context provider's
        rendering, blank-line separated. Empty when no provider has anything."""
        context_parts = [
            rendered for ctx in self.context if (rendered := ctx.render(transcript))
        ]
        return "\n\n".join(context_parts)

    @override
    def evaluate(self, transcript: Transcript) -> Scoring:
        _ = self._require_model()
        context_text = self.render_context(transcript)

        if not context_text:
            return Scoring(
                score=0.0,
                metadata={"error": "No context available for evaluation"},
                continue_task=False,
            )

        # Evaluate each criterion
        total_score: float = 0.0
        metadata: Metadata = {}

        for i, item in enumerate(self.rubric):
            criterion = item["criterion"]
            weight = float(item["weight"])

            # Ask LLM if criterion is met
            is_met, reasoning = self._evaluate_criterion(context_text, criterion)

            criterion_key = f"criterion_{i}_{criterion[:30]}"
            if is_met:
                total_score += weight
                metadata[criterion_key] = f"✓: {reasoning}"
            else:
                metadata[criterion_key] = f"✗: {reasoning}"

        # Store final score and context
        metadata["context"] = context_text
        metadata["final_score"] = str(total_score)
        metadata["pass_threshold"] = f"> {self.continue_threshold}"

        return Scoring(
            score=total_score,
            metadata=metadata,
            continue_task=total_score > self.continue_threshold,
        )

    def _require_model(self) -> str:
        if self.model is None:
            raise ValueError(
                "RubricJudge has no model: pass model=, or set rubric_judge_model in the run config."
            )
        return self.model

    def _evaluate_criterion(self, context: str, criterion: str) -> tuple[bool, str]:
        """Evaluate a single criterion using the LLM.

        Returns:
            Tuple of (is_met, reasoning) where is_met is True if the criterion
            is met, and reasoning is the LLM's explanation.
        """
        prompt = self.criterion_prompt(context, criterion)

        spec = spec_for(self._require_model())
        params: dict[str, Any] = {
            "model": spec.litellm_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": _MAX_TOKENS,
            "api_key": self._api_key(spec),
        }
        if api_base := proxy_api_base(provider_for(spec)):
            params["api_base"] = api_base
        if spec.supports_sampling_params:
            params["temperature"] = self.temperature

        import litellm

        response = cast(litellm.ModelResponse, _completion_with_retries(params))
        choice = response.choices[0]
        content = choice.message.content
        if not content:
            raise RubricJudgeError(
                f"{self.model} returned no text (finish_reason={choice.finish_reason!r})"
            )
        return self.parse_reply(content)

    def _api_key(self, spec: ModelSpec) -> str | None:
        """``api_key``, else a placeholder when a proxy is in use and litellm
        would find no key in the environment."""
        if self.api_key or not spec.requires_api_key:
            return self.api_key
        if not os.environ.get("KAROTTE_PROXY_URL"):
            return None
        import litellm

        if litellm.validate_environment(spec.litellm_model)["keys_in_environment"]:
            return None
        return PROXY_PLACEHOLDER_KEY

    @staticmethod
    def criterion_prompt(context: str, criterion: str) -> str:
        """The prompt asking whether ``context`` meets ``criterion``, answered
        in the format :meth:`parse_reply` reads."""
        return dedent(f"""
            You are evaluating whether the following content meets a specific criterion.

            Content to evaluate:
            {context}

            Criterion:
            {criterion}

            Does the content meet this criterion? Respond with ONLY "YES" or "NO", followed by a brief explanation on a new line.

            Format:
            YES
            [brief explanation]

            or

            NO
            [brief explanation]""")

    @staticmethod
    def parse_reply(content: str) -> tuple[bool, str]:
        """``(is_met, reasoning)`` from a reply to :meth:`criterion_prompt`:
        met when the first line starts with YES, the rest is the reasoning."""
        lines = content.strip().split("\n", 1)
        decision = lines[0].strip().upper()
        reasoning = lines[1].strip() if len(lines) > 1 else "No explanation provided"
        return decision.startswith("YES"), reasoning


def _completion_with_retries(params: dict[str, Any]) -> Any:
    import litellm

    from karotte.agents.builtin_source import llm_retry_wait

    attempt = 1
    while True:
        try:
            return litellm.completion(**params)
        except Exception as e:
            wait = llm_retry_wait(attempt, e)
            if wait is None:
                raise
            logger.warning(
                "Rubric judge call failed: {cause}; retrying after {wait:.1f}s...",
                cause=type(e).__name__,
                wait=wait,
            )
            time.sleep(wait)
            attempt += 1
