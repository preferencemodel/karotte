from textwrap import dedent
from typing import Any, Final, TypedDict, cast, override

import litellm

from karotte.judges.judge import Judge
from karotte.judges.rubric_context import AnswersContext, RubricContext
from karotte.model_spec import spec_for
from karotte.providers import provider_for, proxy_api_base
from karotte.schemas.scoring import Metadata, Scoring
from karotte.schemas.transcript import Transcript


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
        model: Model id, e.g. "claude-sonnet-5" or any litellm model name.
            Defaults to `RubricJudge.default_model`, then Fable 5.
        api_key: API key for the model. Defaults to `RubricJudge.default_api_key`,
            then to litellm reading the provider's env var (e.g. ANTHROPIC_API_KEY).
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
        self.model: Final = model or RubricJudge.default_model or "claude-fable-5"
        self.api_key: Final = api_key or RubricJudge.default_api_key
        self.context: Final[list[RubricContext]] = context or [AnswersContext()]
        self.continue_threshold: Final = continue_threshold
        self.temperature: Final = temperature

    @override
    def evaluate(self, transcript: Transcript) -> Scoring:
        # Gather context from all providers
        context_parts = [
            rendered for ctx in self.context if (rendered := ctx.render(transcript))
        ]
        context_text = "\n\n".join(context_parts)

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

    def _evaluate_criterion(self, context: str, criterion: str) -> tuple[bool, str]:
        """Evaluate a single criterion using the LLM.

        Returns:
            Tuple of (is_met, reasoning) where is_met is True if the criterion
            is met, and reasoning is the LLM's explanation.
        """
        prompt = dedent(f"""
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

        spec = spec_for(self.model)
        params: dict[str, Any] = {
            "model": spec.litellm_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 200,
            "api_key": self.api_key,
        }
        if api_base := proxy_api_base(provider_for(spec)):
            params["api_base"] = api_base
        if spec.supports_sampling_params:
            params["temperature"] = self.temperature

        try:
            response = cast(litellm.ModelResponse, litellm.completion(**params))
            content = response.choices[0].message.content
            if not content:
                return False, "LLM returned no text"

            # Parse response
            lines = content.strip().split("\n", 1)
            decision = lines[0].strip().upper()
            reasoning = (
                lines[1].strip() if len(lines) > 1 else "No explanation provided"
            )

            is_met = decision.startswith("YES")
            return is_met, reasoning

        except Exception as e:
            error_msg = f"Error evaluating criterion: {str(e)}"
            return False, error_msg
