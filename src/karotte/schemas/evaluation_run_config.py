from typing import Any, Literal, Self

from pydantic import BaseModel, ValidationInfo, model_validator

from karotte.model_spec import (
    SPECIAL_TRAINING_MODEL_NAME,
    SPECIAL_TRAINING_MODEL_PREFIX,
    ReasoningEffort,
    is_special_training_model,
    spec_for,
)

from .http_mcp_server_config import HttpMcpServerConfig
from .websocket_config import WebSocketConfig

__all__ = [
    "SPECIAL_TRAINING_MODEL_NAME",
    "SPECIAL_TRAINING_MODEL_PREFIX",
    "EvaluationRunConfig",
]


class EvaluationRunConfig(BaseModel):
    run_id: str
    task_id: str
    agent: str | None = None
    """Which agent drives the run (e.g. ``builtin``, ``external``,
    ``mistral-vibe``). When unset, resolves to ``external`` for special-training
    models (``pt/`` prefix) and ``builtin`` otherwise; see ``resolved_agent``."""
    model: str
    model_api_key: str | None = None
    """API key for the model. Not required for Vertex AI models."""
    rubric_judge_api_key: str | None = None
    use_hints: bool = True

    reasoning_effort: ReasoningEffort | None = None

    turn_limit: int | None = None
    """Maximum number of turns (model responses) allowed before terminating the run."""

    step_time_limit_seconds: float | list[float] | None = None
    """Wall-clock seconds allowed per step, measured from when the step's
    instructions are issued. A single value applies to every step; a list is
    indexed per step (steps past the end of the list are unlimited). None
    disables the limit and the remaining-time counter."""

    on_step_time_limit: Literal["error", "score"] = "error"
    """What to do when a step exceeds its time limit. "error" aborts the run;
    "score" ends the step and scores what the model produced so far."""

    inject_time_remaining_counter: bool = True
    """Whether to append a "Time remaining: N seconds" note after each turn while a step
    time limit is set. No effect without a limit, or for agents that run the
    step in one opaque subprocess (e.g. the Vibe CLI agent)."""

    step_context_window_limit: int | list[int] | None = None
    """Maximum context-window length (input tokens of the most recent turn)
    allowed per step. A single value applies to every step; a list is indexed
    per step (steps past the end of the list are unlimited). None disables the
    limit and the remaining-context counter."""

    on_step_context_window_limit: Literal["error", "score"] = "error"
    """What to do when a step exceeds its context-window limit. "error" aborts
    the run; "score" ends the step and scores what the model produced so far."""

    inject_context_remaining_counter: bool = True
    """Whether to append a "Context remaining: N" note after each turn while a
    step context-window limit is set. No effect without a limit, or for agents
    that run the step in one opaque subprocess (e.g. the Vibe CLI agent)."""

    mcp_server_config: HttpMcpServerConfig = HttpMcpServerConfig()

    websocket_config: WebSocketConfig = WebSocketConfig()
    """Where to stream the run's progress."""

    transcript_file: str | None = None
    """Where to save the run's transcript."""

    use_fake_model: bool = False
    """Whether to use a fake model for evaluation."""

    extra_config: dict[Any, Any] | None = None
    """Custom configuration that allows hyperparameter search on tasks without
    having to rebuild the container image."""

    backend_uri: str | None = None
    """URI of the backend server. Used for artifact uploads and transcript streaming."""

    save_artifacts: bool = True
    """Whether to save artifacts. If False, save_artifact() will no-op."""

    @property
    def extra_artifact_paths(self) -> list[str]:
        """Absolute container paths to persist as artifacts before scoring.

        Read from `extra_config["extra_artifact_paths"]`, which accepts a single
        string or a list of strings. Lets a run capture extra output file(s)
        without rebuilding the env."""
        raw = (self.extra_config or {}).get("extra_artifact_paths")
        if raw is None:
            return []
        return [raw] if isinstance(raw, str) else raw

    @model_validator(mode="after")
    def validate_extra_artifact_paths(self) -> Self:
        self._validate_str_or_str_list("extra_artifact_paths")
        return self

    @model_validator(mode="after")
    def validate_task_instructions(self) -> Self:
        self._validate_str_or_str_list("extra_task_instructions")
        self._validate_str_or_str_list("task_instructions_override")
        return self

    def _validate_str_or_str_list(self, key: str) -> None:
        raw = (self.extra_config or {}).get(key)
        if raw is None:
            return
        if not (
            isinstance(raw, str)
            or (isinstance(raw, list) and all(isinstance(p, str) for p in raw))
        ):
            msg = (
                f"extra_config[{key!r}] must be a string or list "
                f"of strings, got {raw!r}"
            )
            raise ValueError(msg)

    def _extra_config_entry_for_step(self, key: str, index: int) -> str | None:
        """Resolve a str-or-list `extra_config` entry for a given step index.

        A string applies to every step; a list is indexed per step, with
        out-of-range indices yielding ``None``.
        """
        raw = (self.extra_config or {}).get(key)
        if raw is None:
            return None
        if isinstance(raw, str):
            return raw
        return raw[index] if index < len(raw) else None

    def resolve_step_time_limit(self, index: int) -> float | None:
        """Wall-clock second budget for the step at `index`, or None if unlimited.

        A single value applies to every step; a list is indexed per step, with
        out-of-range indices yielding ``None``.
        """
        limit = self.step_time_limit_seconds
        if isinstance(limit, list):
            return limit[index] if index < len(limit) else None
        return limit

    def resolve_step_context_window_limit(self, index: int) -> int | None:
        """Context-window token budget for the step at `index`, or None if
        unlimited.

        A single value applies to every step; a list is indexed per step, with
        out-of-range indices yielding ``None``.
        """
        limit = self.step_context_window_limit
        if isinstance(limit, list):
            return limit[index] if index < len(limit) else None
        return limit

    def resolve_step_instructions(self, original: str, index: int) -> str:
        """Apply `extra_config` instruction overlays for the step at `index`.

        `task_instructions_override` replaces the step's instructions entirely
        and takes precedence. Otherwise `extra_task_instructions` is appended.
        Both accept a single string (applied to every step) or a list (one
        entry per step); missing list entries leave the step unchanged."""
        override = self._extra_config_entry_for_step(
            "task_instructions_override", index
        )
        if override is not None:
            return override
        extra = self._extra_config_entry_for_step("extra_task_instructions", index)
        if extra:
            return original + "\n\n" + extra
        return original

    @property
    def is_special_training_model(self) -> bool:
        """Whether this run uses the special training rollout path rather than
        calling an LLM provider directly."""
        return is_special_training_model(self.model)

    @property
    def resolved_agent(self) -> str:
        """The agent to drive this run.

        An explicit ``agent`` always wins. Otherwise the ``pt/`` model prefix
        maps to ``external`` (training rollouts) and everything else to
        ``builtin``, so existing configs keep their behavior without setting
        ``agent``."""
        if self.agent is not None:
            return self.agent
        if self.is_special_training_model:
            return "external"
        return "builtin"

    @property
    def applied_reasoning_effort(self) -> str | None:
        """The effort this run actually sends, under the provider's own name.

        None where nothing is sent: the external and CLI agents drive their own
        inference, a fake-model run reaches no provider, and plenty of models
        take no effort parameter at all."""
        if self.use_fake_model or self.resolved_agent != "builtin":
            return None
        return spec_for(self.model).reasoning_effort_value(self.reasoning_effort)

    @model_validator(mode="after")
    def validate_reasoning_effort(self) -> Self:
        """``min`` and ``max`` resolve against any model; a provider's own level
        name has to be one this model accepts."""
        level = self.reasoning_effort
        if level in (None, "min", "max"):
            return self
        levels = spec_for(self.model).reasoning_effort_levels
        if not levels:
            msg = (
                f"{self.model} takes no reasoning_effort; only min and max are accepted"
            )
            raise ValueError(msg)
        if level not in levels:
            accepted = ", ".join(levels)
            msg = f"reasoning_effort={level!r} is not accepted by {self.model}; accepted: {accepted}"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_api_key_required(self, info: ValidationInfo) -> Self:
        """Validate that model_api_key is provided when needed.

        A ``prepare_only`` validation context and ``use_fake_model`` skip the check:
        neither calls the model, so no key is needed.
        """
        if self.use_fake_model or (info.context or {}).get("prepare_only"):
            return self
        if self.model_api_key is None and spec_for(self.model).requires_api_key:
            msg = f"model_api_key is required for model {self.model}"
            raise ValueError(msg)
        return self
