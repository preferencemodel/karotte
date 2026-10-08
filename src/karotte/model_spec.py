"""Everything karotte decides from a model id: output ceiling, reasoning effort,
and which provider quirks apply.

Unknown ids resolve to a spec of safe defaults rather than failing, so a model
nobody has heard of still runs.
"""

import re
from typing import Any, Literal

from pydantic import BaseModel

SPECIAL_TRAINING_MODEL_NAME = "__karotte_special__/training"
SPECIAL_TRAINING_MODEL_PREFIX = "pt/"
"""Models whose name starts with this prefix also use the special training path."""

type ToolCallRepair = Literal["deepseek", "xai"]

type ReasoningEffort = str
"""How hard a run asks its model to think: ``min`` and ``max`` pick the model's
lowest and highest level, anything else is one of the provider's own level names."""

PROVIDER_API_KEY_ENV: dict[str, str] = {
    "mistral": "MISTRAL_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
}
"""Provider prefix to the env var that provider's CLI reads its API key from."""

LITELLM_API_KEY_ENV: dict[str, str] = {
    **PROVIDER_API_KEY_ENV,
    "together_ai": "TOGETHERAI_API_KEY",
}
"""Provider prefix to the env var litellm reads its API key from."""


def api_key_env_var(model: str) -> str | None:
    """The env var holding ``model``'s API key, or ``None`` if it needs none.
    Providers not in :data:`LITELLM_API_KEY_ENV` get ``<PROVIDER>_API_KEY``."""
    spec = spec_for(model)
    if not spec.requires_api_key:
        return None
    if env_var := LITELLM_API_KEY_ENV.get(spec.provider):
        return env_var
    return f"{spec.provider.upper()}_API_KEY" if spec.provider else "ANTHROPIC_API_KEY"


class ModelSpec(BaseModel, frozen=True):
    """What karotte knows about one model."""

    model: str

    model_display_name: str
    """The model's human name, derived from the id."""

    provider: str
    """API provider litellm routes to, or "" when the id names none."""

    provider_display_name: str
    """The provider's human name, or "" when the id names none."""

    litellm_model: str
    """What to send as ``model``; differs where litellm needs an explicit prefix."""

    max_output_tokens: int

    min_reasoning_effort: str | None
    """The provider's name for this model's lowest effort, or None where the
    model takes no effort parameter."""

    max_reasoning_effort: str | None
    """The provider's name for this model's highest effort, or None where the
    model takes no effort parameter."""

    reasoning_effort_levels: tuple[str, ...]
    """Every effort value the model accepts, under the provider's names and
    lowest first; empty where the model takes no effort parameter."""

    reasoning_request: dict[str, Any]
    """Params that ask the model to think and to return its reasoning.
    Providers may override entries or drop ones their API rejects."""

    supports_sampling_params: bool
    """False where temperature and friends are a 400."""

    tool_call_repairs: tuple[ToolCallRepair, ...]

    requires_api_key: bool

    extra_allowed_openai_params: tuple[str, ...] = ()
    """Params to force past litellm's validation, for models it doesn't know yet."""

    def reasoning_effort_value(self, level: ReasoningEffort | None) -> str | None:
        """What to send as ``reasoning_effort`` for ``level``, or None to send
        nothing and leave the provider's default."""
        if level is None:
            return None
        if level == "min":
            return self.min_reasoning_effort
        if level == "max":
            return self.max_reasoning_effort
        return level


def is_special_training_model(model: str) -> bool:
    """Whether a run on this model gets its messages from the training backend
    rather than calling an inference provider."""
    return model == SPECIAL_TRAINING_MODEL_NAME or model.startswith(
        SPECIAL_TRAINING_MODEL_PREFIX
    )


# Claude 4.6 and later, matched by prefix so a point release or dated snapshot
# inherits the family's rules (adaptive thinking, low..max effort, 128k output)
# with no code change. The 4.5 generation takes budget_tokens instead and 400s
# on the thinking parameter, so it stays out.
_CLAUDE_ADAPTIVE_PREFIXES = (
    "claude-fable",
    "claude-haiku-5",
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
)

# Asking for more than a model allows is a hard 400, so unknown ids keep this.
_DEFAULT_MAX_OUTPUT_TOKENS = 64000

# max_tokens covers thinking *and* response text, so it doubles as the thinking
# budget on adaptive-thinking models.
_MAX_OUTPUT_TOKENS: dict[str, int] = {
    "muse-spark-1.3": 128000,
    "Qwen3.8-2.4T-A95B": 128000,
}

_MAX_OUTPUT_TOKENS_PREFIXES: tuple[tuple[str, int], ...] = (
    *((prefix, 128000) for prefix in _CLAUDE_ADAPTIVE_PREFIXES),
    ("gpt-6", 128000),
    ("gpt-5.6", 128000),
)

_CLAUDE_EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")

# The 4.6 generation predates xhigh and answers 400 to it.
_CLAUDE_4_6_PREFIXES = ("claude-opus-4-6", "claude-sonnet-4-6")
_CLAUDE_4_6_EFFORT_LEVELS = ("low", "medium", "high", "max")

# Every accepted level, lowest first, under each provider's own names. `none`,
# where a provider has it, switches reasoning off and is left out. Exact,
# because siblings under the same prefix disagree: o1-mini and o1-preview
# answer 400 to the parameter, litellm validates effort on gemini ids against
# its cost map, and xhigh is grok-4.6 and up.
_REASONING_EFFORT_MODELS: dict[str, tuple[str, ...]] = {
    "o1": ("low", "medium", "high"),
    "o3": ("low", "medium", "high"),
    "gemini-3.1-pro-preview": ("low", "medium", "high"),
    "gemini-3.8-flash": ("low", "medium", "high"),
    "gemini-3.7-flash": ("low", "medium", "high"),
    "grok-4.7": ("low", "medium", "high", "xhigh"),
    "grok-4.6": ("low", "medium", "high", "xhigh"),
    "muse-spark-1.2": ("minimal", "low", "medium", "high", "xhigh"),
    "muse-spark-1.3": ("minimal", "low", "medium", "high", "xhigh", "max"),
    "DeepSeek-V4-Pro-0813": ("high", "max"),
    "GLM-5.3": ("low", "high", "max"),
    # The same checkpoint under Fireworks' spelling of the id.
    "glm-5p3": ("low", "high", "max"),
    "glm-5p3-flash": ("low", "high", "max"),
    "Kimi-K3": ("low", "high", "max"),
    "Qwen3.8-2.4T-A95B": ("low", "medium", "xhigh"),
}

# litellm validates effort against its cost map, which lacks the bridge name
# responses/<model> for any Muse Spark or GPT-6, lacks Gemini 3.8 Flash and
# Grok 4.7 outright, and carries no reasoning_effort at all in its together_ai
# or fireworks_ai param lists. Once allowed through, litellm still maps Gemini's
# effort onto thinkingLevel itself.
_EXTRA_ALLOWED_OPENAI_PARAMS: dict[str, tuple[str, ...]] = dict.fromkeys(
    (
        "muse-spark-1.3",
        "muse-spark-1.2",
        "gemini-3.8-flash",
        "grok-4.7",
        "gpt-6-astra",
        "gpt-6-sol",
        "gpt-6-luna",
        "DeepSeek-V4-Pro-0813",
        "GLM-5.3",
        "glm-5p3",
        "glm-5p3-flash",
        "Qwen3.8-2.4T-A95B",
        "Kimi-K3",
    ),
    ("reasoning_effort",),
)

# (prefix, levels) for families whose snapshots all behave alike, matched in
# order, so gpt-5 has to come after gpt-5.6. Families left out get no effort
# parameter: minimax and the older together_ai models answer 400 to it,
# grok-4.3 and mistral are undocumented, and on Claude 4.5 litellm would turn
# it into budget_tokens thinking those models do not run with today.
_REASONING_EFFORT_PREFIXES: tuple[tuple[str, tuple[str, ...]], ...] = (
    # max is Responses API only; Chat Completions answers 400 to it.
    ("gpt-6", ("low", "medium", "high", "xhigh", "max")),
    ("gpt-5.6", ("low", "medium", "high", "xhigh", "max")),
    ("gpt-5.5", ("low", "medium", "high", "xhigh")),
    ("gpt-5.4", ("low", "medium", "high", "xhigh")),
    ("gpt-5.3", ("low", "medium", "high")),
    ("gpt-5.2", ("low", "medium", "high", "xhigh")),
    ("gpt-5.1", ("low", "medium", "high")),
    ("gpt-5", ("minimal", "low", "medium", "high")),
)

# Opus 4.7 onward and GPT-6 answer 400 to temperature and the other sampling
# parameters.
_NO_SAMPLING_PARAMS_PREFIXES = (
    "gpt-6",
    "claude-fable",
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-haiku-5",
)


def spec_for(model: str) -> ModelSpec:
    """Resolve what karotte knows about ``model``."""
    name = model.split("/")[-1]
    levels = _reasoning_effort_levels(name)
    return ModelSpec(
        model=model,
        model_display_name=_model_display_name(name),
        provider=(provider := _provider(model)),
        provider_display_name=_provider_display_name(provider),
        litellm_model=_litellm_model(model),
        max_output_tokens=_max_output_tokens(name),
        min_reasoning_effort=levels[0] if levels else None,
        max_reasoning_effort=levels[-1] if levels else None,
        reasoning_effort_levels=levels,
        reasoning_request=_reasoning_request(name),
        supports_sampling_params=not name.startswith(_NO_SAMPLING_PARAMS_PREFIXES),
        tool_call_repairs=_tool_call_repairs(model),
        requires_api_key=not (
            model.startswith("vertex_ai/") or is_special_training_model(model)
        ),
        extra_allowed_openai_params=_EXTRA_ALLOWED_OPENAI_PARAMS.get(name, ()),
    )


# Keyed by routing slug, so it grows per provider, not per model.
_PROVIDER_DISPLAY_NAMES: dict[str, str] = {
    "anthropic": "Anthropic",
    "openai": "OpenAI",
    "together_ai": "Together AI",
    "fireworks_ai": "Fireworks AI",
    "vertex_ai": "Google Vertex AI",
    "gemini": "Google",
    "xai": "xAI",
    "minimax": "MiniMax",
    "meta": "Meta",
    "mistral": "Mistral",
    SPECIAL_TRAINING_MODEL_PREFIX.rstrip("/"): "Training",
    SPECIAL_TRAINING_MODEL_NAME.split("/")[0]: "Training",
}


def _provider_display_name(provider: str) -> str:
    return _PROVIDER_DISPLAY_NAMES.get(provider, provider.capitalize())


def _model_display_name(name: str) -> str:
    """``claude-opus-4-5-20251101`` → ``Opus 4.5``, ``gpt-5.6-sol`` → ``GPT-5.6 Sol``,
    ``Kimi-K3`` → ``Kimi K3``."""
    if name.startswith("claude-"):
        family, *version = name.removeprefix("claude-").split("-")
        if version and len(version[-1]) == 8 and version[-1].isdigit():
            version.pop()
        return f"{family.capitalize()} {'.'.join(version)}".rstrip()
    words = name.split("-")
    if words[0] == "gpt" and len(words) > 1:
        words = [f"GPT-{words[1]}", *words[2:]]
    # Fireworks lowercases a family and spells a version's dot as `p` (glm-5p3,
    # kimi-k2p6); read those back as the vendor's own spelling.
    if words[0] == "glm":
        words[0] = "GLM"
    words = [re.sub(r"(?<=\d)p(?=\d)", ".", word) for word in words]
    return " ".join(word.capitalize() if word.islower() else word for word in words)


def _provider(model: str) -> str:
    prefix, separator, _ = model.partition("/")
    if separator:
        return prefix
    return "anthropic" if model.startswith("claude") else ""


def _litellm_model(model: str) -> str:
    # A bare Claude id names its provider so litellm routes a release it has
    # never heard of; for ids it knows the prefix changes nothing.
    if model.startswith("claude"):
        return f"anthropic/{model}"
    # Meta only returns reasoning summaries on the Responses API, which litellm
    # reaches through its "responses/" infix.
    if model.startswith("meta/muse-spark"):
        return model.replace("meta/", "meta/responses/", 1)
    # Same for GPT-6, which litellm's own Responses bridge misses because that
    # bridge only matches "gpt-5." names. Chat Completions would also take
    # max_tokens, which every OpenAI reasoning model answers 400 to.
    if model.startswith("openai/gpt-6"):
        return model.replace("openai/", "openai/responses/", 1)
    return model


def _reasoning_request(name: str) -> dict[str, Any]:
    if name.startswith(_CLAUDE_ADAPTIVE_PREFIXES):
        return {"thinking": {"type": "adaptive", "display": "summarized"}}
    # Gemini 2.0 answers 400 to thinkingConfig, so only 3.x asks for summaries.
    if name.startswith("gemini-3"):
        return {"thinkingConfig": {"includeThoughts": True}}
    # litellm's Responses bridge turns this into reasoning.summary and maps the
    # returned summary onto reasoning_content. It is also what builds the
    # reasoning object at all: without it litellm drops reasoning_effort on the
    # floor for a Responses-routed model instead of folding it into
    # reasoning.effort. Its string path has no "max" branch, hence gpt-5.6 too.
    if name.startswith(("muse-spark", "gpt-6", "gpt-5.6")):
        return {"extra_body": {"reasoning_summary": "detailed"}}
    return {}


def _reasoning_effort_levels(name: str) -> tuple[str, ...]:
    if name.startswith(_CLAUDE_4_6_PREFIXES):
        return _CLAUDE_4_6_EFFORT_LEVELS
    if name.startswith(_CLAUDE_ADAPTIVE_PREFIXES):
        return _CLAUDE_EFFORT_LEVELS
    if (exact := _REASONING_EFFORT_MODELS.get(name)) is not None:
        return exact
    return _prefix_value(name, _REASONING_EFFORT_PREFIXES) or ()


def _max_output_tokens(name: str) -> int:
    if (exact := _MAX_OUTPUT_TOKENS.get(name)) is not None:
        return exact
    prefixed = _prefix_value(name, _MAX_OUTPUT_TOKENS_PREFIXES)
    return prefixed if prefixed is not None else _DEFAULT_MAX_OUTPUT_TOKENS


def _prefix_value[T](name: str, rules: tuple[tuple[str, T], ...]) -> T | None:
    for prefix, value in rules:
        if name.startswith(prefix):
            return value
    return None


def _tool_call_repairs(model: str) -> tuple[ToolCallRepair, ...]:
    repairs: list[ToolCallRepair] = []
    if "deepseek" in model.lower():
        repairs.append("deepseek")
    if model.lower().startswith("xai/"):
        repairs.append("xai")
    return tuple(repairs)
