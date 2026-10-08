"""The models this karotte release supports."""

from pydantic import BaseModel

from karotte.model_spec import (
    SPECIAL_TRAINING_MODEL_NAME,
    SPECIAL_TRAINING_MODEL_PREFIX,
    ModelSpec,
    spec_for,
)


class ModelFamily(BaseModel, frozen=True):
    """A prefix under which any model id is accepted."""

    prefix: str
    description: str


class ModelCatalog(BaseModel, frozen=True):
    models: tuple[ModelSpec, ...]
    families: tuple[ModelFamily, ...]


# The first entry is the default in model pickers.
CATALOG_MODEL_IDS: tuple[str, ...] = (
    "claude-opus-5-5",
    "claude-fable-5-1",
    "claude-fable-5",
    "claude-opus-5",
    "claude-sonnet-5-5",
    "claude-sonnet-5",
    "claude-haiku-5-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-opus-4-5-20251101",
    "claude-sonnet-4-5-20250929",
    "claude-haiku-4-5-20251001",
    "together_ai/moonshotai/Kimi-K3",
    "together_ai/moonshotai/Kimi-K2.6",
    "together_ai/moonshotai/Kimi-K2.5",
    "together_ai/deepseek-ai/DeepSeek-V4-Pro-0813",
    "together_ai/deepseek-ai/DeepSeek-V3.1",
    "fireworks_ai/accounts/fireworks/models/glm-5p3",
    "fireworks_ai/accounts/fireworks/models/glm-5p3-flash",
    "together_ai/zai-org/GLM-5.2",
    "together_ai/zai-org/GLM-5.1",
    "together_ai/Qwen/Qwen3.8-2.4T-A95B",
    "openai/o1",
    "openai/o3",
    "openai/gpt-5",
    "openai/gpt-5.1",
    "openai/gpt-5.2",
    "openai/gpt-5.4",
    "openai/gpt-5.5",
    "openai/gpt-6-astra",
    "openai/gpt-6-sol",
    "openai/gpt-6-luna",
    "openai/gpt-5.6",
    "openai/gpt-5.6-sol",
    "openai/gpt-5.6-terra",
    "openai/gpt-5.6-luna",
    "minimax/MiniMax-M3",
    "minimax/MiniMax-M2.5",
    "vertex_ai/gemini-3.1-pro-preview",
    "vertex_ai/gemini-3.8-flash",
    "vertex_ai/gemini-3.7-flash",
    "xai/grok-4.7",
    "xai/grok-4.6",
    "xai/grok-4.3",
    "meta/muse-spark-1.3",
    "meta/muse-spark-1.2",
    SPECIAL_TRAINING_MODEL_NAME,
)

MODEL_FAMILIES: tuple[ModelFamily, ...] = (
    ModelFamily(
        prefix=SPECIAL_TRAINING_MODEL_PREFIX,
        description="Training checkpoint; the training backend brokers inference.",
    ),
    ModelFamily(prefix="together_ai/", description="Any model Together AI hosts."),
    ModelFamily(prefix="fireworks_ai/", description="Any model Fireworks AI hosts."),
)


def catalog() -> ModelCatalog:
    """Every named model with its capabilities, plus the open-ended families."""
    return ModelCatalog(
        models=tuple(spec_for(model) for model in CATALOG_MODEL_IDS),
        families=MODEL_FAMILIES,
    )
