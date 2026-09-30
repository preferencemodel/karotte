"""Shared model/credential resolution for agents.

CLI agents need a provider-native model name plus credential environment
variables, whereas the builtin litellm loop consumes the raw ``model`` string
directly. This module is the single place that maps a litellm-style model
string (e.g. ``mistral/mistral-medium-3.5``) and API key to what a CLI agent's
subprocess needs, so each agent doesn't reinvent prefix-stripping and
key-to-env-var mapping.
"""

from dataclasses import dataclass, field

from karotte.model_spec import PROVIDER_API_KEY_ENV


@dataclass(frozen=True)
class ResolvedModel:
    """A model string resolved for handing to a CLI agent subprocess."""

    provider: str
    """The provider prefix (e.g. ``mistral``), or ``""`` if none was given."""

    model: str
    """The provider-native model name, with any ``provider/`` prefix stripped."""

    env: dict[str, str] = field(default_factory=dict)
    """Credential environment variables to set for the agent process."""

    key_env: str | None = None
    """Name of the env var the provider's CLI reads its API key from, if known."""


def resolve_model(model: str, api_key: str | None) -> ResolvedModel:
    """Split a litellm-style ``model`` string and place ``api_key`` in the env
    var the resolved provider's CLI expects.

    A model without a ``provider/`` prefix yields an empty provider and no
    credential env var (the caller supplies auth another way)."""
    provider, _, rest = model.partition("/")
    if not rest:
        return ResolvedModel(provider="", model=model)

    key_env = PROVIDER_API_KEY_ENV.get(provider)
    env: dict[str, str] = {}
    if key_env and api_key is not None:
        env[key_env] = api_key

    return ResolvedModel(provider=provider, model=rest, env=env, key_env=key_env)
