"""Pluggable agents for the evaluation runner.

An `Agent` produces a run's transcript one step at a time; the runner keeps the
task lifecycle (hooks, judging, artifacts, event bookkeeping). Builtin and
external agents share `MessageLoopAgent`, differing only in their
`MessageSource` -- builtin drives the litellm API loop itself, external fetches
messages from the backend (training rollouts). Fake replays env-authored
messages.
"""

from karotte.agents.agent import (
    Agent,
    EmptyTurnLimitReachedError,
    RunContext,
    TurnLimitReachedError,
)
from karotte.agents.backend_source import EXTERNAL_MESSAGE_TIMEOUT_S, BackendSource
from karotte.agents.builtin import BuiltinAgent
from karotte.agents.builtin_source import BuiltinSource
from karotte.agents.cli_agent import (
    CliAgent,
    cli_agent_types,
    get_cli_agent_type,
    register_cli_agent,
)
from karotte.agents.external import ExternalAgent
from karotte.agents.fake_source import FakeSource
from karotte.agents.message_loop import MessageLoopAgent
from karotte.agents.message_source import MessageSource

# Import agent modules for their `register_cli_agent` side effects.
from karotte.agents.mistral_vibe import MistralVibeAgent

__all__ = [
    "EXTERNAL_MESSAGE_TIMEOUT_S",
    "Agent",
    "BackendSource",
    "BuiltinAgent",
    "BuiltinSource",
    "CliAgent",
    "EmptyTurnLimitReachedError",
    "ExternalAgent",
    "FakeSource",
    "MessageLoopAgent",
    "MessageSource",
    "MistralVibeAgent",
    "RunContext",
    "TurnLimitReachedError",
    "cli_agent_types",
    "get_cli_agent_type",
    "register_cli_agent",
]
