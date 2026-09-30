"""Schemas and data models for karotte.

Submodule imports are deferred so that ``import karotte.schemas`` is cheap.
"""

from typing import TYPE_CHECKING, Literal

type RunStatus = Literal["pending", "running", "passed", "failed", "error"]

# Static declarations so that type checkers and ``__all__`` are satisfied,
# while the actual imports are deferred until first attribute access.
if TYPE_CHECKING:
    from .chat import (
        ChatCompletionMessageToolCall as ChatCompletionMessageToolCall,
    )
    from .chat import Function as Function
    from .chat import Message as Message
    from .data_mount import DataMount as DataMount
    from .data_mount import MountType as MountType
    from .environment_template import EnvironmentTemplate as EnvironmentTemplate
    from .evaluation_run_config import (
        EvaluationRunConfig as EvaluationRunConfig,
    )
    from .http_mcp_server_config import (
        HttpMcpServerConfig as HttpMcpServerConfig,
    )
    from .scoring import Metadata as Metadata
    from .scoring import Score as Score
    from .scoring import Scoring as Scoring

# Lazy attribute map: name -> (module, attribute)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "ChatCompletionMessageToolCall": (".chat", "ChatCompletionMessageToolCall"),
    "Function": (".chat", "Function"),
    "Message": (".chat", "Message"),
    "EnvironmentTemplate": (".environment_template", "EnvironmentTemplate"),
    "EvaluationRunConfig": (".evaluation_run_config", "EvaluationRunConfig"),
    "HttpMcpServerConfig": (".http_mcp_server_config", "HttpMcpServerConfig"),
    "DataMount": (".data_mount", "DataMount"),
    "MountType": (".data_mount", "MountType"),
    "Metadata": (".scoring", "Metadata"),
    "Score": (".scoring", "Score"),
    "Scoring": (".scoring", "Scoring"),
}


def __getattr__(name: str) -> object:
    if name in _LAZY_IMPORTS:
        import importlib

        module_path, attr = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path, __name__)
        value = getattr(module, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ChatCompletionMessageToolCall",
    "DataMount",
    "EnvironmentTemplate",
    "EvaluationRunConfig",
    "Function",
    "HttpMcpServerConfig",
    "Message",
    "Metadata",
    "MountType",
    "RunStatus",
    "Score",
    "Scoring",
]
