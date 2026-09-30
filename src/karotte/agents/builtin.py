from typing import final

from karotte.agents.builtin_source import BuiltinSource
from karotte.agents.message_loop import MessageLoopAgent
from karotte.schemas.evaluation_run_config import EvaluationRunConfig


@final
class BuiltinAgent(MessageLoopAgent):
    """karotte drives the model itself via the litellm API loop."""

    def __init__(self, config: EvaluationRunConfig) -> None:
        super().__init__(BuiltinSource(config))
