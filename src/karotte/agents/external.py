from typing import final

from karotte.agents.backend_source import BackendSource
from karotte.agents.message_loop import MessageLoopAgent
from karotte.backend_client import BackendClient


@final
class ExternalAgent(MessageLoopAgent):
    """An external model produces the messages; karotte fetches them from the
    backend and still drives the step loop and tool execution."""

    def __init__(self, backend_client: BackendClient, run_id: str) -> None:
        super().__init__(BackendSource(backend_client, run_id))
