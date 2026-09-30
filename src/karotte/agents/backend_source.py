import time
from collections.abc import AsyncGenerator
from typing import final

from litellm import ChatCompletionToolParam
from loguru import logger

from karotte.backend_client import BackendClient
from karotte.schemas.chat import Message
from karotte.schemas.transcript import Event

# Maximum time to wait for a message from the model before giving up.
EXTERNAL_MESSAGE_TIMEOUT_S = 120 * 60


@final
class BackendSource:
    """Long-polls the backend for the next message produced by an external model
    (training rollouts). karotte still drives the step loop and executes the tool
    calls the message contains."""

    def __init__(self, backend_client: BackendClient, run_id: str) -> None:
        self.backend_client = backend_client
        self.run_id = run_id

    async def collect(
        self,
        messages: list[Message],  # pyright: ignore[reportUnusedParameter]
        tools: list[ChatCompletionToolParam],  # pyright: ignore[reportUnusedParameter]
    ) -> AsyncGenerator[Event]:
        start = time.time()
        last_log = time.time()
        while True:
            elapsed = time.time() - start
            if elapsed > EXTERNAL_MESSAGE_TIMEOUT_S:
                raise TimeoutError(
                    f"Timed out after {elapsed:.0f}s waiting for a response from the model"
                )
            if time.time() - last_log > 30:
                logger.warning(f"Still waiting for new message, {elapsed:.0f}s elapsed")
                last_log = time.time()
            # get_message is a long-poll operation so we don't need extra sleep
            msg_added_evt = await self.backend_client.get_message(self.run_id)
            if msg_added_evt is not None:
                yield msg_added_evt
                await self.backend_client.delete_message(self.run_id)
                return
