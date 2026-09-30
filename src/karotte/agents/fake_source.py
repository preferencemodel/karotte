from collections.abc import AsyncGenerator
from typing import final

from litellm import ChatCompletionToolParam

from karotte.schemas.chat import Message
from karotte.schemas.transcript import Event, MessageAddedEvent


@final
class FakeSource:
    """Replays a fixed list of environment-authored messages, one per turn."""

    def __init__(self, messages: list[Message]) -> None:
        self._messages = messages
        self._i_message = 0

    async def collect(
        self,
        messages: list[Message],  # pyright: ignore[reportUnusedParameter]
        tools: list[ChatCompletionToolParam],  # pyright: ignore[reportUnusedParameter]
    ) -> AsyncGenerator[Event]:
        if self._i_message >= len(self._messages):
            raise RuntimeError("Not enough messages defined for fake model.")

        yield MessageAddedEvent(message=self._messages[self._i_message])
        self._i_message += 1
