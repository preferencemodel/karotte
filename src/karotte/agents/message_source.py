from collections.abc import AsyncGenerator
from typing import Protocol

from litellm import ChatCompletionToolParam

from karotte.schemas.chat import Message
from karotte.schemas.transcript import Event


class MessageSource(Protocol):
    """Produces the model's next response as a stream of transcript events.

    The runner passes the current conversation and tool schemas each turn; a
    source is otherwise responsible only for talking to its backend. Yielded
    events are raw -- the runner applies transcript / run-state bookkeeping.
    """

    def collect(
        self,
        messages: list[Message],
        tools: list[ChatCompletionToolParam],
    ) -> AsyncGenerator[Event]: ...
