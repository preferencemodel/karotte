from typing import Any, Literal

from pydantic import BaseModel

type Role = Literal["assistant", "user", "system", "tool", "function"]


class Function(BaseModel):
    name: str | None
    arguments: str = ""


class ChatCompletionDeltaToolCall(BaseModel):
    id: str | None
    function: Function
    type: str | None
    index: int


class Delta(BaseModel):
    content: str | None
    role: Role | None
    tool_calls: list[ChatCompletionDeltaToolCall] | None
    reasoning_content: str | None = None
    # add fields as necessary from litellm.types.utils.Delta


class ChatCompletionMessageToolCall(BaseModel):
    id: str
    function: Function
    type: str


class Message(BaseModel):
    content: str | list[dict[str, Any]] | None = None
    role: Role = "assistant"
    tool_calls: list[ChatCompletionMessageToolCall] | None = None
    reasoning_content: str | None = None
    tool_call_id: str | None = None
    # add fields as necessary from litellm.Message
