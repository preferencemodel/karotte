"""The effort karotte resolves must survive litellm and reach the request body."""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import litellm
import pytest
from litellm import CustomStreamWrapper

from karotte.model_spec import spec_for
from karotte.providers import provider_for

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "end_turn",
            "description": "End the turn",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]


def _capture_request_body(model: str, effort: str) -> dict[str, Any]:
    bodies: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("content-length", 0))
            bodies.append(json.loads(self.rfile.read(n)))
            self.send_response(500)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":{"message":"capture"}}')

        def log_message(self, format: str, *args: Any) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        spec = spec_for(model)
        params: dict[str, Any] = {
            "stream": True,
            "stream_options": {"include_usage": True},
            "model": spec.litellm_model,
            "messages": [{"role": "user", "content": "Go."}],
            "tools": _TOOLS,
            "tool_choice": "auto",
            "max_tokens": spec.max_output_tokens,
            "timeout": 5,
            "allowed_openai_params": [
                "tools",
                "tool_choice",
                *spec.extra_allowed_openai_params,
            ],
            "api_key": "test_key",
            "api_base": f"http://127.0.0.1:{server.server_port}",
        }
        provider_for(spec).apply_reasoning(spec, effort, params)

        async def go():
            response = await litellm.acompletion(**params)
            assert isinstance(response, CustomStreamWrapper)
            async for _ in response:
                pass

        litellm.num_retries = 0
        with pytest.raises(Exception):
            asyncio.run(go())
    finally:
        server.shutdown()
    assert bodies, "no request reached the capture server"
    return bodies[0]


@pytest.mark.parametrize(
    ("model", "effort"),
    [
        ("openai/gpt-5.6-sol", "low"),
        ("openai/gpt-5.6-sol", "medium"),
        ("openai/gpt-5.6-sol", "max"),
        ("openai/gpt-5.6", "max"),
        ("openai/gpt-6-astra", "max"),
        ("openai/gpt-6-sol", "max"),
        ("openai/gpt-6-luna", "max"),
        ("meta/muse-spark-1.3", "max"),
    ],
)
def test_responses_body_carries_effort(model: str, effort: str):
    body = _capture_request_body(model, effort)
    assert body["reasoning"]["effort"] == effort


@pytest.mark.parametrize(
    ("model", "effort"),
    [
        ("xai/grok-4.7", "xhigh"),
        ("xai/grok-4.6", "xhigh"),
        ("together_ai/Qwen/Qwen3.8-2.4T-A95B", "xhigh"),
        ("together_ai/moonshotai/Kimi-K3", "max"),
    ],
)
def test_chat_body_carries_effort(model: str, effort: str):
    body = _capture_request_body(model, effort)
    assert body["reasoning_effort"] == effort
