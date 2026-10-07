"""The loopback forwarder a CLI agent reaches its model provider through."""

import asyncio
import logging
import socket
from collections.abc import AsyncIterator

import aiohttp
import pytest
import pytest_asyncio
from aiohttp import web

from karotte.model_forwarder import ModelForwarder

_KEY = "sk-real"


async def _echo(request: web.Request) -> web.Response:
    return web.json_response(
        {
            "method": request.method,
            "path_qs": request.rel_url.path_qs,
            "authorization": request.headers.get("Authorization"),
            "x_api_key": request.headers.get("x-api-key"),
            "custom": request.headers.get("x-custom"),
            "body": (await request.read()).decode(),
        },
        status=201,
    )


async def _endless(request: web.Request) -> web.StreamResponse:
    response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await response.prepare(request)
    try:
        while True:
            await response.write(b"data: x\n\n")
            await asyncio.sleep(0.01)
    except ConnectionResetError:
        # The forwarder hung up after its agent did.
        return response


async def _stream(request: web.Request) -> web.StreamResponse:
    response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await response.prepare(request)
    for i in range(3):
        await response.write(f"data: {i}\n\n".encode())
    await response.write_eof()
    return response


@pytest_asyncio.fixture
async def upstream() -> AsyncIterator[str]:
    app = web.Application()
    app.router.add_route("*", "/v1/echo", _echo)
    app.router.add_get("/v1/stream", _stream)
    app.router.add_get("/v1/endless", _endless)
    runner = web.AppRunner(app)
    await runner.setup()
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    await web.SockSite(runner, sock).start()
    yield f"http://127.0.0.1:{sock.getsockname()[1]}"
    await runner.cleanup()


@pytest_asyncio.fixture
async def forwarder(upstream: str) -> AsyncIterator[ModelForwarder]:
    forwarder = ModelForwarder(upstream, _KEY)
    _ = await forwarder.start()
    yield forwarder
    await forwarder.stop()


class TestModelForwarder:
    @pytest.mark.asyncio
    async def test_listens_on_loopback(self, forwarder: ModelForwarder):
        assert forwarder.url.startswith("http://127.0.0.1:")

    @pytest.mark.asyncio
    async def test_relays_with_the_real_key(self, forwarder: ModelForwarder):
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{forwarder.url}/v1/echo?a=1",
                data=b'{"model": "m"}',
                headers={
                    "Authorization": "Bearer model_api_key",
                    "x-api-key": "model_api_key",
                    "x-custom": "kept",
                },
            ) as response:
                assert response.status == 201
                seen = await response.json()
        assert seen == {
            "method": "POST",
            "path_qs": "/v1/echo?a=1",
            "authorization": f"Bearer {_KEY}",
            "x_api_key": None,
            "custom": "kept",
            "body": '{"model": "m"}',
        }

    @pytest.mark.asyncio
    async def test_streams_server_sent_events(self, forwarder: ModelForwarder):
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{forwarder.url}/v1/stream") as response:
                assert response.headers["Content-Type"] == "text/event-stream"
                body = await response.text()
        assert body == "data: 0\n\ndata: 1\n\ndata: 2\n\n"

    @pytest.mark.asyncio
    async def test_agent_hanging_up_mid_stream_is_not_an_error(
        self, forwarder: ModelForwarder, caplog: pytest.LogCaptureFixture
    ):
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{forwarder.url}/v1/endless") as response:
                _ = await response.content.readany()
            await asyncio.sleep(0.2)
            async with session.get(f"{forwarder.url}/v1/stream") as response:
                assert response.status == 200
        assert not [r for r in caplog.records if r.levelno >= logging.ERROR]

    @pytest.mark.asyncio
    async def test_passes_upstream_errors_through(self, forwarder: ModelForwarder):
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{forwarder.url}/v1/missing") as response:
                assert response.status == 404

    @pytest.mark.asyncio
    async def test_unreachable_upstream_is_a_502(self):
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        closed_port = sock.getsockname()[1]
        sock.close()
        forwarder = ModelForwarder(f"http://127.0.0.1:{closed_port}", _KEY)
        url = await forwarder.start()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{url}/v1/models") as response:
                    assert response.status == 502
                    assert _KEY not in await response.text()
        finally:
            await forwarder.stop()
