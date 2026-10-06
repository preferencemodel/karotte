"""A loopback HTTP forwarder from a CLI agent to its model provider.

The student firewall lets a CLI agent reach only localhost, the sandbox's own
addresses and the model proxy. Without a proxy, the harness runs this
forwarder in its own (root) process, which the firewall doesn't restrict: the
agent talks to it on loopback, and it relays each request to the provider with
the real API key. The agent is handed a placeholder key, so the key never
reaches the student.
"""

import socket
from typing import Final

import aiohttp
from aiohttp import web
from loguru import logger

_HOP_BY_HOP: Final = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)

_DROPPED_REQUEST_HEADERS: Final = _HOP_BY_HOP | {
    "host",
    "content-length",
    "authorization",
    "x-api-key",
}

_DROPPED_RESPONSE_HEADERS: Final = _HOP_BY_HOP | {"content-length"}


class ModelForwarder:
    """Relays every request under its URL to ``upstream`` with ``api_key``.

    ``upstream`` is the provider's API root without ``/v1``, e.g.
    ``https://api.mistral.ai``, like a ``--proxy`` URL; a request for
    ``<url>/v1/chat/completions`` goes to ``<upstream>/v1/chat/completions``.
    Responses stream back as they arrive, so server-sent events pass through.
    """

    def __init__(self, upstream: str, api_key: str) -> None:
        self._upstream: str = upstream.rstrip("/")
        self._api_key: str = api_key
        self._session: aiohttp.ClientSession | None = None
        self._runner: web.AppRunner | None = None
        self.url: str = ""

    async def start(self) -> str:
        """Listen on an unused loopback port; returns the forwarder's URL."""
        # Bodies pass through byte for byte, compressed or not.
        self._session = aiohttp.ClientSession(
            auto_decompress=False, timeout=aiohttp.ClientTimeout(total=None)
        )
        app = web.Application(client_max_size=0)
        app.router.add_route("*", "/{tail:.*}", self._forward)
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        await web.SockSite(self._runner, sock).start()
        port: int = sock.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        logger.info(
            "Forwarding model calls from {} to {}; the agent gets a placeholder key",
            self.url,
            self._upstream,
        )
        return self.url

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def _forward(self, request: web.Request) -> web.StreamResponse:
        assert self._session is not None
        headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower() not in _DROPPED_REQUEST_HEADERS
        }
        headers["Authorization"] = f"Bearer {self._api_key}"
        try:
            upstream = await self._session.request(
                request.method,
                self._upstream + request.rel_url.path_qs,
                headers=headers,
                data=request.content if request.body_exists else None,
                allow_redirects=False,
            )
        except aiohttp.ClientError as e:
            logger.warning("Model forwarder could not reach {}: {}", self._upstream, e)
            return web.Response(status=502, text=f"karotte model forwarder: {e}")

        async with upstream:
            response = web.StreamResponse(
                status=upstream.status, reason=upstream.reason
            )
            for k, v in upstream.headers.items():
                if k.lower() not in _DROPPED_RESPONSE_HEADERS:
                    response.headers.add(k, v)
            await response.prepare(request)
            try:
                async for chunk in upstream.content.iter_any():
                    await response.write(chunk)
                await response.write_eof()
            except ConnectionResetError:
                # The agent hung up, e.g. once it had read the end of a stream
                # it cares about; drop the rest of the upstream response.
                pass
        return response
