"""A loopback HTTP forwarder from a CLI agent to its model provider.

The student firewall lets a CLI agent reach only localhost, the sandbox's own
addresses and the model proxy. Without a proxy, the harness runs this
forwarder in its own (root) process, which the firewall doesn't restrict: the
agent talks to it on loopback, and it relays each request to the provider with
the real API key. The agent is handed a placeholder key, so the key never
reaches the student. An agent with ``proxy_via_forwarder`` reaches a proxy this
way too when the run has a key.
"""

import asyncio
import json
import os
import re
import socket
import zlib
from pathlib import Path
from typing import TYPE_CHECKING, Final

import aiohttp
from aiohttp import web
from loguru import logger

if TYPE_CHECKING:
    from google.auth.credentials import Credentials

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
    "x-goog-api-key",
}

_DROPPED_RESPONSE_HEADERS: Final = _HOP_BY_HOP | {"content-length"}

_LOGGED_ERROR_BYTES: Final = 2000
"""How much of an error response's body the forwarder logs."""


class ModelForwarder:
    """Relays every request under its URL to ``upstream`` with ``api_key``.

    ``upstream`` is the provider's API root without ``/v1``, e.g.
    ``https://api.mistral.ai``, like a ``--proxy`` URL; a request for
    ``<url>/v1/chat/completions`` goes to ``<upstream>/v1/chat/completions``.
    Responses stream back as they arrive, so server-sent events pass through.
    """

    def __init__(self, upstream: str, api_key: str) -> None:
        self.upstream: str = upstream.rstrip("/")
        self._api_key: str = api_key
        self._session: aiohttp.ClientSession | None = None
        self._runner: web.AppRunner | None = None
        self.url: str = ""

    async def start(self) -> str:
        """Listen on an unused loopback port; returns the forwarder's URL."""
        # Bodies pass through byte for byte, compressed or not, so the agent
        # alone says which encodings it accepts. aiohttp would otherwise ask
        # for gzip itself, and an agent that never asked can't decode it.
        self._session = aiohttp.ClientSession(
            auto_decompress=False,
            skip_auto_headers=("Accept-Encoding",),
            timeout=aiohttp.ClientTimeout(total=None),
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
            self.upstream,
        )
        return self.url

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
        if self._session is not None:
            await self._session.close()
            self._session = None

    def cli_env(self) -> dict[str, str]:
        """Extra environment for the CLI agent this forwarder serves."""
        return {}

    def step_artifacts(self) -> list[Path]:
        """Files to save as artifacts after each step."""
        return []

    def _target(self, request: web.Request) -> str | None:
        """Where ``request`` goes upstream, or None to refuse it."""
        return self.upstream + request.rel_url.path_qs

    async def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    async def _forward(self, request: web.Request) -> web.StreamResponse:
        assert self._session is not None
        target = self._target(request)
        if target is None:
            logger.warning(
                "Model forwarder refused {} {}", request.method, request.rel_url.path
            )
            return web.Response(status=403, text="karotte model forwarder: refused")
        headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower() not in _DROPPED_REQUEST_HEADERS
        }
        headers.update(await self._auth_headers())
        try:
            upstream = await self._session.request(
                request.method,
                target,
                headers=headers,
                data=request.content if request.body_exists else None,
                allow_redirects=False,
            )
        except aiohttp.ClientError as e:
            logger.warning("Model forwarder could not reach {}: {}", self.upstream, e)
            return web.Response(status=502, text=f"karotte model forwarder: {e}")

        async with upstream:
            response = web.StreamResponse(
                status=upstream.status, reason=upstream.reason
            )
            for k, v in upstream.headers.items():
                if k.lower() not in _DROPPED_RESPONSE_HEADERS:
                    response.headers.add(k, v)
            await response.prepare(request)
            # Only the agent sees an error response, and it may not show why
            # the call failed, so the start of the body is logged as well.
            error_body = bytearray()
            try:
                async for chunk in upstream.content.iter_any():
                    if upstream.status >= 400:
                        error_body += chunk[: _LOGGED_ERROR_BYTES - len(error_body)]
                    await response.write(chunk)
                await response.write_eof()
            except ConnectionResetError:
                # The agent hung up, e.g. once it had read the end of a stream
                # it cares about; drop the rest of the upstream response.
                pass
            if upstream.status >= 400:
                logger.warning(
                    "Model forwarder: {} {} got {} {}: {}",
                    request.method,
                    request.rel_url.path,
                    upstream.status,
                    upstream.reason,
                    _error_text(
                        bytes(error_body), upstream.headers.get("Content-Encoding")
                    ),
                )
        return response


def _error_text(body: bytes, encoding: str | None) -> str:
    """The start of an error response's body, for the log.

    ``body`` is the start of the body as the provider sent it, so a gzip or
    deflate stream is cut short; decompress what is there. Any other encoding
    is reported by name.
    """
    if encoding in ("gzip", "deflate"):
        try:
            # wbits 47 reads either a gzip or a zlib header.
            body = zlib.decompressobj(wbits=47).decompress(body)
        except zlib.error:
            return f"<{len(body)} bytes of {encoding}>"
    elif encoding not in (None, "identity"):
        return f"<{len(body)} bytes of {encoding}>"
    return body[:_LOGGED_ERROR_BYTES].decode(errors="replace")


class GeminiForwarder(ModelForwarder):
    """Sends the key the way the Gemini API expects it."""

    async def _auth_headers(self) -> dict[str, str]:
        return {"x-goog-api-key": self._api_key}


_GEMINI_API_GENERATE: Final = re.compile(
    r"/v1beta/models/([\w.-]+):(generateContent|streamGenerateContent)"
)


class VertexForwarder(ModelForwarder):
    """Turns Gemini API calls into Vertex AI calls, authorized with the
    harness's own Google credentials."""

    def __init__(
        self,
        project: str,
        location: str,
        credentials: "Credentials",
        *,
        upstream: str | None = None,
    ) -> None:
        host = (
            "https://aiplatform.googleapis.com"
            if location == "global"
            else f"https://{location}-aiplatform.googleapis.com"
        )
        super().__init__(upstream or host, "")
        self.project: str = project
        self.location: str = location
        self._credentials: Credentials = credentials
        self._refreshing: asyncio.Lock = asyncio.Lock()

    def _target(self, request: web.Request) -> str | None:
        """Only content generation: the credentials reach all of Vertex AI."""
        match = _GEMINI_API_GENERATE.fullmatch(request.rel_url.path)
        if request.method != "POST" or match is None:
            return None
        model, method = match.groups()
        query = request.rel_url.raw_query_string
        location = f"/v1/projects/{self.project}/locations/{self.location}"
        target = f"{self.upstream}{location}/publishers/google/models/{model}:{method}"
        return f"{target}?{query}" if query else target

    async def _auth_headers(self) -> dict[str, str]:
        async with self._refreshing:
            if not self._credentials.valid:
                from google.auth.transport.requests import Request

                await asyncio.to_thread(self._credentials.refresh, Request())
        headers: dict[str, str] = {}
        self._credentials.apply(headers)
        return headers


def vertex_forwarder() -> VertexForwarder:
    """A forwarder with the credentials, project and location litellm would use
    for Vertex AI, except that the location defaults to ``global``."""
    try:
        import google.auth
    except ImportError as e:
        msg = "Vertex AI models need google-auth; install karotte[gcs]."
        raise RuntimeError(msg) from e

    scopes = ["https://www.googleapis.com/auth/cloud-platform"]
    if raw := os.environ.get("VERTEXAI_CREDENTIALS"):
        info = json.loads(Path(raw).read_text() if os.path.exists(raw) else raw)
        credentials, project = google.auth.load_credentials_from_dict(
            info, scopes=scopes
        )
    else:
        credentials, project = google.auth.default(scopes=scopes)
    project = os.environ.get("VERTEXAI_PROJECT") or project
    if not project:
        msg = "No Google Cloud project for Vertex AI; set VERTEXAI_PROJECT."
        raise ValueError(msg)
    location = (
        os.environ.get("VERTEXAI_LOCATION")
        or os.environ.get("VERTEX_LOCATION")
        or "global"
    )
    return VertexForwarder(project, location, credentials)
