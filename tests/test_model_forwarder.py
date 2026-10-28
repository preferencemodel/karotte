"""The loopback forwarder a CLI agent reaches its model provider through."""

import asyncio
import gzip
import json
import logging
import socket
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import TYPE_CHECKING, cast

import aiohttp
import pytest
import pytest_asyncio
from aiohttp import web
from loguru import logger
from yarl import URL

if TYPE_CHECKING:
    from google.auth.credentials import Credentials

from karotte.model_forwarder import (
    GeminiForwarder,
    ModelForwarder,
    VertexForwarder,
    _error_text,  # pyright: ignore[reportPrivateUsage]
    vertex_forwarder,
)

_KEY = "sk-real"


async def _echo(request: web.Request) -> web.Response:
    return web.json_response(
        {
            "method": request.method,
            "path_qs": request.rel_url.path_qs,
            "authorization": request.headers.get("Authorization"),
            "x_api_key": request.headers.get("x-api-key"),
            "goog_api_key": request.headers.get("x-goog-api-key"),
            "user_project": request.headers.get("x-goog-user-project"),
            "custom": request.headers.get("x-custom"),
            "body": (await request.read()).decode(),
        },
        status=201,
    )


async def _accept_encoding(request: web.Request) -> web.Response:
    return web.json_response(request.headers.get("Accept-Encoding"))


async def _rate_limited(request: web.Request) -> web.Response:
    del request
    return web.json_response({"error": "slow down"}, status=429)


async def _rate_limited_gzip(request: web.Request) -> web.Response:
    del request
    return web.Response(
        body=gzip.compress(b'{"error": "slow down"}'),
        status=429,
        headers={"Content-Encoding": "gzip", "Content-Type": "application/json"},
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
    app.router.add_get("/v1/accept-encoding", _accept_encoding)
    app.router.add_post("/v1/limited", _rate_limited)
    app.router.add_post("/v1/limited-gzip", _rate_limited_gzip)
    app.router.add_route("*", "/v1beta/{tail:.*}", _echo)
    app.router.add_route("*", "/v1/projects/{tail:.*}", _echo)
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
            "goog_api_key": None,
            "user_project": None,
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
    @pytest.mark.parametrize("sent", [None, "br"])
    async def test_asks_upstream_only_for_the_agents_encodings(
        self, forwarder: ModelForwarder, sent: str | None
    ):
        headers = {} if sent is None else {"Accept-Encoding": sent}
        async with aiohttp.ClientSession(
            skip_auto_headers=("Accept-Encoding",)
        ) as session:
            async with session.get(
                f"{forwarder.url}/v1/accept-encoding", headers=headers
            ) as response:
                assert await response.json() == sent

    @pytest.mark.asyncio
    async def test_passes_upstream_errors_through(self, forwarder: ModelForwarder):
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{forwarder.url}/v1/missing") as response:
                assert response.status == 404

    @pytest.mark.asyncio
    @pytest.mark.parametrize("path", ["/v1/limited", "/v1/limited-gzip"])
    async def test_logs_why_upstream_refused(
        self, forwarder: ModelForwarder, path: str
    ):
        warnings: list[str] = []
        handler = logger.add(warnings.append, level="WARNING", format="{message}")
        try:
            async with aiohttp.ClientSession(auto_decompress=False) as session:
                async with session.post(f"{forwarder.url}{path}") as response:
                    assert response.status == 429
                async with session.post(f"{forwarder.url}/v1/echo") as response:
                    assert response.status == 201
        finally:
            logger.remove(handler)
        reason = '{"error": "slow down"}'
        assert warnings == [
            f"Model forwarder: POST {path} got 429 Too Many Requests: {reason}\n"
        ]

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


@pytest.mark.parametrize(
    ("body", "encoding", "text"),
    [
        (b"plain", None, "plain"),
        # Without the gzip trailer, as when the log takes only the start.
        (gzip.compress(b"cut short")[:-8], "gzip", "cut short"),
        (b"not gzip", "gzip", "<8 bytes of gzip>"),
        (b"\x8b\x00", "br", "<2 bytes of br>"),
    ],
    ids=["plain", "gzip-cut-short", "not-gzip", "br"],
)
def test_error_text(body: bytes, encoding: str | None, text: str):
    assert _error_text(body, encoding) == text


class _Credentials:
    """Stands in for google-auth credentials: each refresh mints a new token."""

    def __init__(self) -> None:
        self.token: str | None = None
        self.refreshes: int = 0
        self.quota_project_id: str = "quota-project"

    @property
    def valid(self) -> bool:
        return self.token is not None

    def refresh(self, request: object) -> None:
        del request
        self.refreshes += 1
        self.token = f"token-{self.refreshes}"

    def apply(self, headers: dict[str, str]) -> None:
        headers["authorization"] = f"Bearer {self.token}"
        headers["x-goog-user-project"] = self.quota_project_id


def _as_credentials(credentials: _Credentials) -> "Credentials":
    return cast("Credentials", cast(object, credentials))


def _default(
    credentials: _Credentials, project: str | None
) -> Callable[..., tuple["Credentials", str | None]]:
    def default(scopes: list[str]) -> tuple["Credentials", str | None]:
        del scopes
        return _as_credentials(credentials), project

    return default


async def _post(url: str) -> dict[str, object]:
    async with aiohttp.ClientSession() as session:
        async with session.post(
            url,
            data=b"{}",
            headers={"x-goog-api-key": "placeholder", "Authorization": "Bearer x"},
        ) as response:
            assert response.status == 201
            return await response.json()


class TestGeminiForwarder:
    @pytest.mark.asyncio
    async def test_sends_the_real_key_as_a_gemini_api_key(self, upstream: str):
        forwarder = GeminiForwarder(upstream, _KEY)
        url = await forwarder.start()
        try:
            seen = await _post(
                f"{url}/v1beta/models/gemini-3.8-flash:streamGenerateContent?alt=sse"
            )
        finally:
            await forwarder.stop()
        assert seen["goog_api_key"] == _KEY
        assert seen["authorization"] is None
        assert seen["path_qs"] == (
            "/v1beta/models/gemini-3.8-flash:streamGenerateContent?alt=sse"
        )


class TestVertexForwarder:
    @pytest.mark.asyncio
    async def test_turns_gemini_api_calls_into_vertex_calls(self, upstream: str):
        credentials = _Credentials()
        forwarder = VertexForwarder(
            "proj",
            "global",
            _as_credentials(credentials),
            upstream=upstream,
        )
        url = await forwarder.start()
        try:
            seen = await _post(
                f"{url}/v1beta/models/gemini-4-argon:streamGenerateContent?alt=sse"
            )
            _ = await _post(f"{url}/v1beta/models/gemini-4-argon:generateContent")
        finally:
            await forwarder.stop()
        assert seen["path_qs"] == (
            "/v1/projects/proj/locations/global/publishers/google/models/"
            "gemini-4-argon:streamGenerateContent?alt=sse"
        )
        assert seen["authorization"] == "Bearer token-1"
        assert seen["user_project"] == "quota-project"
        assert seen["goog_api_key"] is None
        assert credentials.refreshes == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("method", "path"),
        [
            ("DELETE", "/v1/projects/other/locations/global/endpoints/1"),
            ("POST", "/v1/projects/other/locations/global/customJobs"),
            ("GET", "/v1beta/models"),
            ("GET", "/v1beta/models/gemini-4-argon:generateContent"),
            ("POST", "/v1beta/models/gemini-4-argon:predict"),
            ("POST", "/v1beta/models/a/b:generateContent"),
            ("POST", "/v1beta/models/..%2F..%2Fendpoints%2F1:generateContent"),
        ],
    )
    async def test_refuses_anything_but_generating_content(
        self, upstream: str, method: str, path: str
    ):
        credentials = _Credentials()
        forwarder = VertexForwarder(
            "proj", "global", _as_credentials(credentials), upstream=upstream
        )
        url = await forwarder.start()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method, URL(f"{url}{path}", encoded=True)
                ) as response:
                    assert response.status == 403
        finally:
            await forwarder.stop()
        assert credentials.refreshes == 0

    @pytest.mark.parametrize(
        ("location", "host"),
        [
            ("global", "https://aiplatform.googleapis.com"),
            ("us-central1", "https://us-central1-aiplatform.googleapis.com"),
        ],
    )
    def test_calls_the_locations_endpoint(self, location: str, host: str):
        forwarder = VertexForwarder("proj", location, _as_credentials(_Credentials()))
        assert forwarder.upstream == host


class TestVertexForwarderFromEnv:
    @pytest.fixture(autouse=True)
    def no_vertex_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in (
            "VERTEXAI_PROJECT",
            "VERTEXAI_LOCATION",
            "VERTEX_LOCATION",
            "VERTEXAI_CREDENTIALS",
        ):
            monkeypatch.delenv(name, raising=False)

    def test_uses_default_credentials_and_their_project(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        credentials = _Credentials()
        monkeypatch.setattr("google.auth.default", _default(credentials, "adc-project"))
        forwarder = vertex_forwarder()
        assert (forwarder.project, forwarder.location) == ("adc-project", "global")

    def test_env_overrides_project_and_location(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            "google.auth.default", _default(_Credentials(), "adc-project")
        )
        monkeypatch.setenv("VERTEXAI_PROJECT", "env-project")
        monkeypatch.setenv("VERTEX_LOCATION", "europe-west4")
        forwarder = vertex_forwarder()
        assert (forwarder.project, forwarder.location) == (
            "env-project",
            "europe-west4",
        )

    @pytest.mark.parametrize("as_file", [False, True])
    def test_reads_credentials_from_vertexai_credentials(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, as_file: bool
    ):
        info = {"type": "service_account", "project_id": "sa-project"}
        loaded: list[dict[str, str]] = []

        def load(info: dict[str, str], scopes: list[str]) -> tuple[object, str]:
            del scopes
            loaded.append(info)
            return _Credentials(), info["project_id"]

        monkeypatch.setattr("google.auth.load_credentials_from_dict", load)
        value = json.dumps(info)
        if as_file:
            (tmp_path / "sa.json").write_text(value)
            value = str(tmp_path / "sa.json")
        monkeypatch.setenv("VERTEXAI_CREDENTIALS", value)
        assert vertex_forwarder().project == "sa-project"
        assert loaded == [info]

    def test_without_a_project_says_how_to_set_one(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr("google.auth.default", _default(_Credentials(), None))
        with pytest.raises(ValueError, match="VERTEXAI_PROJECT"):
            _ = vertex_forwarder()
