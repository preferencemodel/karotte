"""Tests for the websocket_handler module."""

import asyncio
from collections import deque

import anyio
import pytest
from websockets.asyncio.server import ServerConnection, serve

from karotte.terminal.websocket_handler import listen_to_websocket
from tests.conftest import find_free_port


@pytest.mark.asyncio
async def test_listen_to_websocket_receives_events():
    """Test that listen_to_websocket receives events from the server."""
    port = find_free_port()
    event_queue: deque[str] = deque()
    shutdown_event = anyio.Event()
    server_ready = asyncio.Event()
    client_done = asyncio.Event()

    async def handler(websocket: ServerConnection) -> None:
        await websocket.send('{"type": "test", "data": "hello"}')
        await websocket.send('{"type": "test", "data": "world"}')
        await websocket.close()

    async def run_server():
        async with serve(handler, "127.0.0.1", port):
            server_ready.set()
            await client_done.wait()

    async def run_client():
        await server_ready.wait()
        try:
            await listen_to_websocket(
                f"127.0.0.1:{port}",
                event_queue,
                shutdown_event,
            )
        finally:
            client_done.set()

    await asyncio.gather(run_server(), run_client())

    assert len(event_queue) == 2
    assert "hello" in event_queue[0]
    assert "world" in event_queue[1]


@pytest.mark.asyncio
async def test_listen_to_websocket_handles_connection_closed_ok():
    """Test that ConnectionClosedOK (normal close) is handled gracefully.

    ConnectionClosedOK is a subclass of ConnectionClosed, so catching ConnectionClosed
    should handle both normal closes and error closes.
    """
    port = find_free_port()
    event_queue: deque[str] = deque()
    shutdown_event = anyio.Event()
    status_updates: list[str] = []
    server_ready = asyncio.Event()
    client_done = asyncio.Event()

    async def handler(websocket: ServerConnection) -> None:
        await websocket.send('{"type": "test"}')
        # Normal close
        await websocket.close(1000, "Normal closure")

    async def run_server():
        async with serve(handler, "127.0.0.1", port):
            server_ready.set()
            await client_done.wait()

    async def run_client():
        await server_ready.wait()
        try:
            await listen_to_websocket(
                f"127.0.0.1:{port}",
                event_queue,
                shutdown_event,
                status_callback=lambda s: status_updates.append(s),
            )
        finally:
            client_done.set()

    await asyncio.gather(run_server(), run_client())

    # Should have received the event before close
    assert len(event_queue) == 1
    # Status should show disconnected (not "completed" — run may still be scoring)
    assert "disconnected" in status_updates
    assert "completed" not in status_updates


@pytest.mark.asyncio
async def test_listen_to_websocket_handles_connection_closed_error():
    """Test that ConnectionClosedError (abnormal close) is handled gracefully.

    ConnectionClosedError is a subclass of ConnectionClosed, so catching ConnectionClosed
    should handle both normal closes and error closes.
    """
    port = find_free_port()
    event_queue: deque[str] = deque()
    shutdown_event = anyio.Event()
    status_updates: list[str] = []
    server_ready = asyncio.Event()
    client_done = asyncio.Event()

    async def handler(websocket: ServerConnection) -> None:
        await websocket.send('{"type": "test"}')
        # Abnormal close (e.g., server error)
        await websocket.close(1011, "Server error")

    async def run_server():
        async with serve(handler, "127.0.0.1", port):
            server_ready.set()
            await client_done.wait()

    async def run_client():
        await server_ready.wait()
        try:
            await listen_to_websocket(
                f"127.0.0.1:{port}",
                event_queue,
                shutdown_event,
                status_callback=lambda s: status_updates.append(s),
            )
        finally:
            client_done.set()

    await asyncio.gather(run_server(), run_client())

    # Should have received the event before close
    assert len(event_queue) == 1
    # Status should show disconnected (graceful handling of close)
    assert "disconnected" in status_updates
    assert "completed" not in status_updates


@pytest.mark.asyncio
async def test_listen_to_websocket_status_callback():
    """Test that status callback is called with correct statuses."""
    port = find_free_port()
    event_queue: deque[str] = deque()
    shutdown_event = anyio.Event()
    status_updates: list[str] = []
    server_ready = asyncio.Event()
    client_done = asyncio.Event()

    async def handler(websocket: ServerConnection) -> None:
        await websocket.send('{"type": "test"}')
        await websocket.close()

    async def run_server():
        async with serve(handler, "127.0.0.1", port):
            server_ready.set()
            await client_done.wait()

    async def run_client():
        await server_ready.wait()
        try:
            await listen_to_websocket(
                f"127.0.0.1:{port}",
                event_queue,
                shutdown_event,
                status_callback=lambda s: status_updates.append(s),
            )
        finally:
            client_done.set()

    await asyncio.gather(run_server(), run_client())

    # Should have connecting -> connected -> disconnected
    assert "connecting" in status_updates
    assert "connected" in status_updates
    assert "disconnected" in status_updates
