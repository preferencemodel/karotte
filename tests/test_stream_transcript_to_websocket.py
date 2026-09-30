"""Tests for the websocket transcript streaming functionality."""

import asyncio
import socket
from contextlib import closing
from typing import cast

import anyio
import pytest
import websockets
from pydantic import TypeAdapter

from karotte.schemas.chat import Message
from karotte.schemas.transcript import (
    Event,
    MessageAddedEvent,
    TaskStartedEvent,
)
from karotte.schemas.websocket_config import WebSocketConfig
from karotte.transcript_streaming.stream_transcript_to_websocket import (
    stream_transcript_to_websocket,
)

EventAdapter = TypeAdapter(Event)


def find_free_port() -> int:
    """Find a free port on localhost."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.listen(1)
        port = cast(int, s.getsockname()[1])
    return port


@pytest.mark.asyncio
async def test_single_client_receives_events():
    """Test that a single client receives all events."""
    port = find_free_port()
    config = WebSocketConfig(host="127.0.0.1", port=port)

    events_to_send = [
        TaskStartedEvent(run_id="run_1", task_id="task_1", n_steps=10),
        MessageAddedEvent(message=Message(role="user", content="Hello")),
        MessageAddedEvent(message=Message(role="assistant", content="Hi there")),
    ]

    received_events: list[str] = []
    server_ready = asyncio.Event()

    async def run_server():
        send, recv = anyio.create_memory_object_stream[Event]()
        async with anyio.create_task_group() as tg:
            tg.start_soon(stream_transcript_to_websocket, config, recv)
            # Give server time to start
            await asyncio.sleep(0.2)
            server_ready.set()

            # Wait for client to connect
            await asyncio.sleep(0.3)

            for event in events_to_send:
                await send.send(event)

            # Give time for events to be sent
            await asyncio.sleep(0.5)
            await send.aclose()

    async def run_client():
        await server_ready.wait()
        async with websockets.connect(f"ws://127.0.0.1:{port}/events") as ws:
            async for message in ws:
                received_events.append(str(message))
                if len(received_events) >= len(events_to_send):
                    break

    await asyncio.gather(run_server(), run_client())

    assert len(received_events) == len(events_to_send)
    # Verify events are valid JSON and match expected types
    for i, event_json in enumerate(received_events):
        parsed = EventAdapter.validate_json(event_json)
        assert type(parsed) is type(events_to_send[i])


@pytest.mark.asyncio
async def test_late_joining_client_receives_past_events():
    """Test that a client connecting after events have been sent receives all past events."""
    port = find_free_port()
    config = WebSocketConfig(host="127.0.0.1", port=port)

    events_to_send = [
        TaskStartedEvent(run_id="run_1", task_id="task_1", n_steps=10),
        MessageAddedEvent(message=Message(role="user", content="Hello")),
    ]

    received_events: list[str] = []
    server_ready = asyncio.Event()
    events_sent = asyncio.Event()

    async def run_server():
        send, recv = anyio.create_memory_object_stream[Event]()
        async with anyio.create_task_group() as tg:
            tg.start_soon(stream_transcript_to_websocket, config, recv)
            await asyncio.sleep(0.2)
            server_ready.set()

            # Send events before client connects
            for event in events_to_send:
                await send.send(event)
            events_sent.set()

            # Wait for client to receive, then close
            await asyncio.sleep(1.0)
            await send.aclose()

    async def run_client():
        await server_ready.wait()
        await events_sent.wait()
        # Connect AFTER events have been sent
        await asyncio.sleep(0.3)

        async with websockets.connect(f"ws://127.0.0.1:{port}/events") as ws:
            async for message in ws:
                received_events.append(str(message))
                if len(received_events) >= len(events_to_send):
                    break

    await asyncio.gather(run_server(), run_client())

    # Late-joining client should receive all past events
    assert len(received_events) == len(events_to_send)


@pytest.mark.asyncio
async def test_multiple_clients_receive_all_events():
    """Test that multiple clients each receive all events."""
    port = find_free_port()
    config = WebSocketConfig(host="127.0.0.1", port=port)

    events_to_send = [
        TaskStartedEvent(run_id="run_1", task_id="task_1", n_steps=10),
        MessageAddedEvent(message=Message(role="user", content="Hello")),
    ]

    client1_events: list[str] = []
    client2_events: list[str] = []
    server_ready = asyncio.Event()

    async def run_server():
        send, recv = anyio.create_memory_object_stream[Event]()
        async with anyio.create_task_group() as tg:
            tg.start_soon(stream_transcript_to_websocket, config, recv)
            await asyncio.sleep(0.2)
            server_ready.set()

            # Wait for both clients to connect
            await asyncio.sleep(0.5)

            for event in events_to_send:
                await send.send(event)

            # Give time for broadcast
            await asyncio.sleep(0.5)
            await send.aclose()

    async def run_client(events_list: list[str]):
        await server_ready.wait()
        async with websockets.connect(f"ws://127.0.0.1:{port}/events") as ws:
            async for message in ws:
                events_list.append(str(message))
                if len(events_list) >= len(events_to_send):
                    break

    await asyncio.gather(
        run_server(),
        run_client(client1_events),
        run_client(client2_events),
    )

    assert len(client1_events) == len(events_to_send)
    assert len(client2_events) == len(events_to_send)


@pytest.mark.asyncio
async def test_client_disconnect_does_not_affect_other_clients():
    """Test that one client disconnecting doesn't affect other clients."""
    port = find_free_port()
    config = WebSocketConfig(host="127.0.0.1", port=port)

    events_to_send = [
        TaskStartedEvent(run_id="run_1", task_id="task_1", n_steps=10),
        MessageAddedEvent(message=Message(role="user", content="First")),
        MessageAddedEvent(message=Message(role="user", content="Second")),
        MessageAddedEvent(message=Message(role="user", content="Third")),
    ]

    client1_events: list[str] = []
    client2_events: list[str] = []
    server_ready = asyncio.Event()
    client1_done = asyncio.Event()

    async def run_server():
        send, recv = anyio.create_memory_object_stream[Event]()
        async with anyio.create_task_group() as tg:
            tg.start_soon(stream_transcript_to_websocket, config, recv)
            await asyncio.sleep(0.2)
            server_ready.set()

            # Wait for clients to connect
            await asyncio.sleep(0.3)

            # Send first event
            await send.send(events_to_send[0])
            await asyncio.sleep(0.2)

            # Wait for client1 to disconnect
            await client1_done.wait()
            await asyncio.sleep(0.1)

            # Send remaining events after client1 disconnects
            for event in events_to_send[1:]:
                await send.send(event)
                await asyncio.sleep(0.1)

            await asyncio.sleep(0.3)
            await send.aclose()

    async def run_client1():
        """Client that disconnects after first event."""
        await server_ready.wait()
        async with websockets.connect(f"ws://127.0.0.1:{port}/events") as ws:
            message = await ws.recv()
            client1_events.append(str(message))
        client1_done.set()

    async def run_client2():
        """Client that stays connected for all events."""
        await server_ready.wait()
        async with websockets.connect(f"ws://127.0.0.1:{port}/events") as ws:
            async for message in ws:
                client2_events.append(str(message))
                if len(client2_events) >= len(events_to_send):
                    break

    await asyncio.gather(run_server(), run_client1(), run_client2())

    # Client 1 only got first event before disconnecting
    assert len(client1_events) == 1
    # Client 2 got all events
    assert len(client2_events) == len(events_to_send)


@pytest.mark.asyncio
async def test_server_shutdown_closes_connections():
    """Test that closing the event stream properly shuts down the server."""
    port = find_free_port()
    config = WebSocketConfig(host="127.0.0.1", port=port)

    server_ready = asyncio.Event()
    connection_closed = asyncio.Event()

    async def run_server():
        send, recv = anyio.create_memory_object_stream[Event]()
        async with anyio.create_task_group() as tg:
            tg.start_soon(stream_transcript_to_websocket, config, recv)
            await asyncio.sleep(0.2)
            server_ready.set()

            # Wait for client to connect, then close stream
            await asyncio.sleep(0.5)
            await send.aclose()

    async def run_client():
        await server_ready.wait()
        try:
            async with websockets.connect(f"ws://127.0.0.1:{port}/events") as ws:
                # Wait for server to close
                async for _ in ws:
                    pass
        except websockets.ConnectionClosed:
            pass
        connection_closed.set()

    await asyncio.gather(run_server(), run_client())

    # Connection should be closed after server shuts down
    assert connection_closed.is_set()


@pytest.mark.asyncio
async def test_events_are_valid_json():
    """Test that events are serialized as valid JSON."""
    port = find_free_port()
    config = WebSocketConfig(host="127.0.0.1", port=port)

    event = TaskStartedEvent(
        run_id="run_1",
        task_id="task_with_special_chars_<>&\"'",
        n_steps=10,
    )

    received_events: list[str] = []

    async def run_server():
        send, recv = anyio.create_memory_object_stream[Event]()
        async with anyio.create_task_group() as tg:
            tg.start_soon(stream_transcript_to_websocket, config, recv)
            await asyncio.sleep(0.2)
            await send.send(event)
            await asyncio.sleep(0.5)
            await send.aclose()

    async def run_client():
        await asyncio.sleep(0.3)
        async with websockets.connect(f"ws://127.0.0.1:{port}/events") as ws:
            message = await ws.recv()
            received_events.append(str(message))

    await asyncio.gather(run_server(), run_client())

    assert len(received_events) == 1
    # Verify it's valid JSON and can be deserialized back
    parsed = TaskStartedEvent.model_validate_json(received_events[0])
    assert parsed.task_id == event.task_id
    assert parsed.run_id == event.run_id


@pytest.mark.asyncio
async def test_empty_event_stream():
    """Test that an empty event stream doesn't cause issues."""
    port = find_free_port()
    config = WebSocketConfig(host="127.0.0.1", port=port)

    received_events: list[str] = []
    server_ready = asyncio.Event()

    async def run_server():
        send, recv = anyio.create_memory_object_stream[Event]()
        async with anyio.create_task_group() as tg:
            tg.start_soon(stream_transcript_to_websocket, config, recv)
            await asyncio.sleep(0.2)
            server_ready.set()
            # Close immediately without sending anything
            await asyncio.sleep(0.3)
            await send.aclose()

    async def run_client():
        await server_ready.wait()
        try:
            async with websockets.connect(f"ws://127.0.0.1:{port}/events") as ws:
                async for message in ws:
                    received_events.append(str(message))
        except websockets.ConnectionClosed:
            pass

    await asyncio.gather(run_server(), run_client())

    assert len(received_events) == 0


@pytest.mark.asyncio
async def test_rapid_event_sending():
    """Test that rapid event sending doesn't lose events."""
    port = find_free_port()
    config = WebSocketConfig(host="127.0.0.1", port=port)

    num_events = 100
    events_to_send = [
        MessageAddedEvent(message=Message(role="user", content=f"Message {i}"))
        for i in range(num_events)
    ]

    received_events: list[str] = []

    async def run_server():
        send, recv = anyio.create_memory_object_stream[Event]()
        async with anyio.create_task_group() as tg:
            tg.start_soon(stream_transcript_to_websocket, config, recv)
            await asyncio.sleep(0.2)

            # Send all events rapidly
            for event in events_to_send:
                await send.send(event)

            await asyncio.sleep(1.0)
            await send.aclose()

    async def run_client():
        await asyncio.sleep(0.3)
        async with websockets.connect(f"ws://127.0.0.1:{port}/events") as ws:
            async for message in ws:
                received_events.append(str(message))
                if len(received_events) >= num_events:
                    break

    await asyncio.gather(run_server(), run_client())

    assert len(received_events) == num_events


@pytest.mark.asyncio
async def test_server_waits_for_client_before_completing():
    """Test that the server waits for at least one client to connect before completing.

    This tests the fix for crashes that occur before TaskStartedEvent - the server
    should wait for a client to connect and receive events rather than exiting immediately.
    """
    port = find_free_port()
    config = WebSocketConfig(host="127.0.0.1", port=port)

    events_to_send = [
        TaskStartedEvent(run_id="run_1", task_id="task_1", n_steps=10),
        MessageAddedEvent(message=Message(role="user", content="Hello")),
    ]

    received_events: list[str] = []
    server_completed = asyncio.Event()

    async def run_server():
        send, recv = anyio.create_memory_object_stream[Event]()
        async with anyio.create_task_group() as tg:
            tg.start_soon(stream_transcript_to_websocket, config, recv)
            await asyncio.sleep(0.2)

            # Send all events immediately (simulating a crash scenario)
            for event in events_to_send:
                await send.send(event)

            # Close stream - server should wait for client
            await send.aclose()
        server_completed.set()

    async def run_client():
        # Wait a bit before connecting (simulating slow TUI startup)
        await asyncio.sleep(0.5)
        async with websockets.connect(f"ws://127.0.0.1:{port}/events") as ws:
            async for message in ws:
                received_events.append(str(message))
                if len(received_events) >= len(events_to_send):
                    break

    await asyncio.gather(run_server(), run_client())

    # Client should have received all events even though it connected late
    assert len(received_events) == len(events_to_send)
    assert server_completed.is_set()


@pytest.mark.asyncio
async def test_server_times_out_if_no_client_connects(monkeypatch: pytest.MonkeyPatch):
    """Test that the server eventually times out if no client ever connects."""
    import time

    monkeypatch.setattr(
        "karotte.transcript_streaming.stream_transcript_to_websocket.NO_CLIENT_TIMEOUT_S",
        0.5,
    )

    port = find_free_port()
    config = WebSocketConfig(host="127.0.0.1", port=port)

    events_to_send = [
        TaskStartedEvent(run_id="run_1", task_id="task_1", n_steps=10),
    ]

    start_time = time.time()

    async def run_server():
        send, recv = anyio.create_memory_object_stream[Event]()
        async with anyio.create_task_group() as tg:
            tg.start_soon(stream_transcript_to_websocket, config, recv)
            await asyncio.sleep(0.1)

            for event in events_to_send:
                await send.send(event)

            # Close stream - server should wait for client but eventually timeout
            await send.aclose()

    # No client connects - server should give up after NO_CLIENT_TIMEOUT_S
    await asyncio.wait_for(run_server(), timeout=5)

    elapsed = time.time() - start_time
    # Server should have waited (not exited immediately) but eventually timed out
    assert 0.5 <= elapsed < 5
