"""Stream evaluation run events via websocket using only the websockets library."""

import asyncio
import time
from typing import final

import anyio
import anyio.streams.memory
from karotte.schemas.transcript import Event as TaskEvent
from karotte.schemas.websocket_config import WebSocketConfig
from loguru import logger
from websockets import ConnectionClosed
from websockets.asyncio.server import Server, ServerConnection, serve

SEND_EVENT_LOOP_INTERVAL_S = 0.1
NO_CLIENT_TIMEOUT_S = 10.0


@final
class WebsocketBroadcaster:
    def __init__(self) -> None:
        self._events: list[TaskEvent] = []
        self._clients: dict[ServerConnection, int] = {}  # client -> cursor
        self._shutdown = asyncio.Event()
        self._stop_broadcasting_immediately = False
        self._stop_broadcasting_when_all_caught_up = False
        self._finished_broadcasting = False
        self._broadcast_task: asyncio.Task[None] | None = None
        self._server: Server | None = None

    async def handler(self, websocket: ServerConnection) -> None:
        """Handle a new websocket connection."""
        if self._stop_broadcasting_immediately:
            await websocket.close()
            return

        self._clients[websocket] = 0

        try:
            # Keep connection open until shutdown or client disconnects
            await self._shutdown.wait()
        except ConnectionClosed:
            pass
        finally:
            self._clients.pop(websocket, None)

    async def queue_event_for_broadcasting(self, event: TaskEvent) -> None:
        """Queue an event for broadcasting to all clients."""
        self._events.append(event)

    async def wait_for_broadcasting_to_complete(self, timeout_s: float) -> None:
        """Wait for all clients to receive all events.

        If no clients are connected, waits up to timeout_s for at least one client
        to connect and receive all events before giving up.
        """
        self._stop_broadcasting_when_all_caught_up = True
        start_time = time.time()

        while time.time() - start_time < timeout_s and not self._finished_broadcasting:
            await asyncio.sleep(0.1)

    async def broadcast_events(self) -> None:
        """Broadcast events to all connected WebSocket clients."""
        while not self._stop_broadcasting_immediately:
            await self._send_outstanding_events_to_all_clients()

            if self._stop_broadcasting_when_all_caught_up:
                # Only consider "caught up" if we have at least one client that received events
                # This prevents exiting early before any client has connected
                if self._clients and all(
                    self._client_caught_up(client) for client in self._clients
                ):
                    break

            await asyncio.sleep(SEND_EVENT_LOOP_INTERVAL_S)

        self._finished_broadcasting = True

    async def _send_outstanding_events_to_all_clients(self) -> None:
        """Send any pending events to all clients."""
        for client in list(self._clients.keys()):
            await self._send_outstanding_events_to_client(client)

    async def _send_outstanding_events_to_client(
        self, client: ServerConnection
    ) -> None:
        """Send all pending events to a specific client."""
        while (
            client in self._clients
            and not self._stop_broadcasting_immediately
            and self._clients[client] < len(self._events)
        ):
            try:
                event = self._events[self._clients[client]]
                await client.send(event.model_dump_json())
                self._clients[client] += 1
            except ConnectionClosed:
                self._clients.pop(client, None)
                break
            except Exception as e:
                if client not in self._clients:
                    break
                logger.exception(f"Error broadcasting event to client: {e}")

    def _client_caught_up(self, client: ServerConnection) -> bool:
        """Check if a client has received all events."""
        if client not in self._clients:
            return True
        return self._clients[client] == len(self._events)

    async def start(self, host: str, port: int) -> None:
        """Start the websocket server and broadcast loop."""
        self._server = await serve(self.handler, host, port)
        self._broadcast_task = asyncio.create_task(self.broadcast_events())

    async def stop(self) -> None:
        """Stop the broadcaster and close all connections."""
        self._stop_broadcasting_immediately = True
        self._shutdown.set()

        # Close all client connections
        for client in list(self._clients.keys()):
            try:
                await client.close()
            except Exception as e:
                logger.debug(f"Error closing WebSocket client: {e}")

        # Wait for broadcast task to finish
        if self._broadcast_task:
            self._broadcast_task.cancel()
            try:
                await self._broadcast_task
            except asyncio.CancelledError:
                pass

        # Close the server
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    async def __aenter__(self) -> "WebsocketBroadcaster":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001 # pyright:ignore[reportMissingParameterType]
        await self.stop()


async def stream_transcript_to_websocket(
    config: WebSocketConfig,
    event_stream: anyio.streams.memory.MemoryObjectReceiveStream[TaskEvent],
) -> None:
    """Stream events from an event stream to connected websocket clients."""
    broadcaster = WebsocketBroadcaster()

    try:
        await broadcaster.start(config.host, config.port)

        async for event in event_stream:
            await broadcaster.queue_event_for_broadcasting(event)

        await broadcaster.wait_for_broadcasting_to_complete(NO_CLIENT_TIMEOUT_S)
    finally:
        await broadcaster.stop()
