import time
from collections import deque
from collections.abc import Callable

import anyio
import websockets
from websockets.exceptions import ConnectionClosed


async def listen_to_websocket(
    address: str,
    event_queue: deque[str],
    shutdown_event: anyio.Event,
    status_callback: Callable[[str], None] | None = None,
) -> None:
    """Listens to a websocket and writes incoming events to the queue."""
    url = f"ws://{address}/events"
    start_time = time.time()
    timeout = 60

    if status_callback:
        status_callback("connecting")

    while True:
        try:
            async with websockets.connect(url) as websocket:
                if status_callback:
                    status_callback("connected")

                async for message in websocket:
                    event_queue.append(str(message))
                break
        except (KeyboardInterrupt, ConnectionClosed):
            if status_callback:
                status_callback("disconnected")
            shutdown_event.set()
            return
        except Exception:
            if time.time() - start_time > timeout:
                if status_callback:
                    status_callback("failed")
                shutdown_event.set()
                return

            await anyio.sleep(1)

    if status_callback:
        status_callback("disconnected")
