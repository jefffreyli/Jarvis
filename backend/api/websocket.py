"""
Websocket server that streams harness events to the frontend so it can show the agent
loop live. The HarnessBroadcaster is the agent's observer: each event is serialized,
kept in a short replay buffer, and queued for every connected browser. A new client
first receives the replay (so a turn already in progress shows up), then live events.
The server only listens on localhost and only accepts browsers from the allowed origins,
so other websites cannot read what Jarvis is doing.
"""

import asyncio
import json
import logging
from collections import deque

from aiohttp import WSMsgType, web

from agent.trace import HarnessEvent

logger = logging.getLogger(__name__)

HARNESS_PATH = "/harness"
REPLAY_EVENTS = 500
HEARTBEAT_SECONDS = 30


class HarnessBroadcaster:
    """Sends every harness event to the connected browsers."""

    def __init__(self, port: int, allowed_origins: tuple[str, ...]):
        """
        Args:
            port: Local port to listen on.
            allowed_origins: Browser origins allowed to connect, such as the frontend's URL.
        """
        self.port = port
        self.allowed_origins = allowed_origins
        self._runner: web.AppRunner | None = None
        self._recent: deque[str] = deque(maxlen=REPLAY_EVENTS)
        self._clients: dict[web.WebSocketResponse, asyncio.Queue[str]] = {}

    async def start(self) -> None:
        """Start serving the websocket on localhost."""
        app = web.Application()
        app.router.add_get(HARNESS_PATH, self._handle_client)
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        await web.TCPSite(self._runner, "127.0.0.1", self.port).start()
        logger.info("Harness websocket listening on port %d", self.port)

    async def stop(self) -> None:
        """Disconnect every client and shut down the server."""
        for client in list(self._clients):
            await client.close()
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    def publish(self, event: HarnessEvent) -> None:
        """
        Queue an event for every connected client. Used as the agent's observer.

        Args:
            event: What the agent just did.
        """
        payload = json.dumps(event.to_dict())
        self._recent.append(payload)
        for queue in self._clients.values():
            queue.put_nowait(payload)

    async def _handle_client(self, request: web.Request) -> web.WebSocketResponse:
        """
        Accept a browser connection and stream events to it until it disconnects.

        Args:
            request: The websocket upgrade request.
        Returns:
            The finished websocket response.
        """
        origin = request.headers.get("Origin", "")
        if origin not in self.allowed_origins:
            logger.warning("Rejected harness connection from origin %r", origin)
            raise web.HTTPForbidden()

        client = web.WebSocketResponse(heartbeat=HEARTBEAT_SECONDS)
        try:
            await client.prepare(request)
        except ConnectionResetError:
            # The browser left before the handshake finished.
            return client

        queue: asyncio.Queue[str] = asyncio.Queue()
        for payload in self._recent:
            queue.put_nowait(payload)
        self._clients[client] = queue
        sender = asyncio.create_task(self._send_events(client, queue))

        try:
            # Clients only listen, so incoming messages are ignored until the socket closes.
            async for message in client:
                if message.type == WSMsgType.ERROR:
                    break
        finally:
            del self._clients[client]
            sender.cancel()
        return client

    async def _send_events(self, client: web.WebSocketResponse, queue: asyncio.Queue[str]) -> None:
        """
        Send queued events to one client in order.

        Args:
            client: The browser's websocket.
            queue: Events waiting to be sent to it.
        """
        while True:
            payload = await queue.get()
            try:
                await client.send_str(payload)
            except ConnectionError:
                return
