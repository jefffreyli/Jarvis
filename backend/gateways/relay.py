"""
Keeps Jarvis connected to its AWS relay, a WebSocket API that Alexa and Linq reach from
the internet. The connection is opened from this Mac, so no tunnel or open port is
needed. It authenticates with a shared secret and reconnects whenever API Gateway drops
it (API Gateway closes idle connections after 10 minutes, so a heartbeat is sent, and
every connection after 2 hours). Each incoming message is handed to the handler
registered for its "type", in a background task so a long agent turn never holds up
other messages. Gateways answer by sending messages with an "action", which API
Gateway routes to the relay's Lambda.
"""

import asyncio
import json
import logging
import ssl
from collections.abc import Awaitable, Callable
from typing import Any

import certifi
from websockets.asyncio.client import ClientConnection, connect, process_exception
from websockets.exceptions import ConnectionClosed, InvalidStatus

logger = logging.getLogger(__name__)

SECRET_HEADER = "x-jarvis-secret"
HEARTBEAT_SECONDS = 300
SEND_WAIT_SECONDS = 10
# API Gateway rejects websocket frames from clients larger than 32 KB.
MAX_MESSAGE_BYTES = 32_000

MessageHandler = Callable[[dict[str, Any]], Awaitable[None]]


class RelayError(Exception):
    """Raised when a message cannot be sent to the relay."""


class RelayClient:
    """A persistent connection to the AWS relay."""

    def __init__(self, url: str, secret: str, notify: Callable[[str], None] = print):
        """
        Args:
            url: The relay's websocket URL, like "wss://abc.execute-api.us-east-1.amazonaws.com/prod".
            secret: The shared secret the relay checks on connect.
            notify: Shows a one-line note about the connection, by default in the terminal.
        """
        self.url = url
        self.secret = secret
        self.notify = notify
        self._handlers: dict[str, MessageHandler] = {}
        self._websocket: ClientConnection | None = None
        self._connected = asyncio.Event()
        self._runner: asyncio.Task | None = None
        self._tasks: set[asyncio.Task] = set()

    def on(self, message_type: str, handler: MessageHandler) -> None:
        """
        Register the handler for one type of incoming message.

        Args:
            message_type: The message's "type" field, like "alexa.ask".
            handler: Called with each message of that type.
        """
        self._handlers[message_type] = handler

    async def start(self) -> None:
        """Connect in the background and keep the connection up until stopped."""
        self._runner = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Close the connection and cancel any running handlers."""
        for task in list(self._tasks):
            task.cancel()
        if self._runner is not None:
            self._runner.cancel()
            try:
                await self._runner
            except asyncio.CancelledError:
                pass
            self._runner = None

    async def send(self, message: dict[str, Any]) -> None:
        """
        Send a message to the relay, waiting briefly if it is reconnecting.

        Args:
            message: The message, which needs an "action" for API Gateway to route it.
        """
        data = json.dumps(message)
        if len(data.encode()) > MAX_MESSAGE_BYTES:
            raise RelayError(f"Message is larger than the relay's {MAX_MESSAGE_BYTES} byte limit")

        try:
            await asyncio.wait_for(self._connected.wait(), SEND_WAIT_SECONDS)
        except asyncio.TimeoutError:
            raise RelayError("Not connected to the relay") from None

        try:
            await self._websocket.send(data)
        except ConnectionClosed as e:
            raise RelayError(f"The relay connection closed: {e}") from None

    async def _run(self) -> None:
        """Connect, reconnecting with backoff, and dispatch messages until cancelled."""
        connected_before = False
        try:
            # Iterating connect() reconnects after transient failures. Websocket pings
            # are off since API Gateway does not count them as activity.
            options = {
                "additional_headers": {SECRET_HEADER: self.secret},
                "ping_interval": None,
                "process_exception": _retry_unless_ssl_error,
            }
            if self.url.startswith("wss://"):
                # python.org builds of Python on macOS ship without root certificates.
                options["ssl"] = ssl.create_default_context(cafile=certifi.where())
            async for websocket in connect(self.url, **options):
                if connected_before:
                    logger.info("Reconnected to the relay")
                else:
                    self.notify("[Relay] Connected. Alexa and texting can reach Jarvis.")
                    connected_before = True

                self._websocket = websocket
                self._connected.set()
                heartbeat = asyncio.create_task(self._heartbeat(websocket))
                try:
                    async for raw in websocket:
                        self._dispatch(raw)
                except ConnectionClosed:
                    logger.info("The relay connection closed, reconnecting")
                finally:
                    heartbeat.cancel()
                    self._connected.clear()
                    self._websocket = None
        except InvalidStatus as e:
            logger.exception("The relay refused the connection")
            self.notify(f"[Relay] The relay refused the connection (HTTP {e.response.status_code}). Check RELAY_SECRET.")
        except ssl.SSLError as e:
            logger.exception("The relay's certificate could not be verified")
            self.notify(f"[Relay] Alexa and texting are off: the relay's certificate could not be verified ({e}).")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("The relay connection failed")
            self.notify(f"[Relay] Alexa and texting are off: {e}")

    def _dispatch(self, raw: str | bytes) -> None:
        """
        Start the registered handler for one incoming message.

        Args:
            raw: The message as received.
        """
        try:
            message = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Ignored a relay message that is not JSON")
            return

        if not isinstance(message, dict):
            logger.warning("Ignored a relay message that is not an object")
            return
        handler = self._handlers.get(message.get("type"))
        if handler is None:
            logger.warning("Ignored a relay message with unknown type %r", message.get("type"))
            return

        task = asyncio.create_task(self._run_handler(handler, message))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _run_handler(self, handler: MessageHandler, message: dict[str, Any]) -> None:
        """
        Run a handler, logging any failure so one bad message cannot go unnoticed.

        Args:
            handler: The handler.
            message: The message it handles.
        """
        try:
            await handler(message)
        except Exception:
            logger.exception("Handling a %s message failed", message.get("type"))

    async def _heartbeat(self, websocket: ClientConnection) -> None:
        """
        Send a message every few minutes so API Gateway does not close the idle connection.

        Args:
            websocket: The connection to keep alive.
        """
        while True:
            await asyncio.sleep(HEARTBEAT_SECONDS)
            try:
                await websocket.send(json.dumps({"action": "ping"}))
            except ConnectionClosed:
                return


def _retry_unless_ssl_error(error: Exception) -> Exception | None:
    """
    Decide whether a failed connection attempt is retried. Certificate problems do not
    fix themselves, so they are raised instead of retried forever.

    Args:
        error: Why the attempt failed.
    Returns:
        The error to raise, or None to retry.
    """
    if isinstance(error, ssl.SSLError):
        return error
    return process_exception(error)
