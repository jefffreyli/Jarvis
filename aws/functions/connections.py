"""
A lambda function that handles every event on the websocket between AWS to local.
- $connect: admits the Jarvis backend only when it presents the shared secret and records its connection
- $disconnect: forgets it
- every other message is routed by its "action": "reply" stores Jarvis's answer to an Alexa request
- "ping" is the backend's heartbeat that keeps the connection open
A reply is logged under the trace of the Alexa request it answers, and that log is
forwarded to the backend.
"""

import hmac
import json
import logging
import os

import relay_log
import store

logger = logging.getLogger("jarvis.connections")

SECRET_HEADER = "x-jarvis-secret"


def handler(event: dict, context) -> dict:
    """
    Handle one websocket route event from API Gateway.

    Args:
        event: The route event.
        context: The Lambda context (unused).
    Returns:
        The status for API Gateway; anything but 200 on $connect refuses the connection.
    """
    route = event["requestContext"]["routeKey"]
    connection_id = event["requestContext"]["connectionId"]

    # The connection is not open yet on $connect, or no longer on $disconnect, so these
    # are only logged to CloudWatch.
    if route == "$connect":
        source_ip = event["requestContext"].get("identity", {}).get("sourceIp")
        if not _has_secret(event):
            logger.warning("Refused a connection from %s without the shared secret", source_ip)
            return {"statusCode": 401}
        store.save_backend(connection_id)
        logger.info("Backend connected from %s on connection %s", source_ip, connection_id)
        return {"statusCode": 200}

    if route == "$disconnect":
        store.forget_backend(connection_id)
        logger.info(
            "Backend connection %s closed: %s %s",
            connection_id,
            event["requestContext"].get("disconnectStatusCode"),
            event["requestContext"].get("disconnectReason"),
        )
        return {"statusCode": 200}

    try:
        message = json.loads(event.get("body") or "")
    except json.JSONDecodeError:
        logger.warning("Ignored a message that is not JSON")
        return {"statusCode": 400}

    action = message.get("action")
    if action == "reply":
        request_id = message.get("request_id")
        text = message.get("text")
        if not isinstance(request_id, str) or not isinstance(text, str):
            logger.warning("Ignored a malformed reply")
            return {"statusCode": 400}
        _save_reply(event, request_id, text)
    elif action != "ping":
        logger.warning("Ignored a message with unknown action %r", action)
    return {"statusCode": 200}


def _save_reply(event: dict, request_id: str, text: str) -> None:
    """
    Store the backend's reply to an Alexa request, forwarding the log to the backend.

    Args:
        event: The route event that carried the reply.
        request_id: The Alexa request being answered.
        text: What Jarvis says.
    """
    relay_log.begin(f"alexa:{request_id[:8]}")
    try:
        store.save_reply(request_id, text)
        logger.info(
            "Stored the reply to request %s (%d chars), %s ms after API Gateway received it",
            request_id,
            len(text),
            relay_log.gateway_delay_ms(event),
        )
    finally:
        relay_log.flush()


def _has_secret(event: dict) -> bool:
    """
    Check that a connecting client sent the shared secret.

    Args:
        event: The $connect event.
    Returns:
        True if the secret header matches RELAY_SECRET.
    """
    presented = ""
    for name, value in (event.get("headers") or {}).items():
        if name.lower() == SECRET_HEADER:
            presented = value
    return hmac.compare_digest(presented.encode(), os.environ["RELAY_SECRET"].encode())
