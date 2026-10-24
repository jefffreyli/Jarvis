"""
Receives Linq's message.received webhooks and forwards them to the Jarvis backend
through the relay websocket. The raw body is passed along base64 encoded with Linq's
signature headers, so the backend can check the signature itself. When the backend is
offline the webhook gets a 503, which Linq retries.
"""

import base64
import logging

import store

logger = logging.getLogger()
logger.setLevel(logging.INFO)

SIGNATURE_HEADERS = ("webhook-id", "webhook-timestamp", "webhook-signature")


def handler(event: dict, context) -> dict:
    """
    Forward one webhook request to the backend.

    Args:
        event: The HTTP API request event.
        context: The Lambda context (unused).
    Returns:
        200 once forwarded, or 503 when the backend is offline.
    """
    body = event.get("body") or ""
    if event.get("isBase64Encoded"):
        encoded_body = body
    else:
        encoded_body = base64.b64encode(body.encode()).decode()

    # HTTP APIs lowercase header names.
    request_headers = event.get("headers") or {}
    headers = {}
    for name in SIGNATURE_HEADERS:
        if name in request_headers:
            headers[name] = request_headers[name]

    try:
        store.send_to_backend({"type": "linq.webhook", "headers": headers, "body": encoded_body})
    except store.BackendOffline:
        logger.warning("Backend is offline, so Linq will retry the webhook")
        return {"statusCode": 503}
    return {"statusCode": 200}
