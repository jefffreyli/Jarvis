"""
Receives Linq's message.received webhooks and forwards them to the Jarvis backend
through the relay websocket. The raw body is passed along base64 encoded with Linq's
signature headers, so the backend can check the signature itself. When the backend is
offline the webhook gets a 503, which Linq retries. Each webhook gets a trace,
"linq:<id>", which travels with it so the backend logs under the same trace.
"""

import base64
import logging
import uuid

import relay_log
import store

logger = logging.getLogger("jarvis.linq_webhook")

SIGNATURE_HEADERS = ("webhook-id", "webhook-timestamp", "webhook-signature")


def handler(event: dict, context) -> dict:
    """
    Forward one webhook request to the backend, along with this invocation's log.

    Args:
        event: The HTTP API request event.
        context: The Lambda context (unused).
    Returns:
        200 once forwarded, or 503 when the backend is offline.
    """
    trace_id = uuid.uuid4().hex[:8]
    relay_log.begin(f"linq:{trace_id}")
    try:
        return _forward(event, trace_id)
    except Exception:
        logger.exception("Forwarding the webhook failed")
        raise
    finally:
        relay_log.flush()


def _forward(event: dict, trace_id: str) -> dict:
    """
    Forward one webhook request to the backend.

    Args:
        event: The HTTP API request event.
        trace_id: The id the backend logs this webhook under.
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

    context = event.get("requestContext") or {}
    logger.info(
        "Linq webhook %s from %s (API Gateway request %s, %s ms in API Gateway, %d bytes)",
        headers.get("webhook-id"),
        context.get("http", {}).get("sourceIp"),
        context.get("requestId"),
        relay_log.gateway_delay_ms(event),
        len(body),
    )

    message = {"type": "linq.webhook", "trace_id": trace_id, "headers": headers, "body": encoded_body}
    try:
        store.send_to_backend(message)
    except store.BackendOffline:
        logger.warning("Backend is offline, so Linq will retry the webhook")
        return {"statusCode": 503}
    return {"statusCode": 200}
