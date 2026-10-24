"""
Shared state for the relay Lambdas, kept in one DynamoDB table: the websocket
connection of the Jarvis backend that is online, replies the backend sent for Alexa
requests, and the request each Alexa user is still waiting on. Records other than the
backend's connection expire after an hour. Also sends messages down to the backend
through API Gateway's connection management API.
"""

import json
import os
import time

import boto3
from botocore.exceptions import ClientError

TABLE = boto3.resource("dynamodb").Table(os.environ["TABLE_NAME"])
BACKEND_KEY = "backend"
RECORD_LIFETIME_SECONDS = 3600


class BackendOffline(Exception):
    """Raised when no Jarvis backend is connected to the relay."""


# Backend connection


def save_backend(connection_id: str) -> None:
    """
    Remember the backend's connection, replacing any older one.

    Args:
        connection_id: API Gateway's id for the websocket connection.
    """
    TABLE.put_item(Item={"pk": BACKEND_KEY, "connection_id": connection_id})


def forget_backend(connection_id: str) -> None:
    """
    Forget the backend's connection, unless a newer one has replaced it.

    Args:
        connection_id: The connection that closed.
    """
    try:
        TABLE.delete_item(
            Key={"pk": BACKEND_KEY},
            ConditionExpression="connection_id = :id",
            ExpressionAttributeValues={":id": connection_id},
        )
    except ClientError as e:
        if e.response["Error"]["Code"] != "ConditionalCheckFailedException":
            raise


def send_to_backend(message: dict) -> None:
    """
    Send a message to the connected backend.

    Args:
        message: The message, serialized as JSON.
    """
    item = TABLE.get_item(Key={"pk": BACKEND_KEY}, ConsistentRead=True).get("Item")
    if item is None:
        raise BackendOffline()

    # send the request to the APIGateway to send to agent
    connection_id = item["connection_id"]
    client = boto3.client("apigatewaymanagementapi", endpoint_url=os.environ["CONNECTIONS_URL"])
    try:
        client.post_to_connection(ConnectionId=connection_id, Data=json.dumps(message).encode())
    except client.exceptions.GoneException:
        forget_backend(connection_id)
        raise BackendOffline() from None


# Alexa replies and pending requests


def save_reply(request_id: str, text: str) -> None:
    """
    Store the backend's reply to an Alexa request.

    Args:
        request_id: The request being answered.
        text: What Jarvis says.
    """
    _put(f"reply#{request_id}", {"text": text})


def get_reply(request_id: str) -> str | None:
    """
    Look up the reply to an Alexa request.

    Args:
        request_id: The request.
    Returns:
        The reply text, or None if it has not arrived.
    """
    item = _get(f"reply#{request_id}")
    if item is None:
        return None
    return item["text"]


def delete_reply(request_id: str) -> None:
    """
    Delete a reply once it has been spoken.

    Args:
        request_id: The request it answered.
    """
    TABLE.delete_item(Key={"pk": f"reply#{request_id}"})


def save_pending(user_id: str, request_id: str) -> None:
    """
    Remember the request an Alexa user is waiting on after Alexa's time limit ran out.

    Args:
        user_id: The Alexa user.
        request_id: The unanswered request.
    """
    _put(f"pending#{user_id}", {"request_id": request_id})


def get_pending(user_id: str) -> str | None:
    """
    Look up the request an Alexa user is waiting on.

    Args:
        user_id: The Alexa user.
    Returns:
        The request id, or None if nothing is pending.
    """
    item = _get(f"pending#{user_id}")
    if item is None:
        return None
    return item["request_id"]


def delete_pending(user_id: str) -> None:
    """
    Clear the request an Alexa user was waiting on.

    Args:
        user_id: The Alexa user.
    """
    TABLE.delete_item(Key={"pk": f"pending#{user_id}"})


def _put(key: str, fields: dict) -> None:
    """
    Store a record that expires after RECORD_LIFETIME_SECONDS.

    Args:
        key: The record's key.
        fields: The record's other attributes.
    """
    item = {"pk": key, "expires_at": int(time.time()) + RECORD_LIFETIME_SECONDS}
    item.update(fields)
    TABLE.put_item(Item=item)


def _get(key: str) -> dict | None:
    """
    Read a record, treating an expired one as missing, since DynamoDB deletes
    expired records lazily.

    Args:
        key: The record's key.
    Returns:
        The record, or None.
    """
    item = TABLE.get_item(Key={"pk": key}, ConsistentRead=True).get("Item")
    if item is None:
        return None
    if int(item["expires_at"]) < time.time():
        return None
    return item
