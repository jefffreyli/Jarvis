"""
Configuration for the agent.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

# Paths. Everything is logged to LOG_PATH, including the AWS Lambdas and canvas-mcp.
LOG_PATH = PROJECT_ROOT / "jarvis.log"
DATABASE_PATH = PROJECT_ROOT / "data" / "jarvis.db"
ALEXA_TRANSCRIPT_PATH = PROJECT_ROOT / "data" / "alexa-transcript.txt"
SKILLS_DIR = PROJECT_ROOT / "backend" / "skills"
MESSAGES_DATABASE_PATH = Path.home() / "Library" / "Messages" / "chat.db"

# OpenAI
MODEL = "gpt-6-sol"
EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_TIMEOUT_SECONDS = 120.0
OPENAI_MAX_RETRIES = 3

# Notion's hosted MCP server, which OpenAI calls during replies. Sign in once with
# `python3 sign_in_notion.py`; the OAuth tokens are kept in NOTION_TOKEN_PATH.
NOTION_MCP_URL = "https://mcp.notion.com/mcp"
NOTION_TOKEN_PATH = PROJECT_ROOT / "data" / "notion_oauth.json"
OAUTH_CALLBACK_PORT = 8789

# Canvas through canvas-mcp (github.com/vishalsachdev/canvas-mcp), which Jarvis runs
# locally. CANVAS_SERVER_ENV makes the server register only its read-only student tools,
# and makes its log plain lines (no banner or colors), since they go into jarvis.log.
CANVAS_SERVER_ENV = {
    "CANVAS_ROLE": "student",
    "ALLOWED_WRITE_TOOLS": "none",
    "FASTMCP_SHOW_SERVER_BANNER": "false",
    "FASTMCP_ENABLE_RICH_LOGGING": "false",
}

# Voice mode (Smallest AI)
WAKE_PHRASE = "hey jarvis"
SPEECH_TO_TEXT_MODEL = "pulse"
SPEECH_LANGUAGE = "en"
TEXT_TO_SPEECH_VOICE = "magnus"

# Phone calls (Smallest AI voice agent). The cloned voice must be used with a Waves model.
CALL_VOICE_ID = "voice_mXgEAmbEQl"
CALL_VOICE_MODEL = "waves_lightning_v3_1"

# Harness websocket that streams the agent's progress to the frontend
HARNESS_PORT = 8788
HARNESS_ALLOWED_ORIGINS = ("http://localhost:3000", "http://127.0.0.1:3000")


def openai_api_key() -> str:
    """
    Read the OpenAI API key from the environment.

    Returns:
        The API key.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key == "":
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to the .env file at the project root.")
    return api_key


def relay_settings() -> tuple[str, str]:
    """
    Read the AWS relay settings, which Alexa and texting reach Jarvis through.

    Returns:
        The relay's websocket URL and its shared secret.
    """
    url = os.environ.get("RELAY_URL", "")
    secret = os.environ.get("RELAY_SECRET", "")
    if url == "" or secret == "":
        raise RuntimeError(
            "Alexa and texting are off. To turn them on, deploy the relay in aws/ and add "
            "RELAY_URL and RELAY_SECRET to the .env file at the project root."
        )
    return url, secret


def canvas_settings() -> tuple[str, str, str]:
    """
    Read the Canvas settings for the canvas-mcp server.

    Returns:
        The command that starts canvas-mcp, the Canvas API token and the Canvas API URL.
    """
    command = os.environ.get("CANVAS_MCP_COMMAND", "canvas-mcp-server")
    token = os.environ.get("CANVAS_API_TOKEN", "")
    url = os.environ.get("CANVAS_API_URL", "")
    if token == "" or url == "":
        raise RuntimeError(
            "Canvas is off. To turn it on, install canvas-mcp and add CANVAS_API_TOKEN and "
            "CANVAS_API_URL (ending in /api/v1) to the .env file at the project root. Set "
            "CANVAS_MCP_COMMAND if canvas-mcp-server is not on your PATH."
        )
    return command, token, url


def gmail_credentials() -> tuple[str, str]:
    """
    Read the Gmail address and app password from the environment.

    Returns:
        The address and the app password.
    """
    address = os.environ.get("GMAIL_ADDRESS", "")
    app_password = os.environ.get("GMAIL_APP_PASSWORD", "")
    if address == "" or app_password == "":
        raise RuntimeError(
            "Gmail is not set up. Add GMAIL_ADDRESS and GMAIL_APP_PASSWORD (a Google app "
            "password) to the .env file at the project root."
        )
    return address, app_password


def smallest_api_key() -> str:
    """
    Read the Smallest AI API key, used for phone calls, transcription and speech.

    Returns:
        The API key.
    """
    api_key = os.environ.get("SMALLEST_API_KEY", "")
    if api_key == "":
        raise RuntimeError("SMALLEST_API_KEY is not set. Add it to the .env file at the project root.")
    return api_key


def smallest_call_settings() -> tuple[str, str]:
    """
    Read the settings for phone calls placed through a Smallest AI voice agent.

    Returns:
        The voice agent's id, and the caller ID number ("" for Smallest AI's default).
    """
    agent_id = os.environ.get("SMALLEST_AGENT_ID", "")
    from_number = os.environ.get("SMALLEST_FROM_NUMBER", "")
    if agent_id == "":
        raise RuntimeError(
            "Phone calls are not set up. Add SMALLEST_AGENT_ID (a voice agent whose prompt "
            "uses {{objective}}) to the .env file at the project root."
        )
    return agent_id, from_number


def linq_settings() -> tuple[str, list[str], str]:
    """
    Read the Linq texting settings from the environment.

    Returns:
        The API key, the phone numbers allowed to text Jarvis, and the relay's webhook URL.
    """
    api_key = os.environ.get("LINQ_API_KEY", "")
    allowed = os.environ.get("LINQ_ALLOWED_NUMBERS", "")
    webhook_url = os.environ.get("LINQ_WEBHOOK_URL", "")

    allowed_numbers = []
    for number in allowed.split(","):
        if number.strip() != "":
            allowed_numbers.append(number.strip())

    if api_key == "" or len(allowed_numbers) == 0 or webhook_url == "":
        raise RuntimeError(
            "Texting is off. To turn it on, add LINQ_API_KEY, LINQ_ALLOWED_NUMBERS (your "
            "phone numbers, comma-separated) and LINQ_WEBHOOK_URL (from the relay deploy) "
            "to the .env file at the project root."
        )
    return api_key, allowed_numbers, webhook_url
