"""
Remote MCP servers that the model provider calls on Jarvis's behalf, such as Notion's
hosted server. Jarvis never talks to these servers itself: it hands the provider the
server's URL and an OAuth access token with each request. McpOAuth gets that token. A
one-time sign-in registers Jarvis as an OAuth client, opens the browser so the user can
approve access, and catches the redirect on a local port (PKCE, no client secret
needed). The tokens are saved to a file, and each request refreshes the access token
when it is about to expire.
"""

import asyncio
import base64
import hashlib
import json
import logging
import secrets
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlparse

import httpx
from aiohttp import web

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_SECONDS = 30
SIGN_IN_TIMEOUT_SECONDS = 300
CALLBACK_PATH = "/callback"
# Refresh this long before the access token expires, so it never expires mid-request.
REFRESH_MARGIN_SECONDS = 120


class McpAuthError(Exception):
    """Raised when there is no usable access token for an MCP server."""


class McpOAuth:
    """Gets and refreshes an OAuth access token for one remote MCP server."""

    def __init__(self, server_url: str, token_path: Path, callback_port: int):
        """
        Args:
            server_url: The MCP server's URL, like "https://mcp.notion.com/mcp".
            token_path: The file the client registration and tokens are saved in.
            callback_port: Local port that receives the browser redirect during sign-in.
        """
        self.server_url = server_url
        self.token_path = token_path
        self.callback_port = callback_port
        self._refresh_lock = asyncio.Lock()

    def signed_in(self) -> bool:
        """
        Check whether sign-in has been done.

        Returns:
            True if a token file exists.
        """
        return self.token_path.is_file()

    async def access_token(self) -> str:
        """
        Get a current access token, refreshing it first if it is about to expire.

        Returns:
            The access token.
        """
        # One refresh at a time, since a refresh token may only be usable once.
        async with self._refresh_lock:
            saved = self._load()
            if saved["expires_at"] - REFRESH_MARGIN_SECONDS > time.time():
                return saved["access_token"]
            if saved.get("refresh_token") is None:
                raise McpAuthError(f"The access token for {self.server_url} expired. Sign in again.")

            fields = {"grant_type": "refresh_token", "refresh_token": saved["refresh_token"]}
            tokens = await self._request_tokens(saved, fields)
            self._save_tokens(saved, tokens)
            logger.info("Refreshed the access token for %s", self.server_url)
            return tokens["access_token"]

    async def sign_in(self) -> None:
        """Register Jarvis with the server's OAuth provider, ask the user to approve access in the browser, and save the tokens."""
        metadata = await self._discover()
        redirect_uri = f"http://127.0.0.1:{self.callback_port}{CALLBACK_PATH}"
        client = await self._register(metadata, redirect_uri)

        verifier = secrets.token_urlsafe(64)
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
        state = secrets.token_urlsafe(32)
        query = {
            "response_type": "code",
            "client_id": client["client_id"],
            "redirect_uri": redirect_uri,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": state,
            "resource": self.server_url,
        }
        authorize_url = f"{metadata['authorization_endpoint']}?{urlencode(query)}"

        print(f"Opening your browser to approve access. If it does not open, visit:\n{authorize_url}\n")
        webbrowser.open(authorize_url)
        code = await _wait_for_code(self.callback_port, state)

        saved = {
            "token_endpoint": metadata["token_endpoint"],
            "client_id": client["client_id"],
            "client_secret": client.get("client_secret"),
        }
        fields = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "code_verifier": verifier,
        }
        tokens = await self._request_tokens(saved, fields)
        self._save_tokens(saved, tokens)

    async def _discover(self) -> dict[str, Any]:
        """
        Read the OAuth endpoints of the server's authorization server.

        Returns:
            The authorization server metadata (RFC 8414).
        """
        parsed = urlparse(self.server_url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as http:
            resource = await http.get(f"{origin}/.well-known/oauth-protected-resource")
            resource.raise_for_status()
            issuer = resource.json()["authorization_servers"][0].rstrip("/")
            response = await http.get(f"{issuer}/.well-known/oauth-authorization-server")
            response.raise_for_status()
            return response.json()

    async def _register(self, metadata: dict[str, Any], redirect_uri: str) -> dict[str, Any]:
        """
        Register Jarvis as an OAuth client (RFC 7591 dynamic client registration).

        Args:
            metadata: The authorization server metadata.
            redirect_uri: Where the browser is sent after the user approves.
        Returns:
            The registration, with "client_id" and possibly "client_secret".
        """
        body = {
            "client_name": "Jarvis",
            "redirect_uris": [redirect_uri],
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "token_endpoint_auth_method": "none",
        }
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as http:
            response = await http.post(metadata["registration_endpoint"], json=body)
        if response.status_code >= 400:
            raise McpAuthError(f"Registering with {self.server_url} failed: HTTP {response.status_code} {response.text[:300]}")
        return response.json()

    async def _request_tokens(self, saved: dict[str, Any], fields: dict[str, str]) -> dict[str, Any]:
        """
        Exchange a code or refresh token at the token endpoint.

        Args:
            saved: The saved client registration, with the token endpoint.
            fields: The grant's form fields.
        Returns:
            The token response, with "access_token" and usually "refresh_token" and "expires_in".
        """
        form = dict(fields)
        form["client_id"] = saved["client_id"]
        form["resource"] = self.server_url
        if saved.get("client_secret"):
            form["client_secret"] = saved["client_secret"]

        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as http:
            try:
                response = await http.post(saved["token_endpoint"], data=form)
            except httpx.HTTPError as e:
                raise McpAuthError(f"Could not reach {saved['token_endpoint']}: {e}") from None
        if response.status_code >= 400:
            raise McpAuthError(
                f"Getting a token for {self.server_url} failed: HTTP {response.status_code} {response.text[:300]}"
            )
        return response.json()

    def _load(self) -> dict[str, Any]:
        """
        Read the saved registration and tokens.

        Returns:
            The saved fields.
        """
        try:
            return json.loads(self.token_path.read_text())
        except FileNotFoundError:
            raise McpAuthError(f"Not signed in to {self.server_url}") from None

    def _save_tokens(self, saved: dict[str, Any], tokens: dict[str, Any]) -> None:
        """
        Save new tokens next to the client registration, readable only by the user.

        Args:
            saved: The client registration fields.
            tokens: The token endpoint's response.
        """
        updated = dict(saved)
        updated["access_token"] = tokens["access_token"]
        # Some servers rotate refresh tokens; others keep the old one valid.
        if "refresh_token" in tokens:
            updated["refresh_token"] = tokens["refresh_token"]
        expires_in = tokens.get("expires_in", 3600)
        updated["expires_at"] = time.time() + expires_in

        self.token_path.parent.mkdir(parents=True, exist_ok=True)
        self.token_path.write_text(json.dumps(updated, indent=2))
        self.token_path.chmod(0o600)


@dataclass(frozen=True)
class HostedMcpServer:
    """A remote MCP server the model provider may call during a reply."""

    # Short name the provider shows in calls, like "notion".
    label: str
    url: str
    # Tells the model what the server is for.
    description: str
    auth: McpOAuth


async def _wait_for_code(port: int, state: str) -> str:
    """
    Serve the OAuth redirect on localhost until the browser delivers the code.

    Args:
        port: The local port to listen on.
        state: The state value sent with the request, which the redirect must echo.
    Returns:
        The authorization code.
    """
    result: asyncio.Future[str] = asyncio.get_running_loop().create_future()

    async def handle(request: web.Request) -> web.Response:
        if request.query.get("state") != state:
            return web.Response(text="This sign-in link is out of date. Start again from the terminal.", status=400)
        if "error" in request.query:
            if not result.done():
                result.set_exception(McpAuthError(f"Access was not approved: {request.query['error']}"))
            return web.Response(text="Access was not approved. You can close this tab.")
        if not result.done():
            result.set_result(request.query.get("code", ""))
        return web.Response(text="Jarvis is signed in. You can close this tab.")

    app = web.Application()
    app.router.add_get(CALLBACK_PATH, handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    try:
        return await asyncio.wait_for(result, SIGN_IN_TIMEOUT_SECONDS)
    finally:
        await runner.cleanup()
