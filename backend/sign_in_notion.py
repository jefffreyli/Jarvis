"""
Signs Jarvis in to Notion's hosted MCP server. It opens the browser so the user can
approve access to their workspace, then saves the OAuth tokens that main.py uses.
Run once from the backend directory: `python3 sign_in_notion.py`. Run it again to
switch workspaces or after revoking access.
"""

import asyncio

import config
from llm.hosted_mcp import McpAuthError, McpOAuth


def main() -> None:
    """Run the Notion sign-in and report the result."""
    auth = McpOAuth(config.NOTION_MCP_URL, config.NOTION_TOKEN_PATH, config.OAUTH_CALLBACK_PORT)
    try:
        asyncio.run(auth.sign_in())
    except McpAuthError as e:
        print(f"Sign-in failed: {e}")
        return
    except asyncio.TimeoutError:
        print("Sign-in timed out waiting for the browser. Run this again to retry.")
        return
    print(f"Signed in to Notion. Tokens saved to {config.NOTION_TOKEN_PATH}. Restart Jarvis to use them.")


if __name__ == "__main__":
    main()
