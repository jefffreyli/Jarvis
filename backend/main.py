"""
Starts Jarvis. It opens long-term memory and builds the OpenAI client (with Notion's
hosted MCP server when signed in), starts the local canvas-mcp server when Canvas is
set up, and builds the tool registry and the agent loop, then runs the gateways the user can reach Jarvis
through: the terminal chat (which also switches into voice mode) and, when the AWS
relay is configured, the Alexa skill and texting over iMessage through Linq. Every step the agent takes is streamed over a local
websocket so the frontend can show it live, and everything is logged to jarvis.log.
Quitting the terminal chat shuts everything down.
Run from the backend directory: `python3 main.py`.
"""

import asyncio
import logging

from mcp import McpError

import config
import logs
from agent.loop import AgentLoop
from agent.trace import HarnessObserver, report_storage
from api.websocket import HarnessBroadcaster
from gateways.alexa import AlexaGateway
from gateways.linq import LinqClient, LinqError
from gateways.relay import RelayClient
from gateways.terminal import TerminalGateway
from gateways.text import TextGateway
from gateways.transcript import Transcript
from llm.hosted_mcp import HostedMcpServer, McpOAuth
from llm.openai_client import OpenAIClient, OpenAIEmbedder
from memory.episodic import EpisodeStore
from memory.memory import Memory
from memory.procedural import SkillLibrary
from memory.semantic import FactStore
from storage.database import Database
from tools.apple_calendar import create_event, delete_event, list_calendars, list_events, update_event
from tools.base import BaseTool
from tools.contacts import lookup_contact
from tools.filesystem import read_file, view_image
from tools.gmail import draft_email, organize_emails, read_email, read_thread, search_emails, send_email
from tools.imessage import list_chats, read_messages, search_messages, send_imessage, view_attachment
from tools.local_mcp import LocalMcpServer
from tools.memory import memory_tools
from tools.phone import check_phone_call, make_phone_call
from tools.registry import ToolRegistry
from tools.shell import run_shell

logger = logging.getLogger(__name__)


def report(message: str) -> None:
    """
    Show a startup message in the terminal and log it.

    Args:
        message: What to say.
    """
    print(message)
    logger.info(message)


def build_mcp_servers() -> list[HostedMcpServer]:
    """
    Build the remote MCP servers OpenAI may call, leaving out any not signed in to.

    Returns:
        The servers.
    """
    notion_auth = McpOAuth(config.NOTION_MCP_URL, config.NOTION_TOKEN_PATH, config.OAUTH_CALLBACK_PORT)
    if not notion_auth.signed_in():
        report("Notion is off. To turn it on, run `python3 sign_in_notion.py` in backend/.")
        return []

    notion = HostedMcpServer(
        label="notion",
        url=config.NOTION_MCP_URL,
        description="The user's Notion workspace: search, read, create and edit pages and databases.",
        auth=notion_auth,
    )
    return [notion]


async def start_canvas() -> LocalMcpServer | None:
    """
    Start the canvas-mcp server, reporting in the terminal if Canvas is off.

    Returns:
        The running server, or None when Canvas is not set up or the server fails to start.
    """
    try:
        command, token, url = config.canvas_settings()
    except RuntimeError as e:
        report(str(e))
        return None

    env = dict(config.CANVAS_SERVER_ENV)
    env["CANVAS_API_TOKEN"] = token
    env["CANVAS_API_URL"] = url
    canvas = LocalMcpServer("canvas", command, env)
    try:
        await canvas.start()
    except (OSError, McpError, TimeoutError) as e:
        logger.exception("Starting the canvas-mcp server failed")
        print(f"Canvas is off: the canvas-mcp server did not start ({e}). See {config.LOG_PATH}.")
        await canvas.stop()
        return None
    return canvas


def build_agent(database: Database, observer: HarnessObserver, extra_tools: list[BaseTool]) -> AgentLoop:
    """
    Build the agent with its model, long-term memory and tools.

    Args:
        database: The database that holds long-term memory.
        observer: Receives an event for each step the agent takes.
        extra_tools: Tools from local MCP servers, added to the built-in ones.
    Returns:
        The agent loop.
    """
    api_key = config.openai_api_key()
    llm = OpenAIClient(
        api_key=api_key,
        model=config.MODEL,
        timeout=config.OPENAI_TIMEOUT_SECONDS,
        max_retries=config.OPENAI_MAX_RETRIES,
        mcp_servers=build_mcp_servers(),
    )
    embedder = OpenAIEmbedder(
        api_key=api_key,
        model=config.EMBEDDING_MODEL,
        timeout=config.OPENAI_TIMEOUT_SECONDS,
        max_retries=config.OPENAI_MAX_RETRIES,
    )
    memory = Memory(
        facts=FactStore(database),
        episodes=EpisodeStore(database, embedder),
        skills=SkillLibrary(config.SKILLS_DIR),
    )

    tools: list[BaseTool] = [
        read_file,
        view_image,
        run_shell,
        list_calendars,
        list_events,
        create_event,
        update_event,
        delete_event,
        lookup_contact,
        list_chats,
        read_messages,
        search_messages,
        view_attachment,
        send_imessage,
        search_emails,
        read_email,
        read_thread,
        send_email,
        draft_email,
        organize_emails,
        make_phone_call,
        check_phone_call,
    ]
    tools += memory_tools(memory)
    tools += extra_tools

    return AgentLoop(llm, ToolRegistry(tools), memory, observer=observer)


def build_relay() -> RelayClient | None:
    """
    Build the connection to the AWS relay, if it is configured.

    Returns:
        The relay client, or None when the relay is not set up.
    """
    try:
        url, secret = config.relay_settings()
    except RuntimeError as e:
        report(str(e))
        return None
    return RelayClient(url, secret)


def build_text_gateway(agent: AgentLoop, relay: RelayClient) -> TextGateway | None:
    """
    Build the texting gateway, if Linq is configured.

    Args:
        agent: The agent that answers texts.
        relay: The relay connection that Linq webhooks arrive on.
    Returns:
        The gateway, or None when texting is not set up.
    """
    try:
        api_key, allowed_numbers, webhook_url = config.linq_settings()
    except RuntimeError as e:
        report(str(e))
        return None
    return TextGateway(agent, LinqClient(api_key), allowed_numbers, relay, webhook_url)


async def start_texting(gateway: TextGateway) -> None:
    """
    Start the texting gateway, reporting the outcome in the terminal.

    Args:
        gateway: The gateway to start.
    """
    try:
        await gateway.start()
    except LinqError as e:
        logger.exception("Starting the texting gateway failed")
        print(f"\n[Text] Texting is off: {e}")
        return
    print()
    report("[Text] Texting is on. Text the Linq number to reach Jarvis.")


async def start_harness(broadcaster: HarnessBroadcaster) -> bool:
    """
    Start the harness websocket, reporting a failure in the terminal.

    Args:
        broadcaster: The websocket server to start.
    Returns:
        True if it started.
    """
    try:
        await broadcaster.start()
    except OSError as e:
        logger.exception("Starting the harness websocket failed")
        print(f"[Harness] Live view is off: {e}")
        return False
    return True


async def run(database: Database, broadcaster: HarnessBroadcaster) -> None:
    """
    Start Canvas, build the agent and serve it until the user quits, then stop Canvas.

    Args:
        database: The database that holds long-term memory.
        broadcaster: Streams the agent's events to the frontend.
    """
    # Canvas starts first because its tools are listed by the running server.
    canvas = await start_canvas()
    try:
        canvas_tools: list[BaseTool] = []
        if canvas is not None:
            canvas_tools += canvas.tools
        agent = build_agent(database, broadcaster.publish, canvas_tools)
        await serve(agent, broadcaster)
    finally:
        if canvas is not None:
            await canvas.stop()


async def serve(agent: AgentLoop, broadcaster: HarnessBroadcaster) -> None:
    """
    Run the terminal chat, with Alexa, texting and the live harness view in the
    background, until the user quits.

    Args:
        agent: The agent behind every gateway.
        broadcaster: Streams the agent's events to the frontend.
    """
    terminal = TerminalGateway(agent)
    harness_started = await start_harness(broadcaster)

    relay = build_relay()
    texting = None
    starting = None
    if relay is not None:
        AlexaGateway(agent, relay, Transcript(config.ALEXA_TRANSCRIPT_PATH))
        texting = build_text_gateway(agent, relay)
        await relay.start()
        if texting is not None:
            starting = asyncio.create_task(start_texting(texting))

    try:
        await terminal.run()
    finally:
        if starting is not None:
            starting.cancel()
        if texting is not None:
            await texting.stop()
        if relay is not None:
            await relay.stop()
        if harness_started:
            await broadcaster.stop()


def main() -> None:
    """Configure logging and run Jarvis."""
    logs.setup_logging(config.LOG_PATH)
    logger.info("Jarvis starting with model %s", config.MODEL)

    database = Database(config.DATABASE_PATH, on_access=report_storage)
    try:
        broadcaster = HarnessBroadcaster(config.HARNESS_PORT, config.HARNESS_ALLOWED_ORIGINS)
        asyncio.run(run(database, broadcaster))
    except RuntimeError as e:
        print(e)
    except KeyboardInterrupt:
        pass
    finally:
        database.close()
    logger.info("Jarvis stopped")
    print("\nGoodbye.")


if __name__ == "__main__":
    main()
