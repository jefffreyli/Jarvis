"""
The terminal chat. TerminalInput reads typed lines on a background thread, so the chat
can wait for typing and for other things (like voice commands) at the same time. The
chat runs each typed message through the agent and prints tool calls, results and
answers as they arrive; typed approvals discard anything typed in advance, so an early
"y" cannot approve a command the user has not seen. Typing "voice" starts voice mode,
in which spoken commands are run and the answers are spoken too, until the user
presses Enter or says "Jarvis, stop listening".
"""

import asyncio
import logging
import sys
import threading

import sounddevice

import config
from agent.loop import AgentLoop, Channel
from agent.models import Message, Role, ToolCall
from agent.prompts import TERMINAL_INSTRUCTIONS, VOICE_INSTRUCTIONS
from agent.session import Session

from .approval import describe_call, is_yes
from .speech import SpeechError
from .voice import VoiceMode

logger = logging.getLogger(__name__)

EXIT_COMMANDS = ("exit", "quit")
VOICE_COMMAND = "voice"
PREVIEW_CHARS = 200


class TerminalInput:
    """Lines typed in the terminal, readable from any coroutine."""

    def __init__(self):
        # None marks the end of input (Ctrl-D).
        self._lines: asyncio.Queue[str | None] = asyncio.Queue()

    def start(self) -> None:
        """Start reading stdin on a background thread."""
        loop = asyncio.get_running_loop()
        thread = threading.Thread(target=self._read_lines, args=(loop,), daemon=True)
        thread.start()

    async def read_line(self, prompt: str = "") -> str | None:
        """
        Wait for the next typed line.

        Args:
            prompt: Text to show before waiting.
        Returns:
            The line without its newline, or None at the end of input.
        """
        if prompt != "":
            print(prompt, end="", flush=True)
        line = await self._lines.get()
        if line is None:
            # Keep the end-of-input marker for the next reader too.
            self._lines.put_nowait(None)
        return line

    def discard_pending(self) -> None:
        """Drop lines typed before now, keeping the end-of-input marker."""
        ended = False
        while not self._lines.empty():
            if self._lines.get_nowait() is None:
                ended = True
        if ended:
            self._lines.put_nowait(None)

    def _read_lines(self, loop: asyncio.AbstractEventLoop) -> None:
        """
        Read stdin line by line, handing each line to the event loop.

        Args:
            loop: The event loop that owns the queue.
        """
        while True:
            line = sys.stdin.readline()
            if line == "":
                loop.call_soon_threadsafe(self._lines.put_nowait, None)
                return
            loop.call_soon_threadsafe(self._lines.put_nowait, line.rstrip("\n"))


class TerminalGateway:
    """A chat with Jarvis in the terminal, by typing or by voice."""

    def __init__(self, agent: AgentLoop):
        """
        Args:
            agent: The agent that answers.
        """
        self.agent = agent
        self.session = Session()
        self.input = TerminalInput()

    async def run(self) -> None:
        """Read and answer typed messages until the user exits."""
        self.input.start()
        print(f"Jarvis ({config.MODEL}). Type '{VOICE_COMMAND}' for voice mode or 'exit' to quit.\n")

        channel = Channel(approve=self._approve_typed, instructions=TERMINAL_INSTRUCTIONS, name="terminal")
        while True:
            line = await self.input.read_line("You: ")
            if line is None:
                break

            line = line.strip()
            if line == "":
                continue
            if line.lower() in EXIT_COMMANDS:
                break
            if line.lower() == VOICE_COMMAND:
                await self._voice_mode()
                continue

            await self._run_turn(line, channel)

    async def _run_turn(self, user_input: str, channel: Channel) -> str | None:
        """
        Run one user message through the agent and print what it produces.

        Args:
            user_input: The user's message.
            channel: Where the message came from.
        Returns:
            The final answer, or None if the turn failed.
        """
        answer = None
        try:
            async for message in self.agent.run(self.session, user_input, channel):
                print_message(message)
                if message.role == Role.ASSISTANT and len(message.tool_calls) == 0:
                    answer = message.content
        except Exception as e:
            # One failed turn should not end the chat. Details go to the log file.
            logger.exception("Turn failed")
            print(f"\n[Error] {type(e).__name__}: {e}\n")
            return None
        return answer

    async def _approve_typed(self, call: ToolCall) -> bool:
        """
        Ask the user in the terminal whether a tool call may run.

        Args:
            call: The tool call awaiting approval.
        Returns:
            True if the user typed yes.
        """
        self.input.discard_pending()
        answer = await self.input.read_line(f"  Allow {call.name}({call.arguments})? [y/N] ")
        if answer is None:
            return False
        return is_yes(answer)

    async def _voice_mode(self) -> None:
        """Talk with Jarvis by voice until the user presses Enter or says to stop."""
        try:
            api_key = config.smallest_api_key()
        except RuntimeError as e:
            print(e)
            return

        voice = VoiceMode(
            api_key,
            wake_phrase=config.WAKE_PHRASE,
            model=config.SPEECH_TO_TEXT_MODEL,
            language=config.SPEECH_LANGUAGE,
            voice=config.TEXT_TO_SPEECH_VOICE,
        )
        try:
            await voice.start()
        except (SpeechError, sounddevice.PortAudioError) as e:
            await voice.stop()
            print(f"Could not start voice mode: {e}")
            return

        print("Voice mode on. Say \"Hey Jarvis\" and then your request.")
        print("Press Enter or say \"Jarvis, stop listening\" to go back to typing.\n")

        self.input.discard_pending()
        conversation = asyncio.create_task(self._voice_conversation(voice))
        enter_pressed = asyncio.create_task(self.input.read_line())
        try:
            await asyncio.wait({conversation, enter_pressed}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            conversation.cancel()
            enter_pressed.cancel()
            await voice.stop()

        if conversation.done() and not conversation.cancelled() and conversation.exception() is not None:
            error = conversation.exception()
            logger.error("Voice mode failed", exc_info=error)
            print(f"\n[Voice] Voice mode stopped: {type(error).__name__}: {error}")
        print("Voice mode off.\n")

    async def _voice_conversation(self, voice: VoiceMode) -> None:
        """
        Run spoken commands and speak the answers until the user says to stop.

        Args:
            voice: The running voice mode.
        """

        async def approve(call: ToolCall) -> bool:
            return await self._approve_spoken(voice, call)

        channel = Channel(approve=approve, instructions=VOICE_INSTRUCTIONS, name="voice")
        while True:
            command = await voice.next_command()
            if command is None:
                return

            answer = await self._run_turn(command, channel)
            if answer is None:
                answer = "Sorry, something went wrong. The details are on screen."
            await voice.speak(answer)

    async def _approve_spoken(self, voice: VoiceMode, call: ToolCall) -> bool:
        """
        Show a tool call on screen and ask the user out loud whether it may run.

        Args:
            voice: The running voice mode.
            call: The tool call awaiting approval.
        Returns:
            True if the user said a clear yes.
        """
        print(f"  Approval needed:\n{describe_call(call)}")
        tool_name = call.name.replace("_", " ")
        answer = await voice.ask(f"I need your OK to use {tool_name}. The details are on screen. Should I go ahead?")

        approved = is_yes(answer)
        if approved:
            print("  Approved.")
        else:
            print("  Not approved.")
        return approved


def print_message(message: Message) -> None:
    """
    Print one message produced by the agent.

    Args:
        message: An assistant or tool message.
    """
    if message.role == Role.TOOL:
        image_note = ""
        if len(message.images) > 0:
            image_note = f" [{len(message.images)} image(s)]"
        print(f"  <- {preview(message.content)}{image_note}")
        return

    for hosted in message.hosted_calls:
        print(f"  -> {hosted.server}: {hosted.name}({hosted.arguments})")
        print(f"  <- {preview(hosted.output)}")
    if message.content != "":
        print(f"\nJarvis: {message.content}\n")
    for call in message.tool_calls:
        print(f"  -> {call.name}({call.arguments})")


def preview(text: str) -> str:
    """
    Shorten text to a single line for display.

    Args:
        text: The text to shorten.
    Returns:
        The first line, cut to PREVIEW_CHARS characters.
    """
    lines = text.strip().splitlines()
    if len(lines) == 0:
        return "(empty)"

    first_line = lines[0]
    if len(first_line) > PREVIEW_CHARS or len(lines) > 1:
        return first_line[:PREVIEW_CHARS] + " ..."
    return first_line
