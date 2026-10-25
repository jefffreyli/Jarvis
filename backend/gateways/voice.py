"""
Voice mode: Jarvis listens through the microphone and answers out loud. The microphone
streams to Smallest AI's speech-to-text the whole time voice mode is on, and
WakeListener reads the transcripts for the wake phrase ("Hey Jarvis"). Nothing is shown
until the phrase is heard; after it, the user's words appear live in the terminal and
become a command when they pause. Saying the phrase alone makes the next sentence the
command. While Jarvis works or speaks, the microphone is muted so it never hears
itself. "Jarvis, stop listening" ends voice mode. If the connection to Smallest AI
drops it is reopened, and voice mode ends after repeated failures.
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from enum import Enum

import websockets

import logs

from .speech import Microphone, SpeechError, SpeechToText, TextToSpeech, Transcript

logger = logging.getLogger(__name__)

# How long after a bare "Hey Jarvis" to wait for the command.
AWAKE_TIMEOUT_SECONDS = 8
# How long to wait for a spoken answer to a question.
ANSWER_TIMEOUT_SECONDS = 15
# A connection that drops sooner than this counts as a failure.
SHORT_CONNECTION_SECONDS = 10
MAX_CONNECTION_FAILURES = 3
LIVE_LABEL = "You (voice): "


class HeardKind(Enum):
    """What a transcript means for voice mode."""

    NOTHING = "nothing"  # Not addressed to Jarvis.
    PARTIAL = "partial"  # A command still being spoken.
    WAKE = "wake"  # The wake phrase alone; the command comes next.
    COMMAND = "command"  # A finished command.
    STOP = "stop"  # The user asked Jarvis to stop listening.


@dataclass(frozen=True)
class Heard:
    """The meaning of one transcript."""

    kind: HeardKind
    text: str = ""


class WakeListener:
    """Finds commands addressed to Jarvis in a stream of transcripts."""

    def __init__(self, wake_phrase: str):
        """
        Args:
            wake_phrase: Words that start a command, like "hey jarvis".
        """
        # Match the words with any punctuation between them, like "Hey, Jarvis."
        words = wake_phrase.lower().split()
        self._wake_pattern = re.compile(r"\b" + r"\W+".join(words) + r"\b\W*", re.IGNORECASE)
        self._stop_pattern = re.compile(r"\b" + re.escape(words[-1]) + r"\W+stop listening\b", re.IGNORECASE)
        # True after a bare wake phrase, while waiting for the command.
        self.awake = False

    def hear(self, transcript: Transcript) -> Heard:
        """
        Interpret the next transcript.

        Args:
            transcript: A partial or final transcript of the current sentence.
        Returns:
            What the transcript means.
        """
        if transcript.is_final and self._stop_pattern.search(transcript.text) is not None:
            self.awake = False
            return Heard(HeardKind.STOP)

        if self.awake:
            command = transcript.text
        else:
            match = self._wake_pattern.search(transcript.text)
            if match is None:
                return Heard(HeardKind.NOTHING)
            command = transcript.text[match.end() :].strip()

        if not transcript.is_final:
            return Heard(HeardKind.PARTIAL, command)
        if command == "":
            self.awake = True
            return Heard(HeardKind.WAKE)

        self.awake = False
        if _is_stop_request(command):
            return Heard(HeardKind.STOP)
        return Heard(HeardKind.COMMAND, command)

    def fall_asleep(self) -> None:
        """Stop waiting for a command after a bare wake phrase."""
        self.awake = False


class VoiceMode:
    """The microphone, transcription and speech for one stretch of voice mode."""

    def __init__(self, api_key: str, wake_phrase: str, model: str, language: str, voice: str):
        """
        Args:
            api_key: Smallest AI API key.
            wake_phrase: Words that start a command, like "hey jarvis".
            model: The speech-to-text model.
            language: The spoken language code.
            voice: The text-to-speech voice.
        """
        self._api_key = api_key
        self._model = model
        self._language = language
        self._listener = WakeListener(wake_phrase)
        self._speaker = TextToSpeech(api_key, voice)
        self._microphone = Microphone()
        self._transcripts: asyncio.Queue[Transcript | Exception] = asyncio.Queue()
        self._listening = False
        self._live_line_shown = False
        self._stream_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Connect to speech-to-text and start the microphone."""
        speech_to_text = SpeechToText(self._api_key, self._model, self._language)
        await speech_to_text.connect()
        self._microphone.start()
        self._stream_task = asyncio.create_task(self._stream(speech_to_text))

    async def stop(self) -> None:
        """Stop the microphone and close the connection."""
        self._clear_live_line()
        self._microphone.stop()
        if self._stream_task is not None:
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass
            self._stream_task = None

    async def next_command(self) -> str | None:
        """
        Listen until the user says a command to Jarvis, showing it live as they speak.

        Returns:
            The command, or None if the user asked Jarvis to stop listening.
        """
        self._set_listening(True)
        try:
            while True:
                timeout = None
                if self._listener.awake:
                    timeout = AWAKE_TIMEOUT_SECONDS
                try:
                    transcript = await self._next_transcript(timeout)
                except asyncio.TimeoutError:
                    self._listener.fall_asleep()
                    self._clear_live_line()
                    continue

                heard = self._listener.hear(transcript)
                if transcript.is_final and heard.kind != HeardKind.NOTHING:
                    logger.info("Heard %s: %s", heard.kind.value, logs.snippet(transcript.text))
                if heard.kind == HeardKind.NOTHING:
                    self._clear_live_line()
                elif heard.kind == HeardKind.PARTIAL or heard.kind == HeardKind.WAKE:
                    self._show_live_line(heard.text)
                elif heard.kind == HeardKind.COMMAND:
                    self._finish_live_line(heard.text)
                    return heard.text
                else:
                    self._clear_live_line()
                    return None
        finally:
            self._set_listening(False)

    async def ask(self, question: str) -> str:
        """
        Ask the user a question out loud and listen for their answer, with no wake phrase.

        Args:
            question: What to ask.
        Returns:
            What they said, or "" if they said nothing in time.
        """
        print(f"Jarvis: {question}")
        await self.speak(question)

        self._set_listening(True)
        try:
            while True:
                try:
                    transcript = await self._next_transcript(ANSWER_TIMEOUT_SECONDS)
                except asyncio.TimeoutError:
                    self._clear_live_line()
                    logger.info("No spoken answer within %d seconds", ANSWER_TIMEOUT_SECONDS)
                    return ""

                if not transcript.is_final:
                    self._show_live_line(transcript.text)
                elif transcript.text != "":
                    self._finish_live_line(transcript.text)
                    logger.info("Heard answer: %s", logs.snippet(transcript.text))
                    return transcript.text
        finally:
            self._set_listening(False)

    async def speak(self, text: str) -> None:
        """
        Say text out loud. A failure is shown rather than raised, since the text is
        already on screen.

        Args:
            text: What to say.
        """
        try:
            await self._speaker.speak(text)
        except SpeechError as e:
            logger.exception("Speaking failed")
            print(f"[Voice] Could not speak the reply: {e}")

    async def _stream(self, speech_to_text: SpeechToText) -> None:
        """
        Send microphone audio and collect transcripts, reconnecting if the connection
        drops. After repeated quick failures, the error is passed to the listener.

        Args:
            speech_to_text: An open connection to use first.
        """
        failures = 0
        while True:
            opened_at = time.monotonic()
            sender = asyncio.create_task(self._send_audio(speech_to_text))
            error: Exception | None = None
            try:
                async for transcript in speech_to_text.transcripts():
                    self._transcripts.put_nowait(transcript)
            except (SpeechError, websockets.ConnectionClosed, OSError) as e:
                error = e
            finally:
                sender.cancel()
                await speech_to_text.close()

            if time.monotonic() - opened_at < SHORT_CONNECTION_SECONDS:
                failures += 1
            else:
                failures = 0
            if failures >= MAX_CONNECTION_FAILURES:
                if error is None:
                    error = SpeechError("The speech-to-text connection keeps closing")
                self._transcripts.put_nowait(error)
                return

            logger.warning("Speech-to-text connection ended (%s); reconnecting", error)
            await asyncio.sleep(1)
            speech_to_text = SpeechToText(self._api_key, self._model, self._language)
            try:
                await speech_to_text.connect()
            except SpeechError as e:
                self._transcripts.put_nowait(e)
                return

    async def _send_audio(self, speech_to_text: SpeechToText) -> None:
        """
        Forward microphone audio to speech-to-text, dropping it while muted.

        Args:
            speech_to_text: The open connection.
        """
        while True:
            chunk = await self._microphone.chunks.get()
            if self._listening:
                await speech_to_text.send_audio(chunk)

    async def _next_transcript(self, timeout: float | None) -> Transcript:
        """
        Wait for the next transcript.

        Args:
            timeout: Seconds to wait, or None to wait forever.
        Returns:
            The transcript.
        """
        item = await asyncio.wait_for(self._transcripts.get(), timeout)
        if isinstance(item, Exception):
            raise item
        return item

    def _set_listening(self, listening: bool) -> None:
        """
        Unmute or mute the microphone. Unmuting drops transcripts of anything heard
        before, such as the end of Jarvis's own speech.

        Args:
            listening: True to send audio to speech-to-text.
        """
        if listening:
            while not self._transcripts.empty():
                item = self._transcripts.get_nowait()
                if isinstance(item, Exception):
                    raise item
        self._listening = listening

    def _show_live_line(self, text: str) -> None:
        """
        Show what the user is saying, replacing the previous partial transcript.

        Args:
            text: The words so far.
        """
        print(f"\r\033[K{LIVE_LABEL}{text}", end="", flush=True)
        self._live_line_shown = True

    def _finish_live_line(self, text: str) -> None:
        """
        Show the final transcript and move to the next line.

        Args:
            text: The words.
        """
        print(f"\r\033[K{LIVE_LABEL}{text}")
        self._live_line_shown = False

    def _clear_live_line(self) -> None:
        """Erase the live transcript line, if one is showing."""
        if self._live_line_shown:
            print("\r\033[K", end="", flush=True)
            self._live_line_shown = False


def _is_stop_request(command: str) -> bool:
    """
    Check whether a command asks Jarvis to stop listening.

    Args:
        command: The words after the wake phrase.
    Returns:
        True for commands like "stop listening".
    """
    words = re.sub(r"[^a-z ]", "", command.lower())
    return " ".join(words.split()) == "stop listening"
