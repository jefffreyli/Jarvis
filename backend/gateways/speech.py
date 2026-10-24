"""
Audio in and out for voice mode. Microphone records 16 kHz mono 16-bit audio from the
Mac's default input device. SpeechToText streams that audio over a websocket to Smallest
AI's Pulse model, which sends back partial transcripts while someone speaks and a final
transcript when they pause. TextToSpeech turns a reply into a WAV file with Smallest
AI's Lightning voices and plays it with macOS's built-in `afplay`.
"""

import asyncio
import json
import tempfile
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode

import httpx
import sounddevice
import websockets

SAMPLE_RATE = 16_000
# 2048 samples of 16-bit audio is 4096 bytes, the chunk size Smallest AI recommends.
CHUNK_SAMPLES = 2048
SPEECH_TO_TEXT_URL = "wss://api.smallest.ai/waves/v1/stt/live"
TEXT_TO_SPEECH_URL = "https://api.smallest.ai/waves/v1/tts"
TEXT_TO_SPEECH_SAMPLE_RATE = 24_000
TEXT_TO_SPEECH_TIMEOUT_SECONDS = 30
MAX_SPOKEN_CHARS = 8000


class SpeechError(Exception):
    """Raised when transcription or speech fails."""


@dataclass(frozen=True)
class Transcript:
    """What the speech-to-text model heard. Partial transcripts are replaced by later ones."""

    text: str
    is_final: bool


class Microphone:
    """Records from the default input device into an asyncio queue of audio chunks."""

    def __init__(self):
        self.chunks: asyncio.Queue[bytes] = asyncio.Queue()
        self._stream: sounddevice.RawInputStream | None = None

    def start(self) -> None:
        """Start recording. macOS asks for microphone permission the first time."""
        loop = asyncio.get_running_loop()

        # sounddevice calls this from its own audio thread, so hand chunks to the event loop.
        def on_audio(data, frames, time, status) -> None:
            loop.call_soon_threadsafe(self.chunks.put_nowait, bytes(data))

        self._stream = sounddevice.RawInputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=CHUNK_SAMPLES,
            callback=on_audio,
        )
        self._stream.start()

    def stop(self) -> None:
        """Stop recording."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


class SpeechToText:
    """One streaming transcription connection to Smallest AI."""

    def __init__(self, api_key: str, model: str, language: str):
        """
        Args:
            api_key: Smallest AI API key.
            model: The speech-to-text model, such as "pulse".
            language: The spoken language code, such as "en".
        """
        query = urlencode(
            {
                "model": model,
                "language": language,
                "sample_rate": SAMPLE_RATE,
                "encoding": "linear16",
            }
        )
        self._url = f"{SPEECH_TO_TEXT_URL}?{query}"
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._connection: websockets.ClientConnection | None = None

    async def connect(self) -> None:
        """Open the websocket."""
        try:
            self._connection = await websockets.connect(self._url, additional_headers=self._headers)
        except (OSError, websockets.InvalidHandshake) as e:
            raise SpeechError(f"Could not connect to Smallest AI speech-to-text: {e}") from None

    async def close(self) -> None:
        """Close the websocket."""
        if self._connection is not None:
            await self._connection.close()
            self._connection = None

    async def send_audio(self, chunk: bytes) -> None:
        """
        Send a chunk of 16 kHz mono 16-bit audio.

        Args:
            chunk: The raw audio.
        """
        await self._connection.send(chunk)

    async def transcripts(self) -> AsyncIterator[Transcript]:
        """
        Receive transcripts until the connection closes.

        Returns:
            An async iterator of partial and final transcripts.
        """
        async for raw in self._connection:
            message = json.loads(raw)
            message_type = message.get("type")
            if message_type == "error":
                raise SpeechError(f"Smallest AI speech-to-text error: {message.get('message')}")
            if message_type != "transcription":
                continue

            text = message.get("transcript")
            if text is None:
                text = ""
            yield Transcript(text=text.strip(), is_final=message.get("is_final") is True)


class TextToSpeech:
    """Speaks text aloud with a Smallest AI voice."""

    def __init__(self, api_key: str, voice: str):
        """
        Args:
            api_key: Smallest AI API key.
            voice: The voice id, such as "magnus".
        """
        self.voice = voice
        self._headers = {"Authorization": f"Bearer {api_key}", "Accept": "audio/wav"}

    async def speak(self, text: str) -> None:
        """
        Say text out loud and wait until it finishes. Cancelling stops playback.

        Args:
            text: What to say.
        """
        text = text.strip()
        if text == "":
            return
        if len(text) > MAX_SPOKEN_CHARS:
            text = text[:MAX_SPOKEN_CHARS]

        audio = await self._synthesize(text)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "speech.wav"
            path.write_bytes(audio)
            await _play(path)

    async def _synthesize(self, text: str) -> bytes:
        """
        Turn text into speech audio.

        Args:
            text: What to say.
        Returns:
            The speech as WAV file bytes.
        """
        body = {
            "text": text,
            "voice_id": self.voice,
            "sample_rate": TEXT_TO_SPEECH_SAMPLE_RATE,
            "output_format": "wav",
        }
        async with httpx.AsyncClient(timeout=TEXT_TO_SPEECH_TIMEOUT_SECONDS) as client:
            try:
                response = await client.post(TEXT_TO_SPEECH_URL, headers=self._headers, json=body)
            except httpx.HTTPError as e:
                raise SpeechError(f"Could not reach Smallest AI text-to-speech: {e}") from None

        if response.status_code != 200:
            raise SpeechError(f"Smallest AI text-to-speech returned HTTP {response.status_code}: {response.text[:300]}")
        return response.content


async def _play(path: Path) -> None:
    """
    Play an audio file through the speakers, stopping it if the caller is cancelled.

    Args:
        path: The audio file.
    """
    process = await asyncio.create_subprocess_exec(
        "afplay",
        str(path),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        await process.wait()
    except asyncio.CancelledError:
        process.kill()
        await process.wait()
        raise
