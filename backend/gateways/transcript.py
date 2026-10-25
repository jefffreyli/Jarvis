"""
A local text file holding the conversation from the current Alexa session. The first
utterance of a new session clears the file and writes a header, and every utterance and
spoken reply after that is appended as one line. Writing is best effort, so a disk
problem never stops Jarvis from answering.
"""

import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class Transcript:
    """The transcript file for the current Alexa session."""

    def __init__(self, path: Path):
        """
        Args:
            path: Where the transcript file lives.
        """
        self.path = path
        self.session_id: str | None = None

    def start(self, session_id: str) -> None:
        """
        Clear the file if this is a new Alexa session.

        Args:
            session_id: Alexa's id for the session the utterance belongs to.
        """
        if session_id == self.session_id:
            return
        self.session_id = session_id
        started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._write(f"Alexa session started {started}\n\n", mode="w")

    def add(self, speaker: str, text: str) -> None:
        """
        Append one line to the transcript.

        Args:
            speaker: Who spoke, like "You" or "Jarvis".
            text: What they said.
        """
        self._write(f"{speaker}: {text}\n", mode="a")

    def _write(self, text: str, mode: str) -> None:
        """
        Write to the transcript file, logging instead of raising on failure.

        Args:
            text: What to write.
            mode: "w" to replace the file, "a" to append.
        """
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, mode, encoding="utf-8") as file:
                file.write(text)
        except OSError:
            logger.exception("Writing the Alexa transcript failed")
