"""
Semantic memory: durable facts about the user, such as preferences, people and
routines. Facts are short sentences stored in the database. They are few enough that
all of them go into every system prompt, so the agent always knows them without
searching.
"""

from dataclasses import dataclass
from datetime import datetime, timezone

from storage.database import Database


@dataclass(frozen=True)
class Fact:
    """One remembered fact about the user."""

    id: int
    text: str


class FactStore:
    """Saves, removes and lists facts about the user."""

    def __init__(self, database: Database):
        """
        Args:
            database: Where facts are stored.
        """
        self._database = database

    def add(self, text: str) -> Fact:
        """
        Remember a new fact.

        Args:
            text: The fact, as a short sentence.
        Returns:
            The saved fact with its id.
        """
        text = text.strip()
        if text == "":
            raise ValueError("A fact cannot be empty")

        created_at = datetime.now(timezone.utc).isoformat()
        fact_id = self._database.write(
            "INSERT INTO facts (created_at, text) VALUES (?, ?)",
            (created_at, text),
        )
        return Fact(fact_id, text)

    def remove(self, fact_id: int) -> Fact | None:
        """
        Forget a fact.

        Args:
            fact_id: The id of the fact to forget.
        Returns:
            The removed fact, or None if no fact has that id.
        """
        rows = self._database.read("SELECT id, text FROM facts WHERE id = ?", (fact_id,))
        if len(rows) == 0:
            return None

        self._database.write("DELETE FROM facts WHERE id = ?", (fact_id,))
        return Fact(rows[0]["id"], rows[0]["text"])

    def all(self) -> list[Fact]:
        """
        List every fact, oldest first.

        Returns:
            All remembered facts.
        """
        facts = []
        for row in self._database.read("SELECT id, text FROM facts ORDER BY id"):
            facts.append(Fact(row["id"], row["text"]))
        return facts
