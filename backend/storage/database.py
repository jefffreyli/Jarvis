"""
A small wrapper around one SQLite database file. It creates the schema on first use
and offers two operations: `write` for statements that change data (each runs in its
own transaction) and `read` for queries. All tables Jarvis stores are defined here in
SCHEMA, so the full data layout can be seen in one place. An optional callback is told
about every read and write, which lets the harness view show storage activity live.
"""

import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    user_text TEXT NOT NULL,
    assistant_text TEXT NOT NULL,
    embedding BLOB,
    embedding_model TEXT
);

CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    text TEXT NOT NULL
);
"""


class Database:
    """A connection to Jarvis's SQLite database."""

    def __init__(self, path: Path, on_access: Callable[[str], None] | None = None):
        """
        Open the database, creating the file and tables if they do not exist.

        Args:
            path: Location of the database file.
            on_access: Called with "read" or "write" after each successful query.
        """
        self._on_access = on_access
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path)
        self._connection.row_factory = sqlite3.Row
        self._connection.executescript(SCHEMA)

    def write(self, sql: str, parameters: tuple[Any, ...] = ()) -> int:
        """
        Run a statement that changes data, committing it immediately.

        Args:
            sql: The SQL statement, with ? placeholders.
            parameters: Values for the placeholders.
        Returns:
            The id of the inserted row, for INSERT statements.
        """
        with self._connection:
            cursor = self._connection.execute(sql, parameters)
        self._report("write")
        return cursor.lastrowid

    def read(self, sql: str, parameters: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        """
        Run a query.

        Args:
            sql: The SQL query, with ? placeholders.
            parameters: Values for the placeholders.
        Returns:
            The matching rows. Columns can be read by name, like row["text"].
        """
        cursor = self._connection.execute(sql, parameters)
        rows = cursor.fetchall()
        self._report("read")
        return rows

    def _report(self, operation: str) -> None:
        """
        Tell the access callback about a finished query, if there is one.

        Args:
            operation: "read" or "write".
        """
        if self._on_access is not None:
            self._on_access(operation)

    def close(self) -> None:
        """Close the connection."""
        self._connection.close()
