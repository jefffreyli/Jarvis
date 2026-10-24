"""
Episodic memory: what was said in past conversations, searchable by meaning. After
each turn, the user's message and the final answer are saved together as one episode,
and the episode's text is embedded into a vector stored next to it in SQLite. A search
embeds the query and ranks every episode by cosine similarity with numpy, which is fast
enough at personal scale that no vector database is needed. Episodes are saved before
they are embedded, so an embeddings outage never loses a conversation: episodes with a
missing (or outdated model's) vector are embedded on the next search.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from llm.base import Embedder
from storage.database import Database

logger = logging.getLogger(__name__)

# Embedding inputs are capped well under the model's limit (about 8k tokens).
MAX_EMBED_CHARS = 20_000
EMBED_BATCH_SIZE = 100


@dataclass(frozen=True)
class Episode:
    """One past exchange: a user message and the assistant's final answer."""

    id: int
    created_at: datetime
    user_text: str
    assistant_text: str


class EpisodeStore:
    """Saves conversation turns and finds the ones most related to a query."""

    def __init__(self, database: Database, embedder: Embedder):
        """
        Args:
            database: Where episodes and their vectors are stored.
            embedder: Turns episode text and queries into vectors.
        """
        self._database = database
        self._embedder = embedder

    async def save(self, session_id: str, user_text: str, assistant_text: str) -> None:
        """
        Save one turn, then try to embed it. If embedding fails, the episode is still
        saved and gets embedded on the next search.

        Args:
            session_id: The conversation the turn belongs to.
            user_text: The user's message.
            assistant_text: The assistant's final answer.
        """
        created_at = datetime.now(timezone.utc).isoformat()
        self._database.write(
            "INSERT INTO episodes (session_id, created_at, user_text, assistant_text) VALUES (?, ?, ?, ?)",
            (session_id, created_at, user_text, assistant_text),
        )

        try:
            await self._embed_missing()
        except Exception:
            logger.exception("Embedding episodes failed; will retry on the next search")

    async def search(self, query: str, limit: int) -> list[tuple[Episode, float]]:
        """
        Find the episodes most similar in meaning to a query.

        Args:
            query: What to look for.
            limit: Maximum number of episodes to return.
        Returns:
            (episode, similarity) pairs, most similar first. Similarity is from -1 to 1.
        """
        await self._embed_missing()

        rows = self._database.read(
            "SELECT id, created_at, user_text, assistant_text, embedding FROM episodes WHERE embedding_model = ?",
            (self._embedder.model,),
        )
        if len(rows) == 0:
            return []

        query_vectors = await self._embedder.embed([query])
        query_vector = _normalize(np.array(query_vectors[0], dtype=np.float32))

        episode_vectors = []
        for row in rows:
            episode_vectors.append(np.frombuffer(row["embedding"], dtype=np.float32))
        matrix = np.stack(episode_vectors)

        # Vectors are stored normalized, so a dot product is the cosine similarity.
        similarities = matrix @ query_vector
        best_first = np.argsort(-similarities)[:limit]

        results = []
        for index in best_first:
            row = rows[index]
            episode = Episode(
                id=row["id"],
                created_at=datetime.fromisoformat(row["created_at"]),
                user_text=row["user_text"],
                assistant_text=row["assistant_text"],
            )
            results.append((episode, float(similarities[index])))
        return results

    async def _embed_missing(self) -> None:
        """Embed every episode that has no vector from the current embedding model."""
        rows = self._database.read(
            "SELECT id, user_text, assistant_text FROM episodes "
            "WHERE embedding IS NULL OR embedding_model != ?",
            (self._embedder.model,),
        )

        for start in range(0, len(rows), EMBED_BATCH_SIZE):
            batch = rows[start : start + EMBED_BATCH_SIZE]

            texts = []
            for row in batch:
                texts.append(_episode_text(row["user_text"], row["assistant_text"]))
            vectors = await self._embedder.embed(texts)

            for row, vector in zip(batch, vectors):
                normalized = _normalize(np.array(vector, dtype=np.float32))
                self._database.write(
                    "UPDATE episodes SET embedding = ?, embedding_model = ? WHERE id = ?",
                    (normalized.tobytes(), self._embedder.model, row["id"]),
                )


def _episode_text(user_text: str, assistant_text: str) -> str:
    """
    Combine a turn into the text that gets embedded.

    Args:
        user_text: The user's message.
        assistant_text: The assistant's final answer.
    Returns:
        Both sides of the turn, capped at MAX_EMBED_CHARS.
    """
    text = f"User: {user_text}\nAssistant: {assistant_text}"
    return text[:MAX_EMBED_CHARS]


def _normalize(vector: np.ndarray) -> np.ndarray:
    """
    Scale a vector to length 1, so dot products between vectors equal cosine similarity.

    Args:
        vector: The vector to scale.
    Returns:
        The unit-length vector.
    """
    length = np.linalg.norm(vector)
    if length == 0:
        return vector
    return vector / length
