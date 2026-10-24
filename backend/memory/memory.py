"""
Groups the agent's three kinds of long-term memory so they can be passed around as one
object: semantic facts about the user, episodic records of past conversations, and
procedural skills. The agent reads facts and the skill index into every system prompt
and records each finished turn as an episode; the memory tools let the LLM use all three.
"""

from dataclasses import dataclass

from .episodic import EpisodeStore
from .procedural import SkillLibrary
from .semantic import FactStore


@dataclass(frozen=True)
class Memory:
    """The agent's long-term memory."""

    facts: FactStore
    episodes: EpisodeStore
    skills: SkillLibrary
