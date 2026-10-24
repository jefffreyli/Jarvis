"""
Tools that let the LLM use long-term memory: save and forget facts about the user,
search past conversations by meaning, and read or save skills. Unlike the other tool
modules, these tools need the user's Memory, so memory_tools builds them around a given
Memory instead of defining them at module level. Saving a skill asks the user first,
since skills change how the agent behaves in future conversations.
"""

from memory.memory import Memory
from memory.procedural import Skill

from .base import Tool, ToolError, tool

MAX_SEARCH_RESULTS = 20
MAX_EPISODE_SIDE_CHARS = 1_000


def memory_tools(memory: Memory) -> list[Tool]:
    """
    Build the memory tools for one Memory.

    Args:
        memory: The long-term memory the tools read and write.
    Returns:
        The tools, ready to register.
    """

    @tool(requires_approval=False)
    async def remember_fact(fact: str) -> str:
        """
        Save a durable fact about the user to long-term memory, such as a preference,
        a person in their life or a routine. Saved facts appear in every future
        conversation. Do not save temporary details or anything already remembered.

        Args:
            fact: The fact as one short sentence, e.g. "The user is vegetarian."
        Returns:
            A confirmation with the fact's id.
        """
        saved = memory.facts.add(fact)
        return f"Remembered fact #{saved.id}: {saved.text}"

    @tool(requires_approval=False)
    async def forget_fact(fact_id: int) -> str:
        """
        Remove a fact from long-term memory, e.g. when it is wrong or out of date.

        Args:
            fact_id: The fact's id, as shown in the list of remembered facts.
        Returns:
            A confirmation naming the removed fact.
        """
        removed = memory.facts.remove(fact_id)
        if removed is None:
            raise ToolError(f"No fact with id {fact_id}")
        return f"Forgot fact #{removed.id}: {removed.text}"

    @tool(requires_approval=False)
    async def search_memory(query: str, limit: int = 5) -> str:
        """
        Search past conversations with the user by meaning. Use it when earlier
        conversations might help, e.g. "what did we decide about the trip?".

        Args:
            query: What to look for, described in a few words or a sentence.
            limit: Maximum number of past exchanges to return, from 1 to 20.
        Returns:
            The most related past exchanges with their dates, most related first.
        """
        if limit < 1 or limit > MAX_SEARCH_RESULTS:
            raise ToolError(f"limit must be between 1 and {MAX_SEARCH_RESULTS}")

        results = await memory.episodes.search(query, limit)
        if len(results) == 0:
            return "No past conversations found."

        descriptions = []
        for episode, similarity in results:
            date = episode.created_at.astimezone().strftime("%Y-%m-%d %H:%M")
            descriptions.append(
                f"- [{date}, similarity {similarity:.2f}]\n"
                f"  User: {_shorten(episode.user_text)}\n"
                f"  Jarvis: {_shorten(episode.assistant_text)}"
            )
        return "\n".join(descriptions)

    @tool(requires_approval=False)
    async def read_skill(name: str) -> str:
        """
        Load a skill's full instructions. Do this before a task that one of the listed
        skills covers, then follow the instructions.

        Args:
            name: The skill's name, as shown in the list of skills.
        Returns:
            The skill's instructions.
        """
        skill = memory.skills.read(name)
        if skill is None:
            raise ToolError(f"No skill named '{name}'")
        return skill.instructions

    @tool
    async def save_skill(name: str, description: str, instructions: str) -> str:
        """
        Create or replace a skill: instructions for how the user likes a task done, so
        it is done the same way in future conversations. Replacing an existing skill
        overwrites it, so read it first and include anything worth keeping.

        Args:
            name: Lowercase words joined by hyphens, e.g. "weekly-review".
            description: One line saying when to use the skill.
            instructions: The full how-to, in Markdown.
        Returns:
            A confirmation naming the saved skill.
        """
        memory.skills.save(Skill(name, description, instructions))
        return f"Saved skill '{name}'."

    return [remember_fact, forget_fact, search_memory, read_skill, save_skill]


def _shorten(text: str) -> str:
    """
    Cap text at MAX_EPISODE_SIDE_CHARS for search results.

    Args:
        text: The text to shorten.
    Returns:
        The text, cut with " ..." if it was too long.
    """
    if len(text) <= MAX_EPISODE_SIDE_CHARS:
        return text
    return text[:MAX_EPISODE_SIDE_CHARS] + " ..."
