"""
Prompt templates for the agent: the main system prompt and the prompts used to
compact old history into a summary. Prompts are rendered at request time so they can
include fresh values such as the current date, remembered facts, the skill index and
the latest summary.
"""

from datetime import datetime

from memory.procedural import Skill
from memory.semantic import Fact

SYSTEM_PROMPT = """\
You are Jarvis, a personal assistant for a single user.

Be concise and direct. Use the available tools when they help you complete a task, \
and prefer acting over asking when the request is clear. If a tool fails, read the \
error and try a different approach before giving up. Never invent tool results. \
Use the dedicated tools for the user's messages, email, calendar, contacts and files; \
do not use run_shell to read app data directly or to work around something a tool \
cannot do. If no tool can do what is asked, say so.

You have long-term memory. When the user tells you something lasting about themselves \
(preferences, people, routines), save it with remember_fact. When earlier conversations \
might help, use search_memory. When the user explains how they like a task done, offer \
to save it as a skill.

Current date and time: {now}
"""

FACTS_SECTION = """
What you know about the user (fact ids are for forget_fact):
{facts}
"""

SKILLS_SECTION = """
Skills you can follow. Before a task one of these covers, load it with read_skill:
{skills}
"""

CHANNEL_SECTION = """
How the user is talking to you right now:
{instructions}
"""

TERMINAL_INSTRUCTIONS = "The user is typing in a terminal chat."

VOICE_INSTRUCTIONS = """\
The user is speaking to you, and your reply will be read aloud. Their message is a \
speech transcript, so it may contain misheard words; read it for the most likely \
meaning. Reply in one to three short spoken sentences: no markdown, lists, links, \
emoji or code. Round numbers and times the way a person would say them.\
"""

TEXT_INSTRUCTIONS = """\
The user is texting you over iMessage. Reply like a text message: short, plain text, \
no markdown formatting. Use line breaks rather than bullet symbols for lists.\
"""

SUMMARY_SECTION = """
Summary of the earlier conversation (older messages were compacted):
{summary}
"""

COMPACTION_SYSTEM_PROMPT = """\
You compact conversations between a user and their assistant, Jarvis. Write a summary \
that lets Jarvis continue the conversation without the original messages. Preserve the \
user's goals, preferences and personal details, decisions made, important facts and \
tool results, and any unfinished tasks or open questions. Be concise and write in plain \
prose or short bullet points.
"""

PREVIOUS_SUMMARY_SECTION = """
The conversation below continues from this earlier summary. Merge it into your summary:
{summary}
"""

COMPACTION_REQUEST = "Summarize the conversation above. Reply with only the summary."


def build_system_prompt(
    summary: str | None = None,
    facts: list[Fact] | None = None,
    skills: list[Skill] | None = None,
    channel_instructions: str = "",
    now: datetime | None = None,
) -> str:
    """
    Render the main system prompt.

    Args:
        summary: Summary of compacted history to include, if any.
        facts: Remembered facts about the user, if any.
        skills: Skills the agent can load, if any.
        channel_instructions: How the user is talking to Jarvis right now, if known.
        now: Time to show the model. Defaults to the current local time.
    Returns:
        The rendered system prompt.
    """
    if now is None:
        now = datetime.now().astimezone()

    formatted_time = now.strftime("%A, %B %d, %Y %H:%M %Z")
    prompt = SYSTEM_PROMPT.format(now=formatted_time)

    if facts:
        fact_lines = []
        for fact in facts:
            fact_lines.append(f"- [#{fact.id}] {fact.text}")
        prompt += FACTS_SECTION.format(facts="\n".join(fact_lines))

    if skills:
        skill_lines = []
        for skill in skills:
            skill_lines.append(f"- {skill.name}: {skill.description}")
        prompt += SKILLS_SECTION.format(skills="\n".join(skill_lines))

    if channel_instructions != "":
        prompt += CHANNEL_SECTION.format(instructions=channel_instructions)

    if summary is not None:
        prompt += SUMMARY_SECTION.format(summary=summary)
    return prompt


def build_compaction_prompt(previous_summary: str | None) -> str:
    """
    Render the system prompt for a compaction request.

    Args:
        previous_summary: The existing summary to merge in, if any.
    Returns:
        The rendered compaction system prompt.
    """
    prompt = COMPACTION_SYSTEM_PROMPT

    if previous_summary is not None:
        prompt += PREVIOUS_SUMMARY_SECTION.format(summary=previous_summary)
    return prompt
