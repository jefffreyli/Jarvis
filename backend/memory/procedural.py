"""
Procedural memory: skills, which are Markdown how-to docs the user writes by hand or the
agent saves (such as how the user likes their weekly review done). Each skill is one
Markdown file in the skills directory, like `skills/weekly-review.md`: the file name is
the skill's name, and the file starts with a header giving a one-line description. The
index of names and descriptions goes into every system prompt; the full text is only
loaded when the agent reads a skill.
"""

import re
from dataclasses import dataclass
from pathlib import Path

# Lowercase words joined by hyphens. Also keeps names from escaping the skills directory.
SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
HEADER_FENCE = "---"


@dataclass(frozen=True)
class Skill:
    """A how-to doc the agent can follow."""

    name: str
    description: str
    instructions: str


class SkillLibrary:
    """Reads and writes skill files in one directory."""

    def __init__(self, directory: Path):
        """
        Args:
            directory: Folder holding the skill files. Created if missing.
        """
        directory.mkdir(parents=True, exist_ok=True)
        self._directory = directory

    def all(self) -> list[Skill]:
        """
        Load every skill, skipping files that are not valid skills.

        Returns:
            Skills sorted by name.
        """
        skills = []
        for path in sorted(self._directory.glob("*.md")):
            if SKILL_NAME_PATTERN.match(path.stem) is None:
                continue
            skill = _parse_skill(path.stem, path.read_text(encoding="utf-8"))
            if skill is not None:
                skills.append(skill)
        return skills

    def read(self, name: str) -> Skill | None:
        """
        Load one skill by name.

        Args:
            name: The skill's name.
        Returns:
            The skill, or None if there is no valid skill with that name.
        """
        path = self._path(name)
        if not path.is_file():
            return None
        return _parse_skill(name, path.read_text(encoding="utf-8"))

    def save(self, skill: Skill) -> None:
        """
        Create or replace a skill.

        Args:
            skill: The skill to write.
        """
        description = skill.description.strip()
        if description == "" or "\n" in description:
            raise ValueError("A skill needs a one-line description")

        text = (
            f"{HEADER_FENCE}\n"
            f"description: {description}\n"
            f"{HEADER_FENCE}\n\n"
            f"{skill.instructions.strip()}\n"
        )
        self._path(skill.name).write_text(text, encoding="utf-8")

    def _path(self, name: str) -> Path:
        """
        Get the file path for a skill name.

        Args:
            name: The skill's name.
        Returns:
            The path of its Markdown file.
        """
        if SKILL_NAME_PATTERN.match(name) is None:
            raise ValueError(f"Invalid skill name '{name}'. Use lowercase words joined by hyphens.")
        return self._directory / f"{name}.md"


def _parse_skill(name: str, text: str) -> Skill | None:
    """
    Parse a skill file: a header between --- lines with a description, then the
    instructions.

    Args:
        name: The skill's name, from its file name.
        text: The file's contents.
    Returns:
        The skill, or None if the header is missing or incomplete.
    """
    lines = text.splitlines()
    if len(lines) == 0 or lines[0].strip() != HEADER_FENCE:
        return None

    header = {}
    body_start = None
    for index in range(1, len(lines)):
        line = lines[index]
        if line.strip() == HEADER_FENCE:
            body_start = index + 1
            break
        key, separator, value = line.partition(":")
        if separator == ":":
            header[key.strip()] = value.strip()

    if body_start is None:
        return None
    if header.get("description", "") == "":
        return None

    instructions = "\n".join(lines[body_start:]).strip()
    return Skill(name, header["description"], instructions)
