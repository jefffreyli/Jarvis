"""
Uses the @tool decorator that turns an async function into one.
The decorator reads the function's signature and docstring to build a Pydantic model
for its arguments. That model provides the JSON schema shown to the LLM and validates
the LLM's arguments before the function runs. The first part of the docstring becomes
the tool's description and the Args section becomes the parameter descriptions. A tool
returns text, or a ToolOutput when it also has images for the LLM to see.
"""

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, get_type_hints

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from agent.models import Image


@dataclass(frozen=True)
class ToolOutput:
    """What a tool returns: text, plus any images for the LLM to look at."""

    text: str
    images: tuple[Image, ...] = ()


ToolFunction = Callable[..., Awaitable[str | ToolOutput]]
ApprovalRule = bool | Callable[..., bool]

ARGS_HEADERS = ("Args:", "Params:", "Parameters:")
OTHER_HEADERS = ("Returns:", "Return:", "Raises:", "Yields:")


class ToolError(Exception):
    """Raised when a tool call cannot run, such as an unknown tool or invalid arguments."""


class Tool:
    """An async function the LLM can call, with a validated argument schema."""

    def __init__(self, function: ToolFunction, requires_approval: ApprovalRule):
        """
        Args:
            function: The async function to expose. Every parameter needs a type hint.
            requires_approval: Whether the user must approve each call before it runs.
                Either a fixed bool, or a function that receives the same arguments as
                the tool and returns whether this particular call needs approval.
        """
        if not inspect.iscoroutinefunction(function):
            raise TypeError(f"Tool '{function.__name__}' must be an async function")

        parameter_names = list(inspect.signature(function).parameters)
        description, parameter_descriptions = _parse_docstring(function.__doc__, parameter_names)
        if description == "":
            raise TypeError(f"Tool '{function.__name__}' needs a docstring describing what it does")

        self.name = function.__name__
        self.description = description
        self.requires_approval = requires_approval
        self._function = function
        self._args_model = _build_args_model(function, parameter_descriptions)

    def schema(self) -> dict[str, Any]:
        """
        Describe the tool for the LLM.

        Returns:
            A {"name", "description", "parameters"} dict, where parameters is a JSON schema.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self._args_model.model_json_schema(),
        }

    def parse_arguments(self, arguments: str) -> BaseModel:
        """
        Validate the LLM's JSON arguments against the tool's parameters.

        Args:
            arguments: JSON-encoded arguments from the tool call.
        Returns:
            The validated arguments.
        """
        # Models sometimes send an empty string for tools with no parameters.
        if arguments.strip() == "":
            arguments = "{}"

        try:
            return self._args_model.model_validate_json(arguments)
        except ValidationError as e:
            message = f"Invalid arguments for '{self.name}': {_describe_errors(e)}"
            if _has_unknown_arguments(e):
                message += f". Valid parameters: {self._parameter_list()}"
            raise ToolError(message) from e

    def _parameter_list(self) -> str:
        """
        List the tool's parameter names, for error messages.

        Returns:
            The names separated by commas, or "none".
        """
        names = list(self._args_model.model_fields)
        if len(names) == 0:
            return "none"
        return ", ".join(names)

    def needs_approval(self, args: BaseModel) -> bool:
        """
        Decide whether this call must be approved by the user before it runs.

        Args:
            args: Arguments returned by parse_arguments.
        Returns:
            True if the user must approve the call.
        """
        if isinstance(self.requires_approval, bool):
            return self.requires_approval
        return self.requires_approval(**_keyword_arguments(args))

    async def run(self, args: BaseModel) -> ToolOutput:
        """
        Call the tool's function with validated arguments.

        Args:
            args: Arguments returned by parse_arguments.
        Returns:
            The tool's output. Plain text results are wrapped in a ToolOutput.
        """
        output = await self._function(**_keyword_arguments(args))
        if isinstance(output, str):
            return ToolOutput(output)
        if isinstance(output, ToolOutput):
            return output
        raise ToolError(f"Tool '{self.name}' returned {type(output).__name__} instead of str or ToolOutput")


def tool(function: ToolFunction | None = None, *, requires_approval: ApprovalRule = True) -> Any:
    """
    Turn an async function into a Tool. Use as @tool, or as @tool(requires_approval=...)
    to set when the user must approve a call.

    Args:
        function: The function being decorated, when used as a bare @tool.
        requires_approval: A bool, or a function taking the tool's arguments that decides
            per call. Defaults to True.
    Returns:
        A Tool, or a decorator that makes one when called with options.
    """

    def make_tool(decorated: ToolFunction) -> Tool:
        """Wrap the decorated function in a Tool."""
        return Tool(decorated, requires_approval)

    if function is None:
        return make_tool
    return make_tool(function)


def _keyword_arguments(args: BaseModel) -> dict[str, Any]:
    """
    Turn validated arguments into keyword arguments for the tool's functions.

    Args:
        args: Arguments returned by Tool.parse_arguments.
    Returns:
        A parameter name to value map.
    """
    keyword_arguments = {}
    for field_name in type(args).model_fields:
        keyword_arguments[field_name] = getattr(args, field_name)
    return keyword_arguments


def _build_args_model(function: ToolFunction, parameter_descriptions: dict[str, str]) -> type[BaseModel]:
    """
    Build a Pydantic model whose fields match the function's parameters.

    Args:
        function: The tool function.
        parameter_descriptions: Description for each parameter, from the docstring.
    Returns:
        A model class that rejects unknown arguments.
    """
    type_hints = get_type_hints(function)
    fields: dict[str, Any] = {}

    for parameter in inspect.signature(function).parameters.values():
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            raise TypeError(f"Tool '{function.__name__}' cannot use *args or **kwargs")
        if parameter.name not in type_hints:
            raise TypeError(f"Tool '{function.__name__}' parameter '{parameter.name}' needs a type hint")

        if parameter.default is inspect.Parameter.empty:
            default = ...  # Pydantic's marker for a required field.
        else:
            default = parameter.default

        description = parameter_descriptions.get(parameter.name)
        fields[parameter.name] = (type_hints[parameter.name], Field(default, description=description))

    model_name = f"{function.__name__}_arguments"
    return create_model(model_name, __config__=ConfigDict(extra="forbid"), **fields)


def _parse_docstring(docstring: str | None, parameter_names: list[str]) -> tuple[str, dict[str, str]]:
    """
    Split a docstring into the tool description and per-parameter descriptions.

    Args:
        docstring: The function's docstring.
        parameter_names: Names of the function's parameters.
    Returns:
        The description (text before any section header) and a parameter to description map.
    """
    if docstring is None:
        return "", {}

    description_lines = []
    parameter_descriptions: dict[str, str] = {}
    section = "description"
    current_parameter = None

    for line in inspect.cleandoc(docstring).splitlines():
        stripped = line.strip()

        if stripped in ARGS_HEADERS:
            section = "args"
            continue
        if stripped in OTHER_HEADERS:
            section = "other"
            continue

        if section == "description":
            description_lines.append(stripped)
        elif section == "args" and stripped != "":
            name, separator, text = stripped.partition(":")
            if separator == ":" and name in parameter_names:
                current_parameter = name
                parameter_descriptions[name] = text.strip()
            elif current_parameter is not None:
                # A continuation line of the previous parameter's description.
                parameter_descriptions[current_parameter] += " " + stripped

    description = "\n".join(description_lines).strip()
    return description, parameter_descriptions


def _has_unknown_arguments(error: ValidationError) -> bool:
    """
    Check whether a validation error includes arguments the tool does not accept.

    Args:
        error: The validation error.
    Returns:
        True if any argument was unknown.
    """
    for detail in error.errors(include_url=False):
        if detail["type"] == "extra_forbidden":
            return True
    return False


def _describe_errors(error: ValidationError) -> str:
    """
    Summarize a Pydantic validation error in one line the LLM can act on.

    Args:
        error: The validation error.
    Returns:
        Each problem as "field: message", separated by semicolons.
    """
    problems = []
    for detail in error.errors(include_url=False):
        location_parts = []
        for part in detail["loc"]:
            location_parts.append(str(part))
        location = ".".join(location_parts)

        if location == "":
            problems.append(detail["msg"])
        else:
            problems.append(f"{location}: {detail['msg']}")
    return "; ".join(problems)
