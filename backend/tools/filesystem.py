"""
Tools for working with files on the user's computer: reading text files, and viewing
image files, which are converted and shrunk by the images module before the LLM sees them.
"""

import asyncio
from pathlib import Path

from .base import ToolError, ToolOutput, tool
from .images import load_image

MAX_READ_BYTES = 1_000_000


@tool(requires_approval=False)
async def read_file(path: str) -> str:
    """
    Read a UTF-8 text file from the user's computer.

    Args:
        path: Absolute path to the file. A leading ~ means the user's home directory.
    Returns:
        The file's contents.
    """
    file_path = Path(path).expanduser()
    if not file_path.is_file():
        raise ToolError(f"No file found at {file_path}")

    size = file_path.stat().st_size
    if size > MAX_READ_BYTES:
        raise ToolError(f"{file_path} is {size} bytes, over the {MAX_READ_BYTES} byte limit")

    return await asyncio.to_thread(file_path.read_text, encoding="utf-8")


@tool(requires_approval=False)
async def view_image(path: str) -> ToolOutput:
    """
    Look at an image file on the user's computer (JPEG, PNG, HEIC, GIF, TIFF, BMP, WebP).

    Args:
        path: Absolute path to the image. A leading ~ means the user's home directory.
    Returns:
        The image, for you to see.
    """
    file_path = Path(path).expanduser()
    image = await load_image(file_path)
    return ToolOutput(f"Image {file_path.name}:", (image,))
