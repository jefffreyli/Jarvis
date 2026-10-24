"""
Loads image files so the LLM can look at them. Every image is converted to JPEG with
macOS's built-in `sips` tool, which handles formats the model cannot read (like the
iPhone's HEIC), and images larger than MAX_DIMENSION pixels are shrunk. Keeping images
small matters because the whole conversation, images included, is resent on every step.
"""

import asyncio
import re
import tempfile
from pathlib import Path

from agent.models import Image

from .base import ToolError

MAX_DIMENSION = 1024
CONVERT_TIMEOUT_SECONDS = 30
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".heic", ".heif", ".gif", ".tif", ".tiff", ".bmp", ".webp")


def is_image(path: Path) -> bool:
    """
    Check whether a file looks like an image, by its extension.

    Args:
        path: The file.
    Returns:
        True for common image formats.
    """
    return path.suffix.lower() in IMAGE_EXTENSIONS


async def load_image(path: Path) -> Image:
    """
    Load an image file as a JPEG, shrinking it if it is larger than MAX_DIMENSION.

    Args:
        path: The image file.
    Returns:
        The JPEG image.
    """
    if not path.is_file():
        raise ToolError(f"No file found at {path}")
    if not is_image(path):
        raise ToolError(f"{path.name} is not a supported image ({', '.join(IMAGE_EXTENSIONS)})")

    sips_arguments = ["-s", "format", "jpeg"]
    # sips -Z also enlarges small images, so only pass it when shrinking.
    if await _longest_side(path) > MAX_DIMENSION:
        sips_arguments += ["-Z", str(MAX_DIMENSION)]

    with tempfile.TemporaryDirectory() as directory:
        converted = Path(directory) / "image.jpg"
        await _run_sips(sips_arguments + [str(path), "--out", str(converted)])
        if not converted.is_file():
            raise ToolError(f"Could not convert {path.name} to JPEG")
        data = converted.read_bytes()

    return Image("image/jpeg", data)


async def _longest_side(path: Path) -> int:
    """
    Read an image's longest side in pixels.

    Args:
        path: The image file.
    Returns:
        The larger of its width and height.
    """
    output = await _run_sips(["-g", "pixelWidth", "-g", "pixelHeight", str(path)])

    sizes = []
    for match in re.finditer(r"pixel(?:Width|Height): (\d+)", output):
        sizes.append(int(match.group(1)))
    if len(sizes) == 0:
        raise ToolError(f"Could not read the size of {path.name}; it may not be a valid image")
    return max(sizes)


async def _run_sips(arguments: list[str]) -> str:
    """
    Run macOS's sips image tool.

    Args:
        arguments: Arguments after "sips".
    Returns:
        What sips printed.
    """
    process = await asyncio.create_subprocess_exec(
        "sips",
        *arguments,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), CONVERT_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        raise ToolError("Converting the image took too long") from None

    if process.returncode != 0:
        raise ToolError(f"sips failed: {stderr.decode(errors='replace').strip()}")
    return stdout.decode(errors="replace")
