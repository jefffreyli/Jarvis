"""
The run_shell tool runs a command through the user's shell in a fresh process, so
pipes, redirects and globs work but nothing carries over between calls. Commands that
are plainly read-only (a single allowlisted program with no shell operators) run
without approval; everything else asks the user first. Each command gets a fixed time
limit, after which the command and any processes it started are killed. Output is
stdout and stderr combined, labeled with the exit code.
"""

import asyncio
import os
import shlex
import signal
from pathlib import Path

from .base import ToolError, tool

TIMEOUT_SECONDS = 60
MAX_OUTPUT_BYTES = 1_000_000
READ_CHUNK_BYTES = 64 * 1024

# Programs that only read, as long as they are used without any shell operators.
READ_ONLY_PROGRAMS = {
    "cat", "date", "df", "du", "grep", "head", "ls", "pwd", "stat", "tail", "wc",
    "which", "whoami",
}
READ_ONLY_GIT_SUBCOMMANDS = {"diff", "log", "show", "status"}

# Characters that let one command run, chain or substitute another, or write a file.
SHELL_OPERATORS = (";", "&", "|", "`", "$", ">", "<", "(", ")", "\n", "\\")

# Secrets the shell should never see, since the LLM could print them.
HIDDEN_ENV_VARS = ("OPENAI_API_KEY",)


def is_read_only(command: str) -> bool:
    """
    Check whether a command is a single allowlisted read-only program. Anything this
    cannot be sure about counts as not read-only.

    Args:
        command: The shell command string.
    Returns:
        True if the command can safely run without approval.
    """
    for operator in SHELL_OPERATORS:
        if operator in command:
            return False

    try:
        words = shlex.split(command)
    except ValueError:
        return False  # Unbalanced quotes.
    if len(words) == 0:
        return False

    program = words[0]
    if program in READ_ONLY_PROGRAMS:
        return True

    # Only plain `git <subcommand>`, so options like `git -c` placed first still ask.
    if program == "git" and len(words) >= 2 and words[1] in READ_ONLY_GIT_SUBCOMMANDS:
        for word in words:
            if word.startswith("--output"):
                return False  # git diff/log --output writes a file.
        return True

    return False


def needs_approval(command: str, cwd: str) -> bool:
    """
    Approval rule for run_shell: ask unless the command is read-only.

    Args:
        command: The shell command string.
        cwd: The working directory (does not affect the decision).
    Returns:
        True if the user must approve the command.
    """
    del cwd  # Received because approval rules get every tool argument, but unused.
    return not is_read_only(command)


@tool(requires_approval=needs_approval)
async def run_shell(command: str, cwd: str = "~") -> str:
    """
    Run a shell command on the user's computer and return its output. Each call runs in
    a new shell, so `cd` and exported variables do not carry over; use `cwd` or chain
    commands with &&. Commands cannot read input and are killed after 60 seconds.

    Args:
        command: The command to run, e.g. "ls -la | head". Pipes and redirects work.
        cwd: Directory to run in. A leading ~ means the user's home directory.
    Returns:
        The exit code followed by the combined stdout and stderr.
    """
    working_directory = Path(cwd).expanduser()
    if not working_directory.is_dir():
        raise ToolError(f"No directory found at {working_directory}")

    shell = os.environ.get("SHELL", "/bin/sh")
    environment = dict(os.environ)
    for name in HIDDEN_ENV_VARS:
        environment.pop(name, None)

    process = await asyncio.create_subprocess_exec(
        shell,
        "-c",
        command,
        cwd=working_directory,
        env=environment,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        # Own process group, so a timeout can kill everything the command started.
        start_new_session=True,
    )

    output = bytearray()
    timed_out = False
    try:
        await asyncio.wait_for(_collect_output(process, output), TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        timed_out = True
    finally:
        # Also runs if the agent is cancelled, so no command is left running.
        _kill_process_group(process)

    await process.wait()
    return _format_result(output, process.returncode, timed_out)


async def _collect_output(process: asyncio.subprocess.Process, output: bytearray) -> None:
    """
    Read the process's output until it closes, then wait for the process to exit.

    Args:
        process: The running command.
        output: Buffer to append output to. Stops growing at MAX_OUTPUT_BYTES, but
            reading continues so the command is never blocked on a full pipe.
    """
    while True:
        chunk = await process.stdout.read(READ_CHUNK_BYTES)
        if chunk == b"":
            break

        space_left = MAX_OUTPUT_BYTES - len(output)
        if space_left > 0:
            output.extend(chunk[:space_left])

    await process.wait()


def _kill_process_group(process: asyncio.subprocess.Process) -> None:
    """
    Kill the command and everything it started, if anything is still running.

    Args:
        process: The command's shell process, which leads its own process group.
    """
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass  # Everything already exited.


def _format_result(output: bytearray, exit_code: int | None, timed_out: bool) -> str:
    """
    Describe how the command ended, followed by its output.

    Args:
        output: Everything the command printed.
        exit_code: The shell's exit code.
        timed_out: Whether the command was killed for running too long.
    Returns:
        A header line and the output text.
    """
    if timed_out:
        header = f"[Timed out after {TIMEOUT_SECONDS}s and was killed. Output so far:]"
    else:
        header = f"[Exit code {exit_code}]"

    text = output.decode("utf-8", errors="replace").strip()
    if text == "":
        text = "(no output)"
    if len(output) >= MAX_OUTPUT_BYTES:
        text += f"\n[Output stopped being recorded after {MAX_OUTPUT_BYTES} bytes]"

    return header + "\n" + text
