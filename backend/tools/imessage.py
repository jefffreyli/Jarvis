"""
Tools for reading and sending iMessages. Reading uses the Messages app's own SQLite
database (~/Library/Messages/chat.db), opened read-only, which requires Full Disk Access
for the app running Jarvis. Phone numbers and emails are shown as contact names when
the person is in Contacts. Sending goes through the Messages app with AppleScript, and
always asks the user first. A chat can be named by contact name, phone number, email,
or a chat id from list_chats (needed for group chats). Attachments are listed with ids,
and view_attachment shows a photo to the LLM.
"""

import asyncio
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from Foundation import NSData, NSUnarchiver

import config

from .base import ToolError, ToolOutput, tool
from .contacts import Person, contact_book, normalize_handle
from .images import is_image, load_image

APPLE_EPOCH = datetime(2001, 1, 1, tzinfo=timezone.utc)
GROUP_CHAT_STYLE = 43
SEARCH_SCAN_LIMIT = 5_000
MAX_LIMIT = 200
SEND_TIMEOUT_SECONDS = 30
ATTACHMENT_PLACEHOLDER = "￼"

NO_ACCESS_MESSAGE = (
    "Jarvis cannot read the Messages database. Give the app running Jarvis (e.g. Terminal) "
    "Full Disk Access in System Settings > Privacy & Security > Full Disk Access, then restart it."
)

# Arguments are passed through argv rather than pasted into the script, so message text
# can never be interpreted as AppleScript.
SEND_TO_PERSON_SCRIPT = """
on run argv
    tell application "Messages"
        set targetService to 1st account whose service type = iMessage
        send (item 2 of argv) to participant (item 1 of argv) of targetService
    end tell
end run
"""

SEND_TO_CHAT_SCRIPT = """
on run argv
    tell application "Messages"
        send (item 2 of argv) to chat id (item 1 of argv)
    end tell
end run
"""

# Real text messages only: no tapback reactions or group events like renames.
MESSAGE_COLUMNS = """
    message.ROWID AS id,
    message.text AS text,
    message.attributedBody AS attributed_body,
    message.cache_has_attachments AS has_attachments,
    message.is_from_me AS is_from_me,
    message.date AS date,
    handle.id AS sender,
    chat_message_join.chat_id AS chat_id
"""
MESSAGE_JOINS = """
    FROM message
    JOIN chat_message_join ON chat_message_join.message_id = message.ROWID
    LEFT JOIN handle ON handle.ROWID = message.handle_id
"""
ONLY_TEXT_MESSAGES = "message.associated_message_type = 0 AND message.item_type = 0"

ATTACHMENT_COLUMNS = """
    attachment.ROWID AS attachment_id,
    attachment.filename AS filename,
    attachment.mime_type AS mime_type,
    attachment.transfer_name AS transfer_name
"""


@dataclass(frozen=True)
class Chat:
    """One conversation: a 1:1 chat or a group chat."""

    id: int
    guid: str
    identifier: str
    display_name: str
    is_group: bool
    participants: tuple[str, ...]


@dataclass(frozen=True)
class Attachment:
    """A file sent in a message, such as a photo."""

    id: int
    path: Path | None
    mime_type: str
    name: str

    @property
    def is_image(self) -> bool:
        """
        Check whether the attachment is a photo or other image.

        Returns:
            True for images.
        """
        return self.mime_type.startswith("image/") or is_image(Path(self.name))


@dataclass(frozen=True)
class TextMessage:
    """One message in a chat."""

    chat_id: int
    sender: str | None
    is_from_me: bool
    sent_at: datetime
    text: str
    attachments: tuple[Attachment, ...]


class MessagesDatabase:
    """Read-only queries over the Messages app's database."""

    def __init__(self, path: Path):
        """
        Args:
            path: Location of chat.db.
        """
        self._path = path

    def recent_chats(self, limit: int) -> list[Chat]:
        """
        List chats by most recent activity.

        Args:
            limit: Maximum number of chats.
        Returns:
            Chats, most recently active first.
        """
        rows = self._query(
            "SELECT chat.ROWID AS id, MAX(message.date) AS last_date FROM chat "
            "JOIN chat_message_join ON chat_message_join.chat_id = chat.ROWID "
            "JOIN message ON message.ROWID = chat_message_join.message_id "
            "GROUP BY chat.ROWID ORDER BY last_date DESC LIMIT ?",
            (limit,),
        )
        chats = []
        for row in rows:
            chats.append(self.chat_by_id(row["id"]))
        return chats

    def chat_by_guid(self, guid: str) -> Chat | None:
        """
        Find a chat by its chat id (guid).

        Args:
            guid: The chat's guid, as shown by list_chats.
        Returns:
            The chat, or None if there is none.
        """
        rows = self._query("SELECT ROWID AS id FROM chat WHERE guid = ?", (guid,))
        if len(rows) == 0:
            return None
        return self.chat_by_id(rows[0]["id"])

    def direct_chats_with(self, handles: list[str]) -> list[Chat]:
        """
        Find 1:1 chats with any of the given phone numbers or emails.

        Args:
            handles: Phone numbers or emails, in any format.
        Returns:
            Matching 1:1 chats (there can be several, e.g. iMessage and SMS).
        """
        wanted = set()
        for handle in handles:
            wanted.add(normalize_handle(handle))

        rows = self._query(
            "SELECT ROWID AS id, chat_identifier FROM chat WHERE style != ?",
            (GROUP_CHAT_STYLE,),
        )
        chats = []
        for row in rows:
            if normalize_handle(row["chat_identifier"]) in wanted:
                chats.append(self.chat_by_id(row["id"]))
        return chats

    def messages_in(self, chat_ids: list[int], limit: int) -> list[TextMessage]:
        """
        Get the latest messages across one or more chats.

        Args:
            chat_ids: The chats to read.
            limit: Maximum number of messages.
        Returns:
            Messages, oldest first.
        """
        question_marks = []
        for _ in chat_ids:
            question_marks.append("?")
        placeholders = ", ".join(question_marks)
        rows = self._query(
            f"SELECT {MESSAGE_COLUMNS} {MESSAGE_JOINS} "
            f"WHERE chat_message_join.chat_id IN ({placeholders}) AND {ONLY_TEXT_MESSAGES} "
            "ORDER BY message.date DESC LIMIT ?",
            (*chat_ids, limit),
        )
        messages = self._to_messages(rows)
        messages.reverse()
        return messages

    def latest_messages(self, limit: int) -> list[TextMessage]:
        """
        Get the latest messages across all chats.

        Args:
            limit: Maximum number of messages.
        Returns:
            Messages, newest first.
        """
        rows = self._query(
            f"SELECT {MESSAGE_COLUMNS} {MESSAGE_JOINS} "
            f"WHERE {ONLY_TEXT_MESSAGES} ORDER BY message.date DESC LIMIT ?",
            (limit,),
        )
        return self._to_messages(rows)

    def attachment(self, attachment_id: int) -> Attachment | None:
        """
        Find an attachment by id.

        Args:
            attachment_id: The attachment's id, as shown in read_messages.
        Returns:
            The attachment, or None if there is none.
        """
        rows = self._query(
            f"SELECT {ATTACHMENT_COLUMNS} FROM attachment WHERE attachment.ROWID = ?",
            (attachment_id,),
        )
        if len(rows) == 0:
            return None
        return _to_attachment(rows[0])

    def _to_messages(self, rows: list[sqlite3.Row]) -> list[TextMessage]:
        """
        Convert message rows to TextMessages with their attachments, skipping rows with
        no readable content.

        Args:
            rows: Rows selected with MESSAGE_COLUMNS.
        Returns:
            The messages, in the same order.
        """
        message_ids = []
        for row in rows:
            message_ids.append(row["id"])
        attachments_by_message = self._attachments_for(message_ids)

        messages = []
        for row in rows:
            text = _message_text(row["text"], row["attributed_body"])
            attachments = attachments_by_message.get(row["id"], [])

            # Attachments that were never downloaded have no attachment row.
            if text == "" and len(attachments) == 0 and row["has_attachments"]:
                text = "[attachment not on this Mac]"
            if text == "" and len(attachments) == 0:
                continue

            messages.append(
                TextMessage(
                    chat_id=row["chat_id"],
                    sender=row["sender"],
                    is_from_me=bool(row["is_from_me"]),
                    sent_at=_from_apple_time(row["date"]),
                    text=text,
                    attachments=tuple(attachments),
                )
            )
        return messages

    def _attachments_for(self, message_ids: list[int]) -> dict[int, list[Attachment]]:
        """
        Load the attachments of several messages.

        Args:
            message_ids: The messages' ROWIDs.
        Returns:
            Each message's attachments, keyed by message id. Messages without any are absent.
        """
        if len(message_ids) == 0:
            return {}

        question_marks = []
        for _ in message_ids:
            question_marks.append("?")
        placeholders = ", ".join(question_marks)

        rows = self._query(
            f"SELECT message_attachment_join.message_id AS message_id, {ATTACHMENT_COLUMNS} "
            "FROM message_attachment_join "
            "JOIN attachment ON attachment.ROWID = message_attachment_join.attachment_id "
            f"WHERE message_attachment_join.message_id IN ({placeholders}) ORDER BY attachment.ROWID",
            tuple(message_ids),
        )

        attachments_by_message: dict[int, list[Attachment]] = {}
        for row in rows:
            if row["message_id"] not in attachments_by_message:
                attachments_by_message[row["message_id"]] = []
            attachments_by_message[row["message_id"]].append(_to_attachment(row))
        return attachments_by_message

    def chat_by_id(self, chat_id: int) -> Chat:
        """
        Load one chat with its participants.

        Args:
            chat_id: The chat's ROWID.
        Returns:
            The chat.
        """
        chat_row = self._query(
            "SELECT ROWID AS id, guid, chat_identifier, display_name, style FROM chat WHERE ROWID = ?",
            (chat_id,),
        )[0]
        participant_rows = self._query(
            "SELECT handle.id AS handle FROM chat_handle_join "
            "JOIN handle ON handle.ROWID = chat_handle_join.handle_id WHERE chat_handle_join.chat_id = ?",
            (chat_id,),
        )

        participants = []
        for row in participant_rows:
            participants.append(row["handle"])

        display_name = chat_row["display_name"]
        if display_name is None:
            display_name = ""

        return Chat(
            id=chat_row["id"],
            guid=chat_row["guid"],
            identifier=chat_row["chat_identifier"],
            display_name=display_name,
            is_group=chat_row["style"] == GROUP_CHAT_STYLE,
            participants=tuple(participants),
        )

    def _query(self, sql: str, parameters: tuple) -> list[sqlite3.Row]:
        """
        Run a read-only query, opening a fresh connection so results are never stale.

        Args:
            sql: The SQL query, with ? placeholders.
            parameters: Values for the placeholders.
        Returns:
            The matching rows.
        """
        try:
            connection = sqlite3.connect(f"file:{self._path}?mode=ro", uri=True)
        except sqlite3.OperationalError:
            raise ToolError(NO_ACCESS_MESSAGE) from None

        try:
            connection.row_factory = sqlite3.Row
            return connection.execute(sql, parameters).fetchall()
        except sqlite3.OperationalError as e:
            # Without Full Disk Access, the open succeeds but the first read fails.
            if "unable to open" in str(e) or "authorization denied" in str(e):
                raise ToolError(NO_ACCESS_MESSAGE) from None
            raise
        finally:
            connection.close()


messages_database = MessagesDatabase(config.MESSAGES_DATABASE_PATH)


# Tools


@tool(requires_approval=False)
async def list_chats(limit: int = 20) -> str:
    """
    List the user's most recent iMessage/SMS conversations with their latest message.

    Args:
        limit: Maximum number of chats, from 1 to 200.
    Returns:
        One entry per chat, most recent first, each with a chat id.
    """
    _check_limit(limit)
    chats = messages_database.recent_chats(limit)
    if len(chats) == 0:
        return "No chats found."

    entries = []
    for chat in chats:
        name = await _chat_name(chat)
        latest = messages_database.messages_in([chat.id], 1)

        entry = f"- {name}"
        if chat.is_group:
            entry += " (group)"
        if len(latest) > 0:
            entry += f"\n  latest: {await _format_message(latest[0])}"
        entry += f"\n  chat id: {chat.guid}"
        entries.append(entry)
    return "\n".join(entries)


@tool(requires_approval=False)
async def read_messages(chat: str, limit: int = 30) -> str:
    """
    Read the latest messages in a conversation.

    Args:
        chat: A contact name, phone number, email, or chat id from list_chats.
        limit: Maximum number of messages, from 1 to 200.
    Returns:
        The messages, oldest first, with times and senders.
    """
    _check_limit(limit)
    chats = await _find_chats(chat)
    if len(chats) == 0:
        return f"No conversation found with '{chat}'."

    chat_ids = []
    for found in chats:
        chat_ids.append(found.id)

    messages = messages_database.messages_in(chat_ids, limit)
    if len(messages) == 0:
        return f"No messages with '{chat}'."

    lines = []
    for message in messages:
        lines.append(await _format_message(message))
    return "\n".join(lines)


@tool(requires_approval=False)
async def search_messages(query: str, limit: int = 20) -> str:
    """
    Search recent iMessages/SMS for text (case-insensitive). Only the most recent 5,000
    messages are searched.

    Args:
        query: Text to look for, e.g. "dinner".
        limit: Maximum number of matches, from 1 to 200.
    Returns:
        Matching messages, newest first, with their conversation.
    """
    _check_limit(limit)
    query = query.strip().lower()
    if query == "":
        raise ToolError("query cannot be empty")

    chat_names: dict[int, str] = {}
    lines = []
    for message in messages_database.latest_messages(SEARCH_SCAN_LIMIT):
        if query not in message.text.lower():
            continue

        if message.chat_id not in chat_names:
            chat_names[message.chat_id] = await _chat_name(messages_database.chat_by_id(message.chat_id))
        lines.append(f"[{chat_names[message.chat_id]}] {await _format_message(message)}")

        if len(lines) >= limit:
            break

    if len(lines) == 0:
        return f"No recent messages contain '{query}'."
    return "\n".join(lines)


@tool(requires_approval=False)
async def view_attachment(attachment_id: int) -> ToolOutput:
    """
    Look at a photo sent in iMessage. Use the attachment id shown by read_messages,
    e.g. 123 for "[photo #123: IMG_4821.HEIC]".

    Args:
        attachment_id: The attachment's id.
    Returns:
        The photo, for you to see.
    """
    attachment = messages_database.attachment(attachment_id)
    if attachment is None:
        raise ToolError(f"No attachment with id {attachment_id}")
    if not attachment.is_image:
        raise ToolError(f"{attachment.name} is a {attachment.mime_type} file; only photos can be viewed")
    if attachment.path is None or not attachment.path.is_file():
        raise ToolError(f"{attachment.name} is not downloaded on this Mac (it may only be in iCloud)")

    image = await load_image(attachment.path)
    return ToolOutput(f"Photo {attachment.name}:", (image,))


@tool
async def send_imessage(to: str, text: str) -> str:
    """
    Send an iMessage. For a group chat, pass its chat id from list_chats.

    Args:
        to: A contact name, phone number, email, or group chat id.
        text: The message to send.
    Returns:
        A confirmation naming who it was sent to.
    """
    if text.strip() == "":
        raise ToolError("text cannot be empty")

    if _looks_like_chat_id(to):
        await _run_send_script(SEND_TO_CHAT_SCRIPT, to, text)
        return f"Sent to chat {to}."

    handle = await _resolve_handle(to)
    await _run_send_script(SEND_TO_PERSON_SCRIPT, handle, text)
    if handle == to.strip():
        return f"Sent to {handle}."
    return f"Sent to {to} ({handle})."


# Helpers


async def _find_chats(chat: str) -> list[Chat]:
    """
    Find the conversations a chat argument refers to.

    Args:
        chat: A contact name, phone number, email, or chat id.
    Returns:
        The matching chats.
    """
    if _looks_like_chat_id(chat):
        found = messages_database.chat_by_guid(chat)
        if found is None:
            return []
        return [found]

    if _looks_like_handle(chat):
        return messages_database.direct_chats_with([chat])

    person = await _find_one_person(chat)
    return messages_database.direct_chats_with(list(person.phones + person.emails))


async def _resolve_handle(to: str) -> str:
    """
    Turn a recipient into the phone number or email to send to. For a contact with
    several, prefer the one used in the most recent conversation.

    Args:
        to: A contact name, phone number or email.
    Returns:
        The phone number or email.
    """
    if _looks_like_handle(to):
        return to.strip()

    person = await _find_one_person(to)
    handles = list(person.phones + person.emails)
    if len(handles) == 0:
        raise ToolError(f"{person.name} has no phone number or email in Contacts.")
    if len(handles) == 1:
        return handles[0]

    try:
        recent_chats = messages_database.recent_chats(MAX_LIMIT)
    except ToolError:
        recent_chats = []  # Without Full Disk Access we cannot tell which one is used.

    for chat in recent_chats:
        if chat.is_group:
            continue
        for handle in handles:
            if normalize_handle(handle) == normalize_handle(chat.identifier):
                return handle

    raise ToolError(
        f"{person.name} has several numbers/emails: {', '.join(handles)}. Pass the one to use."
    )


async def _find_one_person(name: str) -> Person:
    """
    Find exactly one contact by name.

    Args:
        name: A full or partial name.
    Returns:
        The single matching Person.
    """
    matches = await contact_book.find(name)
    if len(matches) == 0:
        raise ToolError(f"No contact named '{name}'. Use a phone number or email instead.")
    if len(matches) > 1:
        names = []
        for person in matches[:10]:
            names.append(person.name)
        raise ToolError(f"Several contacts match '{name}': {', '.join(names)}. Be more specific.")
    return matches[0]


async def _run_send_script(script: str, target: str, text: str) -> None:
    """
    Run an AppleScript that sends a message through the Messages app.

    Args:
        script: SEND_TO_PERSON_SCRIPT or SEND_TO_CHAT_SCRIPT.
        target: The handle or chat id, passed as the script's first argument.
        text: The message, passed as the script's second argument.
    """
    process = await asyncio.create_subprocess_exec(
        "osascript",
        "-e",
        script,
        target,
        text,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        _, stderr = await asyncio.wait_for(process.communicate(), SEND_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        process.kill()
        raise ToolError("Messages did not respond in time; the message may not have been sent.") from None

    if process.returncode != 0:
        raise ToolError(f"Messages could not send: {stderr.decode().strip()}")


async def _chat_name(chat: Chat) -> str:
    """
    Name a chat the way the user would recognize it.

    Args:
        chat: The chat.
    Returns:
        The group's name, or the participants' contact names.
    """
    if chat.display_name != "":
        return chat.display_name

    handles = list(chat.participants)
    if len(handles) == 0:
        handles = [chat.identifier]

    names = []
    for handle in handles:
        names.append(await _person_name(handle))
    return ", ".join(names)


async def _person_name(handle: str) -> str:
    """
    Show a phone number or email as a contact name when possible.

    Args:
        handle: A phone number or email.
    Returns:
        "Name" if the handle is in Contacts, otherwise the handle itself.
    """
    try:
        name = await contact_book.name_for(handle)
    except ToolError:
        name = None  # Without Contacts access, fall back to raw handles.
    if name is None:
        return handle
    return name


async def _format_message(message: TextMessage) -> str:
    """
    Format one message for the LLM.

    Args:
        message: The message.
    Returns:
        Text like "[2026-10-07 14:03] Sam: see you soon".
    """
    sent_at = message.sent_at.astimezone().strftime("%Y-%m-%d %H:%M")
    if message.is_from_me:
        sender = "Me"
    elif message.sender is None:
        sender = "Unknown"
    else:
        sender = await _person_name(message.sender)

    parts = []
    if message.text != "":
        parts.append(message.text)
    for attachment in message.attachments:
        if attachment.is_image:
            parts.append(f"[photo #{attachment.id}: {attachment.name}]")
        else:
            parts.append(f"[attachment #{attachment.id}: {attachment.name} ({attachment.mime_type})]")
    return f"[{sent_at}] {sender}: {' '.join(parts)}"


def _to_attachment(row: sqlite3.Row) -> Attachment:
    """
    Convert an attachment row to an Attachment.

    Args:
        row: A row selected with ATTACHMENT_COLUMNS.
    Returns:
        The attachment.
    """
    path = None
    if row["filename"] is not None:
        path = Path(row["filename"]).expanduser()

    name = row["transfer_name"]
    if name is None and path is not None:
        name = path.name
    if name is None:
        name = "attachment"

    mime_type = row["mime_type"]
    if mime_type is None:
        mime_type = "unknown type"

    return Attachment(id=row["attachment_id"], path=path, mime_type=mime_type, name=name)


def _message_text(text: str | None, attributed_body: bytes | None) -> str:
    """
    Get a message's text. Newer macOS versions often leave the text column empty and
    store the text only inside attributedBody.

    Args:
        text: The text column.
        attributed_body: The attributedBody column (an archived NSAttributedString).
    Returns:
        The text without attachment placeholders. Empty if there is none.
    """
    if text is None or text == "":
        text = ""
        if attributed_body is not None:
            text = _decode_attributed_body(attributed_body)

    # Attachments are listed separately, so drop the character marking where they sat.
    return text.replace(ATTACHMENT_PLACEHOLDER, "").strip()


def _decode_attributed_body(blob: bytes) -> str:
    """
    Extract the plain text from an archived NSAttributedString.

    Args:
        blob: The attributedBody bytes.
    Returns:
        The text, or "" if it cannot be decoded.
    """
    try:
        data = NSData.dataWithBytes_length_(blob, len(blob))
        attributed_string = NSUnarchiver.unarchiveObjectWithData_(data)
    except Exception:
        return ""
    if attributed_string is None:
        return ""
    return str(attributed_string.string())


def _from_apple_time(value: int) -> datetime:
    """
    Convert a Messages timestamp (time since 2001-01-01, in nanoseconds on modern macOS
    and seconds on old versions) to a datetime.

    Args:
        value: The raw date column.
    Returns:
        The time in UTC.
    """
    seconds = value
    if value > 1_000_000_000_000:
        seconds = value / 1_000_000_000
    return APPLE_EPOCH + timedelta(seconds=seconds)


def _looks_like_chat_id(value: str) -> bool:
    """
    Check whether a value is a chat id (e.g. "iMessage;+;chat123") rather than a person.

    Args:
        value: The value to check.
    Returns:
        True for chat ids.
    """
    return ";" in value


def _looks_like_handle(value: str) -> bool:
    """
    Check whether a value is a phone number or email rather than a name.

    Args:
        value: The value to check.
    Returns:
        True for emails and values that are mostly digits.
    """
    value = value.strip()
    if "@" in value:
        return True

    digit_count = 0
    for character in value:
        if character.isdigit():
            digit_count += 1
    return digit_count >= 7


def _check_limit(limit: int) -> None:
    """
    Make sure a limit argument is in range.

    Args:
        limit: The requested limit.
    """
    if limit < 1 or limit > MAX_LIMIT:
        raise ToolError(f"limit must be between 1 and {MAX_LIMIT}")
