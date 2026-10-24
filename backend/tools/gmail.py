"""
Tools for the user's Gmail over IMAP (reading, drafts, organizing) and SMTP (sending),
signed in with a Google app password from .env. GmailClient opens a fresh connection
for each operation and works in the "All Mail" folder, so an email's id is its stable
IMAP UID there. Searches use Gmail's own search syntax through Gmail's X-GM-RAW IMAP
extension, and labels use X-GM-LABELS. The client is blocking, so tools run it in a
worker thread. Sending asks the user first; reading, drafting and organizing do not
(trashed mail stays recoverable in Gmail's Trash for 30 days).
"""

import asyncio
import email
import email.policy
import imaplib
import re
import shlex
import smtplib
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from email.message import EmailMessage
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from typing import Literal

import config

from .base import ToolError, tool

IMAP_HOST = "imap.gmail.com"
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 465
NETWORK_TIMEOUT_SECONDS = 30
MAX_SEARCH_RESULTS = 50
MAX_THREAD_MESSAGES = 20

OrganizeAction = Literal[
    "mark_read",
    "mark_unread",
    "star",
    "unstar",
    "archive",
    "move_to_inbox",
    "trash",
    "add_label",
    "remove_label",
]

# Actions that are a single IMAP STORE command: (command, value).
STORE_ACTIONS = {
    "mark_read": ("+FLAGS", "(\\Seen)"),
    "mark_unread": ("-FLAGS", "(\\Seen)"),
    "star": ("+FLAGS", "(\\Flagged)"),
    "unstar": ("-FLAGS", "(\\Flagged)"),
    "archive": ("-X-GM-LABELS", "(\\Inbox)"),
    "move_to_inbox": ("+X-GM-LABELS", "(\\Inbox)"),
}

LIST_LINE_PATTERN = re.compile(rb'\((?P<flags>[^)]*)\) (?:"[^"]*"|NIL) (?P<name>.+)')


@dataclass(frozen=True)
class EmailSummary:
    """The headline facts about an email, as shown in search results."""

    id: str
    sent_at: datetime | None
    sender: str
    subject: str
    unread: bool
    labels: tuple[str, ...]


@dataclass(frozen=True)
class Email:
    """A full email."""

    summary: EmailSummary
    to: str
    cc: str
    body: str
    attachments: tuple[str, ...]
    message_id: str
    references: str
    thread_id: str


class GmailClient:
    """Reads, organizes and sends the user's Gmail."""

    def __init__(self, address: str, app_password: str):
        """
        Args:
            address: The Gmail address.
            app_password: A Google app password for that address.
        """
        self.address = address
        self._app_password = app_password
        self._folders: dict[str, str] | None = None

    def search(self, query: str, limit: int) -> list[EmailSummary]:
        """
        Search with Gmail's search syntax.

        Args:
            query: A Gmail search, e.g. "from:sam is:unread newer_than:7d".
            limit: Maximum number of results.
        Returns:
            Matching emails, newest first.
        """
        with self._mailbox() as connection:
            # Sending the query as a literal handles quotes and non-ASCII text safely.
            connection.literal = query.encode("utf-8")
            uids = _search(connection, "CHARSET", "UTF-8", "X-GM-RAW")

            newest_uids = uids[-limit:]
            newest_uids.reverse()
            if len(newest_uids) == 0:
                return []

            fetched = _fetch(connection, newest_uids, "(FLAGS X-GM-LABELS BODY.PEEK[HEADER])")
            summaries = []
            for uid in newest_uids:
                if uid not in fetched:
                    continue
                metadata, header_bytes = fetched[uid]
                headers = email.message_from_bytes(header_bytes, policy=email.policy.default)
                summaries.append(_to_summary(uid, metadata, headers))
            return summaries

    def read(self, email_id: str) -> Email:
        """
        Read one full email without marking it as read.

        Args:
            email_id: The email's id.
        Returns:
            The email.
        """
        with self._mailbox() as connection:
            return _read(connection, email_id)

    def thread(self, email_id: str) -> list[Email]:
        """
        Read every email in the conversation an email belongs to.

        Args:
            email_id: The id of any email in the thread.
        Returns:
            The thread's emails, oldest first (at most MAX_THREAD_MESSAGES).
        """
        with self._mailbox() as connection:
            first = _read(connection, email_id)
            uids = _search(connection, "X-GM-THRID", first.thread_id)

            emails = []
            for uid in uids[-MAX_THREAD_MESSAGES:]:
                emails.append(_read(connection, uid))
            return emails

    def send(self, message: EmailMessage) -> None:
        """
        Send an email. Gmail files a copy in Sent automatically.

        Args:
            message: The email to send.
        """
        try:
            with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=NETWORK_TIMEOUT_SECONDS) as smtp:
                smtp.login(self.address, self._app_password)
                smtp.send_message(message)
        except smtplib.SMTPAuthenticationError:
            raise ToolError(_login_failed_message()) from None

    def save_draft(self, message: EmailMessage) -> None:
        """
        Save an email to Gmail's Drafts for the user to review and send.

        Args:
            message: The draft email.
        """
        with self._mailbox() as connection:
            drafts = _quote(self._folder("\\Drafts"))
            internal_date = imaplib.Time2Internaldate(time.time())
            status, response = connection.append(drafts, "(\\Draft)", internal_date, message.as_bytes())
            if status != "OK":
                raise ToolError(f"Gmail refused the draft: {response}")

    def organize(self, email_ids: list[str], action: OrganizeAction, label: str) -> list[str]:
        """
        Apply an action to several emails.

        Args:
            email_ids: The emails to change.
            action: What to do with them.
            label: The label name, for add_label and remove_label.
        Returns:
            Ids that do not exist, which were skipped.
        """
        with self._mailbox() as connection:
            existing = _search(connection, "UID", ",".join(email_ids))
            missing = []
            for email_id in email_ids:
                if email_id not in existing:
                    missing.append(email_id)
            if len(existing) == 0:
                return missing

            uid_set = ",".join(existing)
            if action in STORE_ACTIONS:
                command, value = STORE_ACTIONS[action]
                _check(connection.uid("STORE", uid_set, command, value))
            elif action == "trash":
                _check(connection.uid("MOVE", uid_set, _quote(self._folder("\\Trash"))))
            elif action == "add_label":
                _check(connection.uid("STORE", uid_set, "+X-GM-LABELS", f"({_quote(label)})"))
            elif action == "remove_label":
                _check(connection.uid("STORE", uid_set, "-X-GM-LABELS", f"({_quote(label)})"))
            return missing

    @contextmanager
    def _mailbox(self) -> Iterator[imaplib.IMAP4_SSL]:
        """
        Open a signed-in IMAP connection with All Mail selected, closing it afterwards.

        Returns:
            The connection, for use in a with block.
        """
        connection = imaplib.IMAP4_SSL(IMAP_HOST, timeout=NETWORK_TIMEOUT_SECONDS)
        try:
            try:
                connection.login(self.address, self._app_password)
            except imaplib.IMAP4.error:
                raise ToolError(_login_failed_message()) from None

            if self._folders is None:
                self._folders = _special_folders(connection)
            _check(connection.select(_quote(self._folder("\\All"))))
            yield connection
        finally:
            try:
                connection.logout()
            except (imaplib.IMAP4.error, OSError):
                pass

    def _folder(self, special_use: str) -> str:
        """
        Get the name of one of Gmail's special folders, which differs by account language.

        Args:
            special_use: The folder's role, e.g. "\\All", "\\Drafts" or "\\Trash".
        Returns:
            The folder's name, e.g. "[Gmail]/All Mail".
        """
        if self._folders is None or special_use not in self._folders:
            raise ToolError(f"Could not find Gmail's {special_use} folder. Is IMAP enabled?")
        return self._folders[special_use]


def build_email(
    sender: str,
    to: str,
    subject: str,
    body: str,
    cc: str,
    reply_to: Email | None,
) -> EmailMessage:
    """
    Build an email, threading it as a reply when reply_to is given.

    Args:
        sender: The From address.
        to: Comma-separated recipients.
        subject: The subject. For replies, an empty subject becomes "Re: <original>".
        body: The plain-text body.
        cc: Comma-separated CC recipients, or "".
        reply_to: The email being replied to, if any.
    Returns:
        The email, ready to send or save as a draft.
    """
    if to.strip() == "":
        raise ToolError("An email needs at least one recipient")

    message = EmailMessage()
    message["From"] = sender
    message["To"] = to
    if cc.strip() != "":
        message["Cc"] = cc

    if reply_to is not None:
        if subject.strip() == "":
            subject = reply_to.summary.subject
            if not subject.lower().startswith("re:"):
                subject = "Re: " + subject
        message["In-Reply-To"] = reply_to.message_id
        message["References"] = (reply_to.references + " " + reply_to.message_id).strip()

    message["Subject"] = subject
    message.set_content(body)
    return message


_gmail_client: GmailClient | None = None


def gmail_client() -> GmailClient:
    """
    Get the shared Gmail client, created from .env credentials on first use.

    Returns:
        The client.
    """
    global _gmail_client
    if _gmail_client is None:
        try:
            address, app_password = config.gmail_credentials()
        except RuntimeError as e:
            raise ToolError(str(e)) from None
        _gmail_client = GmailClient(address, app_password)
    return _gmail_client


# Tools


@tool(requires_approval=False)
async def search_emails(query: str = "in:inbox", limit: int = 10) -> str:
    """
    Search the user's Gmail using Gmail search syntax, e.g. "is:unread",
    "from:sam newer_than:7d", "subject:invoice has:attachment".

    Args:
        query: A Gmail search. Defaults to the inbox.
        limit: Maximum number of results, from 1 to 50.
    Returns:
        Matching emails, newest first, each with an id for other email tools.
    """
    if limit < 1 or limit > MAX_SEARCH_RESULTS:
        raise ToolError(f"limit must be between 1 and {MAX_SEARCH_RESULTS}")

    summaries = await asyncio.to_thread(gmail_client().search, query, limit)
    if len(summaries) == 0:
        return f"No emails match '{query}'."

    lines = []
    for summary in summaries:
        lines.append(_describe_summary(summary))
    return "\n".join(lines)


@tool(requires_approval=False)
async def read_email(email_id: str) -> str:
    """
    Read a full email. Does not mark it as read.

    Args:
        email_id: The email's id from search_emails.
    Returns:
        The email's headers, body and attachment names.
    """
    found = await asyncio.to_thread(gmail_client().read, _check_id(email_id))
    return _describe_email(found)


@tool(requires_approval=False)
async def read_thread(email_id: str) -> str:
    """
    Read the whole conversation an email belongs to, oldest first.

    Args:
        email_id: The id of any email in the thread.
    Returns:
        Every email in the thread.
    """
    emails = await asyncio.to_thread(gmail_client().thread, _check_id(email_id))

    descriptions = []
    for found in emails:
        descriptions.append(_describe_email(found))
    return "\n\n---\n\n".join(descriptions)


@tool
async def send_email(to: str, subject: str, body: str, cc: str = "", reply_to_email_id: str | None = None) -> str:
    """
    Send an email from the user's Gmail. To reply, pass reply_to_email_id so the email
    joins the original thread, and address it to the original sender (and others for
    reply-all) as shown by read_email.

    Args:
        to: Comma-separated recipient addresses.
        subject: The subject. When replying, "" means "Re: <original subject>".
        body: The plain-text body.
        cc: Comma-separated CC addresses.
        reply_to_email_id: The id of the email being replied to, if this is a reply.
    Returns:
        A confirmation.
    """
    client = gmail_client()
    message = await _build(client, to, subject, body, cc, reply_to_email_id)
    await asyncio.to_thread(client.send, message)
    return f"Sent '{message['Subject']}' to {to}."


@tool(requires_approval=False)
async def draft_email(to: str, subject: str, body: str, cc: str = "", reply_to_email_id: str | None = None) -> str:
    """
    Save an email to Gmail Drafts for the user to review and send themselves. Takes the
    same arguments as send_email.

    Args:
        to: Comma-separated recipient addresses.
        subject: The subject. When replying, "" means "Re: <original subject>".
        body: The plain-text body.
        cc: Comma-separated CC addresses.
        reply_to_email_id: The id of the email being replied to, if this is a reply.
    Returns:
        A confirmation.
    """
    client = gmail_client()
    message = await _build(client, to, subject, body, cc, reply_to_email_id)
    await asyncio.to_thread(client.save_draft, message)
    return f"Saved draft '{message['Subject']}' to {to} in Gmail Drafts."


@tool(requires_approval=False)
async def organize_emails(email_ids: list[str], action: OrganizeAction, label: str = "") -> str:
    """
    Organize emails: mark read/unread, star/unstar, archive, move back to the inbox,
    move to trash, or add/remove a label.

    Args:
        email_ids: Ids of the emails to change, from search_emails.
        action: What to do with them.
        label: The label name, required for add_label and remove_label.
    Returns:
        A confirmation, noting any ids that were not found.
    """
    if len(email_ids) == 0:
        raise ToolError("Pass at least one email id")
    checked_ids = []
    for email_id in email_ids:
        checked_ids.append(_check_id(email_id))

    if action in ("add_label", "remove_label"):
        _check_label(label)

    missing = await asyncio.to_thread(gmail_client().organize, checked_ids, action, label)
    done = len(checked_ids) - len(missing)
    result = f"Applied {action} to {done} email(s)."
    if len(missing) > 0:
        result += f" Not found: {', '.join(missing)}."
    return result


# Helpers


async def _build(
    client: GmailClient,
    to: str,
    subject: str,
    body: str,
    cc: str,
    reply_to_email_id: str | None,
) -> EmailMessage:
    """
    Build an email for send_email or draft_email, loading the original when replying.

    Args:
        client: The Gmail client.
        to: Comma-separated recipients.
        subject: The subject.
        body: The plain-text body.
        cc: Comma-separated CC recipients.
        reply_to_email_id: The id of the email being replied to, if any.
    Returns:
        The email.
    """
    reply_to = None
    if reply_to_email_id is not None:
        reply_to = await asyncio.to_thread(client.read, _check_id(reply_to_email_id))
    return build_email(client.address, to, subject, body, cc, reply_to)


def _read(connection: imaplib.IMAP4_SSL, email_id: str) -> Email:
    """
    Fetch and parse one full email on an open connection.

    Args:
        connection: A connection with All Mail selected.
        email_id: The email's UID.
    Returns:
        The email.
    """
    fetched = _fetch(connection, [email_id], "(FLAGS X-GM-LABELS X-GM-THRID BODY.PEEK[])")
    if email_id not in fetched:
        raise ToolError(f"No email with id {email_id}")

    metadata, raw = fetched[email_id]
    return parse_email(email_id, metadata, raw)


def parse_email(email_id: str, metadata: bytes, raw: bytes) -> Email:
    """
    Parse a fetched email.

    Args:
        email_id: The email's UID.
        metadata: The FETCH response line (flags, labels, thread id).
        raw: The full RFC 822 message.
    Returns:
        The email.
    """
    message = email.message_from_bytes(raw, policy=email.policy.default)

    attachments = []
    for attachment in message.iter_attachments():
        filename = attachment.get_filename()
        if filename is None:
            filename = "(unnamed)"
        attachments.append(filename)

    thread_match = re.search(rb"X-GM-THRID (\d+)", metadata)
    thread_id = ""
    if thread_match is not None:
        thread_id = thread_match.group(1).decode()

    return Email(
        summary=_to_summary(email_id, metadata, message),
        to=_header(message, "To"),
        cc=_header(message, "Cc"),
        body=_body_text(message),
        attachments=tuple(attachments),
        message_id=_header(message, "Message-ID"),
        references=_header(message, "References"),
        thread_id=thread_id,
    )


def _to_summary(email_id: str, metadata: bytes, message: email.message.Message) -> EmailSummary:
    """
    Build an EmailSummary from a FETCH response line and the email's headers.

    Args:
        email_id: The email's UID.
        metadata: The FETCH response line.
        message: The parsed email (headers are enough).
    Returns:
        The summary.
    """
    flags_match = re.search(rb"FLAGS \(([^)]*)\)", metadata)
    flags = b""
    if flags_match is not None:
        flags = flags_match.group(1)

    labels: list[str] = []
    labels_match = re.search(rb"X-GM-LABELS \(([^)]*)\)", metadata)
    if labels_match is not None:
        labels = _split_labels(labels_match.group(1).decode("utf-8", errors="replace"))

    sent_at = None
    try:
        sent_at = parsedate_to_datetime(_header(message, "Date"))
    except (TypeError, ValueError):
        pass

    return EmailSummary(
        id=email_id,
        sent_at=sent_at,
        sender=_header(message, "From"),
        subject=_header(message, "Subject"),
        unread=b"\\Seen" not in flags,
        labels=tuple(labels),
    )


def _search(connection: imaplib.IMAP4_SSL, *criteria: str) -> list[str]:
    """
    Run an IMAP UID SEARCH.

    Args:
        connection: A connection with a folder selected.
        criteria: The search criteria.
    Returns:
        Matching UIDs, oldest first.
    """
    status, data = connection.uid("SEARCH", *criteria)
    if status != "OK":
        raise ToolError(f"Gmail search failed: {data}")
    if len(data) == 0 or data[0] is None:
        return []
    return data[0].decode().split()


def _fetch(connection: imaplib.IMAP4_SSL, uids: list[str], items: str) -> dict[str, tuple[bytes, bytes]]:
    """
    Run an IMAP UID FETCH for items that end with one message part (like a header or body).

    Args:
        connection: A connection with a folder selected.
        uids: The UIDs to fetch.
        items: The FETCH items, e.g. "(FLAGS BODY.PEEK[])".
    Returns:
        For each UID found: the response line (flags, labels, ...) and the fetched part.
    """
    status, data = connection.uid("FETCH", ",".join(uids), items)
    if status != "OK":
        raise ToolError(f"Gmail fetch failed: {data}")

    # imaplib returns a list where each message is a (line, part) tuple, sometimes
    # followed by a bytes element holding attributes sent after the part.
    results = {}
    index = 0
    while index < len(data):
        item = data[index]
        if isinstance(item, tuple):
            metadata = item[0]
            part = item[1]
            next_index = index + 1
            if next_index < len(data) and isinstance(data[next_index], bytes):
                metadata += data[next_index]
                index = next_index

            uid_match = re.search(rb"UID (\d+)", metadata)
            if uid_match is not None:
                results[uid_match.group(1).decode()] = (metadata, part)
        index += 1
    return results


def _special_folders(connection: imaplib.IMAP4_SSL) -> dict[str, str]:
    """
    Find Gmail's special folders (All Mail, Drafts, Trash, ...) by their IMAP role.

    Args:
        connection: A signed-in connection.
    Returns:
        A role to folder name map, e.g. {"\\All": "[Gmail]/All Mail"}.
    """
    status, lines = connection.list()
    if status != "OK":
        raise ToolError(f"Gmail could not list folders: {lines}")

    folders = {}
    for line in lines:
        if not isinstance(line, bytes):
            continue
        match = LIST_LINE_PATTERN.match(line)
        if match is None:
            continue

        name = match.group("name").decode()
        if name.startswith('"') and name.endswith('"'):
            name = name[1:-1]
        for flag in match.group("flags").decode().split():
            folders[flag] = name
    return folders


def _body_text(message: email.message.Message) -> str:
    """
    Get an email's body as plain text, converting HTML if there is no plain version.

    Args:
        message: The parsed email.
    Returns:
        The body text.
    """
    part = message.get_body(preferencelist=("plain", "html"))
    if part is None:
        return ""

    try:
        content = part.get_content()
    except (LookupError, ValueError):
        # Unknown or wrong charset: decode the raw bytes as best we can.
        content = part.get_payload(decode=True).decode("utf-8", errors="replace")

    if part.get_content_type() == "text/html":
        content = _html_to_text(content)
    return content.strip()


class _TextExtractor(HTMLParser):
    """Collects the visible text of an HTML document."""

    SKIPPED_TAGS = ("script", "style", "head")
    BLOCK_TAGS = ("p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "table")

    def __init__(self):
        """Start with no text collected."""
        super().__init__()
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        """
        Track skipped sections and line breaks at block elements.

        Args:
            tag: The tag name.
            attrs: The tag's attributes (unused).
        """
        if tag in self.SKIPPED_TAGS:
            self._skip_depth += 1
        elif tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        """
        Leave a skipped section.

        Args:
            tag: The tag name.
        """
        if tag in self.SKIPPED_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        """
        Keep text that is not inside a skipped section.

        Args:
            data: The text.
        """
        if self._skip_depth == 0:
            self.parts.append(data)


def _html_to_text(html: str) -> str:
    """
    Convert HTML to readable plain text.

    Args:
        html: The HTML.
    Returns:
        Its visible text, with runs of blank lines collapsed.
    """
    extractor = _TextExtractor()
    extractor.feed(html)
    text = "".join(extractor.parts)
    return re.sub(r"\n\s*\n+", "\n\n", text)


def _describe_summary(summary: EmailSummary) -> str:
    """
    Format an email summary for the LLM.

    Args:
        summary: The summary.
    Returns:
        One entry with the email's id.
    """
    sent_at = "unknown date"
    if summary.sent_at is not None:
        sent_at = summary.sent_at.astimezone().strftime("%Y-%m-%d %H:%M")

    status = ""
    if summary.unread:
        status = " [unread]"

    labels = ""
    if len(summary.labels) > 0:
        labels = f"\n  labels: {', '.join(summary.labels)}"

    return (
        f"- {summary.subject}{status}\n"
        f"  from: {summary.sender}\n"
        f"  date: {sent_at}{labels}\n"
        f"  id: {summary.id}"
    )


def _describe_email(found: Email) -> str:
    """
    Format a full email for the LLM.

    Args:
        found: The email.
    Returns:
        Its headers, body and attachment names.
    """
    text = _describe_summary(found.summary)
    text += f"\n  to: {found.to}"
    if found.cc != "":
        text += f"\n  cc: {found.cc}"
    if len(found.attachments) > 0:
        text += f"\n  attachments: {', '.join(found.attachments)}"
    return text + "\n\n" + found.body


def _header(message: email.message.Message, name: str) -> str:
    """
    Read a header as text.

    Args:
        message: The parsed email.
        name: The header name.
    Returns:
        The header's value, or "" if it is missing.
    """
    value = message[name]
    if value is None:
        return ""
    return str(value)


def _split_labels(text: str) -> list[str]:
    """
    Split an X-GM-LABELS list like '"\\\\Inbox" "Work stuff" Receipts' into names.

    Args:
        text: The text inside the parentheses.
    Returns:
        The label names, with system labels like Inbox shown without a backslash.
    """
    try:
        raw_labels = shlex.split(text)
    except ValueError:
        raw_labels = text.split()

    labels = []
    for label in raw_labels:
        labels.append(label.lstrip("\\"))
    return labels


def _quote(name: str) -> str:
    """
    Quote a folder or label name for an IMAP command.

    Args:
        name: The name.
    Returns:
        The name in double quotes, with quotes and backslashes escaped.
    """
    escaped = name.replace("\\", "\\\\")
    escaped = escaped.replace('"', '\\"')
    return f'"{escaped}"'


def _check(result: tuple[str, list]) -> None:
    """
    Raise if an IMAP command failed.

    Args:
        result: The (status, data) pair imaplib returns.
    """
    status, data = result
    if status != "OK":
        raise ToolError(f"Gmail command failed: {data}")


def _check_id(email_id: str) -> str:
    """
    Make sure an email id is a UID (digits only), so it cannot alter an IMAP command.

    Args:
        email_id: The id to check.
    Returns:
        The id, without surrounding spaces.
    """
    email_id = email_id.strip()
    if not email_id.isdigit():
        raise ToolError(f"'{email_id}' is not a valid email id. Use an id from search_emails.")
    return email_id


def _check_label(label: str) -> None:
    """
    Make sure a label name is usable over IMAP.

    Args:
        label: The label name.
    """
    if label.strip() == "":
        raise ToolError("Pass a label name")
    if not label.isascii():
        raise ToolError("Only labels with plain ASCII names are supported")


def _login_failed_message() -> str:
    """
    Explain a failed Gmail sign-in.

    Returns:
        What to check.
    """
    return (
        "Gmail sign-in failed. Check GMAIL_ADDRESS and GMAIL_APP_PASSWORD in .env (it must be "
        "a Google app password, not your normal password) and that IMAP is enabled in Gmail."
    )
