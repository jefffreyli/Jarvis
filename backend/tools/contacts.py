"""
Access to the user's macOS Contacts. ContactBook loads every contact's name, phone
numbers and emails on first use (checking Contacts permission) and keeps them for a few
minutes, so lookups are instant. It finds people by name and turns phone numbers or
emails back into names, which the iMessage tools use to show who said what.
"""

import asyncio
import time
from dataclasses import dataclass

import Contacts

from .base import ToolError, tool

CACHE_SECONDS = 600
ACCESS_REQUEST_TIMEOUT_SECONDS = 120
MAX_LOOKUP_RESULTS = 10
PHONE_DIGITS_COMPARED = 10

NO_ACCESS_MESSAGE = (
    "Jarvis does not have access to Contacts. Allow it in System Settings > Privacy & "
    "Security > Contacts for the app running Jarvis (e.g. Terminal)."
)

FETCH_KEYS = [
    Contacts.CNContactGivenNameKey,
    Contacts.CNContactFamilyNameKey,
    Contacts.CNContactNicknameKey,
    Contacts.CNContactOrganizationNameKey,
    Contacts.CNContactPhoneNumbersKey,
    Contacts.CNContactEmailAddressesKey,
]


@dataclass(frozen=True)
class Person:
    """One contact with the ways to reach them."""

    name: str
    nickname: str
    phones: tuple[str, ...]
    emails: tuple[str, ...]


class ContactBook:
    """A cached, searchable copy of the user's contacts."""

    def __init__(self):
        """Create the book. Contacts are not loaded until first use."""
        self._people: list[Person] = []
        self._names_by_handle: dict[str, str] = {}
        self._loaded_at: float | None = None

    async def people(self) -> list[Person]:
        """
        Get every contact, reloading if the cache is older than CACHE_SECONDS.

        Returns:
            All contacts that have a name.
        """
        cache_expired = self._loaded_at is None or time.monotonic() - self._loaded_at > CACHE_SECONDS
        if cache_expired:
            await _ensure_access()
            self._load()
        return self._people

    async def find(self, query: str) -> list[Person]:
        """
        Find contacts by name or nickname. Exact name matches win over partial ones.

        Args:
            query: A full or partial name, e.g. "Sam" or "Sam Lee".
        Returns:
            The matching contacts.
        """
        query = query.strip().lower()
        if query == "":
            return []

        exact = []
        partial = []
        for person in await self.people():
            name = person.name.lower()
            nickname = person.nickname.lower()
            if query == name or query == nickname:
                exact.append(person)
            elif query in name or query in nickname:
                partial.append(person)

        if len(exact) > 0:
            return exact
        return partial

    async def name_for(self, handle: str) -> str | None:
        """
        Find the contact name for a phone number or email.

        Args:
            handle: A phone number or email address in any format.
        Returns:
            The contact's name, or None if no contact has it.
        """
        await self.people()
        return self._names_by_handle.get(normalize_handle(handle))

    def _load(self) -> None:
        """Read every contact from macOS Contacts into the cache."""
        store = Contacts.CNContactStore.alloc().init()
        request = Contacts.CNContactFetchRequest.alloc().initWithKeysToFetch_(FETCH_KEYS)

        people = []

        def visit(contact: Contacts.CNContact, stop: object) -> None:
            """Called by Contacts once per contact."""
            person = _to_person(contact)
            if person is not None:
                people.append(person)

        fetched, error = store.enumerateContactsWithFetchRequest_error_usingBlock_(request, None, visit)
        if not fetched:
            raise ToolError(f"Could not read Contacts: {error}")

        names_by_handle = {}
        for person in people:
            for handle in person.phones + person.emails:
                names_by_handle[normalize_handle(handle)] = person.name

        self._people = people
        self._names_by_handle = names_by_handle
        self._loaded_at = time.monotonic()


def normalize_handle(handle: str) -> str:
    """
    Reduce a phone number or email to a form that matches however it was written.
    Emails are lowercased; phone numbers keep only their last 10 digits, so
    "+1 (555) 123-4567" and "555.123.4567" match.

    Args:
        handle: A phone number or email address.
    Returns:
        The normalized handle.
    """
    handle = handle.strip()
    if "@" in handle:
        return handle.lower()

    digits = ""
    for character in handle:
        if character.isdigit():
            digits += character
    return digits[-PHONE_DIGITS_COMPARED:]


contact_book = ContactBook()


@tool(requires_approval=False)
async def lookup_contact(name: str) -> str:
    """
    Look up someone in the user's contacts to get their phone numbers and emails.

    Args:
        name: A full or partial name or nickname, e.g. "Sam".
    Returns:
        The matching contacts with their phone numbers and emails.
    """
    matches = await contact_book.find(name)
    if len(matches) == 0:
        return f"No contacts match '{name}'."

    lines = []
    for person in matches[:MAX_LOOKUP_RESULTS]:
        phones = "none"
        if len(person.phones) > 0:
            phones = ", ".join(person.phones)

        emails = "none"
        if len(person.emails) > 0:
            emails = ", ".join(person.emails)

        lines.append(f"- {person.name}: phones: {phones}; emails: {emails}")
    if len(matches) > MAX_LOOKUP_RESULTS:
        lines.append(f"({len(matches) - MAX_LOOKUP_RESULTS} more matches; use a more specific name)")
    return "\n".join(lines)


def _to_person(contact: Contacts.CNContact) -> Person | None:
    """
    Convert a Contacts record to a Person.

    Args:
        contact: The CNContact.
    Returns:
        The Person, or None if the contact has no name at all.
    """
    full_name = f"{contact.givenName()} {contact.familyName()}".strip()
    if full_name == "":
        full_name = str(contact.organizationName()).strip()
    if full_name == "":
        return None

    phones = []
    for labeled in contact.phoneNumbers():
        phones.append(str(labeled.value().stringValue()))

    emails = []
    for labeled in contact.emailAddresses():
        emails.append(str(labeled.value()))

    return Person(full_name, str(contact.nickname()), tuple(phones), tuple(emails))


async def _ensure_access() -> None:
    """Make sure Jarvis can read Contacts, asking macOS once if it has not decided."""
    status = _authorization_status()
    if status == Contacts.CNAuthorizationStatusNotDetermined:
        await _request_access()
        # Read the decision back from macOS rather than trusting the callback.
        status = _authorization_status()

    allowed = (Contacts.CNAuthorizationStatusAuthorized, Contacts.CNAuthorizationStatusLimited)
    if status not in allowed:
        raise ToolError(NO_ACCESS_MESSAGE)


def _authorization_status() -> int:
    """
    Read macOS's Contacts permission for this app.

    Returns:
        A CNAuthorizationStatus value.
    """
    return Contacts.CNContactStore.authorizationStatusForEntityType_(Contacts.CNEntityTypeContacts)


async def _request_access() -> None:
    """Show the macOS Contacts permission prompt and wait for the user to answer."""
    loop = asyncio.get_running_loop()
    answered = loop.create_future()

    def on_answer(granted: bool, error: object) -> None:
        """Called by Contacts on its own thread once the user answers the prompt."""
        del granted, error  # The caller reads the status from macOS instead.
        loop.call_soon_threadsafe(answered.set_result, None)

    store = Contacts.CNContactStore.alloc().init()
    store.requestAccessForEntityType_completionHandler_(Contacts.CNEntityTypeContacts, on_answer)
    try:
        await asyncio.wait_for(answered, ACCESS_REQUEST_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        raise ToolError("Timed out waiting for the Contacts permission prompt to be answered.") from None
