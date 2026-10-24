"""
Tools for reading and writing the user's Apple Calendar through EventKit. 
"""

import asyncio
from datetime import datetime, timedelta

import EventKit
from Foundation import NSDate

from .base import ToolError, tool

MAX_RANGE_DAYS = 366
MAX_NOTES_CHARS = 500
ACCESS_REQUEST_TIMEOUT_SECONDS = 120
WRITE_ONLY_CALENDAR_ID = "VIRTUAL_APP_CALENDAR_UUID"

# Event ids are "<EventKit identifier>|<occurrence start>", so one id names one occurrence.
ID_SEPARATOR = "|"

NO_ACCESS_MESSAGE = (
    "Jarvis does not have full access to Calendar. Allow it in System Settings > "
    "Privacy & Security > Calendars for the app running Jarvis (e.g. Terminal)."
)


class CalendarStore:
    """Access to the user's calendars through a single EventKit event store."""

    def __init__(self):
        """Create the wrapper. EventKit is not touched until first use."""
        self._store = None

    async def connect(self) -> EventKit.EKEventStore:
        """
        Get the event store, checking calendar permission the first time.

        Returns:
            The EventKit event store.
        """
        if self._store is None:
            await _ensure_access()
            store = EventKit.EKEventStore.alloc().init()

            # With write-only access, macOS shows one placeholder calendar instead of the
            # real ones. Saves still land in the real default calendar but can never be
            # read back, updated or deleted, so refuse to work in this mode.
            for calendar in store.calendarsForEntityType_(EventKit.EKEntityTypeEvent):
                if calendar.calendarIdentifier() == WRITE_ONLY_CALENDAR_ID:
                    raise ToolError(NO_ACCESS_MESSAGE)

            self._store = store
        return self._store

    async def calendars(self) -> list:
        """
        List every event calendar.

        Returns:
            EKCalendar objects.
        """
        store = await self.connect()
        return list(store.calendarsForEntityType_(EventKit.EKEntityTypeEvent))

    async def find_calendar(self, name_or_id: str | None) -> EventKit.EKCalendar:
        """
        Find a calendar by id or exact title, or the default calendar for new events.

        Args:
            name_or_id: Calendar id or title. None means the default calendar.
        Returns:
            The matching EKCalendar.
        """
        store = await self.connect()
        if name_or_id is None:
            default = store.defaultCalendarForNewEvents()
            if default is None:
                raise ToolError("There is no default calendar. Pass a calendar name.")
            return default

        by_id = store.calendarWithIdentifier_(name_or_id)
        if by_id is not None:
            return by_id

        matches = []
        for calendar in await self.calendars():
            if calendar.title() == name_or_id:
                matches.append(calendar)

        if len(matches) == 0:
            raise ToolError(f"No calendar named '{name_or_id}'. Use list_calendars to see them.")
        if len(matches) > 1:
            raise ToolError(f"Several calendars are named '{name_or_id}'. Pass the calendar id instead.")
        return matches[0]

    async def events_between(self, start: datetime, end: datetime, calendar: EventKit.EKCalendar | None) -> list:
        """
        Find event occurrences that overlap a time range, in start order.

        Args:
            start: Beginning of the range.
            end: End of the range.
            calendar: Only search this calendar, or None for all calendars.
        Returns:
            EKEvent objects, one per occurrence.
        """
        store = await self.connect()

        calendars = None
        if calendar is not None:
            calendars = [calendar]

        predicate = store.predicateForEventsWithStartDate_endDate_calendars_(
            _to_nsdate(start), _to_nsdate(end), calendars
        )
        events = list(store.eventsMatchingPredicate_(predicate))
        events.sort(key=lambda event: event.startDate().timeIntervalSince1970())
        return events

    async def find_occurrence(self, event_id: str) -> EventKit.EKEvent:
        """
        Find the single event occurrence an id refers to.

        Args:
            event_id: An id from list_events or create_event.
        Returns:
            The matching EKEvent occurrence.
        """
        identifier, separator, start_text = event_id.rpartition(ID_SEPARATOR)
        if separator == "" or identifier == "":
            raise ToolError(f"'{event_id}' is not a valid event id. Use the id from list_events.")
        try:
            occurrence_start = _as_local(datetime.fromisoformat(start_text))
        except ValueError:
            raise ToolError(f"'{event_id}' is not a valid event id. Use the id from list_events.") from None

        # Search a small window around the occurrence's start for the same event.
        window = timedelta(seconds=1)
        candidates = await self.events_between(occurrence_start - window, occurrence_start + window, None)
        for event in candidates:
            same_event = event.eventIdentifier() == identifier
            same_start = abs(_from_nsdate(event.startDate()) - occurrence_start) < window
            if same_event and same_start:
                return event

        raise ToolError(f"No event found for id '{event_id}'. It may have been moved or deleted.")

    async def new_event(self) -> EventKit.EKEvent:
        """
        Create an unsaved event attached to the store.

        Returns:
            A blank EKEvent.
        """
        store = await self.connect()
        return EventKit.EKEvent.eventWithEventStore_(store)

    async def save(self, event: EventKit.EKEvent) -> None:
        """
        Save a new or changed event. For a repeating event, only this occurrence changes.

        Args:
            event: The event to save.
        """
        store = await self.connect()
        saved, error = store.saveEvent_span_commit_error_(event, EventKit.EKSpanThisEvent, True, None)
        if not saved:
            raise ToolError(f"Calendar could not save the event: {_error_text(error)}")

    async def remove(self, event: EventKit.EKEvent) -> None:
        """
        Delete an event. For a repeating event, only this occurrence is deleted.

        Args:
            event: The event to delete.
        """
        store = await self.connect()
        removed, error = store.removeEvent_span_commit_error_(event, EventKit.EKSpanThisEvent, True, None)
        if not removed:
            raise ToolError(f"Calendar could not delete the event: {_error_text(error)}")


calendar_store = CalendarStore()


# Tools


@tool(requires_approval=False)
async def list_calendars() -> str:
    """
    List the user's calendars with their account, id and whether they can be edited.

    Returns:
        One line per calendar.
    """
    calendars = await calendar_store.calendars()
    if len(calendars) == 0:
        return "No calendars found."

    default = await calendar_store.find_calendar(None)
    lines = []
    for calendar in calendars:
        line = f"- {calendar.title()} (account: {calendar.source().title()}, id: {calendar.calendarIdentifier()})"
        if not calendar.allowsContentModifications():
            line += " [read-only]"
        if calendar.calendarIdentifier() == default.calendarIdentifier():
            line += " [default]"
        lines.append(line)
    return "\n".join(lines)


@tool(requires_approval=False)
async def list_events(start: datetime, end: datetime, calendar: str | None = None) -> str:
    """
    List calendar events that overlap a time range.

    Args:
        start: Beginning of the range in ISO 8601, e.g. "2026-10-08T00:00". Local time
            unless a UTC offset is given.
        end: End of the range in ISO 8601. At most 366 days after start.
        calendar: Only list events from this calendar (title or id). Omit for all calendars.
    Returns:
        The events in start order.
    """
    _check_range(start, end)
    if end - start > timedelta(days=MAX_RANGE_DAYS):
        raise ToolError(f"The range can be at most {MAX_RANGE_DAYS} days.")

    selected_calendar = None
    if calendar is not None:
        selected_calendar = await calendar_store.find_calendar(calendar)

    events = await calendar_store.events_between(start, end, selected_calendar)
    if len(events) == 0:
        return f"No events between {_format_time(start)} and {_format_time(end)}."

    descriptions = []
    for event in events:
        descriptions.append(_describe_event(event))
    return "\n".join(descriptions)


@tool(requires_approval=False)
async def create_event(
    title: str,
    start: datetime,
    end: datetime,
    calendar: str | None = None,
    location: str = "",
    notes: str = "",
    all_day: bool = False,
) -> str:
    """
    Create a calendar event.

    Args:
        title: The event's title.
        start: Start time in ISO 8601, e.g. "2026-10-08T14:00". Local time unless a UTC
            offset is given. For all-day events, any time on the first day.
        end: End time in ISO 8601. For all-day events, any time on the last day.
        calendar: Calendar title or id to add the event to. Omit for the default calendar.
        location: Where the event takes place.
        notes: Extra details for the event.
        all_day: Whether the event lasts all day instead of at set times.
    Returns:
        The created event, including its id.
    """
    _check_range(start, end, allow_equal=all_day)
    target_calendar = await calendar_store.find_calendar(calendar)
    if not target_calendar.allowsContentModifications():
        raise ToolError(f"The calendar '{target_calendar.title()}' is read-only.")

    event = await calendar_store.new_event()
    event.setCalendar_(target_calendar)
    event.setTitle_(title)
    event.setStartDate_(_to_nsdate(start))
    event.setEndDate_(_to_nsdate(end))
    event.setAllDay_(all_day)
    if location != "":
        event.setLocation_(location)
    if notes != "":
        event.setNotes_(notes)

    await calendar_store.save(event)
    return "Created event:\n" + _describe_event(event)


@tool(requires_approval=False)
async def update_event(
    event_id: str,
    title: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    location: str | None = None,
    notes: str | None = None,
) -> str:
    """
    Change an existing event. Only the fields you pass are changed. For a repeating
    event, only this occurrence changes. Moving the start without an end keeps the
    event's length.

    Args:
        event_id: The event's id from list_events or create_event.
        title: New title.
        start: New start time in ISO 8601. Local time unless a UTC offset is given.
        end: New end time in ISO 8601.
        location: New location. Pass "" to clear it.
        notes: New notes. Pass "" to clear them.
    Returns:
        The updated event, including its new id.
    """
    nothing_to_change = title is None and start is None and end is None and location is None and notes is None
    if nothing_to_change:
        raise ToolError("Pass at least one field to change.")

    event = await calendar_store.find_occurrence(event_id)
    if not event.calendar().allowsContentModifications():
        raise ToolError(f"The calendar '{event.calendar().title()}' is read-only.")

    old_start = _from_nsdate(event.startDate())
    old_end = _from_nsdate(event.endDate())

    new_start = old_start
    if start is not None:
        new_start = start

    if end is not None:
        new_end = end
    elif start is not None:
        new_end = new_start + (old_end - old_start)
    else:
        new_end = old_end

    _check_range(new_start, new_end, allow_equal=bool(event.isAllDay()))

    if title is not None:
        event.setTitle_(title)
    if location is not None:
        event.setLocation_(location)
    if notes is not None:
        event.setNotes_(notes)
    event.setStartDate_(_to_nsdate(new_start))
    event.setEndDate_(_to_nsdate(new_end))

    await calendar_store.save(event)
    return "Updated event:\n" + _describe_event(event)


@tool(requires_approval=False)
async def delete_event(event_id: str) -> str:
    """
    Delete an event. For a repeating event, only this occurrence is deleted.

    Args:
        event_id: The event's id from list_events or create_event.
    Returns:
        A confirmation naming the deleted event.
    """
    event = await calendar_store.find_occurrence(event_id)
    if not event.calendar().allowsContentModifications():
        raise ToolError(f"The calendar '{event.calendar().title()}' is read-only.")

    description = _describe_event(event)
    await calendar_store.remove(event)
    return "Deleted event:\n" + description


# Helpers


async def _ensure_access() -> None:
    """Make sure Jarvis has full calendar access, asking macOS once if it has not decided."""
    status = _authorization_status()
    if status == EventKit.EKAuthorizationStatusNotDetermined:
        await _request_access()
        # The request's callback is unreliable, so read the decision back from macOS.
        status = _authorization_status()

    if status != EventKit.EKAuthorizationStatusFullAccess:
        raise ToolError(NO_ACCESS_MESSAGE)


def _authorization_status() -> int:
    """
    Read macOS's calendar permission for this app.

    Returns:
        An EKAuthorizationStatus value.
    """
    return EventKit.EKEventStore.authorizationStatusForEntityType_(EventKit.EKEntityTypeEvent)


async def _request_access() -> None:
    """Show the macOS calendar permission prompt and wait for the user to answer."""
    loop = asyncio.get_running_loop()
    answered = loop.create_future()

    def on_answer(granted: bool, error: object) -> None:
        """Called by EventKit on its own thread once the user answers the prompt."""
        del granted, error  # Unreliable; the caller reads the status from macOS instead.
        loop.call_soon_threadsafe(answered.set_result, None)

    store = EventKit.EKEventStore.alloc().init()
    store.requestFullAccessToEventsWithCompletion_(on_answer)
    try:
        await asyncio.wait_for(answered, ACCESS_REQUEST_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        raise ToolError("Timed out waiting for the Calendar permission prompt to be answered.") from None


def _check_range(start: datetime, end: datetime, allow_equal: bool = False) -> None:
    """
    Make sure a time range ends after it starts.

    Args:
        start: Start of the range.
        end: End of the range.
        allow_equal: Whether start and end may be the same (used for one-day all-day events).
    """
    if _as_local(end) < _as_local(start):
        raise ToolError("The end time is before the start time.")
    if _as_local(end) == _as_local(start) and not allow_equal:
        raise ToolError("The end time must be after the start time.")


def _describe_event(event: EventKit.EKEvent) -> str:
    """
    Describe one event occurrence for the LLM.

    Args:
        event: The event occurrence.
    Returns:
        A multi-line description including its id.
    """
    start = _from_nsdate(event.startDate())
    end = _from_nsdate(event.endDate())

    if event.isAllDay():
        when = f"all day {start.date()}"
        if end.date() != start.date():
            when = f"all day {start.date()} to {end.date()}"
    else:
        when = f"{_format_time(start)} to {_format_time(end)}"

    lines = [f"- {event.title()}"]
    lines.append(f"  when: {when}")
    lines.append(f"  calendar: {event.calendar().title()}")
    if event.location():
        lines.append(f"  location: {event.location()}")
    if event.notes():
        notes = event.notes()
        if len(notes) > MAX_NOTES_CHARS:
            notes = notes[:MAX_NOTES_CHARS] + " ..."
        lines.append(f"  notes: {notes}")
    if event.hasRecurrenceRules():
        lines.append("  repeats: yes")
    lines.append(f"  id: {event.eventIdentifier()}{ID_SEPARATOR}{start.isoformat()}")
    return "\n".join(lines)


def _as_local(value: datetime) -> datetime:
    """
    Attach the local timezone to a time that has none.

    Args:
        value: A time with or without a UTC offset.
    Returns:
        A timezone-aware time.
    """
    if value.tzinfo is None:
        return value.astimezone()
    return value


def _to_nsdate(value: datetime) -> NSDate:
    """
    Convert a Python time to an NSDate.

    Args:
        value: The time. Without a UTC offset it is treated as local time.
    Returns:
        The equivalent NSDate.
    """
    return NSDate.dateWithTimeIntervalSince1970_(_as_local(value).timestamp())


def _from_nsdate(value: NSDate) -> datetime:
    """
    Convert an NSDate to a local, timezone-aware Python time.

    Args:
        value: The NSDate.
    Returns:
        The equivalent time in the local timezone.
    """
    return datetime.fromtimestamp(value.timeIntervalSince1970()).astimezone()


def _format_time(value: datetime) -> str:
    """
    Format a time for display.

    Args:
        value: The time.
    Returns:
        Text like "Thu 2026-10-08 14:00".
    """
    return _as_local(value).strftime("%a %Y-%m-%d %H:%M")


def _error_text(error: object) -> str:
    """
    Get a readable message from an NSError.

    Args:
        error: The NSError, or None.
    Returns:
        The error's description.
    """
    if error is None:
        return "unknown error"
    return str(error.localizedDescription())
