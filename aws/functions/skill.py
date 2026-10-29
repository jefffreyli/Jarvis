"""
A lambda function that handles requests to Alexa.
The Alexa skill's endpoint. Each utterance is sent to the Jarvis backend through the
relay websocket, and the Lambda waits for the reply to show up in the table. Alexa
gives a skill 8 seconds to answer, so if Jarvis is slower the user hears that it is
still working and the request is remembered; opening the skill or saying "any update"
then speaks the answer once it has arrived. The skill has a single catch-all intent, so
"any update" is recognized here from the words themselves. Each invocation is logged under the trace
"alexa:<first 8 characters of its request id>", which the backend logs under too.
"""

import logging
import os
import time
import uuid

import relay_log
import store

logger = logging.getLogger("jarvis.skill")

# Leaves headroom under Alexa's 8 second limit for the Lambda's own work.
REPLY_WAIT_SECONDS = 6.5
POLL_SECONDS = 0.2
MAX_SPEECH_CHARS = 8000
REPROMPT = "Anything else?"
STOP_INTENTS = ("AMAZON.StopIntent", "AMAZON.CancelIntent", "AMAZON.NavigateHomeIntent")
UPDATE_PHRASES = (
    "any update",
    "any updates",
    "any update yet",
    "any updates yet",
    "give me an update",
    "are you done",
    "are you done yet",
    "is it done",
    "is it done yet",
)


def handler(event: dict, context) -> dict:
    """
    Answer one request from Alexa, forwarding this invocation's log to the backend.

    Args:
        event: The Alexa request envelope.
        context: The Lambda context (unused).
    Returns:
        The Alexa response envelope.
    """
    started = time.monotonic()
    request_id = uuid.uuid4().hex
    relay_log.begin(f"alexa:{request_id[:8]}")
    try:
        response = _handle(event, request_id, started)
        speech = response["response"].get("outputSpeech", {}).get("text", "")
        logger.info(
            "Responded after %.2fs (session ends: %s): %s",
            time.monotonic() - started,
            response["response"].get("shouldEndSession"),
            relay_log.snippet(speech),
        )
        return response
    except Exception:
        logger.exception("Handling the Alexa request failed")
        raise
    finally:
        relay_log.flush()


def _handle(event: dict, request_id: str, started: float) -> dict:
    """
    Answer one request from Alexa.

    Args:
        event: The Alexa request envelope.
        request_id: The id the backend's reply will carry, if the request goes to it.
        started: When this request arrived, on the monotonic clock.
    Returns:
        The Alexa response envelope.
    """
    application_id = event["context"]["System"]["application"]["applicationId"]
    if application_id != os.environ["SKILL_ID"]:
        raise ValueError(f"Request is for another skill: {application_id}")

    user_id = event["context"]["System"]["user"]["userId"]
    request = event["request"]
    intent_name = request.get("intent", {}).get("name", "")
    session = event.get("session", {})
    logger.info(
        "Alexa %s %s (Alexa request %s, session %s, new session: %s)",
        request["type"],
        intent_name,
        request.get("requestId"),
        session.get("sessionId", "")[-8:],
        session.get("new"),
    )

    if request["type"] == "LaunchRequest":
        return _check_pending(user_id, "Hi, I'm Jarvis. What can I do for you?")
    if request["type"] == "SessionEndedRequest":
        # Alexa explains here when it rejected one of our responses.
        logger.info("Session ended: %s %s", request.get("reason"), request.get("error"))
        return {"version": "1.0", "response": {}}
    if request["type"] != "IntentRequest":
        return _speak("Sorry, I can't do that.")

    intent = request["intent"]["name"]
    if intent == "JarvisIntent":
        query = request["intent"].get("slots", {}).get("query", {}).get("value", "")
        if query.strip() == "":
            logger.info("JarvisIntent arrived with an empty query slot")
            return _speak("Sorry, I didn't catch that.")
        if _normalize(query) in UPDATE_PHRASES:
            return _check_pending(user_id, "There's nothing I'm still working on.")
        session_id = event["session"]["sessionId"]
        return _ask(request_id, user_id, session_id, query, started)
    if intent == "AMAZON.HelpIntent":
        return _speak("Just ask me anything, like what's on my calendar today.")
    if intent in STOP_INTENTS:
        return _speak("Goodbye.", end_session=True)
    logger.warning("Unhandled intent %s", intent)
    return _speak("Sorry, I didn't catch that.")


def _ask(request_id: str, user_id: str, session_id: str, text: str, started: float) -> dict:
    """
    Send an utterance to Jarvis and speak its reply, or say it is still working.

    Args:
        request_id: The id the backend's reply will carry.
        user_id: The Alexa user.
        session_id: Alexa's id for the current session.
        text: What the user said.
        started: When this request arrived, on the monotonic clock.
    Returns:
        The Alexa response envelope.
    """
    message = {
        "type": "alexa.ask",
        "request_id": request_id,
        "trace_id": request_id[:8],
        "user_id": user_id,
        "session_id": session_id,
        "text": text,
    }
    logger.info("Asking Jarvis: %s", relay_log.snippet(text))
    try:
        store.send_to_backend(message)
    except store.BackendOffline:
        logger.warning("Jarvis is offline, so the request was not sent")
        return _speak("Jarvis is offline right now.", end_session=True)

    sent = time.monotonic()
    deadline = started + REPLY_WAIT_SECONDS
    polls = 0
    while time.monotonic() < deadline:
        reply = store.get_reply(request_id)
        polls += 1
        if reply is not None:
            store.delete_reply(request_id)
            logger.info("Jarvis replied after %.2fs (%d polls)", time.monotonic() - sent, polls)
            return _speak(reply)
        time.sleep(POLL_SECONDS)

    store.save_pending(user_id, request_id)
    logger.info("No reply within %.1fs, so request %s is now pending", REPLY_WAIT_SECONDS, request_id)
    return _speak("I'm still working on that. Say any update in a bit to hear the answer.")


def _check_pending(user_id: str, nothing_pending: str) -> dict:
    """
    Speak the answer to the request the user is waiting on, if it has arrived.

    Args:
        user_id: The Alexa user.
        nothing_pending: What to say when the user is not waiting on anything.
    Returns:
        The Alexa response envelope.
    """
    request_id = store.get_pending(user_id)
    if request_id is None:
        return _speak(nothing_pending)

    reply = store.get_reply(request_id)
    if reply is None:
        logger.info("Pending request %s has no reply yet", request_id)
        return _speak("I'm still working on it.")

    store.delete_pending(user_id)
    store.delete_reply(request_id)
    logger.info("Speaking the reply to pending request %s", request_id)
    return _speak(reply)


def _normalize(text: str) -> str:
    """
    Lowercase an utterance and drop punctuation and extra spaces, for phrase matching.

    Args:
        text: What the user said.
    Returns:
        The normalized text.
    """
    kept = ""
    for character in text.lower():
        if character.isalnum() or character.isspace():
            kept += character
    return " ".join(kept.split())


def _speak(text: str, end_session: bool = False) -> dict:
    """
    Build a response that speaks text. The session stays open by default so the user
    can follow up without saying the wake word again. An open session carries a
    reprompt, which makes Alexa open the microphone after speaking and close the
    session if the user stays quiet, instead of leaving it open with the mic off.

    Args:
        text: What Alexa says.
        end_session: Whether to close the session afterwards.
    Returns:
        The Alexa response envelope.
    """
    response = {
        "outputSpeech": {"type": "PlainText", "text": text[:MAX_SPEECH_CHARS]},
        "shouldEndSession": end_session,
    }
    if not end_session:
        response["reprompt"] = {"outputSpeech": {"type": "PlainText", "text": REPROMPT}}
    return {"version": "1.0", "response": response}
