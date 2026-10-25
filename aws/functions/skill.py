"""
A lambda function that handles requests to Alexa.
The Alexa skill's endpoint. Each utterance is sent to the Jarvis backend through the
relay websocket, and the Lambda waits for the reply to show up in the table. Alexa
gives a skill 8 seconds to answer, so if Jarvis is slower the user hears that it is
still working and the request is remembered; opening the skill or saying "any update"
then speaks the answer once it has arrived.
"""

import logging
import os
import time
import uuid

import store

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Leaves headroom under Alexa's 8 second limit for the Lambda's own work.
REPLY_WAIT_SECONDS = 6.5
POLL_SECONDS = 0.2
MAX_SPEECH_CHARS = 8000
STOP_INTENTS = ("AMAZON.StopIntent", "AMAZON.CancelIntent", "AMAZON.NavigateHomeIntent")


def handler(event: dict, context) -> dict:
    """
    Answer one request from Alexa.

    Args:
        event: The Alexa request envelope.
        context: The Lambda context (unused).
    Returns:
        The Alexa response envelope.
    """
    started = time.monotonic()
    application_id = event["context"]["System"]["application"]["applicationId"]
    if application_id != os.environ["SKILL_ID"]:
        raise ValueError(f"Request is for another skill: {application_id}")

    user_id = event["context"]["System"]["user"]["userId"]
    request = event["request"]
    intent_name = request.get("intent", {}).get("name", "")
    logger.info("Alexa request %s %s", request["type"], intent_name)

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
        return _ask(user_id, query, started)
    if intent == "UpdateIntent":
        return _check_pending(user_id, "There's nothing I'm still working on.")
    if intent == "AMAZON.HelpIntent":
        return _speak("Just ask me anything, like what's on my calendar today.")
    if intent in STOP_INTENTS:
        return _speak("Goodbye.", end_session=True)
    logger.info("Unhandled intent %s", intent)
    return _speak("Sorry, I didn't catch that.")


def _ask(user_id: str, text: str, started: float) -> dict:
    """
    Send an utterance to Jarvis and speak its reply, or say it is still working.

    Args:
        user_id: The Alexa user.
        text: What the user said.
        started: When this request arrived, on the monotonic clock.
    Returns:
        The Alexa response envelope.
    """
    request_id = uuid.uuid4().hex
    message = {"type": "alexa.ask", "request_id": request_id, "user_id": user_id, "text": text}
    try:
        store.send_to_backend(message)
    except store.BackendOffline:
        return _speak("Jarvis is offline right now.", end_session=True)

    deadline = started + REPLY_WAIT_SECONDS
    while time.monotonic() < deadline:
        reply = store.get_reply(request_id)
        if reply is not None:
            store.delete_reply(request_id)
            return _speak(reply)
        time.sleep(POLL_SECONDS)

    store.save_pending(user_id, request_id)
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
        return _speak("I'm still working on it.")

    store.delete_pending(user_id)
    store.delete_reply(request_id)
    return _speak(reply)


def _speak(text: str, end_session: bool = False) -> dict:
    """
    Build a response that speaks text. The session stays open by default so the user
    can follow up without saying the wake word again.

    Args:
        text: What Alexa says.
        end_session: Whether to close the session afterwards.
    Returns:
        The Alexa response envelope.
    """
    return {
        "version": "1.0",
        "response": {
            "outputSpeech": {"type": "PlainText", "text": text[:MAX_SPEECH_CHARS]},
            "shouldEndSession": end_session,
        },
    }
