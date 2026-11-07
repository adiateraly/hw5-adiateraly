"""A clearly labelled FAKE chat model for tests and offline demos. NOT a real LLM.

It follows simple keyword rules to imitate what a tool-capable model would do,
so tests and CI need no API key. Logs made with it must say 'fake model'.
"""
import json
import re

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

ASK_DETAILS = "What problem are you having, and what have you already tried?"
STEP_KEYWORDS = [("enabled", "Enabled Wi-Fi"),
                 ("forgot the saved network", "Forgot the saved network"),
                 ("reconnect", "Tried reconnecting")]
_n = 0


def _call(name, args):
    global _n
    _n += 1
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"fake-{_n}"}])


def _steps(texts):
    low = " ".join(texts).lower()
    return [label for kw, label in STEP_KEYWORDS if kw in low]


def _topic(texts):
    low = " ".join(texts).lower()
    if "wifi" in low or "wi-fi" in low:
        return "wifi"
    if "password" in low or "recovery" in low:
        return "password"
    return None


def decide(messages):
    system = next((m.content for m in messages if isinstance(m, SystemMessage)), "")
    idx = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
    text = messages[idx].content
    low = text.lower()
    earlier_h = [m.content for m in messages[:idx] if isinstance(m, HumanMessage)]
    earlier_ai = [m.content for m in messages[:idx] if isinstance(m, AIMessage) and m.content]
    done = {m.name: json.loads(m.content) for m in messages[idx + 1:] if isinstance(m, ToolMessage)}
    lang = re.search(r"reply_language = (\w+)", system)

    pref = re.search(r"prefer replies in (\w+)", text, re.I)
    if pref and "remember_preference" not in done:
        return _call("remember_preference", {"name": "reply_language", "value": pref.group(1)})

    ticket = ("prepare a" in low and "ticket" in low) or (earlier_ai and earlier_ai[-1] == ASK_DETAILS)
    conditional = "if it is operating" in low
    wants_status = ("check" in low and ("down" in low or "status" in low)) or "known problem" in low
    topic = _topic(earlier_h + [text])
    service = "account_recovery" if _topic([text]) == "password" else "wifi"

    if wants_status and "check_service_status" not in done:
        return _call("check_service_status", {"service": service})

    if ticket:
        steps = _steps(earlier_h + [text])
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", text)
                     if not re.search(r"check whether|prepare a|if it is", s, re.I)]
        problem = " ".join(sentences).strip()
        status = done.get("check_service_status")
        if conditional and status:
            st = status["status"]
            if st == "outage":
                return AIMessage(content=f"{status['message']} Your request said to prepare a ticket only if the "
                                         "service was operating, so I have not prepared one.")
            if st == "unknown":
                return AIMessage(content="I could not determine the service status, so I cannot confirm it is "
                                         "operating. Would you like a ticket draft explaining that the status "
                                         "could not be checked?")
        if not (topic and steps and problem):
            return AIMessage(content=ASK_DETAILS)
        if "prepare_ticket" not in done:
            return _call("prepare_ticket", {
                "problem": problem, "steps_tried": steps, "category": topic,
                "service_status": status["status"] if status else "not_checked",
                "reference_ids": ["GUIDE-WIFI" if topic == "wifi" else "GUIDE-PASSWORD"]})
        t = done["prepare_ticket"]
        known = (f"The demo status source reports the service is {status['status']}. " if status
                 else "Service status was not checked. ")
        return AIMessage(content=f"{known}I prepared a local draft ({t['ticket_id']}, file {t['file']}) with the "
                                 "problem and the steps you tried. Nothing has been sent.")

    if wants_status:
        return AIMessage(content=done["check_service_status"]["message"])

    if "language" in low:
        known = f"You asked me to reply in {lang.group(1)}." if lang else \
            "No preference or previous problem has been recorded for you."
        if "help with" in low:
            known += " I can help with Wi-Fi troubleshooting, password recovery guidance, service status, and local ticket drafts."
        if not lang and not earlier_h and "previous problem" in low:
            known = "No preference or previous problem has been recorded for you."
        return AIMessage(content=known)

    if "which" in low and ("steps" in low or "problem" in low):
        prior = _topic(earlier_h)
        steps = _steps(earlier_h)
        if not prior:
            return AIMessage(content="I have no earlier problem recorded in this conversation.")
        names = " and ".join(s.lower() for s in steps) or "no steps"
        return AIMessage(content=f"You reported a {'Wi-Fi' if prior == 'wifi' else 'password'} problem and "
                                 f"said you already did this: {names}.")

    if "remember_preference" in done and not (topic and "find_instructions" not in done):
        return AIMessage(content=f"Saved: replies in {done['remember_preference']['saved']['reply_language']}.")

    if topic and "find_instructions" not in done:
        return _call("find_instructions", {"topic": topic})
    if "find_instructions" in done:
        r = done["find_instructions"]
        saved = ""
        if "remember_preference" in done:
            saved = f"Saved: replies in {done['remember_preference']['saved']['reply_language']}. "
        return AIMessage(content=saved + f"{r['title']} [{r['instruction_id']}]: " + " ".join(r["steps"]))
    return AIMessage(content="I can help with Wi-Fi, passwords, service status and local ticket drafts.")


class ScriptedFakeModel(BaseChatModel):
    """FAKE model used by tests/CI. Not a language model."""

    @property
    def _llm_type(self) -> str:
        return "scripted-fake"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=decide(messages))])
