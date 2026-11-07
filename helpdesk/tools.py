"""The three helpdesk tools (+ remember_preference), as plain Python functions.

Every function works when called directly, without an agent or a model.
Values the *program* controls (user id, thread id, status snapshot) are passed
by the application, never chosen by the model.
"""
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

_TOPIC_ALIASES = {"wifi": "wifi", "wi-fi": "wifi", "password": "password"}
_SERVICE_ALIASES = {
    "wifi": "wifi", "wi-fi": "wifi",
    "account_recovery": "account_recovery", "password": "account_recovery",
}
TOPIC_REFERENCE = {"wifi": "GUIDE-WIFI", "password": "GUIDE-PASSWORD"}


def _load(name: str) -> dict:
    with open(DATA_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def find_instructions(topic: str) -> dict:
    """Retrieval: look up Wi-Fi or password instructions in data/helpdesk.json."""
    key = _TOPIC_ALIASES.get(str(topic).strip().lower())
    if key is None:
        return {"ok": False, "error": f"Unknown topic {topic!r}. Use 'wifi' or 'password'."}
    entry = _load("helpdesk.json")["instructions"][key]
    return {"ok": True, "topic": key, "instruction_id": entry["instruction_id"],
            "title": entry["title"], "steps": entry["steps"], "source": "helpdesk.json"}


def check_service_status(service: str, profile: str = "normal") -> dict:
    """Look up a service in one named snapshot of data/service_status.json.

    status is 'operational', 'outage' or 'unknown' (status source unavailable).
    'unknown' is NOT operational and NOT a confirmed outage.
    """
    key = _SERVICE_ALIASES.get(str(service).strip().lower())
    if key is None:
        return {"ok": False, "error": f"Unknown service {service!r}. Use 'wifi' or 'account_recovery'."}
    profiles = _load("service_status.json")["profiles"]
    if profile not in profiles:
        return {"ok": False, "error": f"Unknown status profile {profile!r}."}
    entry = profiles[profile][key]
    return {"ok": True, "service": key, "status": entry["status"],
            "message": entry["message"], "profile": profile,
            "source": "service_status.json"}


def prepare_ticket(problem: str, steps_tried: list, category: str,
                   service_status: str = "not_checked", reference_ids: list | None = None,
                   *, user_id: str, thread_id: str, status_profile: str,
                   tickets_dir: str | Path = "outputs/tickets") -> dict:
    """Save a LOCAL ticket draft. Nothing is sent anywhere.

    Refuses (and asks) when the problem or the steps already tried are missing,
    instead of inventing them.
    """
    missing = []
    if not str(problem or "").strip():
        missing.append("problem description")
    steps = [str(s).strip() for s in (steps_tried or []) if str(s).strip()]
    if not steps:
        missing.append("steps already tried")
    if missing:
        return {"ok": False, "error": "missing_information", "missing": missing,
                "question": "What problem are you having, and what have you already tried?"}
    if category not in TOPIC_REFERENCE:
        return {"ok": False, "error": f"Unknown category {category!r}. Use 'wifi' or 'password'."}

    ticket = dict(_load("ticket_template.json")["ticket"])
    ticket.update({
        "ticket_id": "TCK-" + uuid.uuid4().hex[:8],
        "status": "draft",
        "user_id": user_id,
        "thread_id": thread_id,
        "category": category,
        "summary": str(problem).strip(),
        "steps_tried": steps,
        "service_status": service_status,
        "status_profile": status_profile,
        "reference_ids": list(reference_ids or [TOPIC_REFERENCE[category]]),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": "Local draft only. Not sent to any helpdesk.",
    })
    out = Path(tickets_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{ticket['ticket_id']}.json"
    path.write_text(json.dumps(ticket, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"ok": True, "ticket_id": ticket["ticket_id"], "file": str(path).replace("\\", "/"),
            "ticket": ticket}


def remember_preference(user_id: str, name: str, value: str, memory) -> dict:
    """Write to USER memory (persistent, per user). Only reply_language is allowed."""
    if name != "reply_language":
        return {"ok": False, "error": "Only 'reply_language' can be remembered."}
    value = str(value).strip()[:50]
    if not value:
        return {"ok": False, "error": "Empty value."}
    memory.set(user_id, name, value)
    return {"ok": True, "saved": {name: value}, "target": "user_memory"}


def execute_tool(name: str, args: dict, ctx: dict) -> dict:
    """Run one tool request from the model. ctx holds values the PROGRAM controls."""
    try:
        if name == "find_instructions":
            return find_instructions(**args)
        if name == "check_service_status":
            return check_service_status(args.get("service", ""), ctx["status_profile"])
        if name == "prepare_ticket":
            return prepare_ticket(
                args.get("problem", ""), args.get("steps_tried", []),
                args.get("category", ""), args.get("service_status", "not_checked"),
                args.get("reference_ids"), user_id=ctx["user_id"],
                thread_id=ctx["thread_id"], status_profile=ctx["status_profile"],
                tickets_dir=ctx["tickets_dir"])
        if name == "remember_preference":
            return remember_preference(ctx["user_id"], args.get("name", ""),
                                       args.get("value", ""), ctx["memory"])
        return {"ok": False, "error": f"Unknown tool {name!r}."}
    except Exception as exc:  # report to the model instead of crashing the graph
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
