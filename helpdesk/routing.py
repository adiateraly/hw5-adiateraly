"""Plain-Python decisions for Part 1. No LangGraph and no LLM in this file.

Keeping these rules as ordinary functions makes them easy to test and shows
clearly which decisions are *yours* and which are made by LangGraph.
"""
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_helpdesk(path: Path = DATA_DIR / "helpdesk.json") -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def topics_in(message: str, helpdesk: dict | None = None) -> set[str]:
    """Return the topics ('wifi', 'password') mentioned in the message.

    Matching ignores upper/lower case and uses topic_aliases from helpdesk.json.
    """
    helpdesk = helpdesk or load_helpdesk()
    text = message.lower()
    return {
        topic
        for topic, aliases in helpdesk["topic_aliases"].items()
        if any(alias in text for alias in aliases)
    }


def classify(message: str, clarify: bool = False, helpdesk: dict | None = None) -> str:
    """Original rules: wifi first, then password, else other.

    With clarify=True a message mentioning both topics gets category 'both'.
    """
    found = topics_in(message, helpdesk)
    if clarify and {"wifi", "password"} <= found:
        return "both"
    if "wifi" in found:
        return "wifi"
    if "password" in found:
        return "password"
    return "other"


def format_instructions(entry: dict) -> str:
    steps = "\n".join(f"{i}. {s}" for i, s in enumerate(entry["steps"], 1))
    return f"{entry['title']} [{entry['instruction_id']}]\n{steps}"


OTHER_REPLY = "Is your problem about Wi-Fi or about a password?"
CLARIFY_REPLY = (
    "Is the problem connecting to Wi-Fi, or recovering your account password? "
    "Which one do you want to solve first?"
)
