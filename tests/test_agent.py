"""Graph tests with the FAKE model. No API key, no network."""
import pytest

pytest.importorskip("langgraph")
from langchain_core.messages import AIMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult

from helpdesk.app import Helpdesk
from helpdesk.fake_model import ScriptedFakeModel

COND = ("I enabled Wi-Fi and forgot the saved network, but Campus-Demo still will not connect. "
        "Check whether Wi-Fi is down. If it is operating, prepare a ticket about this problem.")


def make(tmp_path, profile="normal", model=None, **kw):
    return Helpdesk(model or ScriptedFakeModel(), tmp_path / "store", profile,
                    tmp_path / "tickets", **kw)


def names(r):
    return [c["name"] for c in r["tool_requests"]]


@pytest.mark.parametrize("profile,expect_ticket", [("normal", True), ("wifi_outage", False),
                                                   ("status_unavailable", False)])
def test_same_request_different_snapshot(tmp_path, profile, expect_ticket):
    # Would catch: always making a ticket regardless of the tool result (S06/S07/S08).
    hd = make(tmp_path, profile)
    r = hd.send("demo-user-a", "t", COND)
    hd.close()
    assert ("prepare_ticket" in names(r)) is expect_ticket
    assert names(r)[0] == "check_service_status"
    assert bool(r["ticket_files"]) is expect_ticket


def test_missing_details_asks_question(tmp_path):
    hd = make(tmp_path)
    r = hd.send("demo-user-a", "t", "Prepare a support ticket.")
    hd.close()
    assert "prepare_ticket" not in names(r) and "already tried" in r["reply"]


def test_conversation_memory_within_thread(tmp_path):
    hd = make(tmp_path)
    hd.send("demo-user-a", "t", "I cannot connect to Wi-Fi. I enabled it and forgot the saved network.")
    r = hd.send("demo-user-a", "t", "Which steps did I already try?")
    hd.close()
    assert r["messages_in_state_before"] > 0 and "enabled wi-fi" in r["reply"].lower()


def test_memory_survives_restart(tmp_path):
    # Would catch: in-memory checkpointer instead of SQLite.
    hd = make(tmp_path)
    hd.send("demo-user-a", "t", "Please remember that I prefer replies in Kazakh. "
                                "My issue is Wi-Fi. I enabled it and tried reconnecting.")
    hd.close()
    hd2 = make(tmp_path)  # a brand-new object on the same storage = a restart
    r = hd2.send("demo-user-a", "t", "Which problem and steps did we discuss?")
    hd2.close()
    assert "Wi-Fi" in r["reply"] and r["messages_in_state_before"] > 0
    assert "user_memory" in r["information_sources"]


def test_preference_in_new_thread_but_not_old_messages(tmp_path):
    hd = make(tmp_path)
    hd.send("demo-user-a", "a1", "Please remember that I prefer replies in Kazakh. I have a Wi-Fi problem.")
    r = hd.send("demo-user-a", "a2", "Which language did I ask you to use?")
    hd.close()
    assert "Kazakh" in r["reply"]
    assert r["messages_in_state_before"] == 0  # separate thread: no old transcript


def test_other_user_sees_nothing(tmp_path):
    # Would catch: preference or thread shared between users.
    hd = make(tmp_path)
    hd.send("demo-user-a", "same", "Please remember that I prefer replies in Kazakh. I have a Wi-Fi problem.")
    r = hd.send("demo-user-b", "same", "Have I told you a preferred language or a previous problem?")
    hd.close()
    assert "Kazakh" not in r["reply"] and r["messages_in_state_before"] == 0
    assert "No preference" in r["reply"]


class LoopingModel(BaseChatModel):
    @property
    def _llm_type(self):
        return "looping-fake"

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        msg = AIMessage(content="", tool_calls=[{"name": "find_instructions", "args": {"topic": "wifi"},
                                                 "id": f"c{len(messages)}"}])
        return ChatResult(generations=[ChatGeneration(message=msg)])


def test_loop_limit_stops_runaway_model(tmp_path):
    # Would catch: a missing or broken loop limit (infinite model/tool loop).
    hd = make(tmp_path, model=LoopingModel(), max_rounds=3)
    r = hd.send("demo-user-a", "t", "hello")
    hd.close()
    assert r["loop_limit_reached"] is True and len(r["tool_requests"]) == 4
