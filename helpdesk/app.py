"""Ties together: model + tools graph, checkpointer (conversation) and user memory."""
import json
import shutil
import sqlite3
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from .agent import MAX_TOOL_ROUNDS, build_agent_graph
from .memory import UserMemory


def thread_key(user_id: str, thread_id: str) -> str:
    """The checkpointer key. Namespaced by user so two users can never share a thread."""
    return f"{user_id}/{thread_id}"


def reset_storage(storage_dir) -> None:
    shutil.rmtree(storage_dir, ignore_errors=True)


class Helpdesk:
    def __init__(self, model, storage_dir="local-storage", profile="normal",
                 tickets_dir="outputs/tickets", max_rounds=MAX_TOOL_ROUNDS):
        self.profile = profile
        storage = Path(storage_dir)
        storage.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(storage / "checkpoints.sqlite", check_same_thread=False)
        self.memory = UserMemory(storage / "user_memory.sqlite")
        self.graph = build_agent_graph(model, self.memory, SqliteSaver(self._conn),
                                       tickets_dir, max_rounds)

    def close(self):
        self._conn.close()

    def send(self, user_id: str, thread_id: str, message: str) -> dict:
        config = {"configurable": {"thread_id": thread_key(user_id, thread_id),
                                   "user_id": user_id, "raw_thread_id": thread_id,
                                   "status_profile": self.profile}}
        before = self.graph.get_state(config).values
        prior = len(before.get("messages", [])) if before else 0
        steps, requests, results = [], [], []
        for chunk in self.graph.stream({"messages": [HumanMessage(message)]}, config,
                                       stream_mode="updates"):
            for node, update in chunk.items():
                update = update or {}
                step = {"node": node}
                if node == "load_user_memory":
                    step.update(read_from="user_memory", user_memory=update.get("user_memory"))
                elif node == "call_model":
                    m = update["messages"][-1]
                    calls = [{"name": c["name"], "args": c["args"]} for c in m.tool_calls]
                    requests += calls
                    step.update(tool_requests=calls, content=m.content)
                elif node in ("run_tools", "stop_at_limit"):
                    rs = [{"name": m.name, "result": json.loads(m.content)}
                          for m in update["messages"] if isinstance(m, ToolMessage)]
                    results += rs
                    step.update(tool_results=rs)
                    if node == "run_tools":
                        step["tool_rounds"] = update.get("tool_rounds")
                steps.append(step)
        final = self.graph.get_state(config).values
        reply = next(m.content for m in reversed(final["messages"]) if isinstance(m, AIMessage))
        used = {r["name"] for r in requests}
        sources = []
        if prior:
            sources.append("conversation_state (checkpointer)")
        if final.get("user_memory"):
            sources.append("user_memory")
        if used & {"find_instructions", "check_service_status"}:
            sources.append("reference_file (helpdesk.json / service_status.json)")
        tickets = [r["result"]["file"] for r in results if r["name"] == "prepare_ticket"
                   and r["result"].get("ok")]
        return {"user_id": user_id, "thread_id": thread_id, "thread_key": thread_key(user_id, thread_id),
                "status_profile": self.profile, "message": message,
                "messages_in_state_before": prior, "nodes_visited": [s["node"] for s in steps],
                "steps": steps, "tool_requests": requests, "tool_results": results,
                "information_sources": sources, "ticket_files": tickets,
                "loop_limit_reached": bool(final.get("limit_reached")), "reply": reply}
