"""Part 1: a fixed-route LangGraph helpdesk. No model, no API key."""
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .routing import (CLARIFY_REPLY, OTHER_REPLY, classify, format_instructions,
                      load_helpdesk)


class HelpdeskState(TypedDict, total=False):
    message: str   # the user's original message (never changed)
    category: str  # set by read_request
    reply: str     # set by a help node


def build_graph(clarify: bool = False):
    """Build and compile the Part 1 graph.

    clarify=False -> original graph (S04); clarify=True -> with clarify_problem (S05).
    """
    helpdesk = load_helpdesk()

    def read_request(state: HelpdeskState) -> dict:
        return {"category": classify(state["message"], clarify, helpdesk)}

    def wifi_help(state: HelpdeskState) -> dict:
        return {"reply": format_instructions(helpdesk["instructions"]["wifi"])}

    def password_help(state: HelpdeskState) -> dict:
        return {"reply": format_instructions(helpdesk["instructions"]["password"])}

    def other_help(state: HelpdeskState) -> dict:
        return {"reply": OTHER_REPLY}

    def clarify_problem(state: HelpdeskState) -> dict:
        return {"reply": CLARIFY_REPLY}

    def choose_next(state: HelpdeskState) -> str:
        # Conditional edge: returns the NAME of the next node. Changes no state.
        return {
            "wifi": "wifi_help",
            "password": "password_help",
            "both": "clarify_problem",
        }.get(state["category"], "other_help")

    g = StateGraph(HelpdeskState)
    g.add_node("read_request", read_request)
    g.add_node("wifi_help", wifi_help)
    g.add_node("password_help", password_help)
    g.add_node("other_help", other_help)
    targets = ["wifi_help", "password_help", "other_help"]
    if clarify:
        g.add_node("clarify_problem", clarify_problem)
        targets.append("clarify_problem")

    g.add_edge(START, "read_request")
    g.add_conditional_edges("read_request", choose_next, targets)
    for node in targets:
        g.add_edge(node, END)
    return g.compile()


def run_with_trace(message: str, clarify: bool = False) -> dict:
    """Run the graph and record each node and the full state after it."""
    graph = build_graph(clarify)
    state: dict = {"message": message}
    trace = []
    for chunk in graph.stream({"message": message}, stream_mode="updates"):
        for node, update in chunk.items():
            state = {**state, **update}
            trace.append({"node": node, "update": update, "state_after": dict(state)})
    return {
        "graph": "with_clarification" if clarify else "original",
        "message": message,
        "nodes_visited": [t["node"] for t in trace],
        "trace": trace,
        "final_state": state,
    }
