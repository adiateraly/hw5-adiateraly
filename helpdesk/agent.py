"""Parts 2-3: model + tools graph with a loop limit, conversation and user memory."""
import json
import os
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from .tools import execute_tool

MAX_TOOL_ROUNDS = 5  # loop limit: at most 5 rounds of tool execution per user message


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]  # CONVERSATION memory (checkpointed)
    user_memory: dict      # loaded from USER memory at the start of every turn
    tool_rounds: int       # tool rounds used in this turn
    limit_reached: bool


# ---- schemas the model sees (the program injects user/thread/profile itself) ----
class _FindArgs(BaseModel):
    topic: str = Field(description="'wifi' or 'password'")


class _StatusArgs(BaseModel):
    service: str = Field(description="'wifi' or 'account_recovery'")


class _TicketArgs(BaseModel):
    problem: str = Field(description="The user's problem in their own words. Never invent details.")
    steps_tried: list[str] = Field(description="Steps the USER said they already tried. Never invent any.")
    category: str = Field(description="'wifi' or 'password'")
    service_status: str = Field(default="not_checked",
                                description="'operational', 'outage', 'unknown' or 'not_checked'")
    reference_ids: list[str] = Field(default_factory=list,
                                     description="e.g. GUIDE-WIFI if instructions were used")


class _RememberArgs(BaseModel):
    name: str = Field(description="Only 'reply_language' is supported")
    value: str = Field(description="The language, e.g. 'Kazakh'")


def _schema_tool(name, description, args):
    return StructuredTool.from_function(func=lambda **kw: None, name=name,
                                        description=description, args_schema=args)


TOOL_SCHEMAS = [
    _schema_tool("find_instructions", "Look up Wi-Fi or password instructions in the local helpdesk guide.", _FindArgs),
    _schema_tool("check_service_status", "Check whether wifi or account_recovery is operational, in outage, or unknown.", _StatusArgs),
    _schema_tool("prepare_ticket", "Save a LOCAL ticket draft. Needs the problem and the steps the user already tried.", _TicketArgs),
    _schema_tool("remember_preference", "Save the user's reply-language preference for future conversations.", _RememberArgs),
]


def build_system_prompt(user_memory: dict) -> str:
    lang = user_memory.get("reply_language")
    memory_text = (f"Saved user memory: reply_language = {lang}. Reply in {lang}."
                   if lang else "Saved user memory: nothing recorded for this user.")
    return (
        "You are a university IT helpdesk assistant (practice/fictional data).\n"
        "Scope: Wi-Fi troubleshooting, password recovery guidance, service status lookup, "
        "and preparing LOCAL ticket drafts. Nothing is ever sent to a real helpdesk.\n"
        "Rules:\n"
        "- Use find_instructions for steps; never make up instructions.\n"
        "- Use check_service_status before deciding about a ticket when the user asks. "
        "Password problems use service 'account_recovery'. status 'unknown' is NOT operational "
        "and NOT an outage: say you could not determine it and ask if they want a draft.\n"
        "- If the user said to prepare a ticket only IF the service is operating, and it is in outage "
        "or unknown, do NOT prepare it. Do not invent repair times.\n"
        "- prepare_ticket needs the problem and steps the USER already tried. If either is missing, "
        "ask: what problem are you having and what have you already tried? Never invent steps.\n"
        "- Never ask for or store passwords, API keys or recovery codes.\n"
        "- When the user states a preferred reply language, call remember_preference.\n"
        "- Use the earlier messages of this conversation to answer questions about what was discussed.\n"
        "- Say clearly that a ticket is only a local draft.\n"
        + memory_text
    )


def build_agent_graph(model, memory, checkpointer, tickets_dir="outputs/tickets",
                      max_rounds: int = MAX_TOOL_ROUNDS):
    llm = model.bind_tools(TOOL_SCHEMAS)

    def load_user_memory(state: AgentState, config: RunnableConfig) -> dict:
        user_id = config["configurable"]["user_id"]
        return {"user_memory": memory.get_all(user_id), "tool_rounds": 0, "limit_reached": False}

    def call_model(state: AgentState) -> dict:
        prompt = SystemMessage(build_system_prompt(state.get("user_memory", {})))
        return {"messages": [llm.invoke([prompt] + state["messages"])]}

    def run_tools(state: AgentState, config: RunnableConfig) -> dict:
        cfg = config["configurable"]
        ctx = {"user_id": cfg["user_id"], "thread_id": cfg["raw_thread_id"],
               "status_profile": cfg["status_profile"], "tickets_dir": tickets_dir,
               "memory": memory}
        out = []
        for call in state["messages"][-1].tool_calls:
            result = execute_tool(call["name"], call["args"], ctx)
            out.append(ToolMessage(content=json.dumps(result, ensure_ascii=False),
                                   tool_call_id=call["id"], name=call["name"]))
        return {"messages": out, "tool_rounds": state.get("tool_rounds", 0) + 1}

    def stop_at_limit(state: AgentState) -> dict:
        # Every pending tool request must get an answer, or the next model call is invalid.
        out = [ToolMessage(content=json.dumps({"ok": False, "error": "loop limit reached, not executed"}),
                           tool_call_id=c["id"], name=c["name"])
               for c in state["messages"][-1].tool_calls]
        out.append(AIMessage(content="I stopped because I reached the limit of tool calls for one "
                                     "request. Please rephrase or try a smaller request."))
        return {"messages": out, "limit_reached": True}

    def route_after_model(state: AgentState) -> str:
        # The LOOP LIMIT is enforced here, in the conditional edge.
        if not state["messages"][-1].tool_calls:
            return END
        if state.get("tool_rounds", 0) >= max_rounds:
            return "stop_at_limit"
        return "run_tools"

    g = StateGraph(AgentState)
    g.add_node("load_user_memory", load_user_memory)
    g.add_node("call_model", call_model)
    g.add_node("run_tools", run_tools)
    g.add_node("stop_at_limit", stop_at_limit)
    g.add_edge(START, "load_user_memory")
    g.add_edge("load_user_memory", "call_model")
    g.add_conditional_edges("call_model", route_after_model,
                            ["run_tools", "stop_at_limit", END])
    g.add_edge("run_tools", "call_model")
    g.add_edge("stop_at_limit", END)
    return g.compile(checkpointer=checkpointer)


def build_real_model():
    """OpenAI chat model from environment variables (.env supported)."""
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI
    load_dotenv()
    name = os.environ.get("MODEL_NAME")
    if not os.environ.get("OPENAI_API_KEY") or not name:
        raise SystemExit("Set OPENAI_API_KEY and MODEL_NAME in .env (see .env.example), "
                         "or use --fake-model for offline tests.")
    return ChatOpenAI(model=name, temperature=0), f"openai:{name}"
