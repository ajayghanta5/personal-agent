"""LangGraph agent: recall -> reason -> approval gate -> act -> respond.

Flow for each user turn:
  1. ``recall``      pull relevant long-term memories for the user's message
  2. ``agent``       LLM reasons and optionally calls tools
  3. ``gate``        if a tool call needs approval, pause here and ask the user
  4. ``act``        run approved / read-only tool calls, loop back to ``agent``
  5. ``respond``     final answer; extract durable facts into memory

Public API:
  run_turn(user_message, session_id="default")
      -> {"reply": str, "approvals_needed": [{"action_id", "tool", "args", "summary"}]}
  confirm_action(action_id, approved=True)
      -> {"reply": str, "approvals_needed": []}

When a tool with needs_approval=True is requested, the graph halts and
``run_turn`` returns the confirmation request. ``confirm_action`` resumes it:
on approval the tool runs and the result is summarized; on denial nothing runs.

Conversation history is kept in-memory per session_id (bounded). For
production, replace _SESSIONS with a LangGraph checkpointer + Postgres.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict


class AgentState(TypedDict, total=False):
    """Graph state. Keys: messages, session_id, memory_context,
    approvals_needed, halt."""
    messages: Annotated[list, add_messages]
    session_id: str
    memory_context: str
    approvals_needed: list
    halt: bool


from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from . import tools as tools_mod
from . import memory as memory_mod
from .llm import LLM_MISSING_MESSAGE, get_llm, llm_configured

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful personal AI agent. You have tools for web search,
a calculator, long-term memory (remember_fact / recall_facts), reminders, notes,
and sending email.

Guidelines:
- Use recall_facts when the user asks about their preferences, past
  conversations, or anything you might have stored — don't guess.
- Proactively remember durable facts (preferences, routines, goals) with
  remember_fact when the user shares them.
- For current events, prices, or anything time-sensitive, use web_search.
- Reminders need a time like "in 30 minutes" or an ISO timestamp.
- send_email always needs the user's explicit approval — never claim you sent
  an email before it is approved and executed.
- Be concise and warm. If a tool fails, say so plainly and suggest an alternative.
"""


def _last_human_text(messages: list[BaseMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m.content if isinstance(m.content, str) else str(m.content)
    return ""


def _last_ai_text(messages: list[BaseMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, AIMessage) and m.content:
            return m.content if isinstance(m.content, str) else str(m.content)
    return "I wasn't able to produce a reply."


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def recall_memories(state: dict) -> dict:
    """Node 1: fetch relevant long-term memories for the latest user message."""
    query = _last_human_text(state["messages"])
    facts = memory_mod.recall(query, k=5)
    context = "\n".join(f"- {f['text']}" for f in facts)
    return {"memory_context": context}


def agent_reason(state: dict) -> dict:
    """Node 2: LLM reasons over the conversation (+ memories) and may call tools."""
    llm = get_llm().bind_tools(tools_mod.as_langchain_tools())
    system = SYSTEM_PROMPT
    if state.get("memory_context"):
        system += f"\n\nRelevant memories about the user:\n{state['memory_context']}"
    response = llm.invoke([SystemMessage(content=system)] + state["messages"])
    return {"messages": [response]}


@dataclass
class PendingApproval:
    action_id: str
    session_id: str
    tool_name: str
    tool_args: dict
    summary: str


# In-memory store of actions awaiting user confirmation.
_PENDING: dict[str, PendingApproval] = {}


def approval_gate(state: dict) -> dict:
    """Node 3: intercept tool calls that need approval; pause the graph.

    Returns approvals_needed (possibly empty) and halt=True when the graph
    must stop and wait for confirm_action().
    """
    last = state["messages"][-1]
    approvals: list[dict] = []
    if isinstance(last, AIMessage) and last.tool_calls:
        for tc in last.tool_calls:
            spec = tools_mod.get_tool(tc["name"])
            if spec is None or not spec.needs_approval:
                continue
            action_id = uuid.uuid4().hex[:12]
            summary = (
                spec.approval_summary(tc["args"])
                if spec.approval_summary
                else f"Run {spec.name} with {tc['args']}"
            )
            _PENDING[action_id] = PendingApproval(
                action_id=action_id,
                session_id=state.get("session_id", "default"),
                tool_name=spec.name,
                tool_args=tc["args"],
                summary=summary,
            )
            approvals.append({
                "action_id": action_id,
                "tool": spec.name,
                "args": tc["args"],
                "summary": summary,
            })
    return {"approvals_needed": approvals, "halt": bool(approvals)}


def _route(state: dict) -> str:
    if state.get("halt"):
        return "end"
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "act"
    return "respond"


def respond(state: dict) -> dict:
    """Node 5: finalize the turn; extract durable facts into long-term memory."""
    reply = _last_ai_text(state["messages"])
    try:
        facts = memory_mod.extract_facts(_last_human_text(state["messages"]), reply)
        for fact in facts:
            memory_mod.store_fact(fact, session_id=state.get("session_id", "default"))
    except Exception as exc:  # noqa: BLE001 - memory must never break a turn
        logger.warning("Fact extraction/store failed: %s", exc)
    return {}  # reply is already the last message in state


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        g = StateGraph(AgentState)
        g.add_node("recall", recall_memories)
        g.add_node("agent", agent_reason)
        g.add_node("gate", approval_gate)
        g.add_node("act", ToolNode(tools_mod.as_langchain_tools()))
        g.add_node("respond", respond)
        g.set_entry_point("recall")
        g.add_edge("recall", "agent")
        g.add_edge("agent", "gate")
        g.add_conditional_edges("gate", _route, {"act": "act", "respond": "respond", "end": END})
        g.add_edge("act", "agent")
        g.add_edge("respond", END)
        _graph = g.compile()
    return _graph


# In-memory conversation history per session (bounded). Swap for a real
# checkpointer + Postgres for multi-process / production use.
_SESSIONS: dict[str, list[BaseMessage]] = {}
_MAX_HISTORY = 40


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_turn(user_message: str, session_id: str = "default") -> dict[str, Any]:
    """Run one user turn through the agent.

    Returns {"reply": str, "approvals_needed": [...]}. approvals_needed is a
    list of {"action_id", "tool", "args", "summary"} when a tool call is
    waiting for user confirmation — call confirm_action() to resume.
    """
    if not llm_configured():
        return {"reply": LLM_MISSING_MESSAGE, "approvals_needed": []}

    tools_mod.SESSION_CTX.set(session_id)
    history = _SESSIONS.get(session_id, [])
    try:
        result = _get_graph().invoke(
            {
                "messages": [*history, HumanMessage(content=user_message)],
                "session_id": session_id,
            },
            {"recursion_limit": 20},
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Agent turn failed")
        return {"reply": f"Something went wrong running the agent: {exc}", "approvals_needed": []}

    _SESSIONS[session_id] = result["messages"][-_MAX_HISTORY:]
    return {
        "reply": _last_ai_text(result["messages"]),
        "approvals_needed": result.get("approvals_needed") or [],
    }


def confirm_action(action_id: str, approved: bool = True) -> dict[str, Any]:
    """Resume a paused turn: run (or cancel) a tool call awaiting approval."""
    pending = _PENDING.pop(action_id, None)
    if pending is None:
        return {
            "reply": "I couldn't find that pending action — it may have already been handled.",
            "approvals_needed": [],
        }
    if not approved:
        return {
            "reply": f"Cancelled: {pending.summary}. Nothing was executed.",
            "approvals_needed": [],
        }

    tools_mod.SESSION_CTX.set(pending.session_id)
    try:
        output = tools_mod.execute_tool(pending.tool_name, pending.tool_args)
    except Exception as exc:  # noqa: BLE001
        return {"reply": f"The approved action failed: {exc}", "approvals_needed": []}

    # Turn the raw tool result into a natural reply.
    if llm_configured():
        try:
            llm = get_llm()
            msg = llm.invoke([
                SystemMessage(content=(
                    "You are a helpful personal assistant. Summarize what happened "
                    "in one or two sentences."
                )),
                HumanMessage(content=(
                    f"The user approved and I ran '{pending.tool_name}' with "
                    f"arguments {pending.tool_args}.\nResult:\n{output}\n\n"
                    "Summarize the outcome for the user."
                )),
            ])
            return {"reply": msg.content, "approvals_needed": []}
        except Exception as exc:  # noqa: BLE001
            logger.warning("Approval summary failed: %s", exc)
    return {"reply": f"Done: {pending.summary}.\n\nResult:\n{output}", "approvals_needed": []}
