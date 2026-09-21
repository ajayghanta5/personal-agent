"""Tool definitions for the personal agent.

Every tool is registered in REGISTRY with metadata the graph uses:

- ``read_only``: safe to run without asking (search, calculator, ...).
- ``needs_approval``: the graph pauses and asks the user to confirm before
  running it. The user confirms via ``graph.confirm_action(action_id)``.

The current session id is carried in SESSION_CTX (a context var set by
``graph.run_turn``), so tool functions don't need it as an LLM-visible arg.

Tools are exposed to the LLM as LangChain StructuredTools, and can also be
executed directly via ``execute_tool(name, args)`` (used by the approval
resume path and by tests).
"""
from __future__ import annotations

import ast
import logging
import math
import operator
import os
import re
import smtplib
import sqlite3
import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from typing import Any, Callable

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from . import DATA_DIR
from . import memory as memory_mod

logger = logging.getLogger(__name__)

# Session id for the current turn; set by graph.run_turn() before invoking.
SESSION_CTX: ContextVar[str] = ContextVar("agent_session_id", default="default")

# ---------------------------------------------------------------------------
# SQLite storage (reminders + notes)
# ---------------------------------------------------------------------------


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(DATA_DIR / "agent.db")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS reminders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            text TEXT NOT NULL,
            due_at TEXT NOT NULL,
            created_at TEXT NOT NULL,
            fired INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """
    )
    return conn


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Calculator (safe arithmetic — no exec/eval of arbitrary code)
# ---------------------------------------------------------------------------

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.FloorDiv: operator.floordiv,
}
_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}

# Whitelisted names the calculator may use.
_MATH = {
    "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "log": math.log, "log10": math.log10, "exp": math.exp, "pow": pow,
    "abs": abs, "round": round, "floor": math.floor, "ceil": math.ceil,
    "factorial": math.factorial, "pi": math.pi, "e": math.e,
}


def _check_node(node: ast.AST) -> None:
    """Reject any AST node that isn't plain arithmetic."""
    if isinstance(node, ast.Expression):
        _check_node(node.body)
    elif isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        _check_node(node.left)
        _check_node(node.right)
    elif isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
        _check_node(node.operand)
    elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return
    elif isinstance(node, ast.Name) and node.id in _MATH:
        return
    elif (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _MATH
        and callable(_MATH[node.func.id])
        and not node.keywords
    ):
        for arg in node.args:
            _check_node(arg)
    else:
        raise ValueError(f"not allowed here: {ast.dump(node)[:60]}")


def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression safely (no code execution)."""
    try:
        tree = ast.parse(expression, mode="eval")
        _check_node(tree)
        result = eval(  # noqa: S307 - AST was whitelisted above; no builtins
            compile(tree, "<calculator>", "eval"),
            {"__builtins__": {}},
            dict(_MATH),
        )
    except (SyntaxError, ValueError) as exc:
        return f"I couldn't evaluate that: {exc}. Try a plain arithmetic expression like '(2+3)*4'."
    except Exception as exc:  # noqa: BLE001 - e.g. division by zero
        return f"Calculation error: {exc}"
    return str(result)


class CalculatorInput(BaseModel):
    expression: str = Field(description="Arithmetic expression, e.g. '(2+3)*4' or 'sqrt(16) + 2**5'.")


# ---------------------------------------------------------------------------
# Web search (read-only)
# ---------------------------------------------------------------------------


def _ddgs():
    # `ddgs` is the current name of the DuckDuckGo search package
    # (formerly `duckduckgo-search`); support both just in case.
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS
    return DDGS()


def web_search(query: str, max_results: int = 5) -> str:
    """Search the web and return titles, URLs and snippets."""
    try:
        with _ddgs() as ddgs:
            results = list(ddgs.text(query, max_results=max(1, min(max_results, 10))))
    except Exception as exc:  # noqa: BLE001 - network flakiness shouldn't crash a turn
        return f"Web search failed: {exc}"
    if not results:
        return "No results found."
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r.get('title', 'untitled')}\n   {r.get('href', '')}\n   {r.get('body', '')}")
    return "\n".join(lines)


class WebSearchInput(BaseModel):
    query: str = Field(description="The search query.")
    max_results: int = Field(default=5, description="How many results to return (1-10).")


# ---------------------------------------------------------------------------
# Memory tools
# ---------------------------------------------------------------------------


def remember_fact(fact: str) -> str:
    """Save a concise fact about the user to long-term memory."""
    fact = fact.strip()
    if not fact:
        return "Nothing to remember — the fact was empty."
    memory_mod.store_fact(fact, session_id=SESSION_CTX.get())
    return f"Remembered: {fact}"


def recall_facts(query: str) -> str:
    """Look up facts previously stored in long-term memory."""
    facts = memory_mod.recall(query, k=5)
    if not facts:
        return "I don't have any memories matching that."
    return "\n".join(f"- {f['text']}" for f in facts)


class FactInput(BaseModel):
    fact: str = Field(description="One concise fact about the user, e.g. 'User prefers evening workouts.'")


class RecallInput(BaseModel):
    query: str = Field(description="What to look up in memory.")


# ---------------------------------------------------------------------------
# Reminders (SQLite + background watcher thread)
# ---------------------------------------------------------------------------

_RELATIVE_RE = re.compile(
    r"^\s*in\s+(\d+)\s*(seconds?|minutes?|hours?|days?)\s*$", re.IGNORECASE
)


def parse_when(when: str) -> datetime:
    """Parse 'in 30 minutes' / 'in 2 hours' / ISO-8601 into an aware datetime.

    Raises ValueError with a helpful message when the format isn't understood.
    """
    when = when.strip()
    m = _RELATIVE_RE.match(when)
    if m:
        amount, unit = int(m.group(1)), m.group(2).lower()
        delta = {
            "second": timedelta(seconds=amount), "seconds": timedelta(seconds=amount),
            "minute": timedelta(minutes=amount), "minutes": timedelta(minutes=amount),
            "hour": timedelta(hours=amount), "hours": timedelta(hours=amount),
            "day": timedelta(days=amount), "days": timedelta(days=amount),
        }[unit]
        return datetime.now(timezone.utc) + delta
    try:
        dt = datetime.fromisoformat(when)
    except ValueError:
        raise ValueError(
            f"I didn't understand the time '{when}'. "
            "Use something like 'in 30 minutes', 'in 2 hours', or an ISO time like '2026-09-22T09:00'."
        )
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    if dt <= datetime.now(timezone.utc):
        raise ValueError(f"That time is in the past: '{when}'.")
    return dt


def add_reminder(text: str, when: str) -> str:
    """Schedule a reminder. `when` like 'in 30 minutes' or ISO-8601."""
    try:
        due = parse_when(when)
    except ValueError as exc:
        return str(exc)
    with _db() as conn:
        conn.execute(
            "INSERT INTO reminders (session_id, text, due_at, created_at) VALUES (?, ?, ?, ?)",
            (SESSION_CTX.get(), text.strip(), due.isoformat(), _now_iso()),
        )
    start_reminder_watcher()  # make sure the background thread is running
    return f"Reminder set for {due.strftime('%Y-%m-%d %H:%M %Z')}: {text.strip()}"


def list_reminders() -> str:
    """List upcoming (not yet fired) reminders."""
    with _db() as conn:
        rows = conn.execute(
            "SELECT text, due_at FROM reminders WHERE fired = 0 ORDER BY due_at"
        ).fetchall()
    if not rows:
        return "You have no upcoming reminders."
    return "\n".join(
        f"- {r['text']} (at {r['due_at']})" for r in rows
    )


class ReminderInput(BaseModel):
    text: str = Field(description="What to remind the user about.")
    when: str = Field(description="When: 'in 30 minutes', 'in 2 hours', or ISO-8601 like '2026-09-22T09:00'.")


def check_due_reminders() -> list[dict]:
    """Fire due reminders: mark them fired and log them. Returns what fired."""
    now = _now_iso()
    fired: list[dict] = []
    with _db() as conn:
        rows = conn.execute(
            "SELECT id, session_id, text, due_at FROM reminders WHERE fired = 0 AND due_at <= ?",
            (now,),
        ).fetchall()
        for r in rows:
            conn.execute("UPDATE reminders SET fired = 1 WHERE id = ?", (r["id"],))
            fired.append(dict(r))
    if fired:
        log = DATA_DIR / "due_reminders.log"
        with open(log, "a") as f:
            for r in fired:
                line = f"[{now}] REMINDER (session={r['session_id']}): {r['text']}\n"
                f.write(line)
                print(line, end="")  # visible in server logs too
    return fired


_watcher_started = False


def _watch_loop(poll_seconds: int) -> None:
    while True:
        try:
            check_due_reminders()
        except Exception as exc:  # noqa: BLE001 - watcher must never die
            logger.warning("Reminder watcher error: %s", exc)
        time.sleep(poll_seconds)


def start_reminder_watcher(poll_seconds: int = 30) -> None:
    """Start the background thread that fires due reminders (idempotent)."""
    global _watcher_started
    if _watcher_started:
        return
    _watcher_started = True
    thread = threading.Thread(
        target=_watch_loop, args=(poll_seconds,), daemon=True, name="reminder-watcher"
    )
    thread.start()


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


def save_note(title: str, content: str) -> str:
    """Save a note for later."""
    with _db() as conn:
        conn.execute(
            "INSERT INTO notes (session_id, title, content, created_at) VALUES (?, ?, ?, ?)",
            (SESSION_CTX.get(), title.strip(), content.strip(), _now_iso()),
        )
    return f"Note saved: {title.strip()}"


def search_notes(query: str) -> str:
    """Search saved notes by keyword."""
    like = f"%{query.strip()}%"
    with _db() as conn:
        rows = conn.execute(
            "SELECT title, content, created_at FROM notes "
            "WHERE title LIKE ? OR content LIKE ? ORDER BY created_at DESC LIMIT 5",
            (like, like),
        ).fetchall()
    if not rows:
        return f"No notes matching '{query}'."
    return "\n\n".join(
        f"**{r['title']}** ({r['created_at'][:10]})\n{r['content']}" for r in rows
    )


class NoteInput(BaseModel):
    title: str = Field(description="Short title for the note.")
    content: str = Field(description="The note content.")


class NoteSearchInput(BaseModel):
    query: str = Field(description="Keyword to search notes for.")


# ---------------------------------------------------------------------------
# Email (example of an APPROVAL-GATED tool)
# ---------------------------------------------------------------------------


def send_email(to: str, subject: str, body: str) -> str:
    """Send an email via SMTP. Requires approval; requires SMTP env config."""
    host = os.getenv("SMTP_HOST")
    user = os.getenv("SMTP_USER")
    password = os.getenv("SMTP_PASS")
    sender = os.getenv("SMTP_FROM", user)
    if not all([host, user, password]):
        raise RuntimeError(
            "Email is not configured. Set SMTP_HOST, SMTP_USER and SMTP_PASS "
            "in your .env file to enable sending email."
        )
    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)
    with smtplib.SMTP(host, int(os.getenv("SMTP_PORT", "587"))) as smtp:
        smtp.starttls()
        smtp.login(user, password)
        smtp.send_message(msg)
    return f"Email sent to {to} with subject '{subject}'."


class EmailInput(BaseModel):
    to: str = Field(description="Recipient email address.")
    subject: str = Field(description="Email subject.")
    body: str = Field(description="Email body.")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass
class ToolSpec:
    name: str
    description: str
    func: Callable[..., str]
    args_schema: type[BaseModel] | None = None
    read_only: bool = False
    needs_approval: bool = False
    approval_summary: Callable[[dict], str] | None = None
    langchain_tool: StructuredTool = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.langchain_tool = StructuredTool.from_function(
            func=self.func,
            name=self.name,
            description=self.description,
            args_schema=self.args_schema,
        )


def _email_summary(args: dict) -> str:
    return f"Send email to {args.get('to')} with subject '{args.get('subject')}'"


REGISTRY: dict[str, ToolSpec] = {}


def _register(spec: ToolSpec) -> ToolSpec:
    REGISTRY[spec.name] = spec
    return spec


_register(ToolSpec("calculator", "Evaluate an arithmetic expression.", calculator,
                   CalculatorInput, read_only=True))
_register(ToolSpec("web_search", "Search the web for current information.", web_search,
                   WebSearchInput, read_only=True))
_register(ToolSpec("remember_fact", "Save a concise fact about the user to long-term memory.",
                   remember_fact, FactInput))
_register(ToolSpec("recall_facts", "Look up facts previously stored in long-term memory.",
                   recall_facts, RecallInput, read_only=True))
_register(ToolSpec("add_reminder", "Schedule a reminder for the user.", add_reminder,
                   ReminderInput))
_register(ToolSpec("list_reminders", "List the user's upcoming reminders.", list_reminders,
                   read_only=True))
_register(ToolSpec("save_note", "Save a note for later.", save_note, NoteInput))
_register(ToolSpec("search_notes", "Search saved notes by keyword.", search_notes,
                   NoteSearchInput, read_only=True))
_register(ToolSpec("send_email", "Send an email via SMTP. Requires user approval first.",
                   send_email, EmailInput, needs_approval=True,
                   approval_summary=_email_summary))


def get_tool(name: str) -> ToolSpec | None:
    return REGISTRY.get(name)


def as_langchain_tools() -> list[StructuredTool]:
    """Tools in the form the LLM binds to."""
    return [spec.langchain_tool for spec in REGISTRY.values()]


def execute_tool(name: str, args: dict[str, Any]) -> str:
    """Run a tool directly (used by the approval-resume path and tests)."""
    spec = get_tool(name)
    if spec is None:
        raise ValueError(f"Unknown tool: {name}")
    if spec.args_schema is not None:
        validated = spec.args_schema(**args)
        return spec.func(**validated.model_dump())
    return spec.func(**args)
