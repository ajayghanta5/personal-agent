# Personal AI Agent

A chat-based personal agent built with **LangGraph** + **FastAPI**: it searches the web,
does math, remembers facts about you across sessions (ChromaDB), sets reminders
(SQLite + background watcher), keeps notes, and sends email — with an **approval
gate** that pauses the graph before any sensitive action.

## Quickstart

```bash
cd ~/workspace/personal-agent

# 1. Create a venv and install
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. Configure the LLM (pick one)
cp .env.example .env
# Option A: OpenAI (or any compatible provider) -> set OPENAI_API_KEY
# Option B: local model, e.g. Ollama -> set OPENAI_BASE_URL=http://localhost:11434/v1

# 3. Run
.venv/bin/uvicorn src.app:app --reload
# open http://127.0.0.1:8000
```

Without an LLM key the server still starts: tools, memory and the API work,
and chat politely tells you what to configure.

## Configuration

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `OPENAI_API_KEY` | for chat | — | LLM API key (OpenAI or compatible) |
| `OPENAI_BASE_URL` | alternative | — | Custom endpoint, e.g. Ollama `http://localhost:11434/v1` |
| `MODEL_NAME` | no | `gpt-4o-mini` | Chat model name |
| `SMTP_HOST/PORT/USER/PASS/FROM` | for email | — | Enables the approval-gated `send_email` tool |
| `AGENT_DATA_DIR` | no | `./data` | Where ChromaDB + SQLite live |

## Project structure

```
src/
  app.py              FastAPI app: chat UI at /, POST /chat, POST /approve
  agent/
    graph.py          LangGraph StateGraph: recall -> reason -> approval gate -> act -> respond
                      run_turn() / confirm_action() are the public API
    tools.py          Tool registry: web_search, calculator, remember/recall facts,
                      reminders (+background watcher), notes, send_email (approval-gated)
    memory.py         ChromaDB long-term memory; LLM fact extraction; prune_stale()
    llm.py            ChatOpenAI from env; friendly error when unconfigured
evals/
  prompts.json        20 test prompts (memory, tools, reminders, approval, general)
  run.py              Runner: transcripts to evals/results/; --judge for LLM scoring
```

## How the approval gate works

1. The LLM requests a tool marked `needs_approval=True` (today: `send_email`).
2. The `gate` node pauses the graph and `run_turn()` returns the request as
   `approvals_needed: [{action_id, tool, args, summary}]` — nothing executes.
3. The UI shows an Approve/Deny card; `POST /approve` resumes via
   `confirm_action(action_id, approved=...)`. Denying cancels cleanly.

Add `needs_approval=True` to any `ToolSpec` in `tools.py` to gate a new tool.

## Evals

```bash
.venv/bin/python evals/run.py --filter memory   # one category
.venv/bin/python evals/run.py --judge           # LLM scores each reply 1-5
```

## What to build next

- **Gmail / Calendar tools** — real "personal agent" value; both read-only first,
  then approval-gated actions (send, delete). OAuth tokens go in the Secure Vault,
  never in `.env`.
- **Telegram UI** — a bot polling loop calling `run_turn()`; approvals arrive as
  inline Approve/Deny buttons.
- **Persistent sessions** — replace the in-memory `_SESSIONS` dict with a LangGraph
  `checkpointer` + Postgres so history survives restarts and multiple workers.
- **Memory upgrades** — implement the `prune_stale()` strategy fully (semantic
  dedup, contradiction superseding, pinned facts) and run it on a schedule.
- **Eval CI** — run `evals/run.py --judge` on a schedule; alert when scores drop.
- **Voice interface** — speech-to-text in, TTS out, for a hands-free agent.
