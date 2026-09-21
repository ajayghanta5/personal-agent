"""FastAPI front-end for the personal agent.

Run from the repo root:
    uvicorn src.app:app --reload

Endpoints:
  GET  /          minimal chat UI (single HTML page, no build step)
  POST /chat      {"message": str, "session_id": "default"}
                  -> {"reply": str, "approvals_needed": [...]}
  POST /approve   {"action_id": str, "approved": bool, "session_id": "default"}
                  -> {"reply": str, "approvals_needed": []}
  GET  /health    {"ok": true}
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make `agent` importable whether this runs as `src.app` or `app`.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from agent import tools as tools_mod
from agent.graph import confirm_action, run_turn

app = FastAPI(title="Personal Agent")


class ChatIn(BaseModel):
    message: str
    session_id: str = "default"


class ApproveIn(BaseModel):
    action_id: str
    approved: bool = True
    session_id: str = "default"


@app.on_event("startup")
def _startup() -> None:
    tools_mod.start_reminder_watcher()


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/chat")
def chat(body: ChatIn) -> dict:
    return run_turn(body.message, session_id=body.session_id)


@app.post("/approve")
def approve(body: ApproveIn) -> dict:
    return confirm_action(body.action_id, approved=body.approved)


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Personal Agent</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 720px; margin: 0 auto; padding: 16px; background: #f7f7f8; }
  h1 { font-size: 1.4rem; }
  #log { display: flex; flex-direction: column; gap: 10px; margin: 16px 0; min-height: 40vh; }
  .msg { padding: 10px 14px; border-radius: 12px; max-width: 85%; white-space: pre-wrap; }
  .user { align-self: flex-end; background: #2563eb; color: #fff; }
  .bot { align-self: flex-start; background: #fff; border: 1px solid #e5e7eb; }
  .approval { align-self: flex-start; background: #fffbeb; border: 1px solid #f59e0b; padding: 10px 14px; border-radius: 12px; max-width: 85%; }
  .approval button { margin: 6px 6px 0 0; padding: 6px 12px; cursor: pointer; }
  #form { display: flex; gap: 8px; }
  #input { flex: 1; padding: 10px; border-radius: 8px; border: 1px solid #d1d5db; font-size: 1rem; }
  button.send { padding: 10px 18px; border-radius: 8px; border: none; background: #2563eb; color: #fff; font-size: 1rem; cursor: pointer; }
</style>
</head>
<body>
<h1>Personal Agent</h1>
<div id="log"></div>
<form id="form">
  <input id="input" autocomplete="off" placeholder="Ask me anything..." />
  <button class="send" type="submit">Send</button>
</form>
<script>
const log = document.getElementById('log');
const input = document.getElementById('input');
const sessionId = 'web-' + Math.random().toString(36).slice(2, 8);

function addMsg(text, cls) {
  const div = document.createElement('div');
  div.className = 'msg ' + cls;
  div.textContent = text;
  log.appendChild(div);
  log.scrollIntoView({block: 'end'});
  return div;
}
function addApproval(a) {
  const div = document.createElement('div');
  div.className = 'approval';
  div.innerHTML = '<b>Approval needed</b><br>' + escapeHtml(a.summary);
  const ok = document.createElement('button');
  ok.textContent = 'Approve';
  ok.onclick = () => decide(a.action_id, true, div);
  const no = document.createElement('button');
  no.textContent = 'Deny';
  no.onclick = () => decide(a.action_id, false, div);
  div.appendChild(document.createElement('br'));
  div.appendChild(ok); div.appendChild(no);
  log.appendChild(div);
}
function escapeHtml(s) {
  return s.replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
async function decide(action_id, approved, div) {
  div.querySelectorAll('button').forEach(b => b.disabled = true);
  const r = await fetch('/approve', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({action_id, approved, session_id: sessionId})
  });
  const data = await r.json();
  addMsg(data.reply, 'bot');
}
document.getElementById('form').onsubmit = async (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  addMsg(text, 'user');
  const r = await fetch('/chat', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({message: text, session_id: sessionId})
  });
  const data = await r.json();
  addMsg(data.reply, 'bot');
  (data.approvals_needed || []).forEach(addApproval);
};
addMsg('Hi! I can search the web, do math, remember things about you, set reminders, keep notes, and send email (with your approval). What do you need?', 'bot');
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML
