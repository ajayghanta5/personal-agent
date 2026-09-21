"""Eval runner: execute prompts through run_turn, save transcripts.

Usage:
    python evals/run.py                  # run all prompts, save transcript
    python evals/run.py --judge          # also score each reply with the LLM
    python evals/run.py --filter memory  # only run one category

Transcripts go to evals/results/results-<timestamp>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from agent.graph import run_turn
from agent.llm import get_llm, llm_configured

JUDGE_PROMPT = """You are grading an AI agent's reply.

Prompt given to the agent: {prompt}
Expectation: {expectation}
Agent's reply: {reply}
Approval requests returned: {approvals}

Score 1-5: 5 = fully meets the expectation, 1 = completely misses it.
Reply with just the number, then a one-line reason."""


def judge(prompt: str, expectation: str, reply: str, approvals: list) -> str:
    llm = get_llm()
    return llm.invoke(JUDGE_PROMPT.format(
        prompt=prompt, expectation=expectation, reply=reply,
        approvals=json.dumps(approvals),
    )).content.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", action="store_true",
                        help="Score each reply with the LLM against its expectation.")
    parser.add_argument("--filter", default=None,
                        help="Only run prompts in this category (e.g. memory).")
    args = parser.parse_args()

    prompts = json.loads((REPO / "evals" / "prompts.json").read_text())
    if args.filter:
        prompts = [p for p in prompts if p["category"] == args.filter]
    if args.judge and not llm_configured():
        sys.exit("Cannot judge: no LLM configured. Set OPENAI_API_KEY or OPENAI_BASE_URL.")

    results = []
    for p in prompts:
        session = p.get("session", f"eval-{p['id']}")
        print(f"[{p['id']}] {p['prompt'][:60]}...")
        try:
            out = run_turn(p["prompt"], session_id=session)
        except Exception as exc:  # noqa: BLE001 - record failures, don't stop the run
            out = {"reply": f"EVAL ERROR: {exc}", "approvals_needed": []}
        record = {
            "id": p["id"], "category": p["category"],
            "prompt": p["prompt"], "expectation": p["expectation"],
            "reply": out["reply"], "approvals_needed": out["approvals_needed"],
        }
        if args.judge:
            try:
                record["judge"] = judge(p["prompt"], p["expectation"],
                                        out["reply"], out["approvals_needed"])
            except Exception as exc:  # noqa: BLE001
                record["judge"] = f"JUDGE ERROR: {exc}"
            print(f"    judge: {record['judge'][:80]}")
        results.append(record)

    out_dir = REPO / "evals" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"results-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nRan {len(results)} prompts. Transcript saved to {path}")


if __name__ == "__main__":
    main()
