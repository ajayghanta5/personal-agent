"""LLM configuration.

Reads OPENAI_API_KEY / OPENAI_BASE_URL / MODEL_NAME from the environment
(see .env.example). OPENAI_BASE_URL lets you point at any OpenAI-compatible
endpoint, e.g. Ollama at http://localhost:11434/v1.

If neither a key nor a base URL is set, the agent cannot reason — but the
rest of the system (tools, memory, API) still works, and run_turn() returns
a friendly message instead of crashing.
"""
from __future__ import annotations

import os
from functools import lru_cache

LLM_MISSING_MESSAGE = (
    "I don't have a language model configured yet, so I can't chat properly. "
    "Set OPENAI_API_KEY in your .env file (or OPENAI_BASE_URL to use a local "
    "model like Ollama, e.g. http://localhost:11434/v1), then restart the server. "
    "See .env.example for details."
)


def llm_configured() -> bool:
    """True when there is enough config to construct a chat model."""
    return bool(os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_BASE_URL"))


@lru_cache(maxsize=1)
def get_llm():
    """Build (once) the chat model used by the agent.

    Raises:
        RuntimeError: with a friendly setup message when no LLM is configured.
    """
    if not llm_configured():
        raise RuntimeError(LLM_MISSING_MESSAGE)

    from langchain_openai import ChatOpenAI

    # Local endpoints (Ollama etc.) usually accept any non-empty key.
    api_key = os.getenv("OPENAI_API_KEY") or "ollama"
    return ChatOpenAI(
        model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        api_key=api_key,
        base_url=os.getenv("OPENAI_BASE_URL") or None,
        temperature=0.7,
    )
