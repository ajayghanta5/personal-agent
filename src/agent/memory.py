"""Long-term memory backed by ChromaDB (local persistent store).

Design notes:
- We store concise *extracted facts* ("User prefers evening workouts"),
  not raw conversation turns. Raw turns bloat the index and pollute recall.
- Embeddings use Chroma's default local ONNX model (no API key needed).
  If that model can't load (e.g. no network on first run), we fall back to a
  deterministic hash embedding so the skeleton keeps working offline.
- Facts are namespaced by session_id in metadata but recalled globally,
  so memories persist across sessions.
"""
from __future__ import annotations

import hashlib
import logging
import math
import uuid
from datetime import datetime, timedelta, timezone

from . import DATA_DIR
from .llm import get_llm, llm_configured

logger = logging.getLogger(__name__)

COLLECTION_NAME = "facts"

_client = None
_collection = None


class _HashEmbedding:
    """Dependency-free fallback embedding.

    Deterministic token-hash vector, L2-normalized. Good enough for a
    skeleton and for offline smoke tests; swap in a real embedding model
    for production recall quality.
    """

    def __init__(self, dim: int = 256):
        self._dim = dim

    def __call__(self, input):  # Chroma calls it with `input`
        return [self._embed(text) for text in input]

    # Chroma's query path calls these (not __call__) in recent versions.
    def embed_query(self, input):
        return self.__call__(input)

    def embed_documents(self, input):
        return self.__call__(input)

    def _embed(self, text: str):
        vec = [0.0] * self._dim
        for token in text.lower().split():
            # Hash each token into a few buckets to reduce collisions.
            for salt in range(3):
                h = int(hashlib.md5(f"{salt}:{token}".encode()).hexdigest(), 16)
                vec[h % self._dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def name(self) -> str:  # Chroma uses this for collection config
        return "hash-fallback"


def _embedding_function():
    """Default local embedding model, with offline fallback."""
    try:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        ef = DefaultEmbeddingFunction()
        ef(["connectivity ping"])  # forces model download now, not mid-query
        return ef
    except Exception as exc:  # noqa: BLE001 - any failure -> fallback
        logger.warning("Local embedding model unavailable (%s); using hash fallback.", exc)
        return _HashEmbedding()


def _get_collection():
    global _client, _collection
    if _collection is None:
        import chromadb  # lazy: keeps `import agent.memory` light

        _client = chromadb.PersistentClient(path=str(DATA_DIR / "chroma"))
        _collection = _client.get_or_create_collection(
            COLLECTION_NAME, embedding_function=_embedding_function()
        )
    return _collection


def store_fact(text: str, session_id: str = "default", source: str = "chat") -> str:
    """Store one concise fact. Returns the fact id."""
    fact_id = uuid.uuid4().hex
    _get_collection().add(
        ids=[fact_id],
        documents=[text.strip()],
        metadatas=[{
            "session_id": session_id,
            "source": source,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }],
    )
    return fact_id


def recall(query: str, k: int = 5) -> list[dict]:
    """Return up to k relevant facts as dicts: text, session_id, distance."""
    if not query.strip():
        return []
    try:
        res = _get_collection().query(query_texts=[query], n_results=k)
    except Exception as exc:  # noqa: BLE001 - never break a turn on memory
        logger.warning("Memory recall failed: %s", exc)
        return []
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    out = []
    for doc, meta, dist in zip(docs, metas, dists):
        out.append({
            "text": doc,
            "session_id": (meta or {}).get("session_id", "default"),
            "distance": dist,
        })
    return out


_EXTRACT_PROMPT = """You extract durable facts about the user from a conversation turn.

Rules:
- Output ONE concise fact per line, in third person (e.g. "User prefers evening workouts.").
- Only include facts likely to stay true and be useful later: preferences, names,
  recurring routines, goals, constraints.
- Do NOT include one-off task details, and do NOT invent facts that aren't stated.
- If there is nothing worth remembering, output exactly: NONE

Conversation:
User: {user}
Assistant: {assistant}

Facts:"""


def extract_facts(user_message: str, assistant_reply: str = "") -> list[str]:
    """Use the LLM to distill a turn into concise facts.

    Returns [] when no LLM is configured (or on any error) — memory
    extraction is best-effort and must never break a conversation turn.
    """
    if not llm_configured():
        return []
    try:
        llm = get_llm()
        raw = llm.invoke(
            _EXTRACT_PROMPT.format(user=user_message, assistant=assistant_reply)
        ).content.strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Fact extraction failed: %s", exc)
        return []
    if raw.upper().startswith("NONE"):
        return []
    return [line.strip(" -•\t") for line in raw.splitlines() if line.strip()]


def prune_stale(max_age_days: int = 180) -> int:
    """Delete facts older than max_age_days. Returns the number removed.

    Strategy (skeleton):
    - Time-based expiry is the cheap, predictable baseline: preferences do go
      stale, and a wrong old fact is worse than no fact.
    - What this does NOT do yet (future work): semantic dedup ("User likes
      tea" vs "User's favourite drink is tea"), contradiction detection
      ("User moved from Hyderabad to Bengaluru" should supersede the old
      city), and pinning facts the user marks as permanent.
    - A production version would run this on a schedule and log what it
      removed for the user to review.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    col = _get_collection()
    try:
        data = col.get(include=["metadatas"])
    except Exception as exc:  # noqa: BLE001
        logger.warning("Prune failed: %s", exc)
        return 0
    stale_ids = [
        _id for _id, meta in zip(data.get("ids", []), data.get("metadatas", []))
        if (meta or {}).get("created_at", "") < cutoff.isoformat()
    ]
    if stale_ids:
        col.delete(ids=stale_ids)
    return len(stale_ids)
