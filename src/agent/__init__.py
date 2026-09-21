"""Personal AI agent package.

Shared setup: loads .env, resolves the data directory used by
memory (ChromaDB) and tools (SQLite).
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# <repo>/src/agent/__init__.py -> parents[2] == <repo>
ROOT_DIR = Path(__file__).resolve().parents[2]

# All runtime data (vector store, sqlite db, logs) lives here.
DATA_DIR = Path(os.getenv("AGENT_DATA_DIR") or (ROOT_DIR / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

__all__ = ["ROOT_DIR", "DATA_DIR"]
