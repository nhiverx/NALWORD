"""
Mock knowledge base for the HITL document pipeline demo.

In production this is replaced by a FAISS vector index populated from an
organization's own reference documents. Here we use a deterministic in-memory
store with lightweight keyword overlap scoring so the demo is fully offline.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List

from app.models import Source

# ---------------------------------------------------------------------------
# Load entries from JSON file on disk
# ---------------------------------------------------------------------------

_KB_PATH = Path(__file__).parent.parent / "data" / "knowledge_base.json"


def _load_entries() -> list[dict]:
    try:
        with open(_KB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


_ENTRIES: list[dict] = _load_entries()


# ---------------------------------------------------------------------------
# Lightweight similarity scoring (keyword overlap)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z]{3,}", text.lower())


def _score(query_tokens: list[str], entry: dict) -> float:
    """
    Simple overlap between query tokens and entry searchable fields.
    Returns a float in [0, 1].
    """
    corpus_text = " ".join([
        entry.get("title") or "",
        entry.get("snippet") or "",
        entry.get("full_text") or "",
        entry.get("category") or "",
        " ".join(t for t in entry.get("tags", []) if t),
    ])
    corpus_tokens = set(_tokenize(corpus_text))
    if not corpus_tokens:
        return 0.0

    query_set = set(query_tokens)
    overlap = len(query_set & corpus_tokens)
    # Jaccard-ish: overlap / union, capped at 1
    score = overlap / max(len(query_set | corpus_tokens), 1)
    return min(score * 5.0, 1.0)   # scale up for display readability


def retrieve(query: str, top_k: int = 4, source_type: str | None = None) -> List[Source]:
    """
    Return the top-k most relevant knowledge-base entries for a given query.

    Args:
        query:       Free-text query (e.g. extracted input document text).
        top_k:       Number of results to return.
        source_type: Optional filter ("case_study", "reference", etc.)

    Returns:
        List of Source objects sorted by relevance descending.
    """
    entries = _ENTRIES
    if source_type:
        entries = [e for e in entries if e.get("source_type") == source_type]

    query_tokens = _tokenize(query)
    scored = [(e, _score(query_tokens, e)) for e in entries]
    scored.sort(key=lambda x: x[1], reverse=True)

    results: List[Source] = []
    for entry, score in scored[:top_k]:
        results.append(Source(
            id=entry["id"],
            title=entry["title"],
            snippet=entry["snippet"],
            relevance_score=round(score, 3),
            source_type=entry.get("source_type", "unknown"),
            category=entry.get("category"),
            value=entry.get("value"),
            status=entry.get("status"),
        ))

    return results


def get_all_entries() -> list[dict]:
    """Return raw entries (for admin/debug views)."""
    return _ENTRIES
