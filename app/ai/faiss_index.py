"""
FAISS Vector Index — Semantic Search over the Knowledge Base
============================================================
Manages a FAISS flat/IVF index for fast similarity search over
embedded knowledge-base documents.

Index lifecycle:
  1. build_index()  — embed all KB documents and build/save the index
  2. FAISSIndex()   — load the saved index for search
  3. search()       — return top-k chunks for a query

Index files saved to settings.faiss_index_path/:
  - index.faiss    : the FAISS binary index
  - metadata.json  : chunk → document metadata mapping

Prerequisites:
    pip install faiss-cpu numpy
    (or faiss-gpu for GPU-accelerated search)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _get_faiss():
    try:
        import faiss
        return faiss
    except ImportError:
        raise RuntimeError(
            "faiss-cpu is required. Run: pip install faiss-cpu"
        )


@dataclass
class SearchResult:
    """A single FAISS search hit with metadata."""
    chunk_id: int
    score: float           # cosine similarity [0, 1]; higher = more relevant
    text: str
    document_id: str
    title: str
    source_type: str
    category: Optional[str]
    value: Optional[str]
    status: Optional[str]


class FAISSIndex:
    """
    Wraps a FAISS index with metadata for semantic search over KB chunks.

    Thread-safe for concurrent reads (FAISS is GIL-safe for search).
    Write operations (add, rebuild) should be serialized externally.
    """

    def __init__(self, index_dir: str):
        self.index_dir   = Path(index_dir)
        self._faiss_idx  = None
        self._metadata: List[dict] = []
        self._loaded     = False

    def load(self) -> "FAISSIndex":
        """Load index from disk. Call once at startup."""
        faiss = _get_faiss()

        index_path = self.index_dir / "index.faiss"
        meta_path  = self.index_dir / "metadata.json"

        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {index_path}. "
                "Run build_index() first or use the demo knowledge_base.py fallback."
            )

        self._faiss_idx = faiss.read_index(str(index_path))
        with open(meta_path, "r", encoding="utf-8") as f:
            self._metadata = json.load(f)

        self._loaded = True
        logger.info(
            "FAISS index loaded: %d vectors, dim=%d, path=%s",
            self._faiss_idx.ntotal, self._faiss_idx.d, index_path,
        )
        return self

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[SearchResult]:
        """
        Return top-k most similar chunks for a query embedding.

        Args:
            query_vector: Shape (dim,) float32, L2-normalized.
            top_k:        Number of results.

        Returns:
            List of SearchResult, sorted by score descending.
        """
        if not self._loaded:
            raise RuntimeError("Index not loaded. Call load() first.")

        faiss = _get_faiss()

        q = query_vector.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(q)  # ensure unit norm for IP search

        scores, indices = self._faiss_idx.search(q, top_k)

        results: List[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._metadata):
                continue
            meta = self._metadata[idx]
            results.append(SearchResult(
                chunk_id=int(idx),
                score=float(score),     # inner product ≈ cosine similarity for normalized vecs
                text=meta.get("text", ""),
                document_id=meta.get("document_id", ""),
                title=meta.get("title", ""),
                source_type=meta.get("source_type", "unknown"),
                category=meta.get("category"),
                value=meta.get("value"),
                status=meta.get("status"),
            ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results

    @property
    def total_vectors(self) -> int:
        return self._faiss_idx.ntotal if self._faiss_idx else 0


# ---------------------------------------------------------------------------
# Index builder — run offline to (re)build from KB documents
# ---------------------------------------------------------------------------

def build_index(
    kb_entries: List[dict],
    embedding_client,
    output_dir: str,
    dim: int = 768,
) -> FAISSIndex:
    """
    Build a FAISS flat inner-product index from KB entries.

    Args:
        kb_entries:        List of dicts from knowledge_base.json.
        embedding_client:  EmbeddingClient instance.
        output_dir:        Directory to save index.faiss and metadata.json.
        dim:               Embedding dimension (must match the model).

    Returns:
        Loaded FAISSIndex ready for search.
    """
    from app.ai.embeddings import chunk_document
    faiss = _get_faiss()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_chunks: List[str] = []
    all_metadata: List[dict] = []

    logger.info("Chunking %d KB entries...", len(kb_entries))
    for entry in kb_entries:
        full_text = entry.get("full_text") or entry.get("snippet", "")
        chunks = chunk_document(full_text)

        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadata.append({
                "text":           chunk,
                "document_id":    entry["id"],
                "title":          entry.get("title", ""),
                "source_type":    entry.get("source_type", "unknown"),
                "category":       entry.get("category"),
                "value":          entry.get("value"),
                "status":         entry.get("status"),
            })

    logger.info("Embedding %d chunks...", len(all_chunks))
    vectors = embedding_client.embed_batch(all_chunks)   # shape (N, dim)

    # Build flat inner-product index (cosine similarity on L2-normalized vecs)
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    faiss.write_index(index, str(output_path / "index.faiss"))
    with open(output_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2, ensure_ascii=False)

    logger.info(
        "FAISS index built: %d vectors written to %s", index.ntotal, output_path
    )

    loaded_idx = FAISSIndex(str(output_path))
    loaded_idx.load()
    return loaded_idx
