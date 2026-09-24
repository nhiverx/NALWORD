"""
Embedding Generation — llama.cpp Server Integration
====================================================
Generates dense vector embeddings for documents and queries using a locally
hosted llama.cpp server. No external API dependency; fully air-gappable.

Architecture:
  - llama.cpp runs as an HTTP server (llama-server --embedding)
  - We POST text to /embedding and receive float vectors
  - Vectors are stored in FAISS (see faiss_index.py)

Setup:
    # Download a GGUF embedding model (e.g., nomic-embed-text-v1.5.Q4_K_M.gguf)
    # Start: llama-server --model ./model.gguf --embedding --port 8080

Prerequisites:
    pip install httpx numpy
"""

from __future__ import annotations

import logging
import time
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


def _get_httpx():
    try:
        import httpx
        return httpx
    except ImportError:
        raise RuntimeError("httpx is required. Run: pip install httpx")


class EmbeddingClient:
    """
    Client for llama.cpp /embedding endpoint.

    Generates L2-normalized float32 embeddings suitable for FAISS cosine search.
    """

    def __init__(self, settings):
        self.server_url = settings.llama_cpp_server_url.rstrip("/")
        self.batch_size  = settings.embedding_batch_size
        self.dim         = settings.embedding_dim

    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single string. Returns shape (dim,) float32 array."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of strings in batches.

        Returns:
            np.ndarray of shape (len(texts), dim), dtype float32, L2-normalized.
        """
        httpx = _get_httpx()
        all_vectors = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            t0 = time.perf_counter()

            try:
                with httpx.Client(timeout=30.0) as client:
                    resp = client.post(
                        f"{self.server_url}/embedding",
                        json={"content": batch if len(batch) > 1 else batch[0]},
                    )
                    resp.raise_for_status()
                    data = resp.json()

            except httpx.ConnectError:
                raise RuntimeError(
                    f"Cannot connect to llama.cpp server at {self.server_url}. "
                    "Ensure llama-server is running with --embedding flag."
                )
            except Exception as exc:
                raise RuntimeError(f"Embedding request failed: {exc}") from exc

            # llama.cpp returns {"embedding": [...]} for single or list of lists for batch
            raw = data.get("embedding") or data.get("embeddings")
            if raw is None:
                raise ValueError(f"Unexpected response from llama.cpp: {data}")

            # Normalize to list-of-lists
            if isinstance(raw[0], (int, float)):
                raw = [raw]

            vectors = np.array(raw, dtype=np.float32)
            vectors = self._l2_normalize(vectors)
            all_vectors.append(vectors)

            elapsed = time.perf_counter() - t0
            logger.debug("Embedded %d texts in %.2fs", len(batch), elapsed)

        return np.vstack(all_vectors)

    @staticmethod
    def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
        """Normalize rows to unit length for cosine similarity in FAISS."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)   # avoid division by zero
        return vectors / norms

    def health_check(self) -> bool:
        """Return True if the llama.cpp server is reachable."""
        httpx = _get_httpx()
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(f"{self.server_url}/health")
                return resp.status_code == 200
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Text chunking utility — used during KB ingestion
# ---------------------------------------------------------------------------

def chunk_document(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
) -> List[str]:
    """
    Split a document into overlapping chunks suitable for embedding.

    Strategy: split on sentence boundaries where possible; fall back to
    character-level splitting for very long sentences.

    Args:
        text:       Full document text.
        chunk_size: Target chunk size in characters.
        overlap:    Character overlap between adjacent chunks.

    Returns:
        List of text chunks.
    """
    import re

    # Collapse excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    # Split on sentence-ending punctuation followed by whitespace
    sentence_pattern = re.compile(r"(?<=[.!?])\s+")
    sentences = sentence_pattern.split(text)

    chunks: List[str] = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) <= chunk_size:
            current = (current + " " + sentence).strip()
        else:
            if current:
                chunks.append(current)
            # If the sentence itself exceeds chunk_size, hard-split it
            if len(sentence) > chunk_size:
                for start in range(0, len(sentence), chunk_size - overlap):
                    chunks.append(sentence[start : start + chunk_size])
                current = ""
            else:
                # Start new chunk with overlap from previous
                overlap_text = current[-overlap:] if overlap and current else ""
                current = (overlap_text + " " + sentence).strip()

    if current:
        chunks.append(current)

    return [c for c in chunks if c.strip()]
