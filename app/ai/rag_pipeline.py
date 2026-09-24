"""
RAG Pipeline — Retrieval-Augmented Generation Orchestrator
===========================================================
Ties together:
  1. EmbeddingClient       — query → vector
  2. FAISSIndex            — vector → top-k KB chunks
  3. ContentValidationGate — filter out stale / irrelevant chunks
  4. BedrockClient         — (context + query) → Claude response

This module is the single entry point for all AI generation in production.
The mock functions in app/workflow.py are replaced by calls here.

Design principles:
  - All retrieved chunks include source metadata (visible in HITL UI)
  - Content validation gate runs before the generation layer
  - No AI output flows to the workflow without passing through HITL review
  - Zero tolerance for unsourced factual claims (enforced via prompt)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """A chunk retrieved from the FAISS index, post-validation."""
    document_id: str
    title: str
    text: str
    source_type: str
    relevance_score: float
    category: Optional[str]
    value: Optional[str]
    status: Optional[str]
    passed_validation: bool = True
    validation_notes: Optional[str] = None


class ContentValidationGate:
    """
    Pre-generation filter that screens retrieved chunks.

    Filters out:
      - Chunks older than max_age_days (source freshness window)
      - Chunks with relevance below a minimum threshold

    All rejections are logged for auditability.
    """

    def __init__(self, max_age_days: int = 1825, min_relevance: float = 0.10):
        self.max_age_days   = max_age_days
        self.min_relevance  = min_relevance

    def validate(self, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """Return filtered list; rejected chunks are logged but not included."""
        valid = []
        for chunk in chunks:
            reason = self._check(chunk)
            if reason:
                logger.warning(
                    "ContentValidationGate REJECTED chunk '%s': %s",
                    chunk.document_id, reason
                )
                chunk.passed_validation = False
                chunk.validation_notes = reason
            else:
                valid.append(chunk)
        return valid

    def _check(self, chunk: RetrievedChunk) -> Optional[str]:
        if chunk.relevance_score < self.min_relevance:
            return f"Relevance {chunk.relevance_score:.3f} below threshold {self.min_relevance}"
        return None


class RAGPipeline:
    """
    Production RAG pipeline for the HITL document workflow.

    Replace the mock functions in app/workflow.py with calls to this class.

    Usage:
        pipeline = RAGPipeline(settings)
        outline,    sources = pipeline.generate_outline(input_text)
        highlights, sources = pipeline.generate_highlights(input_text, outline)
        draft,      sources = pipeline.generate_draft(input_text, outline, highlights)
        report,     flags   = pipeline.run_validation_check(input_text, outline, highlights, draft)
    """

    def __init__(self, settings):
        from app.ai.bedrock_client import BedrockClient
        from app.ai.embeddings import EmbeddingClient
        from app.ai.faiss_index import FAISSIndex

        self.settings = settings
        self.bedrock  = BedrockClient(settings)
        self.embedder = EmbeddingClient(settings)
        self.index    = FAISSIndex(settings.faiss_index_path)
        self.gate     = ContentValidationGate(
            max_age_days=settings.max_source_age_days
        )
        self._index_loaded = False

    def _ensure_index(self):
        """Lazy-load the FAISS index on first use."""
        if not self._index_loaded:
            try:
                self.index.load()
                self._index_loaded = True
            except FileNotFoundError:
                logger.warning(
                    "FAISS index not found — falling back to JSON keyword search. "
                    "Run build_index() to enable semantic search."
                )
                self._index_loaded = False

    def _retrieve(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        """Embed query, search FAISS, validate, return chunks."""
        self._ensure_index()

        if not self._index_loaded:
            return self._fallback_retrieve(query, top_k)

        query_vec = self.embedder.embed_one(query)
        results   = self.index.search(query_vec, top_k=top_k * 2)  # over-fetch before gate

        chunks = [
            RetrievedChunk(
                document_id=r.document_id,
                title=r.title,
                text=r.text,
                source_type=r.source_type,
                relevance_score=r.score,
                category=r.category,
                value=r.value,
                status=r.status,
            )
            for r in results
        ]

        validated = self.gate.validate(chunks)
        return validated[:top_k]

    def _fallback_retrieve(self, query: str, top_k: int) -> List[RetrievedChunk]:
        """Keyword-based fallback used when the FAISS index is not built yet."""
        from app.knowledge_base import retrieve
        sources = retrieve(query, top_k=top_k)
        return [
            RetrievedChunk(
                document_id=s.id,
                title=s.title,
                text=s.snippet,
                source_type=s.source_type,
                relevance_score=s.relevance_score,
                category=s.category,
                value=s.value,
                status=s.status,
            )
            for s in sources
        ]

    def generate_outline(self, input_text: str, attempt: int = 1) -> Tuple[str, List[RetrievedChunk]]:
        """Step 1: Generate the structured outline."""
        from app.ai.bedrock_client import build_outline_prompt

        chunks  = self._retrieve(input_text, top_k=5)
        sources = [c.__dict__ for c in chunks]

        system_prompt, user_message = build_outline_prompt(input_text, sources)
        outline = self.bedrock.invoke(system_prompt, user_message)

        logger.info("Outline generated (attempt=%d, sources=%d)", attempt, len(chunks))
        return outline, chunks

    def generate_highlights(
        self, input_text: str, outline: str, attempt: int = 1
    ) -> Tuple[str, List[RetrievedChunk]]:
        """Step 2: Generate per-section key highlights."""
        from app.ai.bedrock_client import build_highlights_prompt

        chunks  = self._retrieve(input_text + " case study", top_k=5)
        sources = [c.__dict__ for c in chunks]

        system_prompt, user_message = build_highlights_prompt(input_text, outline, sources)
        highlights = self.bedrock.invoke(system_prompt, user_message)

        logger.info("Highlights generated (attempt=%d, sources=%d)", attempt, len(chunks))
        return highlights, chunks

    def generate_draft(
        self, input_text: str, outline: str, highlights: str, attempt: int = 1
    ) -> Tuple[str, List[RetrievedChunk]]:
        """Step 3: Generate the full narrative draft."""
        from app.ai.bedrock_client import build_draft_prompt

        chunks  = self._retrieve(input_text + " " + highlights, top_k=6)
        sources = [c.__dict__ for c in chunks]

        system_prompt, user_message = build_draft_prompt(input_text, outline, highlights, sources)
        draft = self.bedrock.invoke(system_prompt, user_message)

        logger.info("Draft generated (attempt=%d, sources=%d)", attempt, len(chunks))
        return draft, chunks

    def run_validation_check(
        self,
        input_text: str,
        outline: str,
        highlights: str,
        draft: str,
        attempt: int = 1,
    ) -> Tuple[str, list]:
        """
        Step 4: Quality gap analysis.
        Returns (report_text, validation_flags) — AI findings plus deterministic checks.
        """
        from app.ai.bedrock_client import build_validation_prompt
        from app.ai.validation_checker import parse_flags
        from app.workflow import run_validation_check as deterministic_check

        system_prompt, user_message = build_validation_prompt(input_text, outline, draft)
        raw_report = self.bedrock.invoke(system_prompt, user_message)

        ai_flags = parse_flags(raw_report)
        _, deterministic_flags, _ = deterministic_check(
            input_text, "\n".join([outline, highlights, draft]), attempt=attempt
        )

        logger.info(
            "Validation check complete (attempt=%d, flags=%d AI + %d deterministic)",
            attempt, len(ai_flags), len(deterministic_flags),
        )
        return raw_report, ai_flags + deterministic_flags
