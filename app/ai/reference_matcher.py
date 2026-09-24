"""
Reference Matcher — Case Study to Requirement Matching
======================================================
Retrieves and scores knowledge-base case studies against the requirements of
an input document using semantic similarity + explicit relevance criteria.

Relevance criteria:
  - Scope similarity (similar type of work)
  - Recency (within the configured freshness window — enforced by ContentValidationGate)
  - Scale (comparable size and complexity)
  - Category (evaluated separately by the reviewer)

This module generates the mapping used by both the highlights generator and
the case study section writer.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MatchedReference:
    """A case study record matched to specific input requirements."""
    document_id: str
    title: str
    category: str
    value: Optional[str]
    status: Optional[str]
    relevance_score: float
    matched_requirements: List[str] = field(default_factory=list)
    rating_highlights: List[str] = field(default_factory=list)
    differentiator_notes: str = ""
    suitable_for_citation: bool = True
    citation_order: int = 0


class ReferenceMatcher:
    """
    Matches knowledge-base case studies to input document requirements.

    Produces a ranked, citation-ready list of references for the case study section.
    """

    def __init__(self, rag_pipeline=None):
        """
        Args:
            rag_pipeline: RAGPipeline instance.
                          Falls back to keyword KB search if None.
        """
        self.pipeline = rag_pipeline

    def match(
        self,
        input_text: str,
        extracted_requirements: list,
        max_citations: int = 3,
    ) -> List[MatchedReference]:
        """
        Find the best case study citations for the given input document.

        Args:
            input_text:              Full input document text.
            extracted_requirements:  List of RequirementItem objects.
            max_citations:           Max citations to include.

        Returns:
            Sorted list of MatchedReference, best-first.
        """
        chunks = self._retrieve_case_studies(input_text)

        # Group chunks by source document
        doc_map: dict[str, dict] = {}
        for chunk in chunks:
            did = chunk.document_id
            if did not in doc_map:
                doc_map[did] = {
                    "document_id": did,
                    "title": chunk.title,
                    "category": chunk.category or "General",
                    "value": chunk.value,
                    "status": chunk.status,
                    "scores": [],
                    "texts": [],
                }
            doc_map[did]["scores"].append(chunk.relevance_score)
            doc_map[did]["texts"].append(chunk.text)

        matched: List[MatchedReference] = []
        for did, info in doc_map.items():
            avg_score = sum(info["scores"]) / len(info["scores"])
            matched.append(MatchedReference(
                document_id=did,
                title=info["title"],
                category=info["category"],
                value=info["value"],
                status=info["status"],
                relevance_score=round(avg_score, 3),
                matched_requirements=self._match_requirements(
                    info["texts"], extracted_requirements
                ),
                rating_highlights=self._extract_rating_language(info["texts"]),
                differentiator_notes=self._identify_differentiators(info["texts"]),
                suitable_for_citation=self._is_citable(info),
            ))

        # Sort by relevance, then completed work first
        matched.sort(
            key=lambda c: (c.relevance_score, 1 if c.status == "completed" else 0),
            reverse=True,
        )

        citable = [c for c in matched if c.suitable_for_citation]
        for i, ref in enumerate(citable[:max_citations]):
            ref.citation_order = i + 1

        return matched[:max_citations]

    def _retrieve_case_studies(self, input_text: str) -> list:
        """Retrieve case study chunks (FAISS or keyword fallback)."""
        if self.pipeline:
            return self.pipeline._retrieve(input_text + " case study", top_k=10)

        from app.knowledge_base import retrieve
        from app.ai.rag_pipeline import RetrievedChunk
        sources = retrieve(input_text, top_k=10, source_type="case_study")
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

    @staticmethod
    def _match_requirements(texts: list[str], requirements: list) -> List[str]:
        """Return refs of requirements whose key words appear in the case study texts."""
        combined = " ".join(texts).lower()
        matched = []
        for req in requirements:
            key_words = re.findall(r"[a-z]{4,}", req.text.lower())[:5]
            if any(w in combined for w in key_words):
                matched.append(req.ref)
        return matched[:5]

    @staticmethod
    def _extract_rating_language(texts: list[str]) -> List[str]:
        """Extract stakeholder rating mentions from text."""
        highlights = []
        rating_pattern = re.compile(
            r"rated?[^.]*?(Excellent|Very Good|Good|Satisfactory|Poor)",
            re.IGNORECASE,
        )
        for text in texts:
            for match in rating_pattern.finditer(text):
                snippet = text[max(0, match.start() - 30):match.end() + 30].strip()
                highlights.append(snippet)
        return highlights[:3]

    @staticmethod
    def _identify_differentiators(texts: list[str]) -> str:
        """Extract notable differentiator language (on-time, under budget, etc.)."""
        combined = " ".join(texts)
        patterns = [
            (r"on.time|ahead of schedule", "On-time / ahead-of-schedule delivery"),
            (r"under budget|within budget|no budget overrun", "Cost control / no overruns"),
            (r"zero critical|no critical", "Zero critical defects / incidents"),
            (r"first.time acceptance|accepted.on.first", "First-time acceptance of deliverables"),
            (r"certified", "Certified team available"),
        ]
        found = [label for pat, label in patterns if re.search(pat, combined, re.IGNORECASE)]
        return "; ".join(found)

    @staticmethod
    def _is_citable(info: dict) -> bool:
        """
        Basic checks: must have a title and category.
        Recency check handled by ContentValidationGate upstream.
        """
        return bool(info.get("title") and info.get("category"))
