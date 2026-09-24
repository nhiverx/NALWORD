"""
Document Parser — AI-Powered Requirement Extraction
===================================================
Extracts structured requirements from input documents using a combination of:
  1. Regex heuristics (fast, deterministic — catches numbered section refs)
  2. Claude via Bedrock (catches semantic requirements not in standard format)

The deterministic pass runs first; Claude supplements with anything missed.
All extracted items are deduplicated by ref + text similarity.

Production input: PDF/Word extracted text.
Demo input: pasted plain text.
"""

from __future__ import annotations

import json
import logging
import re
from typing import List

from app.models import RequirementItem
from app.workflow import extract_requirements as _extract_heuristic

logger = logging.getLogger(__name__)


class DocumentParser:
    """
    Two-pass requirement extractor.

    Pass 1: Heuristic regex (always runs, offline-safe)
    Pass 2: Claude via Bedrock (optional, requires AWS credentials)
    """

    def __init__(self, bedrock_client=None):
        """
        Args:
            bedrock_client: BedrockClient instance (optional).
                            If None, only heuristic extraction runs.
        """
        self.bedrock = bedrock_client

    def parse(self, input_text: str, use_ai: bool = True) -> List[RequirementItem]:
        """
        Extract all requirements from the input document.

        Args:
            input_text: Full document text.
            use_ai:     If True and bedrock_client is configured, augment with AI pass.

        Returns:
            Deduplicated list of RequirementItems.
        """
        heuristic_items = _extract_heuristic(input_text)
        logger.info("Heuristic extraction: %d items", len(heuristic_items))

        if not use_ai or self.bedrock is None:
            return heuristic_items

        try:
            ai_items = self._ai_extract(input_text, heuristic_items)
            logger.info("AI extraction added %d items", len(ai_items))
            combined = self._deduplicate(heuristic_items + ai_items)
            logger.info("Combined (deduped): %d requirements", len(combined))
            return combined
        except Exception as exc:
            logger.warning("AI extraction failed, using heuristic only: %s", exc)
            return heuristic_items

    def _ai_extract(
        self, input_text: str, existing: List[RequirementItem]
    ) -> List[RequirementItem]:
        """Use Claude to find requirements missed by the heuristic pass."""
        existing_refs = {r.ref for r in existing}
        existing_sample = "\n".join(f"  {r.ref}: {r.text[:60]}" for r in existing[:8])

        system = (
            "You are a requirements analyst. Extract actionable requirements "
            "from the input document. Output ONLY a JSON array — no markdown, no commentary."
        )
        user = f"""
INPUT DOCUMENT:
{input_text[:5000]}

ALREADY FOUND (do not duplicate):
{existing_sample}

TASK:
Find any additional requirements NOT in the above list.
Focus on: mandatory instructions, evaluation criteria, formatting rules,
certification requirements, and deadlines.

Output format — JSON array only:
[
  {{
    "section": "Instructions",
    "ref": "I.3.1",
    "text": "Requirement text (max 200 chars)",
    "type": "mandatory",
    "mandatory": true
  }}
]

Return empty array [] if nothing new found.
"""
        raw = self.bedrock.invoke(system, user, max_tokens=1024, temperature=0.0)

        try:
            raw = re.sub(r"```(?:json)?", "", raw).strip()
            data = json.loads(raw)
            items = []
            for d in data:
                if d.get("ref") in existing_refs:
                    continue
                items.append(RequirementItem(
                    section=d.get("section", "General"),
                    ref=d.get("ref", "—"),
                    text=d.get("text", "")[:300],
                    type=d.get("type", "instruction"),
                    mandatory=bool(d.get("mandatory", False)),
                ))
            return items
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("Could not parse AI extraction JSON: %s", exc)
            return []

    @staticmethod
    def _deduplicate(items: List[RequirementItem]) -> List[RequirementItem]:
        """Remove ref + text-prefix duplicates; keep first occurrence."""
        seen = set()
        unique = []
        for item in items:
            key = f"{item.ref}::{item.text[:40]}"
            if key not in seen:
                seen.add(key)
                unique.append(item)
        return unique

    @staticmethod
    def extract_evaluation_criteria(input_text: str) -> List[dict]:
        """
        Extract evaluation criteria with weights/importance.
        Returns list of dicts: {ref, criterion, importance, description}.
        """
        criteria = []
        # Pattern: "E.2.1 APPROACH (Most Important)"
        pattern = re.compile(
            r"E\.(\d+(?:\.\d+)?)\s+([A-Z /&]+)\s*(?:\(([^)]+)\))?",
            re.IGNORECASE,
        )
        for match in pattern.finditer(input_text):
            ref, name, qualifier = match.groups()
            criteria.append({
                "ref": f"E.{ref}",
                "criterion": name.strip().title(),
                "importance": qualifier or "Standard",
                "description": "",
            })

        return criteria
