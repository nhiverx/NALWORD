"""
Feedback Capture — Rejection & Edit Feedback Loop
===================================================
Captures structured feedback from HITL review decisions and routes it to:
  1. The audit trail (always)
  2. A feedback store for system improvement analytics
  3. Future fine-tuning / RLHF pipeline (production roadmap)

Feedback is the mechanism by which the system improves over time:
  - REJECT decisions with reasons → improve generation prompts
  - EDIT decisions with diffs → identify systematic output weaknesses
  - ACCEPT patterns → reinforce what's working

This module does NOT apply feedback automatically — all improvement
loops require human review and deliberate action. No autoML.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class FeedbackRecord:
    """Structured feedback from one HITL review decision."""
    feedback_id: str
    session_id: str
    step: str
    decision: str               # "reject" | "edit" | "accept" | "alternative"
    feedback_text: Optional[str]
    original_content: str
    edited_content: Optional[str]
    edit_diff_summary: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def has_edits(self) -> bool:
        return bool(self.edited_content and self.edited_content != self.original_content)

    @property
    def has_feedback_text(self) -> bool:
        return bool(self.feedback_text and len(self.feedback_text.strip()) > 5)


class FeedbackStore:
    """
    Collects and categorizes reviewer feedback for system improvement.

    In production this writes to PostgreSQL and feeds a dashboard showing:
      - Most common rejection reasons by step
      - Edit rates per step (acceptance rate target: 70%+)
      - Feedback themes (clustered by keyword)
    """

    def __init__(self, store_path: Optional[str] = None):
        self._records: List[FeedbackRecord] = []
        self._counter = 0
        self._store_path = Path(store_path) if store_path else None
        if self._store_path:
            self._store_path.parent.mkdir(parents=True, exist_ok=True)

    def capture(
        self,
        session_id: str,
        step: str,
        decision: str,
        original_content: str,
        edited_content: Optional[str] = None,
        feedback_text: Optional[str] = None,
    ) -> FeedbackRecord:
        """
        Capture one feedback record.

        Args:
            session_id:       Active session.
            step:             Workflow step (outline/highlights/draft/validation).
            decision:         Human decision.
            original_content: AI-generated content before review.
            edited_content:   Post-edit content (if decision == "edit").
            feedback_text:    Free-text rejection reason or notes.

        Returns:
            FeedbackRecord
        """
        self._counter += 1
        feedback_id = f"fb-{session_id}-{step}-{self._counter:04d}"

        diff_summary = None
        if edited_content and edited_content != original_content:
            diff_summary = self._summarize_diff(original_content, edited_content)

        tags = self._auto_tag(step, decision, feedback_text)

        record = FeedbackRecord(
            feedback_id=feedback_id,
            session_id=session_id,
            step=step,
            decision=decision,
            feedback_text=feedback_text,
            original_content=original_content[:500],
            edited_content=edited_content[:500] if edited_content else None,
            edit_diff_summary=diff_summary,
            tags=tags,
        )
        self._records.append(record)
        self._persist(record)

        logger.info(
            "Feedback captured: id=%s  step=%s  decision=%s  tags=%s",
            feedback_id, step, decision, tags,
        )
        return record

    def get_metrics(self, step: Optional[str] = None) -> Dict:
        """
        Return acceptance/edit/reject rates and common feedback themes.

        These metrics map to the project's success targets:
          - Target: 70%+ sections accepted with <20% edits
          - Target: 0 unsourced factual claims (rejection reason tracking)
        """
        records = self._records
        if step:
            records = [r for r in records if r.step == step]

        total = len(records)
        if total == 0:
            return {"total": 0}

        accept_count    = sum(1 for r in records if r.decision == "accept")
        edit_count      = sum(1 for r in records if r.decision == "edit")
        reject_count    = sum(1 for r in records if r.decision == "reject")
        alt_count       = sum(1 for r in records if r.decision == "alternative")
        feedback_count  = sum(1 for r in records if r.has_feedback_text)

        return {
            "total":           total,
            "accept_rate":     round(accept_count / total, 3),
            "edit_rate":       round(edit_count / total, 3),
            "reject_rate":     round(reject_count / total, 3),
            "alt_rate":        round(alt_count / total, 3),
            "feedback_rate":   round(feedback_count / total, 3),
            "acceptance_with_edits_rate": round((accept_count + edit_count) / total, 3),
            "target_met_70pct": (accept_count + edit_count) / total >= 0.70,
            "top_tags":        self._top_tags(records),
        }

    @staticmethod
    def _summarize_diff(original: str, edited: str) -> str:
        """Produce a short diff summary for logging."""
        orig_words  = set(original.lower().split())
        edit_words  = set(edited.lower().split())
        added       = edit_words - orig_words
        removed     = orig_words - edit_words
        return f"+{len(added)} words, -{len(removed)} words"

    @staticmethod
    def _auto_tag(step: str, decision: str, feedback_text: Optional[str]) -> List[str]:
        """
        Auto-assign tags based on feedback content.
        Used for clustering common rejection themes in analytics.
        """
        tags = [f"step:{step}", f"decision:{decision}"]
        if not feedback_text:
            return tags

        text = feedback_text.lower()
        keyword_tags = {
            "hallucination": ["hallucin", "made up", "wrong fact", "incorrect", "not true"],
            "missing_content": ["missing", "forgot", "need to add", "not included"],
            "tone": ["tone", "voice", "too casual", "too formal", "unprofessional"],
            "requirements": ["requirement", "instruction", "mandatory", "criteria"],
            "format": ["format", "font", "margin", "page", "layout"],
            "sources": ["source", "citation", "no evidence", "unsupported"],
        }
        for tag, keywords in keyword_tags.items():
            if any(k in text for k in keywords):
                tags.append(f"issue:{tag}")

        return tags

    @staticmethod
    def _top_tags(records: List[FeedbackRecord], top_n: int = 5) -> List[str]:
        from collections import Counter
        all_tags = [t for r in records for t in r.tags]
        issue_tags = [t for t in all_tags if t.startswith("issue:")]
        return [tag for tag, _ in Counter(issue_tags).most_common(top_n)]

    def _persist(self, record: FeedbackRecord):
        if not self._store_path:
            return
        try:
            row = {
                "feedback_id":      record.feedback_id,
                "session_id":       record.session_id,
                "step":             record.step,
                "decision":         record.decision,
                "feedback_text":    record.feedback_text,
                "diff_summary":     record.edit_diff_summary,
                "tags":             record.tags,
                "timestamp":        record.timestamp,
            }
            with open(self._store_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.error("Failed to persist feedback: %s", exc)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
import os
_feedback_store: Optional[FeedbackStore] = None


def get_feedback_store() -> FeedbackStore:
    global _feedback_store
    if _feedback_store is None:
        store_path = os.getenv("FEEDBACK_STORE_PATH", "logs/feedback.jsonl")
        _feedback_store = FeedbackStore(store_path=store_path)
    return _feedback_store
