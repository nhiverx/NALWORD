"""
HITL Audit Trail
================
Immutable, append-only log of every human review decision.

Entries are written to:
  1. In-memory list (fast access for current session)
  2. JSONL file on disk (durable; survives restarts)
  3. PostgreSQL audit_log table (production only)

Each entry records WHAT the AI produced, WHAT the human decided,
WHY (free-text feedback), and WHO (reviewer_id + timestamp).

This log is the accountability record demonstrating that every AI output
was reviewed and approved by a human before use.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AuditEntry:
    """One immutable audit trail entry."""

    __slots__ = (
        "entry_id", "session_id", "step", "decision",
        "reviewer_id", "timestamp", "generation_attempt",
        "original_content_hash", "final_content_hash",
        "original_content_preview", "final_content_preview",
        "feedback", "sources_used", "metadata",
    )

    def __init__(
        self,
        entry_id: str,
        session_id: str,
        step: str,
        decision: str,
        reviewer_id: str,
        generation_attempt: int,
        original_content: str,
        final_content: str,
        feedback: Optional[str] = None,
        sources_used: Optional[List[dict]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        import hashlib
        self.entry_id              = entry_id
        self.session_id            = session_id
        self.step                  = step
        self.decision              = decision
        self.reviewer_id           = reviewer_id
        self.timestamp             = datetime.now(timezone.utc).isoformat()
        self.generation_attempt    = generation_attempt
        self.original_content_hash = hashlib.sha256(original_content.encode()).hexdigest()[:16]
        self.final_content_hash    = hashlib.sha256(final_content.encode()).hexdigest()[:16]
        self.original_content_preview = original_content[:200]
        self.final_content_preview    = final_content[:200]
        self.feedback              = feedback
        self.sources_used          = sources_used or []
        self.metadata              = metadata or {}

    def to_dict(self) -> dict:
        return {s: getattr(self, s) for s in self.__slots__}


class AuditTrail:
    """
    Append-only audit trail for HITL decisions.

    Backed by:
      - In-memory list (always)
      - JSONL file (if configured)
      - PostgreSQL (production)
    """

    def __init__(self, log_path: Optional[str] = None):
        self._entries: List[AuditEntry] = []
        self._log_path = Path(log_path) if log_path else None
        self._counter  = 0

        if self._log_path:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Audit trail writing to: %s", self._log_path)

    def record(
        self,
        session_id: str,
        step: str,
        decision: str,
        original_content: str,
        final_content: str,
        reviewer_id: str = "human_reviewer",
        generation_attempt: int = 1,
        feedback: Optional[str] = None,
        sources_used: Optional[List[dict]] = None,
    ) -> AuditEntry:
        """
        Record one HITL decision. Returns the AuditEntry.

        This method is the single authoritative record of every
        human decision made in the workflow.
        """
        self._counter += 1
        entry_id = f"audit-{session_id}-{step}-{self._counter:04d}"

        entry = AuditEntry(
            entry_id=entry_id,
            session_id=session_id,
            step=step,
            decision=decision,
            reviewer_id=reviewer_id,
            generation_attempt=generation_attempt,
            original_content=original_content,
            final_content=final_content,
            feedback=feedback,
            sources_used=sources_used,
        )

        self._entries.append(entry)
        self._write_to_file(entry)
        self._write_to_db(entry)

        logger.info(
            "AUDIT: session=%s  step=%s  decision=%s  reviewer=%s  attempt=%d",
            session_id, step, decision, reviewer_id, generation_attempt,
        )
        return entry

    def get_session_history(self, session_id: str) -> List[AuditEntry]:
        """Return all audit entries for a session, oldest first."""
        return [e for e in self._entries if e.session_id == session_id]

    def get_all(self) -> List[AuditEntry]:
        return list(self._entries)

    def export_session(self, session_id: str) -> List[dict]:
        """Export session history as list of dicts (for API response)."""
        return [e.to_dict() for e in self.get_session_history(session_id)]

    def _write_to_file(self, entry: AuditEntry):
        """Append to JSONL file."""
        if not self._log_path:
            return
        try:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.error("Failed to write audit log: %s", exc)

    def _write_to_db(self, entry: AuditEntry):
        """
        Write to PostgreSQL audit_log table.
        No-op in demo (no DB configured); implement with SQLAlchemy in production.
        """
        pass  # TODO: INSERT INTO audit_log VALUES (...)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_audit_trail: Optional[AuditTrail] = None


def get_audit_trail() -> AuditTrail:
    global _audit_trail
    if _audit_trail is None:
        log_path = os.getenv("AUDIT_LOG_PATH", "logs/audit.jsonl")
        _audit_trail = AuditTrail(log_path=log_path)
    return _audit_trail
