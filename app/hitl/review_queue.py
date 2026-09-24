"""
HITL Review Queue
=================
Manages the ordered queue of AI outputs awaiting human review.

In production this is backed by PostgreSQL so multiple reviewers can
claim and process items. In the demo it wraps the in-memory session dict.

Design:
  - Each queue item holds step, session_id, generated content, and sources
  - Reviewers claim items (lock), then submit decisions
  - Expired locks are automatically released (heartbeat pattern)
  - Priority ordering: items with CRITICAL validation flags surface first

All decisions are forwarded to AuditTrail for permanent storage.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class QueueItemStatus(str, Enum):
    PENDING  = "pending"
    CLAIMED  = "claimed"
    RESOLVED = "resolved"
    EXPIRED  = "expired"


@dataclass
class ReviewQueueItem:
    """One item in the HITL review queue."""
    item_id: str
    session_id: str
    step: str
    content: str
    sources: List[dict]
    generation_attempt: int
    priority: int = 0           # higher = more urgent; validation flags bump priority
    status: QueueItemStatus = QueueItemStatus.PENDING
    claimed_by: Optional[str] = None
    claimed_at: Optional[datetime] = None
    lock_timeout_seconds: int = 1800  # 30-minute claim timeout
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved_at: Optional[datetime] = None

    @property
    def is_expired(self) -> bool:
        if self.status != QueueItemStatus.CLAIMED or not self.claimed_at:
            return False
        elapsed = datetime.now(timezone.utc) - self.claimed_at
        return elapsed.total_seconds() > self.lock_timeout_seconds


class ReviewQueue:
    """
    In-memory HITL review queue.

    Thread-safe for single-process demo use.
    For production: replace _items with DB-backed implementation.
    """

    def __init__(self):
        self._items: Dict[str, ReviewQueueItem] = {}
        self._counter = 0

    def enqueue(
        self,
        session_id: str,
        step: str,
        content: str,
        sources: List[dict],
        attempt: int = 1,
        priority: int = 0,
    ) -> ReviewQueueItem:
        """Add a new AI output to the review queue."""
        self._counter += 1
        item_id = f"{session_id}-{step}-{self._counter}"

        item = ReviewQueueItem(
            item_id=item_id,
            session_id=session_id,
            step=step,
            content=content,
            sources=sources,
            generation_attempt=attempt,
            priority=priority,
        )
        self._items[item_id] = item
        logger.info(
            "Enqueued HITL item: id=%s  step=%s  session=%s  priority=%d",
            item_id, step, session_id, priority,
        )
        return item

    def claim(self, item_id: str, reviewer_id: str) -> Optional[ReviewQueueItem]:
        """
        Claim a review item. Returns None if already claimed or not found.
        Sets a timeout so items are not locked forever.
        """
        item = self._items.get(item_id)
        if not item:
            logger.warning("Claim failed: item %s not found", item_id)
            return None

        self._expire_if_needed(item)

        if item.status == QueueItemStatus.CLAIMED:
            logger.warning("Claim failed: item %s already claimed by %s", item_id, item.claimed_by)
            return None

        if item.status == QueueItemStatus.RESOLVED:
            logger.warning("Claim failed: item %s already resolved", item_id)
            return None

        item.status     = QueueItemStatus.CLAIMED
        item.claimed_by = reviewer_id
        item.claimed_at = datetime.now(timezone.utc)
        logger.info("Item %s claimed by reviewer=%s", item_id, reviewer_id)
        return item

    def resolve(self, item_id: str, reviewer_id: str) -> bool:
        """Mark an item as resolved after a review decision is submitted."""
        item = self._items.get(item_id)
        if not item:
            return False
        if item.claimed_by != reviewer_id:
            logger.warning(
                "Resolve rejected: item %s claimed by %s, not %s",
                item_id, item.claimed_by, reviewer_id,
            )
            return False
        item.status      = QueueItemStatus.RESOLVED
        item.resolved_at = datetime.now(timezone.utc)
        logger.info("Item %s resolved by reviewer=%s", item_id, reviewer_id)
        return True

    def get_pending(self, session_id: Optional[str] = None) -> List[ReviewQueueItem]:
        """Return all pending items, optionally filtered by session."""
        self._release_expired()
        items = [
            i for i in self._items.values()
            if i.status in (QueueItemStatus.PENDING, QueueItemStatus.EXPIRED)
            and (session_id is None or i.session_id == session_id)
        ]
        items.sort(key=lambda i: (-i.priority, i.created_at))
        return items

    def _expire_if_needed(self, item: ReviewQueueItem):
        if item.is_expired:
            item.status = QueueItemStatus.EXPIRED
            item.claimed_by = None
            item.claimed_at = None
            logger.info("Item %s claim expired and released", item.item_id)

    def _release_expired(self):
        for item in list(self._items.values()):
            self._expire_if_needed(item)

    def queue_depth(self, session_id: Optional[str] = None) -> int:
        return len(self.get_pending(session_id))


# ---------------------------------------------------------------------------
# Module-level singleton (for demo; use DI in production)
# ---------------------------------------------------------------------------
_queue = ReviewQueue()


def get_review_queue() -> ReviewQueue:
    return _queue
