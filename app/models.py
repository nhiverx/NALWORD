"""
Pydantic data models for the HITL document pipeline.

Every session tracks the full lifecycle: Document intake → Outline → Highlights → Draft → Validation.
Each step has an associated StepStatus and a HITL review record capturing every human decision.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ReviewDecision(str, Enum):
    ACCEPT = "accept"
    EDIT = "edit"          # accepted with inline edits
    REJECT = "reject"      # rejected with feedback; regeneration required
    ALTERNATIVE = "alternative"  # request a fresh generation attempt
    REDRAFT = "redraft"    # re-generate with reviewer guidance notes


class WorkflowStep(str, Enum):
    UPLOAD = "upload"
    OUTLINE = "outline"
    HIGHLIGHTS = "highlights"
    DRAFT = "draft"
    VALIDATION = "validation"
    COMPLETE = "complete"


class StepStatus(str, Enum):
    PENDING = "pending"             # step not yet started (locked)
    GENERATING = "generating"       # AI is working
    AWAITING_REVIEW = "awaiting_review"  # output ready, awaiting human decision
    APPROVED = "approved"           # human approved; step complete
    REJECTED = "rejected"           # human rejected; needs regeneration


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class Source(BaseModel):
    """A single retrieved knowledge-base entry used to ground AI output."""
    id: str
    title: str
    snippet: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    source_type: str  # "case_study" | "reference" | "profile" | "template"
    category: Optional[str] = None
    value: Optional[str] = None
    status: Optional[str] = None  # "completed" | "ongoing" | None


class RequirementItem(BaseModel):
    """A single requirement extracted from the input document."""
    section: str        # "Instructions" | "Criteria" | "Scope" | "General"
    ref: str            # e.g. "I.2.1"
    text: str
    type: str           # "instruction" | "evaluation_criterion" | "format" | "mandatory"
    mandatory: bool = False


class ValidationFlag(BaseModel):
    """A single gap or warning identified during quality validation."""
    severity: str       # "critical" | "major" | "minor"
    section: str
    description: str
    suggestion: str


class HITLReviewRecord(BaseModel):
    """Immutable audit record for one human review decision."""
    step: WorkflowStep
    decision: ReviewDecision
    original_content: str
    final_content: str
    reviewer_notes: Optional[str] = None
    feedback: Optional[str] = None
    reviewer_id: Optional[str] = None      # ID/name of the human who made this decision
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    generation_attempt: int = 1


# ---------------------------------------------------------------------------
# Session model
# ---------------------------------------------------------------------------

class PipelineSession(BaseModel):
    """Full state for one document pipeline session."""

    session_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    document_title: str = "Untitled Document"

    # --- Input document ---
    input_text: str = ""
    extracted_requirements: List[RequirementItem] = []

    # --- Current position in workflow ---
    current_step: WorkflowStep = WorkflowStep.UPLOAD

    # --- The single persistent working document (refined at every step) ---
    current_document: Optional[str] = None

    # --- Per-step outputs kept for audit (what the AI added at each step) ---
    outline: Optional[str] = None
    highlights: Optional[str] = None
    draft: Optional[str] = None
    validation_report: Optional[str] = None
    validation_flags: List[ValidationFlag] = []

    # --- Sources retrieved for each step ---
    outline_sources: List[Source] = []
    highlights_sources: List[Source] = []
    draft_sources: List[Source] = []

    # --- Step statuses ---
    outline_status: StepStatus = StepStatus.PENDING
    highlights_status: StepStatus = StepStatus.PENDING
    draft_status: StepStatus = StepStatus.PENDING
    validation_status: StepStatus = StepStatus.PENDING

    # --- Generation attempt counters (for retry tracking) ---
    outline_attempt: int = 0
    highlights_attempt: int = 0
    draft_attempt: int = 0
    validation_attempt: int = 0

    # --- Validation score (0-100 estimated quality) ---
    validation_score: int = 0

    # --- HITL audit trail ---
    review_history: List[HITLReviewRecord] = []

    # --- Redraft guidance notes (stored per step, passed to next generation) ---
    outline_redraft_notes: Optional[str] = None
    highlights_redraft_notes: Optional[str] = None
    draft_redraft_notes: Optional[str] = None
    validation_redraft_notes: Optional[str] = None


# ---------------------------------------------------------------------------
# API request/response schemas
# ---------------------------------------------------------------------------

class DocumentUploadRequest(BaseModel):
    input_text: str = Field(min_length=50, description="Full input document text (paste or extracted)")
    document_title: str = Field(default="Untitled Document", max_length=120)


class HITLReviewRequest(BaseModel):
    session_id: str
    step: WorkflowStep
    decision: ReviewDecision
    edited_content: Optional[str] = None   # if decision == EDIT, the modified text
    feedback: Optional[str] = None          # if decision == REJECT, reason
    reviewer_id: Optional[str] = None       # required on ACCEPT/EDIT; logged in audit trail


class RedraftRequest(BaseModel):
    """Request to re-generate a step's output with specific reviewer guidance."""
    session_id: str
    step: WorkflowStep
    notes: str = Field(min_length=5, description="Reviewer guidance for the AI re-draft")
    reviewer_id: Optional[str] = None


class GenerateRequest(BaseModel):
    session_id: str
    step: WorkflowStep


class SessionStateResponse(BaseModel):
    session_id: str
    current_step: WorkflowStep
    document_title: str
    current_document: Optional[str]
    outline: Optional[str]
    outline_status: StepStatus
    outline_sources: List[Source]
    highlights: Optional[str]
    highlights_status: StepStatus
    highlights_sources: List[Source]
    draft: Optional[str]
    draft_status: StepStatus
    draft_sources: List[Source]
    validation_report: Optional[str]
    validation_status: StepStatus
    validation_flags: List[ValidationFlag]
    validation_score: int
    extracted_requirements: List[RequirementItem]
    review_history: List[HITLReviewRecord]
    outline_attempt: int
    highlights_attempt: int
    draft_attempt: int
    validation_attempt: int
