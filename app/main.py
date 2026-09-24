"""
HITL Document Pipeline — FastAPI Application
=============================================
All routes for the human-in-the-loop document workflow.

Session state is stored in-memory (dict) for the demo.
In production, replace with PostgreSQL-backed session storage.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.models import (
    DocumentUploadRequest,
    GenerateRequest,
    HITLReviewRecord,
    HITLReviewRequest,
    PipelineSession,
    RedraftRequest,
    ReviewDecision,
    SessionStateResponse,
    StepStatus,
    WorkflowStep,
)
from app.workflow import (
    extract_requirements,
    generate_draft,
    generate_highlights,
    generate_outline,
    run_validation_check,
)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="HITL Document Pipeline",
    description="AI-assisted document generation workflow with mandatory Human-in-the-Loop controls.",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

_SAMPLE_DOCUMENT_PATH = Path(__file__).parent.parent / "data" / "input_document.txt"

# ---------------------------------------------------------------------------
# In-memory session store  (swap for DB in production)
# ---------------------------------------------------------------------------

_SESSIONS: Dict[str, PipelineSession] = {}


def _get_session(session_id: str) -> PipelineSession:
    session = _SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return session


def _record_review(
    session: PipelineSession,
    step: WorkflowStep,
    decision: ReviewDecision,
    original_content: str,
    final_content: str,
    notes: Optional[str],
    reviewer_id: Optional[str],
    attempt: int,
) -> None:
    session.review_history.append(HITLReviewRecord(
        step=step,
        decision=decision,
        original_content=original_content,
        final_content=final_content,
        reviewer_notes=notes,
        feedback=notes,
        reviewer_id=reviewer_id,
        timestamp=datetime.utcnow(),
        generation_attempt=attempt,
    ))


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/api/sample-document", response_class=PlainTextResponse)
async def sample_document():
    """Return the bundled sample input document."""
    try:
        return _SAMPLE_DOCUMENT_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise HTTPException(404, "Sample document not found.")


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

@app.post("/api/session/new")
async def new_session():
    """Create a new blank pipeline session."""
    session_id = str(uuid.uuid4())[:8]
    _SESSIONS[session_id] = PipelineSession(session_id=session_id)
    return {"session_id": session_id}


@app.get("/api/session/{session_id}", response_model=SessionStateResponse)
async def get_session(session_id: str):
    """Return full session state for the frontend."""
    return _get_session(session_id)


# ---------------------------------------------------------------------------
# Step 0 — Document Intake & Requirement Extraction
# ---------------------------------------------------------------------------

@app.post("/api/document/upload")
async def upload_document(body: DocumentUploadRequest):
    """
    Accept input document text, extract requirements, and advance session to OUTLINE step.
    Returns the new session_id and extracted requirements.
    """
    session_id = str(uuid.uuid4())[:8]
    session = PipelineSession(
        session_id=session_id,
        input_text=body.input_text,
        document_title=body.document_title,
    )

    session.extracted_requirements = extract_requirements(body.input_text)
    session.current_step = WorkflowStep.OUTLINE
    session.outline_status = StepStatus.PENDING

    _SESSIONS[session_id] = session
    return {
        "session_id": session_id,
        "requirements_count": len(session.extracted_requirements),
        "extracted_requirements": [r.model_dump() for r in session.extracted_requirements],
        "current_step": session.current_step,
    }


# ---------------------------------------------------------------------------
# Step 1 — Outline
# ---------------------------------------------------------------------------

@app.post("/api/step/outline/generate")
async def generate_outline_step(body: GenerateRequest):
    """
    AI generates the structured outline.
    Sets status to AWAITING_REVIEW — no content flows forward until a human approves.
    """
    session = _get_session(body.session_id)

    if session.outline_status == StepStatus.APPROVED:
        raise HTTPException(400, "Outline already approved. Reset to regenerate.")

    session.outline_attempt += 1
    session.outline_status = StepStatus.GENERATING

    outline, sources = generate_outline(session.input_text, attempt=session.outline_attempt)

    session.outline = outline
    session.current_document = outline   # initialise the persistent document
    session.outline_sources = sources
    session.outline_status = StepStatus.AWAITING_REVIEW
    session.current_step = WorkflowStep.OUTLINE

    return {
        "outline": session.outline,
        "current_document": session.current_document,
        "sources": [s.model_dump() for s in session.outline_sources],
        "status": session.outline_status,
        "attempt": session.outline_attempt,
    }


@app.post("/api/step/outline/review")
async def review_outline(body: HITLReviewRequest):
    """
    Human HITL decision for the outline step.
    ACCEPT / EDIT → unlock Highlights.
    REJECT / ALTERNATIVE → keep step locked, allow regeneration.
    """
    session = _get_session(body.session_id)

    if session.outline_status != StepStatus.AWAITING_REVIEW:
        raise HTTPException(400, "Outline is not awaiting review.")

    final_content = body.edited_content or session.outline or ""
    _record_review(
        session, WorkflowStep.OUTLINE, body.decision,
        session.outline or "", final_content,
        body.feedback, body.reviewer_id, session.outline_attempt,
    )

    if body.decision in (ReviewDecision.ACCEPT, ReviewDecision.EDIT):
        session.outline = final_content
        session.current_document = final_content
        session.outline_status = StepStatus.APPROVED
        session.current_step = WorkflowStep.HIGHLIGHTS
        session.highlights_status = StepStatus.PENDING
    elif body.decision == ReviewDecision.REJECT:
        session.outline_status = StepStatus.REJECTED
    elif body.decision == ReviewDecision.ALTERNATIVE:
        session.outline_status = StepStatus.PENDING

    return {
        "decision": body.decision,
        "outline_status": session.outline_status,
        "current_step": session.current_step,
    }


# ---------------------------------------------------------------------------
# Step 2 — Key Highlights
# ---------------------------------------------------------------------------

@app.post("/api/step/highlights/generate")
async def generate_highlights_step(body: GenerateRequest):
    session = _get_session(body.session_id)

    if session.outline_status != StepStatus.APPROVED:
        raise HTTPException(400, "Outline must be approved before generating highlights.")

    session.highlights_attempt += 1
    session.highlights_status = StepStatus.GENERATING

    updated_doc, sources = generate_highlights(
        session.input_text, session.current_document or "", attempt=session.highlights_attempt
    )

    session.highlights = updated_doc
    session.current_document = updated_doc
    session.highlights_sources = sources
    session.highlights_status = StepStatus.AWAITING_REVIEW
    session.current_step = WorkflowStep.HIGHLIGHTS

    return {
        "highlights": session.highlights,
        "current_document": session.current_document,
        "sources": [s.model_dump() for s in session.highlights_sources],
        "status": session.highlights_status,
        "attempt": session.highlights_attempt,
    }


@app.post("/api/step/highlights/review")
async def review_highlights(body: HITLReviewRequest):
    session = _get_session(body.session_id)

    if session.highlights_status != StepStatus.AWAITING_REVIEW:
        raise HTTPException(400, "Highlights are not awaiting review.")

    final_content = body.edited_content or session.highlights or ""
    _record_review(
        session, WorkflowStep.HIGHLIGHTS, body.decision,
        session.highlights or "", final_content,
        body.feedback, body.reviewer_id, session.highlights_attempt,
    )

    if body.decision in (ReviewDecision.ACCEPT, ReviewDecision.EDIT):
        session.highlights = final_content
        session.current_document = final_content
        session.highlights_status = StepStatus.APPROVED
        session.current_step = WorkflowStep.DRAFT
        session.draft_status = StepStatus.PENDING
    elif body.decision == ReviewDecision.REJECT:
        session.highlights_status = StepStatus.REJECTED
    elif body.decision == ReviewDecision.ALTERNATIVE:
        session.highlights_status = StepStatus.PENDING

    return {
        "decision": body.decision,
        "highlights_status": session.highlights_status,
        "current_step": session.current_step,
    }


# ---------------------------------------------------------------------------
# Step 3 — Draft Generation
# ---------------------------------------------------------------------------

@app.post("/api/step/draft/generate")
async def generate_draft_step(body: GenerateRequest):
    session = _get_session(body.session_id)

    if session.highlights_status != StepStatus.APPROVED:
        raise HTTPException(400, "Highlights must be approved before generating the draft.")

    session.draft_attempt += 1
    session.draft_status = StepStatus.GENERATING

    updated_doc, sources = generate_draft(
        session.input_text,
        session.current_document or "",
        attempt=session.draft_attempt,
    )

    session.draft = updated_doc
    session.current_document = updated_doc
    session.draft_sources = sources
    session.draft_status = StepStatus.AWAITING_REVIEW
    session.current_step = WorkflowStep.DRAFT

    return {
        "draft": session.draft,
        "current_document": session.current_document,
        "sources": [s.model_dump() for s in session.draft_sources],
        "status": session.draft_status,
        "attempt": session.draft_attempt,
    }


@app.post("/api/step/draft/review")
async def review_draft(body: HITLReviewRequest):
    session = _get_session(body.session_id)

    if session.draft_status != StepStatus.AWAITING_REVIEW:
        raise HTTPException(400, "Draft is not awaiting review.")

    final_content = body.edited_content or session.draft or ""
    _record_review(
        session, WorkflowStep.DRAFT, body.decision,
        session.draft or "", final_content,
        body.feedback, body.reviewer_id, session.draft_attempt,
    )

    if body.decision in (ReviewDecision.ACCEPT, ReviewDecision.EDIT):
        session.draft = final_content
        session.current_document = final_content
        session.draft_status = StepStatus.APPROVED
        session.current_step = WorkflowStep.VALIDATION
        session.validation_status = StepStatus.PENDING
    elif body.decision == ReviewDecision.REJECT:
        session.draft_status = StepStatus.REJECTED
    elif body.decision == ReviewDecision.ALTERNATIVE:
        session.draft_status = StepStatus.PENDING

    return {
        "decision": body.decision,
        "draft_status": session.draft_status,
        "current_step": session.current_step,
    }


# ---------------------------------------------------------------------------
# Step 4 — Quality Validation
# ---------------------------------------------------------------------------

@app.post("/api/step/validation/run")
async def run_validation(body: GenerateRequest):
    session = _get_session(body.session_id)

    if session.draft_status != StepStatus.APPROVED:
        raise HTTPException(400, "Draft must be approved before running validation.")

    session.validation_attempt += 1
    session.validation_status = StepStatus.GENERATING

    report, flags, score = run_validation_check(
        session.input_text,
        session.current_document or "",
        attempt=session.validation_attempt,
    )

    # The validation report is a separate artifact — current_document is left untouched
    session.validation_report = report
    session.validation_flags = flags
    session.validation_score = score
    session.validation_status = StepStatus.AWAITING_REVIEW
    session.current_step = WorkflowStep.VALIDATION

    return {
        "validation_report": session.validation_report,
        "current_document": session.current_document,
        "flags": [f.model_dump() for f in session.validation_flags],
        "estimated_score": session.validation_score,
        "status": session.validation_status,
        "attempt": session.validation_attempt,
    }


@app.post("/api/step/validation/review")
async def review_validation(body: HITLReviewRequest):
    session = _get_session(body.session_id)

    if session.validation_status != StepStatus.AWAITING_REVIEW:
        raise HTTPException(400, "Validation report is not awaiting review.")

    final_content = body.edited_content or session.validation_report or ""
    _record_review(
        session, WorkflowStep.VALIDATION, body.decision,
        session.validation_report or "", final_content,
        body.feedback, body.reviewer_id, session.validation_attempt,
    )

    if body.decision in (ReviewDecision.ACCEPT, ReviewDecision.EDIT):
        session.validation_report = final_content
        session.validation_status = StepStatus.APPROVED
        session.current_step = WorkflowStep.COMPLETE
    elif body.decision == ReviewDecision.REJECT:
        session.validation_status = StepStatus.REJECTED
    elif body.decision == ReviewDecision.ALTERNATIVE:
        session.validation_status = StepStatus.PENDING

    return {
        "decision": body.decision,
        "validation_status": session.validation_status,
        "current_step": session.current_step,
    }


# ---------------------------------------------------------------------------
# Re-Draft endpoints — reviewer provides notes, AI regenerates immediately
# Status stays AWAITING_REVIEW; can be called unlimited times.
# ---------------------------------------------------------------------------

@app.post("/api/step/outline/redraft")
async def redraft_outline(body: RedraftRequest):
    """Re-generate the outline incorporating reviewer guidance notes."""
    session = _get_session(body.session_id)
    if session.outline_status != StepStatus.AWAITING_REVIEW:
        raise HTTPException(400, "Outline must be awaiting review to request a re-draft.")

    _record_review(
        session, WorkflowStep.OUTLINE, ReviewDecision.REDRAFT,
        session.outline or "", session.outline or "",
        body.notes, body.reviewer_id, session.outline_attempt,
    )

    session.outline_redraft_notes = body.notes
    session.outline_attempt += 1
    outline, sources = generate_outline(
        session.input_text, attempt=session.outline_attempt, notes=body.notes
    )
    session.outline = outline
    session.current_document = outline
    session.outline_sources = sources
    session.outline_status = StepStatus.AWAITING_REVIEW

    return {
        "outline": session.outline,
        "current_document": session.current_document,
        "sources": [s.model_dump() for s in session.outline_sources],
        "status": session.outline_status,
        "attempt": session.outline_attempt,
    }


@app.post("/api/step/highlights/redraft")
async def redraft_highlights(body: RedraftRequest):
    """Re-generate highlights incorporating reviewer guidance notes."""
    session = _get_session(body.session_id)
    if session.highlights_status != StepStatus.AWAITING_REVIEW:
        raise HTTPException(400, "Highlights must be awaiting review to request a re-draft.")

    _record_review(
        session, WorkflowStep.HIGHLIGHTS, ReviewDecision.REDRAFT,
        session.highlights or "", session.highlights or "",
        body.notes, body.reviewer_id, session.highlights_attempt,
    )

    session.highlights_redraft_notes = body.notes
    session.highlights_attempt += 1
    updated_doc, sources = generate_highlights(
        session.input_text, session.current_document or "",
        attempt=session.highlights_attempt, notes=body.notes,
    )
    session.highlights = updated_doc
    session.current_document = updated_doc
    session.highlights_sources = sources
    session.highlights_status = StepStatus.AWAITING_REVIEW

    return {
        "highlights": session.highlights,
        "current_document": session.current_document,
        "sources": [s.model_dump() for s in session.highlights_sources],
        "status": session.highlights_status,
        "attempt": session.highlights_attempt,
    }


@app.post("/api/step/draft/redraft")
async def redraft_draft(body: RedraftRequest):
    """Re-generate the narrative draft incorporating reviewer guidance notes."""
    session = _get_session(body.session_id)
    if session.draft_status != StepStatus.AWAITING_REVIEW:
        raise HTTPException(400, "Draft must be awaiting review to request a re-draft.")

    _record_review(
        session, WorkflowStep.DRAFT, ReviewDecision.REDRAFT,
        session.draft or "", session.draft or "",
        body.notes, body.reviewer_id, session.draft_attempt,
    )

    session.draft_redraft_notes = body.notes
    session.draft_attempt += 1
    updated_doc, sources = generate_draft(
        session.input_text, session.current_document or "",
        attempt=session.draft_attempt, notes=body.notes,
    )
    session.draft = updated_doc
    session.current_document = updated_doc
    session.draft_sources = sources
    session.draft_status = StepStatus.AWAITING_REVIEW

    return {
        "draft": session.draft,
        "current_document": session.current_document,
        "sources": [s.model_dump() for s in session.draft_sources],
        "status": session.draft_status,
        "attempt": session.draft_attempt,
    }


@app.post("/api/step/validation/redraft")
async def redraft_validation(body: RedraftRequest):
    """Re-run the validation check incorporating reviewer guidance notes."""
    session = _get_session(body.session_id)
    if session.validation_status != StepStatus.AWAITING_REVIEW:
        raise HTTPException(400, "Validation report must be awaiting review to request a re-check.")

    _record_review(
        session, WorkflowStep.VALIDATION, ReviewDecision.REDRAFT,
        session.validation_report or "", session.validation_report or "",
        body.notes, body.reviewer_id, session.validation_attempt,
    )

    session.validation_redraft_notes = body.notes
    session.validation_attempt += 1
    report, flags, score = run_validation_check(
        session.input_text, session.current_document or "",
        attempt=session.validation_attempt, notes=body.notes,
    )
    session.validation_report = report
    session.validation_flags = flags
    session.validation_score = score
    session.validation_status = StepStatus.AWAITING_REVIEW

    return {
        "validation_report": session.validation_report,
        "current_document": session.current_document,
        "flags": [f.model_dump() for f in session.validation_flags],
        "estimated_score": session.validation_score,
        "status": session.validation_status,
        "attempt": session.validation_attempt,
    }


# ---------------------------------------------------------------------------
# Audit trail
# ---------------------------------------------------------------------------

@app.get("/api/session/{session_id}/audit")
async def get_audit_trail(session_id: str):
    """Return the full HITL decision audit trail for a session."""
    session = _get_session(session_id)
    return {
        "session_id": session_id,
        "document_title": session.document_title,
        "review_count": len(session.review_history),
        "reviews": [r.model_dump() for r in session.review_history],
    }
