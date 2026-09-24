"""
Mock AI workflow engine for the HITL document pipeline demo.

Each function simulates what the production system will do with real models
(Claude via Amazon Bedrock) and a real FAISS index. The outputs are
deterministic template expansions driven by keyword extraction from the
input document.

In production, swap each function body for the corresponding call in
app/ai/rag_pipeline.py.
"""

from __future__ import annotations

import re
import textwrap
from typing import List, Optional, Tuple

from app.knowledge_base import retrieve
from app.models import (
    RequirementItem,
    Source,
    ValidationFlag,
)


# ---------------------------------------------------------------------------
# Requirement extraction
# ---------------------------------------------------------------------------

_SECTION_HEADER = re.compile(r"^SECTION\s+([A-Z])\b", re.IGNORECASE)

_SECTION_NAMES = {
    "I": "Instructions",
    "E": "Criteria",
    "S": "Scope",
}

_MANDATORY_KEYWORDS = [
    "shall", "must", "required", "mandatory", "will provide", "certification",
    "page limit", "font size", "margin", "part i",
]


def extract_requirements(input_text: str) -> List[RequirementItem]:
    """
    Parse the input document into a structured list of RequirementItems.
    Uses regex heuristics to identify sections, refs, and mandatory language.
    """
    items: List[RequirementItem] = []
    current_section = "General"

    for line in input_text.splitlines():
        line = line.strip()
        if not line:
            continue

        header = _SECTION_HEADER.match(line)
        if header:
            current_section = _SECTION_NAMES.get(header.group(1).upper(), "General")
            continue

        # Numbered requirements (e.g. "I.2.1 Responses shall...")
        ref_match = re.match(r"^([A-Z]\.\d+(?:\.\d+)?)\s+(.+)$", line)
        if ref_match:
            ref, text = ref_match.groups()
            is_mandatory = any(kw in text.lower() for kw in _MANDATORY_KEYWORDS)
            if is_mandatory:
                req_type = "mandatory"
            elif current_section == "Criteria":
                req_type = "evaluation_criterion"
            else:
                req_type = "instruction"
            items.append(RequirementItem(
                section=current_section,
                ref=ref,
                text=text[:300],
                type=req_type,
                mandatory=is_mandatory,
            ))
        elif any(kw in line.lower() for kw in _MANDATORY_KEYWORDS) and len(line) > 40:
            items.append(RequirementItem(
                section=current_section,
                ref="—",
                text=line[:300],
                type="mandatory",
                mandatory=True,
            ))

    if not items:
        items = [
            RequirementItem(section="Instructions", ref="I.1", text="Responses shall describe the proposed approach.", type="mandatory", mandatory=True),
            RequirementItem(section="Instructions", ref="I.2", text="A minimum of three case study references is required.", type="mandatory", mandatory=True),
            RequirementItem(section="Criteria", ref="E.1", text="Approach evaluated on feasibility and clarity.", type="evaluation_criterion", mandatory=False),
            RequirementItem(section="Criteria", ref="E.2", text="Case studies evaluated on relevance and recency.", type="evaluation_criterion", mandatory=False),
            RequirementItem(section="Scope", ref="S.1", text="The provider shall deliver a project plan within 30 days.", type="instruction", mandatory=True),
        ]

    return items


def _redraft_block(notes: Optional[str]) -> str:
    return f"\nREVIEWER GUIDANCE APPLIED\n-------------------------\n{notes}\n" if notes else ""


# ---------------------------------------------------------------------------
# Step 1 – Structured Outline
# ---------------------------------------------------------------------------

def generate_outline(input_text: str, attempt: int = 1, notes: Optional[str] = None) -> Tuple[str, List[Source]]:
    """
    Generate a structured outline grounded in the input document.
    Returns (outline_text, sources).
    """
    sources = retrieve(input_text, top_k=3)
    reqs = extract_requirements(input_text)

    instruction_items = [r for r in reqs if r.section == "Instructions"]
    criteria_items = [r for r in reqs if r.section == "Criteria"]

    source_refs = "\n".join(
        f"  [{i+1}] {s.title} ({s.source_type.replace('_',' ').title()}, score={s.relevance_score:.2f})"
        for i, s in enumerate(sources)
    )

    def source_title(i: int) -> str:
        return sources[i].title if len(sources) > i else "KB Entry"

    outline = textwrap.dedent(f"""\
    DOCUMENT OUTLINE — STRUCTURED FRAMEWORK
    =======================================
    Generated against the input document instructions.  Attempt #{attempt}{_redraft_block(notes)}
    Grounded sources: {len(sources)} KB entries retrieved.

    PART I: APPROACH
    ----------------
    1.0  Executive Summary
         1.1  Context and Objectives
         1.2  Solution Overview
         1.3  Key Differentiators

    2.0  Approach
         2.1  Operating Model
         2.2  Methodology and Tools
         2.3  Innovation and Risk Mitigation

    3.0  Management
         3.1  Delivery Plan
         3.2  Team Structure
         3.3  Quality Assurance & Metrics

    PART II: CASE STUDIES
    ---------------------
    4.0  Relevant Experience
         4.1  Case Study #1  [Source: {source_title(0)}]
         4.2  Case Study #2  [Source: {source_title(1)}]
         4.3  Case Study #3  [Source: {source_title(2)}]

    PART III: COST
    --------------
    5.0  Cost Summary
         5.1  Roles and Effort
         5.2  Other Direct Costs
         5.3  Assumptions

    REQUIREMENTS CHECKLIST
    ----------------------
    {"".join(f"  [ ] {r.ref}  {r.text[:80]}...{chr(10)}" for r in (instruction_items + criteria_items)[:6])}

    RETRIEVED SOURCES USED
    ----------------------
    {source_refs}
    """)

    return outline, sources


# ---------------------------------------------------------------------------
# Step 2 – Key Highlights
# ---------------------------------------------------------------------------

def generate_highlights(input_text: str, current_document: str, attempt: int = 1, notes: Optional[str] = None) -> Tuple[str, List[Source]]:
    """
    Enrich the existing working document with per-section key highlights.
    Returns the updated document (not a separate highlights block).
    """
    sources = retrieve(input_text + " " + current_document, top_k=4)

    source_refs = "\n".join(
        f"  [{i+1}] {s.title} | Category: {s.category or 'General'} | Status: {s.status or 'N/A'}"
        for i, s in enumerate(sources)
    )

    first_title = sources[0].title if sources else "Case Study"
    second_title = sources[1].title if len(sources) > 1 else "Case Study #2"
    second_category = sources[1].category if len(sources) > 1 and sources[1].category else "similar engagements"

    highlights_block = textwrap.dedent(f"""\

    ═══════════════════════════════════════════════════════════════
    STEP 2 ADDITION — KEY HIGHLIGHTS  (Attempt #{attempt}){_redraft_block(notes)}
    ═══════════════════════════════════════════════════════════════

    HIGHLIGHTS BY SECTION
    ---------------------
    SECTION 2.0 — APPROACH
      HIGHLIGHT:   Proven, repeatable delivery model with zero critical defects.
      MESSAGE:     The team has delivered {len(sources)} comparable projects; the
                   standard framework reduces schedule risk by an estimated 30%.
      EVIDENCE:    [{first_title}] — rated Excellent.

    SECTION 3.0 — MANAGEMENT
      HIGHLIGHT:   Experienced leadership focused on continuity and clarity.
      MESSAGE:     The proposed project lead brings 12+ years of delivery
                   experience with no budget overruns on comparable work.
      EVIDENCE:    Team profiles on file; certified staff in place.

    SECTION 4.0 — CASE STUDIES
      HIGHLIGHT:   Directly relevant experience across {second_category}.
      MESSAGE:     Three of the last four comparable projects were rated
                   Excellent or Very Good by stakeholders.
      EVIDENCE:    [{second_title}] — cited for "responsive communication
                   and consistently high-quality deliverables."

    DIFFERENTIATORS
    ---------------
    D1: Certified team — all core staff hold relevant professional certifications.
    D2: Domain familiarity — team members have supported similar initiatives.
    D3: Small-team agility with enterprise-grade delivery capacity.
    D4: AI-assisted review tooling reduces errors and cycle time.

    SOURCES USED IN THIS STEP
    -------------------------
    {source_refs}
    """)

    return current_document + highlights_block, sources


# ---------------------------------------------------------------------------
# Step 3 – Narrative Draft
# ---------------------------------------------------------------------------

def generate_draft(input_text: str, current_document: str, attempt: int = 1, notes: Optional[str] = None) -> Tuple[str, List[Source]]:
    """
    Expand the existing document into full narrative prose.
    Returns the updated document with narrative sections added.
    """
    sources = retrieve(input_text + " " + current_document, top_k=5)

    source_refs = "\n".join(
        f"  [{i+1}] {s.title} (relevance={s.relevance_score:.2f}, type={s.source_type})"
        for i, s in enumerate(sources)
    )

    narrative_block = textwrap.dedent(f"""\

    ═══════════════════════════════════════════════════════════════
    STEP 3 ADDITION — NARRATIVE DRAFT  (Attempt #{attempt}){_redraft_block(notes)}
    ═══════════════════════════════════════════════════════════════

    2.1  OPERATING MODEL
    --------------------
    The team proposes a phased, outcome-first approach designed to meet the
    stated objectives with minimal transition risk. Full operating capability
    will be established within 30 days of kickoff, leveraging an existing
    toolchain and a ready-to-start delivery team.

    [SOURCE: {sources[0].title if sources else "Case Study Record"}]
    The delivery model is built on lessons learned from {len(sources)} analogous
    projects, consistently meeting or exceeding milestones with an average
    on-time delivery rate of 97%.

    2.2  METHODOLOGY AND TOOLS
    --------------------------
    A hybrid Agile/Waterfall framework is tailored to the stakeholder's review
    cadence. Sprint cycles are aligned with scheduled checkpoints, ensuring
    continuous visibility into progress.

    Key tools: an issue tracker, a shared documentation wiki, version control,
    and an AI-assisted quality review system.

    [SOURCE: {sources[1].title if len(sources) > 1 else "Reference Sheet"}]
    The tool suite has been validated on projects of similar scope and
    complexity, including environments with strict data-handling policies.

    2.3  INNOVATION AND RISK MITIGATION
    ------------------------------------
    Two practices directly reduce project risk:

    (1) AI-Assisted Quality Review: every deliverable is scanned against the
    requirements matrix before release, eliminating formatting and
    completeness gaps early.

    (2) Flexible Staffing Model: a pre-qualified bench of specialists enables
    rapid scaling (within 5 business days) without impacting core staffing.

    Risk Register: three risks identified; all rated LOW with active
    mitigations defined. The full register is available in Section 3.0.

    [All content grounded in KB. No unsourced factual claims.]

    SOURCES USED IN THIS STEP
    -------------------------
    {source_refs}
    """)

    return current_document + narrative_block, sources


# ---------------------------------------------------------------------------
# Step 4 – Quality Validation
# ---------------------------------------------------------------------------

def run_validation_check(
    input_text: str,
    current_document: str,
    attempt: int = 1,
    notes: Optional[str] = None,
) -> Tuple[str, List[ValidationFlag], int]:
    """
    Cross-check the working document against the extracted requirements.
    Returns (standalone_report, flags, estimated_score 0-100).
    Does NOT modify current_document — the report is a separate artifact.
    """
    reqs = extract_requirements(input_text)
    flags: List[ValidationFlag] = []
    doc_lower = current_document.lower()

    if re.search(r"(?i)page\s+limit", input_text) and "page limit" not in doc_lower:
        flags.append(ValidationFlag(
            severity="critical",
            section="Part I",
            description="Input document specifies a page limit; no explicit page count control noted in draft.",
            suggestion="Add a page count tracker table to the executive summary.",
        ))

    for sec in ["case studies", "management", "approach"]:
        if sec not in doc_lower:
            flags.append(ValidationFlag(
                severity="major",
                section="Document Structure",
                description=f"Required section '{sec.title()}' may be missing from draft content.",
                suggestion=f"Ensure the {sec.title()} section is explicitly labelled per the instructions.",
            ))

    mandatory_reqs = [r for r in reqs if r.mandatory]
    covered = sum(
        1 for req in mandatory_reqs
        if req.text and req.text.split()[0].lower() in doc_lower
    )
    coverage_pct = int((covered / max(len(mandatory_reqs), 1)) * 100)

    if coverage_pct < 80:
        flags.append(ValidationFlag(
            severity="major",
            section="Requirements Matrix",
            description=f"Estimated requirement coverage: {coverage_pct}% of mandatory items addressed.",
            suggestion="Cross-check each instruction against the draft and fill gaps.",
        ))

    flags.append(ValidationFlag(
        severity="minor",
        section="Formatting",
        description="Font and margin specs not verified in this demo environment.",
        suggestion="Confirm font, margins, and header/footer as specified in the instructions.",
    ))

    critical_count = sum(1 for f in flags if f.severity == "critical")
    major_count    = sum(1 for f in flags if f.severity == "major")
    minor_count    = sum(1 for f in flags if f.severity == "minor")

    estimated_score = max(0, 100 - (critical_count * 25) - (major_count * 10) - (minor_count * 3))

    flag_lines = "\n".join(
        f"  [{f.severity.upper()}] {f.section}\n"
        f"  Description : {f.description}\n"
        f"  Suggestion  : {f.suggestion}"
        for f in flags
    )

    overall = (
        "⚠  REVIEW REQUIRED — address critical/major flags before release."
        if (critical_count + major_count) > 0
        else "✓  ESTIMATED PASS — no critical or major gaps detected."
    )

    report = textwrap.dedent(f"""\
    QUALITY VALIDATION REPORT
    =========================
    Attempt #{attempt}{_redraft_block(notes)}
    Requirements checked  : {len(reqs)} total  |  {len(mandatory_reqs)} mandatory
    Estimated coverage    : {coverage_pct}%
    Estimated quality     : {estimated_score} / 100
    Flags                 : {critical_count} critical  |  {major_count} major  |  {minor_count} minor

    VALIDATION FLAGS
    ----------------
    {flag_lines if flag_lines else "  No flags — all requirements appear addressed."}

    OVERALL ASSESSMENT
    ------------------
    {overall}

    DISCLAIMER
    ----------
    This automated check is advisory only. A qualified human reviewer
    must perform the final review before the document is released.
    """)

    return report, flags, estimated_score
