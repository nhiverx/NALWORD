"""
Validation Checker — Automated Document Gap Analysis
====================================================
Cross-checks approved document content against every extracted requirement.

Two-layer check:
  Layer 1: Deterministic rule-based checks (page limits, font, parts, counts)
  Layer 2: AI semantic coverage check via Claude (catches narrative gaps)

Severity levels:
  CRITICAL — mandatory requirement with no coverage; blocks release
  MAJOR    — important requirement with partial/unclear coverage
  MINOR    — formatting or best-practice note

This checker is ADVISORY ONLY. A qualified human reviewer must sign off
before the document is released.
"""

from __future__ import annotations

import logging
import re
from typing import List, Tuple

from app.models import RequirementItem, ValidationFlag

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Layer 1 — Deterministic rule checks
# ---------------------------------------------------------------------------

class DeterministicChecker:
    """Fast, always-available rule-based validation checks."""

    FONT_SPEC_PATTERN    = re.compile(r"arial\s+12|12.point|12pt", re.IGNORECASE)
    MARGIN_SPEC_PATTERN  = re.compile(r"1.inch|one.inch margin", re.IGNORECASE)
    PAGE_LIMIT_PATTERN   = re.compile(r"page.?limit|not.exceed.+page", re.IGNORECASE)
    REFERENCE_COUNT      = re.compile(r"(\d+)\s+(?:case stud(?:y|ies)|references?)", re.IGNORECASE)
    PART_PATTERN         = re.compile(r"part\s+(I{1,3}|IV|[1-4])\b", re.IGNORECASE)

    def check(
        self,
        input_text: str,
        outline: str,
        draft: str,
        requirements: List[RequirementItem],
    ) -> List[ValidationFlag]:
        flags: List[ValidationFlag] = []

        combined_doc = (outline + " " + draft).lower()

        # 1. Page limit awareness
        if self.PAGE_LIMIT_PATTERN.search(input_text):
            if "page" not in combined_doc and "part" not in combined_doc:
                flags.append(ValidationFlag(
                    severity="critical",
                    section="Formatting",
                    description="Input document specifies a page limit; content does not acknowledge part/page structure.",
                    suggestion="Add explicit part headers and page count controls.",
                ))

        # 2. Font/margin requirements
        if self.FONT_SPEC_PATTERN.search(input_text):
            flags.append(ValidationFlag(
                severity="minor",
                section="Formatting",
                description="Input document requires Arial 12pt font — verify in final formatting.",
                suggestion="Confirm font in final Word/PDF output before release.",
            ))
        if self.MARGIN_SPEC_PATTERN.search(input_text):
            flags.append(ValidationFlag(
                severity="minor",
                section="Formatting",
                description="Input document requires 1-inch margins — verify in final document.",
                suggestion="Set 1-inch margins in the final document template.",
            ))

        # 3. Case study reference count
        ref_match = self.REFERENCE_COUNT.search(input_text)
        if ref_match:
            required_count = int(ref_match.group(1))
            found = len(re.findall(r"case study\s+#\d", combined_doc, re.IGNORECASE))
            if found < required_count:
                flags.append(ValidationFlag(
                    severity="major",
                    section="Part II — Case Studies",
                    description=f"Input document requires {required_count} case study references; ~{found} found in draft.",
                    suggestion=f"Add {required_count - found} more case study reference(s) with the required fields.",
                ))

        # 4. Part structure
        parts_required = set(p.upper() for p in self.PART_PATTERN.findall(input_text))
        if parts_required:
            parts_in_outline = set(p.upper() for p in self.PART_PATTERN.findall(outline))
            missing = parts_required - parts_in_outline
            if missing:
                flags.append(ValidationFlag(
                    severity="major",
                    section="Document Structure",
                    description=f"Input document references part(s) not found in outline: {', '.join(sorted(missing))}.",
                    suggestion="Ensure all required parts are explicitly labelled in the outline.",
                ))

        # 5. Mandatory requirement coverage
        mandatory = [r for r in requirements if r.mandatory]
        uncovered = []
        for req in mandatory:
            key_words = re.findall(r"[a-z]{4,}", req.text.lower())[:4]
            if not any(w in combined_doc for w in key_words):
                uncovered.append(req)

        if uncovered:
            flags.append(ValidationFlag(
                severity="major" if len(uncovered) <= 2 else "critical",
                section="Requirements Matrix",
                description=(
                    f"{len(uncovered)} mandatory requirement(s) may not be addressed: "
                    + ", ".join(r.ref for r in uncovered[:5])
                ),
                suggestion="Add a requirements matrix mapping each instruction to a document section.",
            ))

        return flags


# ---------------------------------------------------------------------------
# Layer 2 — AI semantic gap check
# ---------------------------------------------------------------------------

class AIValidationChecker:
    """
    Uses Claude to find narrative gaps missed by deterministic rules.
    Optional — falls back gracefully if Bedrock is not configured.
    """

    def __init__(self, bedrock_client=None):
        self.bedrock = bedrock_client

    def check(
        self,
        input_text: str,
        outline: str,
        draft: str,
    ) -> List[ValidationFlag]:
        """Return AI-identified validation flags."""
        if self.bedrock is None:
            return []

        from app.ai.bedrock_client import build_validation_prompt

        system, user = build_validation_prompt(input_text, outline, draft)
        try:
            raw = self.bedrock.invoke(system, user, max_tokens=2048, temperature=0.1)
            return parse_flags(raw)
        except Exception as exc:
            logger.warning("AI validation check failed: %s", exc)
            return []


def parse_flags(text: str) -> List[ValidationFlag]:
    """Parse `[SEVERITY] Section — Description — Fix` lines from model output."""
    pattern = re.compile(
        r"\[(CRITICAL|MAJOR|MINOR)\]\s+([^\n—]+?)—\s*([^\n—]+?)—\s*([^\n]+)",
        re.IGNORECASE,
    )
    return [
        ValidationFlag(
            severity=severity.lower(),
            section=section.strip(),
            description=desc.strip(),
            suggestion=suggestion.strip(),
        )
        for severity, section, desc, suggestion in pattern.findall(text)
    ]


# ---------------------------------------------------------------------------
# Unified validation service
# ---------------------------------------------------------------------------

class ValidationService:
    """
    Combines deterministic + AI checks and produces the final report.
    Instantiate once; call check() per session.
    """

    def __init__(self, bedrock_client=None):
        self.deterministic = DeterministicChecker()
        self.ai_checker    = AIValidationChecker(bedrock_client)

    def check(
        self,
        input_text: str,
        outline: str,
        highlights: str,
        draft: str,
        requirements: List[RequirementItem],
        attempt: int = 1,
    ) -> Tuple[str, List[ValidationFlag]]:
        """
        Run the full validation check.

        Returns:
            (report_text, flags)
        """
        det_flags = self.deterministic.check(input_text, outline, draft, requirements)
        ai_flags  = self.ai_checker.check(input_text, outline, draft)

        # Deduplicate by description fingerprint
        seen: set[str] = set()
        unique_flags: List[ValidationFlag] = []
        for f in det_flags + ai_flags:
            key = f.description[:60].lower()
            if key not in seen:
                seen.add(key)
                unique_flags.append(f)

        critical = [f for f in unique_flags if f.severity == "critical"]
        major    = [f for f in unique_flags if f.severity == "major"]
        minor    = [f for f in unique_flags if f.severity == "minor"]

        flag_lines = "\n\n".join(
            f"[{f.severity.upper()}]  {f.section}\n"
            f"  Issue:      {f.description}\n"
            f"  Suggestion: {f.suggestion}"
            for f in unique_flags
        ) or "  None identified."

        overall = (
            "REVIEW REQUIRED — Critical/major flags must be addressed before release."
            if critical or major
            else "PASS — No critical or major gaps detected."
        )

        report = (
            f"QUALITY VALIDATION REPORT\n"
            f"=========================\n"
            f"Attempt #{attempt}  |  Layer 1 (deterministic): {len(det_flags)} flags  "
            f"|  Layer 2 (AI): {len(ai_flags)} flags\n\n"
            f"SUMMARY:  Critical={len(critical)}  Major={len(major)}  Minor={len(minor)}\n\n"
            f"FLAGS\n"
            f"-----\n"
            f"{flag_lines}\n\n"
            f"OVERALL ASSESSMENT\n"
            f"------------------\n"
            f"{overall}\n\n"
            f"NOTE: This automated check is advisory only. A qualified human\n"
            f"reviewer must perform the final review before release."
        )

        return report, unique_flags
