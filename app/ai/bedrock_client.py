"""
Amazon Bedrock Client — Claude Integration
==========================================
Wraps boto3 Bedrock Runtime with:
  - Retry logic with exponential backoff
  - Token usage logging
  - LiteLLM proxy routing (when configured)
  - Structured prompt building for each pipeline step

All calls are synchronous; wrap in asyncio.to_thread() for async endpoints.

Prerequisites:
    pip install boto3 anthropic litellm
    AWS credentials configured via env or IAM role.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy imports — only fail at runtime if the package isn't installed
# ---------------------------------------------------------------------------

def _get_boto3():
    try:
        import boto3
        return boto3
    except ImportError:
        raise RuntimeError(
            "boto3 is required for Bedrock integration. "
            "Run: pip install boto3"
        )


def _get_litellm():
    try:
        import litellm
        return litellm
    except ImportError:
        raise RuntimeError(
            "litellm is required when use_litellm_proxy=True. "
            "Run: pip install litellm"
        )


# ---------------------------------------------------------------------------
# Bedrock client
# ---------------------------------------------------------------------------

class BedrockClient:
    """
    Thread-safe client for Claude via Amazon Bedrock.

    Usage:
        client = BedrockClient(settings)
        response = client.invoke(system_prompt, user_message)
    """

    def __init__(self, settings):
        from config import Settings
        self.settings: Settings = settings
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return
        boto3 = _get_boto3()

        session_kwargs: dict[str, Any] = {"region_name": self.settings.aws_region}
        if self.settings.aws_access_key_id:
            session_kwargs["aws_access_key_id"]     = self.settings.aws_access_key_id
            session_kwargs["aws_secret_access_key"] = self.settings.aws_secret_access_key
        if self.settings.aws_session_token:
            session_kwargs["aws_session_token"] = self.settings.aws_session_token

        session = boto3.Session(**session_kwargs)
        self._client = session.client("bedrock-runtime")
        logger.info("Bedrock client initialized (region=%s, model=%s)",
                    self.settings.aws_region, self.settings.bedrock_model_id)

    def invoke(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        max_retries: int = 3,
    ) -> str:
        """
        Invoke Claude via Bedrock Messages API.

        Args:
            system_prompt:  Role/task instructions.
            user_message:   The actual request (input text, retrieved context, etc.)
            max_tokens:     Override for this call.
            temperature:    Override for this call.
            max_retries:    Retry on throttling / transient errors.

        Returns:
            Generated text string.

        Raises:
            RuntimeError: On persistent failure after retries.
        """
        if self.settings.use_litellm_proxy:
            return self._invoke_via_litellm(system_prompt, user_message, max_tokens, temperature)

        self._ensure_client()

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens or self.settings.bedrock_max_tokens,
            "temperature": temperature if temperature is not None else self.settings.bedrock_temperature,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_message}],
        }

        for attempt in range(1, max_retries + 1):
            try:
                t0 = time.perf_counter()
                response = self._client.invoke_model(
                    modelId=self.settings.bedrock_model_id,
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps(body),
                )
                elapsed = time.perf_counter() - t0
                result = json.loads(response["body"].read())

                text = result["content"][0]["text"]
                usage = result.get("usage", {})
                logger.info(
                    "Bedrock call OK  model=%s  input_tokens=%s  output_tokens=%s  latency=%.2fs",
                    self.settings.bedrock_model_id,
                    usage.get("input_tokens", "?"),
                    usage.get("output_tokens", "?"),
                    elapsed,
                )
                return text

            except Exception as exc:
                err_name = type(exc).__name__
                logger.warning("Bedrock attempt %d/%d failed: %s — %s", attempt, max_retries, err_name, exc)
                if attempt < max_retries:
                    backoff = 2 ** attempt
                    logger.info("Retrying in %ds...", backoff)
                    time.sleep(backoff)
                else:
                    raise RuntimeError(f"Bedrock invocation failed after {max_retries} attempts: {exc}") from exc

        raise RuntimeError("Unreachable")

    def _invoke_via_litellm(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: Optional[int],
        temperature: Optional[float],
    ) -> str:
        """Route through the LiteLLM proxy for centralized logging and policy enforcement."""
        litellm = _get_litellm()

        if self.settings.litellm_proxy_url:
            litellm.api_base = self.settings.litellm_proxy_url
        if self.settings.litellm_api_key:
            litellm.api_key = self.settings.litellm_api_key

        response = litellm.completion(
            model=f"bedrock/{self.settings.bedrock_model_id}",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            max_tokens=max_tokens or self.settings.bedrock_max_tokens,
            temperature=temperature if temperature is not None else self.settings.bedrock_temperature,
        )
        return response.choices[0].message.content


# ---------------------------------------------------------------------------
# System prompt templates
# ---------------------------------------------------------------------------

SYSTEM_WRITER_BASE = """\
You are an expert technical writer helping an organization produce structured,
well-sourced business documents from an input brief.

CRITICAL RULES:
1. Every factual claim must be traceable to the provided knowledge base sources.
2. Never invent identifiers, ratings, figures, or personal names.
3. Flag any claim you cannot ground in provided context with [UNVERIFIED].
4. Write in clear, professional prose. No marketing hyperbole.
5. Follow the exact instructions given — do not exceed the requested scope.

The human reviewer will approve, edit, or reject your output before it is used.
You are an advisory AI, not an approver or decision-maker.
"""

SYSTEM_VALIDATION_CHECK = """\
You are a meticulous document quality reviewer. Your task is to systematically
cross-check document content against the requirements in the input brief and flag gaps.

Output format: structured list of findings with severity (CRITICAL/MAJOR/MINOR),
section reference, description, and suggested remediation. Be conservative —
flag potential issues even if uncertain.

The human reviewer makes all final quality determinations.
"""


def _format_sources(sources: list[dict], use_full_text: bool = True) -> str:
    return "\n\n".join(
        f"SOURCE [{i+1}] — {s.get('title','')}\n"
        f"{s.get('full_text', s.get('snippet','')) if use_full_text else s.get('snippet','')}"
        for i, s in enumerate(sources)
    )


def build_outline_prompt(input_text: str, retrieved_sources: list[dict]) -> tuple[str, str]:
    """Build system + user prompt for outline generation."""
    user = f"""
INPUT DOCUMENT (scope, instructions, evaluation criteria):
==========================================================
{input_text[:6000]}

RETRIEVED KNOWLEDGE BASE SOURCES:
=================================
{_format_sources(retrieved_sources)[:4000]}

TASK:
Generate a structured outline that maps to every instruction in the input document.
Structure: Part I (Approach), Part II (Case Studies), Part III (Cost).
For each section, note which KB source supports it and which instruction it addresses.
Output plain text — no markdown, no headers beyond the outline structure itself.
"""
    return SYSTEM_WRITER_BASE, user


def build_highlights_prompt(input_text: str, outline: str, retrieved_sources: list[dict]) -> tuple[str, str]:
    """Build system + user prompt for key-highlight generation."""
    user = f"""
APPROVED OUTLINE:
=================
{outline[:3000]}

INPUT DOCUMENT EVALUATION CRITERIA:
===================================
{input_text[:2000]}

RETRIEVED CASE STUDY SOURCES:
=============================
{_format_sources(retrieved_sources, use_full_text=False)[:3000]}

TASK:
For each major section of the outline, generate:
1. A one-sentence HIGHLIGHT (the core value statement for that section)
2. A MESSAGE (the specific strength that sets this response apart)
3. EVIDENCE (which KB source supports this claim — cite by SOURCE number)
4. DIFFERENTIATORS: list 3-5 organization-wide strengths that apply across all sections.
Ground every claim in the provided sources. Flag any unsupported claim with [UNVERIFIED].
"""
    return SYSTEM_WRITER_BASE, user


def build_draft_prompt(input_text: str, outline: str, highlights: str, retrieved_sources: list[dict]) -> tuple[str, str]:
    """Build system + user prompt for narrative draft generation."""
    user = f"""
APPROVED OUTLINE:
=================
{outline[:2000]}

APPROVED HIGHLIGHTS:
====================
{highlights[:2000]}

RETRIEVED SOURCES:
==================
{_format_sources(retrieved_sources)[:4000]}

TASK:
Draft the full narrative for Section 2.0 (Approach), sub-sections 2.1, 2.2, 2.3.
Requirements:
- Every factual claim must be cited as [SOURCE N] where N is the source number above.
- Write in a clear, professional third-person voice.
- Target: 2-3 paragraphs per sub-section.
- End with a one-sentence statement that there are no unsourced factual claims.
"""
    return SYSTEM_WRITER_BASE, user


def build_validation_prompt(input_text: str, outline: str, draft: str) -> tuple[str, str]:
    """Build system + user prompt for the quality gap check."""
    user = f"""
INPUT DOCUMENT REQUIREMENTS:
============================
{input_text[:4000]}

DOCUMENT CONTENT (outline + draft):
===================================
OUTLINE:
{outline[:2000]}

DRAFT:
{draft[:2000]}

TASK:
Perform a systematic requirements cross-check. For each instruction:
1. Determine if it is addressed in the document content.
2. If addressed — note where.
3. If not addressed — flag as CRITICAL (mandatory) or MAJOR (important).
Also check: page limits, font/margin specs, part structure, case study count.
Output each finding as: [SEVERITY] Section — Description — Suggested fix.
"""
    return SYSTEM_VALIDATION_CHECK, user
