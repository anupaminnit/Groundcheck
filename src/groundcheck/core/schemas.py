"""Pydantic v2 data models shared across GroundCheck.

This is the single source of truth for all public and internal data structures.
See ``docs/SPEC.md`` §4.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Verdict(str, Enum):
    """Per-claim groundedness verdict.

    Attributes:
        SUPPORTED: The evidence fully supports the claim.
        PARTIALLY_SUPPORTED: The evidence supports part of the claim but not all.
        UNSUPPORTED: The evidence says nothing relevant to the claim.
        CONTRADICTED: The evidence contradicts the claim.
        NOT_A_CLAIM: An opinion, hedge, or meta statement — excluded from scoring.
    """

    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    NOT_A_CLAIM = "NOT_A_CLAIM"


class Evidence(BaseModel):
    """A single retrieved evidence chunk supplied by the caller.

    Attributes:
        id: Caller-supplied identifier, or an auto-generated ``"chunk_{i}"``.
        text: The evidence chunk's text.
        metadata: Arbitrary caller-supplied metadata, unused by GroundCheck itself.
    """

    id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Claim(BaseModel):
    """An atomic, standalone factual claim extracted from an answer.

    Attributes:
        id: Identifier, e.g. ``"claim_{i}"``.
        text: The normalized, standalone claim text.
        span_start: Character offset into the *original* answer where this claim's
            source text begins. Load-bearing: policies use it to annotate/redact.
        span_end: Character offset into the *original* answer where this claim's
            source text ends.
    """

    id: str
    text: str
    span_start: int
    span_end: int


class ClaimVerdict(BaseModel):
    """A claim paired with the verifier's judgement of it.

    Attributes:
        claim: The claim being judged.
        verdict: The groundedness verdict.
        confidence: Confidence in the verdict, in ``[0, 1]``.
        evidence_ids: IDs of the evidence chunks that support or contradict the
            verdict. Empty for ``NOT_A_CLAIM`` and (usually) ``UNSUPPORTED``.
        rationale: A one-line reason for the verdict. Empty for NLI-based verdicts.
    """

    claim: Claim
    verdict: Verdict
    confidence: float
    evidence_ids: list[str] = Field(default_factory=list)
    rationale: str = ""


class Action(str, Enum):
    """The policy-decided transformation applied to the answer.

    Attributes:
        PASS: The answer is unchanged.
        ANNOTATE: Inline markers were inserted after flagged claim spans.
        REDACT: Flagged claim spans were removed from the answer.
        BLOCK: The whole answer was replaced with a fallback message.
        ERROR: Verification itself failed; the original answer was returned as-is.
    """

    PASS = "pass"
    ANNOTATE = "annotate"
    REDACT = "redact"
    BLOCK = "block"
    ERROR = "error"


class TokenUsage(BaseModel):
    """Token accounting across all LLM calls made during a single check.

    Attributes:
        input: Total input (prompt) tokens across all LLM calls.
        output: Total output (completion) tokens across all LLM calls.
    """

    input: int = 0
    output: int = 0

    def __add__(self, other: TokenUsage) -> TokenUsage:
        """Return a new ``TokenUsage`` with both operands' counts summed."""
        return TokenUsage(input=self.input + other.input, output=self.output + other.output)


class GuardReport(BaseModel):
    """Full structured result of a single ``Guard.check()``/``Guard.acheck()`` call.

    Attributes:
        grounded_score: Overall groundedness score in ``[0, 1]``, or ``-1`` when
            ``action`` is ``ERROR``. See ``docs/SPEC.md`` §7 for the scoring rule.
        action: The policy-decided transformation actually applied.
        safe_answer: The answer after that transformation. Equal to the original
            answer when ``action`` is ``PASS`` or ``ERROR``.
        claims: Per-claim verdicts.
        verifier: Which verifier produced this report — ``"llm"``, ``"local"``, or
            ``"hybrid"``.
        latency_ms: Wall-clock time the whole check took, in milliseconds.
        tokens: Token usage across all LLM calls made during the check.
        error: The error message when ``action`` is ``ERROR``; ``None`` otherwise.
    """

    grounded_score: float
    action: Action
    safe_answer: str
    claims: list[ClaimVerdict]
    verifier: str
    latency_ms: int
    tokens: TokenUsage
    error: str | None = None
