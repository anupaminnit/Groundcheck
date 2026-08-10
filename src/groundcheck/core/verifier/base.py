"""``Verifier`` protocol: the common interface for all verifier backends.

See ``docs/SPEC.md`` §2.
"""

from __future__ import annotations

from typing import Protocol

from groundcheck.core.schemas import Claim, ClaimVerdict, Evidence, TokenUsage


class Verifier(Protocol):
    """Turns (claim, candidate evidence) pairs into verdicts."""

    async def verify(
        self, pairs: list[tuple[Claim, list[Evidence]]], question: str
    ) -> tuple[list[ClaimVerdict], TokenUsage]:
        """Judge each claim against its candidate evidence.

        Args:
            pairs: One ``(claim, candidates)`` tuple per claim to judge.
            question: The question the answer responds to, if any. Used as
                context by LLM-based verifiers; ignored by NLI-based ones.

        Returns:
            A tuple of one ``ClaimVerdict`` per input pair (same order) and the
            token usage spent judging them (zero for local/NLI verification).
        """
        ...
