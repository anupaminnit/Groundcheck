"""``HybridVerifier``: run NLI on everything, escalate only low-confidence claims to
the LLM judge.

See ``docs/SPEC.md`` §5.5. A claim's NLI confidence *is* the "winning probability"
regardless of which verdict it produced — ``NLIVerifier`` always sets
``confidence = max(entailment, contradiction)`` for the winning candidate — so
escalation is just a band check on ``verdict.confidence``.
"""

from __future__ import annotations

from groundcheck.core.schemas import Claim, ClaimVerdict, Evidence, TokenUsage
from groundcheck.core.verifier.base import Verifier

_Pair = tuple[Claim, list[Evidence]]


class HybridVerifier:
    """NLI-first verification; claims in the escalation band get a second LLM pass."""

    def __init__(
        self,
        nli_verifier: Verifier,
        llm_verifier: Verifier,
        escalation_band: tuple[float, float],
    ) -> None:
        """Initialize the verifier.

        Args:
            nli_verifier: Runs first, against every claim.
            llm_verifier: Runs second, only against claims in ``escalation_band``.
            escalation_band: Inclusive ``(low, high)`` confidence range that
                triggers escalation to ``llm_verifier``.
        """
        self._nli = nli_verifier
        self._llm = llm_verifier
        self._low, self._high = escalation_band

    async def verify(
        self, pairs: list[_Pair], question: str
    ) -> tuple[list[ClaimVerdict], TokenUsage]:
        """Judge each claim: NLI first, then escalate in-band claims to the LLM.

        Args:
            pairs: One ``(claim, candidates)`` tuple per claim to judge.
            question: The question the answer responds to, if any. Only used by
                the LLM verifier for escalated claims.

        Returns:
            A tuple of one ``ClaimVerdict`` per input pair — the LLM's verdict for
            escalated claims, the NLI verdict otherwise — and the combined token
            usage of both verifiers.
        """
        if not pairs:
            return [], TokenUsage()

        nli_verdicts, nli_tokens = await self._nli.verify(pairs, question)
        escalate_indices = [
            i for i, v in enumerate(nli_verdicts) if self._low <= v.confidence <= self._high
        ]
        if not escalate_indices:
            return nli_verdicts, nli_tokens

        escalate_pairs = [pairs[i] for i in escalate_indices]
        llm_verdicts, llm_tokens = await self._llm.verify(escalate_pairs, question)

        result = list(nli_verdicts)
        for idx, llm_verdict in zip(escalate_indices, llm_verdicts, strict=True):
            result[idx] = llm_verdict
        return result, nli_tokens + llm_tokens
