"""``PolicyEngine``: transform an answer given claim verdicts, per the configured policy.

See ``docs/SPEC.md`` §5.6. ``log``, ``annotate``, ``redact``, and ``block`` are all
implemented. Which verdicts count as "flagged" for both ``annotate`` and ``redact``
is controlled by the single ``config.redact_verdicts`` field.
"""

from __future__ import annotations

import re

from groundcheck.config import GuardConfig
from groundcheck.core.errors import ConfigError
from groundcheck.core.schemas import Action, Claim, ClaimVerdict, Verdict

DEFAULT_BLOCK_FALLBACK_MESSAGE = "I couldn't verify this answer against the available sources."

_REDACT_ESCALATION_FRACTION = 0.6

_MARKERS: dict[Verdict, str] = {
    Verdict.UNSUPPORTED: " ⚠[unverified]",
    Verdict.CONTRADICTED: " ⚠[contradicted by sources]",
}

_CONNECTOR_WORDS = (
    "however", "additionally", "moreover", "furthermore", "also",
    "meanwhile", "therefore", "thus", "consequently",
)
_ORPHAN_CONNECTOR_RE = re.compile(
    r"\b(?:" + "|".join(_CONNECTOR_WORDS) + r"),?\s*\.+", re.IGNORECASE
)
_WHITESPACE_RE = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([.,!?])")
_REPEATED_PUNCT_RE = re.compile(r"([.,!?])\1+")


class PolicyEngine:
    """Applies ``config.policy`` to produce a safe answer and an action."""

    def __init__(
        self,
        config: GuardConfig,
        block_fallback_message: str = DEFAULT_BLOCK_FALLBACK_MESSAGE,
    ) -> None:
        """Initialize the engine.

        Args:
            config: Supplies ``policy``, ``redact_verdicts``, and
                ``block_threshold``.
            block_fallback_message: The replacement text used by the ``block``
                policy (and by ``redact``'s escalate-to-block rule).
        """
        self._config = config
        self._block_fallback_message = block_fallback_message

    def apply(
        self, answer: str, verdicts: list[ClaimVerdict], grounded_score: float
    ) -> tuple[str, Action]:
        """Apply ``config.policy`` to produce a safe answer and an action.

        Args:
            answer: The original answer text.
            verdicts: Per-claim verdicts for that answer.
            grounded_score: The answer's overall groundedness score, used by the
                ``block`` policy.

        Returns:
            A tuple of the transformed (or unchanged) answer and the action that
            was actually taken.
        """
        policy = self._config.policy
        if policy == "log":
            return answer, Action.PASS
        if policy == "annotate":
            return self._apply_annotate(answer, verdicts)
        if policy == "redact":
            return self._apply_redact(answer, verdicts)
        if policy == "block":
            return self._apply_block(answer, grounded_score)
        # Unreachable via GuardConfig's Literal-typed field; defensive only.
        raise ConfigError(f"Unknown policy: {policy!r}")

    def _apply_annotate(
        self, answer: str, verdicts: list[ClaimVerdict]
    ) -> tuple[str, Action]:
        flagged = [v for v in verdicts if v.verdict in self._config.redact_verdicts]
        if not flagged:
            return answer, Action.PASS

        flagged.sort(key=lambda v: v.claim.span_start, reverse=True)
        result = answer
        for verdict in flagged:
            pos = _word_boundary_after(result, verdict.claim.span_end)
            marker = _MARKERS.get(verdict.verdict, f" ⚠[{verdict.verdict.value.lower()}]")
            result = result[:pos] + marker + result[pos:]
        return result, Action.ANNOTATE

    def _apply_redact(self, answer: str, verdicts: list[ClaimVerdict]) -> tuple[str, Action]:
        flagged_claims = [v.claim for v in verdicts if v.verdict in self._config.redact_verdicts]
        if not flagged_claims:
            return answer, Action.PASS

        removed_chars = sum(c.span_end - c.span_start for c in flagged_claims)
        if answer and removed_chars / len(answer) > _REDACT_ESCALATION_FRACTION:
            return self._block_answer()

        redacted = _remove_spans(answer, flagged_claims)
        redacted = _clean_redacted_text(redacted)
        return redacted, Action.REDACT

    def _apply_block(self, answer: str, grounded_score: float) -> tuple[str, Action]:
        if grounded_score < self._config.block_threshold:
            return self._block_answer()
        return answer, Action.PASS

    def _block_answer(self) -> tuple[str, Action]:
        return self._block_fallback_message, Action.BLOCK


def _word_boundary_after(text: str, pos: int) -> int:
    while pos < len(text) and text[pos].isalnum():
        pos += 1
    return pos


def _remove_spans(answer: str, claims: list[Claim]) -> str:
    result = answer
    for claim in sorted(claims, key=lambda c: c.span_start, reverse=True):
        result = result[: claim.span_start] + result[claim.span_end :]
    return result


def _clean_redacted_text(text: str) -> str:
    text = _ORPHAN_CONNECTOR_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text)
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    text = _REPEATED_PUNCT_RE.sub(r"\1", text)
    return text.strip()
