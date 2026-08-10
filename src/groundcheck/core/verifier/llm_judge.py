"""``LLMJudgeVerifier``: a single batched LLM call that judges all claims against
their evidence.

See ``docs/SPEC.md`` §5.3. Claims are chunked into batches of at most
``chunk_size`` (default 20) to bound context size; each chunk is one call, with one
JSON-repair retry on invalid output. A chunk that's still invalid after the retry
raises ``VerifierError``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from groundcheck.core.errors import VerifierError
from groundcheck.core.prompts import JUDGE_V1
from groundcheck.core.schemas import Claim, ClaimVerdict, Evidence, TokenUsage, Verdict
from groundcheck.providers.base import LLMProvider

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_REPAIR_SUFFIX = (
    "\n\nYour previous response was not a valid JSON array. Return ONLY a valid "
    "JSON array, with no prose and no code fences."
)

_Pair = tuple[Claim, list[Evidence]]


class _JudgeItem(BaseModel):
    claim_id: str
    verdict: Verdict
    confidence: float
    evidence_ids: list[str] = Field(default_factory=list)
    rationale: str = ""


class LLMJudgeVerifier:
    """A single batched LLM judge call per chunk of at most ``chunk_size`` claims."""

    def __init__(self, provider: LLMProvider, timeout: float = 30.0, chunk_size: int = 20) -> None:
        """Initialize the verifier.

        Args:
            provider: The LLM provider used for judge calls.
            timeout: Per-call timeout, in seconds.
            chunk_size: Maximum claims judged per call; larger claim lists are
                split into multiple chunked calls and merged.
        """
        self._provider = provider
        self._timeout = timeout
        self._chunk_size = chunk_size

    async def verify(
        self, pairs: list[_Pair], question: str
    ) -> tuple[list[ClaimVerdict], TokenUsage]:
        """Judge each claim against its candidate evidence, chunked into batches
        of at most ``chunk_size``.

        Args:
            pairs: One ``(claim, candidates)`` tuple per claim to judge.
            question: The question the answer responds to, if any.

        Returns:
            A tuple of one ``ClaimVerdict`` per input pair and the total token
            usage across all judge calls (including any repair retries).

        Raises:
            VerifierError: A chunk's judge output was still invalid JSON after
                one repair retry.
        """
        if not pairs:
            return [], TokenUsage()

        verdicts: list[ClaimVerdict] = []
        tokens = TokenUsage()
        for chunk in _chunks(pairs, self._chunk_size):
            chunk_verdicts, chunk_tokens = await self._verify_chunk(chunk, question)
            verdicts.extend(chunk_verdicts)
            tokens = tokens + chunk_tokens
        return verdicts, tokens

    async def _verify_chunk(
        self, chunk: list[_Pair], question: str
    ) -> tuple[list[ClaimVerdict], TokenUsage]:
        user = _build_user_prompt(question, chunk)
        raw, tokens = await self._provider.complete_json(JUDGE_V1, user, self._timeout)

        try:
            items = _parse_judge_json(raw)
        except (ValueError, ValidationError):
            raw2, retry_tokens = await self._provider.complete_json(
                JUDGE_V1, user + _REPAIR_SUFFIX, self._timeout
            )
            tokens = tokens + retry_tokens
            try:
                items = _parse_judge_json(raw2)
            except (ValueError, ValidationError) as exc:
                raise VerifierError(
                    f"LLM judge returned invalid JSON after a repair retry: {exc}"
                ) from exc

        return _verdicts_from_items(chunk, items), tokens


def _chunks(pairs: list[_Pair], size: int) -> Iterator[list[_Pair]]:
    for i in range(0, len(pairs), size):
        yield pairs[i : i + size]


def _build_user_prompt(question: str, chunk: list[_Pair]) -> str:
    lines = [f"QUESTION:\n{question or '(none provided)'}", "", "CLAIMS:"]
    for i, (claim, candidates) in enumerate(chunk, start=1):
        lines.append(f"{i}. [{claim.id}] {claim.text}")
        if candidates:
            for ev in candidates:
                lines.append(f"   - evidence[{ev.id}]: {ev.text}")
        else:
            lines.append("   (no candidate evidence)")
    return "\n".join(lines)


def _parse_judge_json(raw: str) -> list[_JudgeItem]:
    data = json.loads(_strip_code_fences(raw))
    if not isinstance(data, list):
        raise ValueError("Expected a JSON array of judge items.")
    return TypeAdapter(list[_JudgeItem]).validate_python(data)


def _strip_code_fences(raw: str) -> str:
    return _CODE_FENCE_RE.sub("", raw.strip()).strip()


def _verdicts_from_items(chunk: list[_Pair], items: list[_JudgeItem]) -> list[ClaimVerdict]:
    by_id = {item.claim_id: item for item in items}
    verdicts = []
    for claim, _ in chunk:
        item = by_id.get(claim.id)
        if item is None:
            verdicts.append(
                ClaimVerdict(
                    claim=claim,
                    verdict=Verdict.UNSUPPORTED,
                    confidence=0.0,
                    rationale="Judge returned no verdict for this claim.",
                )
            )
            continue
        verdicts.append(
            ClaimVerdict(
                claim=claim,
                verdict=item.verdict,
                confidence=item.confidence,
                evidence_ids=item.evidence_ids,
                rationale=item.rationale,
            )
        )
    return verdicts
