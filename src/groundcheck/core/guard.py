"""``Guard``: the orchestration facade tying extraction, matching, verification, and
policy together.

``Guard.acheck()`` is the real async implementation; ``Guard.check()`` is a sync
wrapper that raises if called from inside a running event loop. When
``config.fail_open`` (default ``True``), any exception during ``acheck()`` is caught
and returned as a ``GuardReport(action=Action.ERROR, ...)`` instead of propagating —
verification failures must never crash the caller's pipeline. See ``docs/SPEC.md``
§2 and §5.7.

All three verifiers (``llm``/``local``/``hybrid``) and all four policies
(``log``/``annotate``/``redact``/``block``) are supported as of Phase 3.
``local`` mode needs no provider at all — extraction is ``SentenceClaimExtractor``,
matching is lexical, and verification is a local NLI cross-encoder.

Note: ``provider=`` here is reserved for an ``LLMProvider`` *instance* (mainly for
test injection). To select a provider *by name* with other overrides, pass
``config=GuardConfig(provider="openai", ...)`` instead of flattened kwargs — the
name `provider` can't do both jobs at once.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections.abc import Callable, Sequence
from typing import Any

from groundcheck.config import GuardConfig
from groundcheck.core.claims import ClaimExtractor, LLMClaimExtractor, SentenceClaimExtractor
from groundcheck.core.errors import ConfigError
from groundcheck.core.evidence import EmbeddingMatcher, EvidenceMatcher, LexicalMatcher
from groundcheck.core.policy import DEFAULT_BLOCK_FALLBACK_MESSAGE, PolicyEngine
from groundcheck.core.schemas import (
    Action,
    ClaimVerdict,
    Evidence,
    GuardReport,
    TokenUsage,
    Verdict,
)
from groundcheck.core.verifier.base import Verifier
from groundcheck.core.verifier.hybrid import HybridVerifier
from groundcheck.core.verifier.llm_judge import LLMJudgeVerifier
from groundcheck.core.verifier.nli_local import NLIVerifier
from groundcheck.providers import build_provider
from groundcheck.providers.base import LLMProvider

logger = logging.getLogger(__name__)

_SCORE_WEIGHTS = {
    Verdict.SUPPORTED: 1.0,
    Verdict.PARTIALLY_SUPPORTED: 0.5,
    Verdict.UNSUPPORTED: 0.0,
    Verdict.CONTRADICTED: 0.0,
}

EvidenceInput = str | Evidence | dict[str, Any]
OnBlockCallback = Callable[[GuardReport], Any]


class Guard:
    """Facade: extract claims, match evidence, verify, apply policy, report."""

    def __init__(
        self,
        *,
        verifier: str = "llm",
        policy: str = "log",
        provider: LLMProvider | None = None,
        config: GuardConfig | None = None,
        on_block: OnBlockCallback | None = None,
        block_fallback_message: str = DEFAULT_BLOCK_FALLBACK_MESSAGE,
        **config_overrides: Any,
    ) -> None:
        """Construct a ``Guard``, wiring up extraction, matching, and verification
        for ``config.verifier``.

        Args:
            verifier: Shorthand for ``config.verifier`` when ``config`` isn't
                given. Ignored if ``config`` is passed.
            policy: Shorthand for ``config.policy`` when ``config`` isn't given.
                Ignored if ``config`` is passed.
            provider: An ``LLMProvider`` *instance* to use instead of building one
                from ``config`` (mainly for test injection). Not a provider name —
                see the module docstring for how to select one by name.
            config: A fully-built ``GuardConfig``. If given, ``verifier``/
                ``policy``/``config_overrides`` are ignored.
            on_block: Called with the ``GuardReport`` whenever ``action`` comes
                out as ``BLOCK`` — e.g. to trigger regeneration. May be sync or
                async; exceptions from it are logged and otherwise ignored.
            block_fallback_message: Replacement text used by the ``block`` policy.
            **config_overrides: Any other ``GuardConfig`` field, forwarded to its
                constructor when ``config`` isn't given.

        Raises:
            ConfigError: ``config.verifier`` needs a provider and none was given
                or configured; ``verifier="local"``/``"hybrid"`` needs the
                ``[local]`` extra and it isn't installed; or ``config.verifier``
                is invalid.
        """
        self.config = config or GuardConfig(verifier=verifier, policy=policy, **config_overrides)  # type: ignore[arg-type]

        self._claim_extractor: ClaimExtractor
        self._evidence_matcher: EvidenceMatcher
        self._verifier: Verifier

        if self.config.verifier == "llm":
            llm_provider = provider if provider is not None else build_provider(self.config)
            self._claim_extractor = LLMClaimExtractor(llm_provider, timeout=self.config.timeout_s)
            self._evidence_matcher = EmbeddingMatcher(llm_provider)
            self._verifier = LLMJudgeVerifier(llm_provider, timeout=self.config.timeout_s)
        elif self.config.verifier == "local":
            self._claim_extractor = SentenceClaimExtractor()
            self._evidence_matcher = LexicalMatcher()
            self._verifier = NLIVerifier()
        elif self.config.verifier == "hybrid":
            llm_provider = provider if provider is not None else build_provider(self.config)
            self._claim_extractor = LLMClaimExtractor(llm_provider, timeout=self.config.timeout_s)
            self._evidence_matcher = EmbeddingMatcher(llm_provider)
            nli_verifier = NLIVerifier()
            llm_judge = LLMJudgeVerifier(llm_provider, timeout=self.config.timeout_s)
            self._verifier = HybridVerifier(
                nli_verifier, llm_judge, self.config.hybrid_escalation_band
            )
        else:
            raise ConfigError(f"Unknown verifier: {self.config.verifier!r}")

        self._policy_engine = PolicyEngine(
            self.config, block_fallback_message=block_fallback_message
        )
        self._on_block = on_block

    async def acheck(
        self,
        answer: str,
        evidence: Sequence[EvidenceInput],
        question: str = "",
    ) -> GuardReport:
        """Run the full pipeline: extract claims, match evidence, verify, apply
        the policy, and assemble a report.

        Never raises when ``config.fail_open`` (the default) — verification
        failures come back as ``GuardReport(action="error", ...)`` instead.

        Args:
            answer: The generated answer to verify.
            evidence: The retrieved evidence it should be grounded in — strings,
                ``Evidence`` instances, or ``{"id": ..., "text": ...}`` dicts.
            question: The question the answer responds to, if any. Passed to the
                verifier as context (ignored by NLI-based verification).

        Returns:
            The full structured result of the check.

        Raises:
            Exception: Whatever the pipeline raised, only when
                ``config.fail_open`` is ``False``.
        """
        start = time.perf_counter()
        try:
            evidence_list = _normalize_evidence(evidence)
            claims, extractor_tokens = await self._claim_extractor.extract(answer)

            if claims:
                candidates = await self._evidence_matcher.match(
                    claims, evidence_list, self.config.top_k_evidence
                )
                verdicts, verifier_tokens = await self._verifier.verify(
                    list(zip(claims, candidates, strict=True)), question
                )
            else:
                verdicts, verifier_tokens = [], TokenUsage()

            tokens = extractor_tokens + verifier_tokens
            score = _grounded_score(verdicts)
            safe_answer, action = self._policy_engine.apply(answer, verdicts, score)

            report = GuardReport(
                grounded_score=score,
                action=action,
                safe_answer=safe_answer,
                claims=verdicts,
                verifier=self.config.verifier,
                latency_ms=_elapsed_ms(start),
                tokens=tokens,
            )
        except Exception as exc:
            if not self.config.fail_open:
                raise
            return GuardReport(
                grounded_score=-1.0,
                action=Action.ERROR,
                safe_answer=answer,
                claims=[],
                verifier=self.config.verifier,
                latency_ms=_elapsed_ms(start),
                tokens=TokenUsage(),
                error=str(exc),
            )

        if report.action == Action.BLOCK and self._on_block is not None:
            await _invoke_on_block(self._on_block, report)
        return report

    def check(
        self,
        answer: str,
        evidence: Sequence[EvidenceInput],
        question: str = "",
    ) -> GuardReport:
        """Sync wrapper around ``acheck()``. Same args, same return value.

        Args:
            answer: The generated answer to verify.
            evidence: The retrieved evidence it should be grounded in.
            question: The question the answer responds to, if any.

        Returns:
            The full structured result of the check.

        Raises:
            RuntimeError: Called from inside a running event loop — use
                ``await guard.acheck(...)`` there instead.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.acheck(answer, evidence, question))
        raise RuntimeError(
            "Guard.check() cannot be called from inside a running event loop; "
            "use `await guard.acheck(...)` instead."
        )


async def _invoke_on_block(callback: OnBlockCallback, report: GuardReport) -> None:
    # on_block is a side-effecting regeneration hook, not part of verification — a
    # broken callback must not turn an already-successful report into an error.
    try:
        result = callback(report)
        if inspect.isawaitable(result):
            await result
    except Exception:
        logger.warning("on_block callback raised; ignoring.", exc_info=True)


def _elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def _grounded_score(verdicts: list[ClaimVerdict]) -> float:
    scoreable = [v for v in verdicts if v.verdict != Verdict.NOT_A_CLAIM]
    if not scoreable:
        return 1.0
    return sum(_SCORE_WEIGHTS[v.verdict] for v in scoreable) / len(scoreable)


def _normalize_evidence(evidence: Sequence[EvidenceInput]) -> list[Evidence]:
    normalized: list[Evidence] = []
    for i, item in enumerate(evidence):
        if isinstance(item, Evidence):
            normalized.append(item)
        elif isinstance(item, str):
            normalized.append(Evidence(id=f"chunk_{i}", text=item))
        elif isinstance(item, dict):
            data = dict(item)
            data.setdefault("id", f"chunk_{i}")
            normalized.append(Evidence(**data))
        else:
            raise TypeError(f"Unsupported evidence item type: {type(item)!r}")
    return normalized
