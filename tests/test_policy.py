"""Tests for core.policy.PolicyEngine: log, annotate, redact, and block.

Offset math for annotate (markers must never split a word) and the redact ≥60%
escalation rule are the highest-risk behavior here (per docs/SPEC.md §10), so they
get dedicated coverage.
"""

from __future__ import annotations

from groundcheck.config import GuardConfig
from groundcheck.core.policy import DEFAULT_BLOCK_FALLBACK_MESSAGE, PolicyEngine
from groundcheck.core.schemas import Action, Claim, ClaimVerdict, Verdict


def _claim(text: str, span_start: int, span_end: int, i: int = 0) -> Claim:
    return Claim(id=f"claim_{i}", text=text, span_start=span_start, span_end=span_end)


def _verdict(v: Verdict, claim: Claim | None = None) -> ClaimVerdict:
    claim = claim or _claim("x", 0, 1)
    return ClaimVerdict(claim=claim, verdict=v, confidence=0.5)


# --- log -------------------------------------------------------------------


def test_log_policy_passes_through_unchanged() -> None:
    engine = PolicyEngine(GuardConfig(policy="log"))

    safe_answer, action = engine.apply(
        "original answer", [_verdict(Verdict.CONTRADICTED)], grounded_score=0.0
    )

    assert safe_answer == "original answer"
    assert action == Action.PASS


# --- annotate ----------------------------------------------------------------


def test_annotate_inserts_markers_at_correct_offsets() -> None:
    answer = "Paris is the capital of France. It has 5 million people."
    sentence1 = "Paris is the capital of France."
    sentence2 = "It has 5 million people."
    start1 = answer.index(sentence1)
    end1 = start1 + len(sentence1)
    start2 = answer.index(sentence2)
    end2 = start2 + len(sentence2)

    claim1 = _claim(sentence1, start1, end1, 0)
    claim2 = _claim(sentence2, start2, end2, 1)
    verdicts = [
        _verdict(Verdict.UNSUPPORTED, claim1),
        _verdict(Verdict.CONTRADICTED, claim2),
    ]
    engine = PolicyEngine(GuardConfig(policy="annotate"))

    safe_answer, action = engine.apply(answer, verdicts, grounded_score=0.0)

    assert action == Action.ANNOTATE
    assert safe_answer == (
        "Paris is the capital of France. ⚠[unverified] "
        "It has 5 million people. ⚠[contradicted by sources]"
    )


def test_annotate_marker_never_splits_a_word() -> None:
    answer = "Supercalifragilisticexpialidocious is a long word."
    # Deliberately end the claim span mid-word.
    claim = _claim("Supercalif", 0, len("Supercalif"))
    verdict = _verdict(Verdict.UNSUPPORTED, claim)
    engine = PolicyEngine(GuardConfig(policy="annotate"))

    safe_answer, _ = engine.apply(answer, [verdict], grounded_score=0.0)

    marker_pos = safe_answer.index("⚠")
    before_marker = safe_answer[:marker_pos].rstrip()
    assert before_marker == "Supercalifragilisticexpialidocious"


def test_annotate_with_no_flagged_verdicts_passes_through() -> None:
    answer = "Everything checks out."
    claim = _claim(answer, 0, len(answer))
    verdict = _verdict(Verdict.SUPPORTED, claim)
    engine = PolicyEngine(GuardConfig(policy="annotate"))

    safe_answer, action = engine.apply(answer, [verdict], grounded_score=1.0)

    assert action == Action.PASS
    assert safe_answer == answer


# --- redact ------------------------------------------------------------------


def test_redact_removes_flagged_span_and_cleans_whitespace() -> None:
    answer = (
        "Paris is the capital of France. It has a very well documented and "
        "consistently reported population of roughly two million residents "
        "within the city limits, according to multiple census sources."
    )
    sentence1 = "Paris is the capital of France."
    start1 = answer.index(sentence1)
    end1 = start1 + len(sentence1)
    claim = _claim(sentence1, start1, end1)
    verdict = _verdict(Verdict.CONTRADICTED, claim)
    engine = PolicyEngine(GuardConfig(policy="redact"))

    safe_answer, action = engine.apply(answer, [verdict], grounded_score=0.5)

    assert action == Action.REDACT
    assert sentence1 not in safe_answer
    assert "  " not in safe_answer


def test_redact_drops_orphaned_connector() -> None:
    answer = "Paris is nice. However, it rains a lot in winter. It gets cold too."
    # Deliberately excludes the leading connector and the trailing period, so
    # removing it leaves the documented "However, ." orphan behind.
    fragment = "it rains a lot in winter"
    start = answer.index(fragment)
    end = start + len(fragment)
    claim = _claim(fragment, start, end)
    verdict = _verdict(Verdict.UNSUPPORTED, claim)
    engine = PolicyEngine(GuardConfig(policy="redact"))

    safe_answer, action = engine.apply(answer, [verdict], grounded_score=0.5)

    assert action == Action.REDACT
    assert "However" not in safe_answer
    assert "Paris is nice." in safe_answer
    assert "It gets cold too." in safe_answer


def test_redact_escalates_to_block_when_removal_exceeds_60_percent() -> None:
    answer = "This whole sentence is wrong."
    claim = _claim(answer, 0, len(answer))
    verdict = _verdict(Verdict.CONTRADICTED, claim)
    engine = PolicyEngine(GuardConfig(policy="redact"))

    safe_answer, action = engine.apply(answer, [verdict], grounded_score=0.0)

    assert action == Action.BLOCK
    assert safe_answer == DEFAULT_BLOCK_FALLBACK_MESSAGE


def test_redact_with_no_flagged_claims_passes_through() -> None:
    answer = "Everything checks out."
    claim = _claim(answer, 0, len(answer))
    verdict = _verdict(Verdict.SUPPORTED, claim)
    engine = PolicyEngine(GuardConfig(policy="redact"))

    safe_answer, action = engine.apply(answer, [verdict], grounded_score=1.0)

    assert action == Action.PASS
    assert safe_answer == answer


# --- block -------------------------------------------------------------------


def test_block_replaces_answer_below_threshold() -> None:
    engine = PolicyEngine(GuardConfig(policy="block", block_threshold=0.7))

    safe_answer, action = engine.apply("some answer", [], grounded_score=0.5)

    assert action == Action.BLOCK
    assert safe_answer == DEFAULT_BLOCK_FALLBACK_MESSAGE


def test_block_passes_through_at_or_above_threshold() -> None:
    engine = PolicyEngine(GuardConfig(policy="block", block_threshold=0.7))

    safe_answer, action = engine.apply("some answer", [], grounded_score=0.7)

    assert action == Action.PASS
    assert safe_answer == "some answer"


def test_block_uses_custom_fallback_message() -> None:
    engine = PolicyEngine(GuardConfig(policy="block"), block_fallback_message="custom message")

    safe_answer, action = engine.apply("some answer", [], grounded_score=0.0)

    assert action == Action.BLOCK
    assert safe_answer == "custom message"
