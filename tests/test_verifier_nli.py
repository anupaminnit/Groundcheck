"""Tests for core.verifier.nli_local.NLIVerifier.

The scoring logic (``_pick_winner``, ``_verdict_from_scores``) is pure and tested
directly with crafted probabilities — no model, no torch. The lazy-import behavior
is tested against this environment's actual dependency set (the ``[local]`` extra
is not part of ``dev``, so torch/transformers are genuinely absent here — exactly
what the import-time guarantee is supposed to hold). The real-model correctness
test is marked ``@pytest.mark.integration`` and needs ``pip install -e ".[dev,local]"``.
"""

from __future__ import annotations

import sys

import pytest

from groundcheck.core.errors import ConfigError
from groundcheck.core.schemas import Claim, Evidence, Verdict
from groundcheck.core.verifier.nli_local import (
    NLIVerifier,
    _pick_winner,
    _verdict_from_scores,
)

# --- lazy import ---------------------------------------------------------


def test_importing_nli_local_does_not_pull_in_torch() -> None:
    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules


def test_constructing_without_local_extra_raises_config_error() -> None:
    # Check actual installability, not sys.modules membership — these may be
    # installed but not yet imported by anything in this test run.
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("[local] extra is installed; can't test the ImportError path here.")
    with pytest.raises(ConfigError, match=r"\[local\]"):
        NLIVerifier()


# --- pure scoring logic ---------------------------------------------------


def test_verdict_from_scores_supported() -> None:
    verdict, confidence = _verdict_from_scores(entail=0.9, contra=0.05)
    assert verdict == Verdict.SUPPORTED
    assert confidence == 0.9


def test_verdict_from_scores_contradicted() -> None:
    verdict, confidence = _verdict_from_scores(entail=0.1, contra=0.85)
    assert verdict == Verdict.CONTRADICTED
    assert confidence == 0.85


def test_verdict_from_scores_unsupported_below_both_thresholds() -> None:
    verdict, confidence = _verdict_from_scores(entail=0.4, contra=0.3)
    assert verdict == Verdict.UNSUPPORTED
    assert confidence == 0.4


def test_verdict_from_scores_boundary_is_inclusive() -> None:
    verdict, _ = _verdict_from_scores(entail=0.7, contra=0.0)
    assert verdict == Verdict.SUPPORTED


def test_pick_winner_picks_strongest_signal_in_either_direction() -> None:
    # candidate 0: weak entailment; candidate 1: strong contradiction (the winner)
    scores = [(0.4, 0.4, 0.2), (0.1, 0.1, 0.8)]
    winner = _pick_winner(scores)
    assert winner == (1, 0.1, 0.8)


def test_pick_winner_returns_none_for_no_candidates() -> None:
    assert _pick_winner([]) is None


# --- real-model correctness (Phase 3 acceptance criterion) ---------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_nli_verifier_six_fixture_pairs_all_correct() -> None:
    """2 supported, 2 unsupported, 2 contradicted — all must land correctly."""
    verifier = NLIVerifier()

    fixtures = [
        # (claim text, evidence text, expected verdict)
        (
            "Paris is the capital of France.",
            "Paris is the capital of France.",
            Verdict.SUPPORTED,
        ),
        (
            "Water boils at 100 degrees Celsius at sea level.",
            "At sea level, water boils at 100 degrees Celsius.",
            Verdict.SUPPORTED,
        ),
        (
            "The company was founded in 1998.",
            "The team enjoys hiking on weekends.",
            Verdict.UNSUPPORTED,
        ),
        (
            "The product ships worldwide.",
            "Our office has a large kitchen.",
            Verdict.UNSUPPORTED,
        ),
        (
            "The refund window is 90 days.",
            "The refund window is 30 days.",
            Verdict.CONTRADICTED,
        ),
        (
            "The meeting is on Tuesday.",
            "The meeting was moved to Thursday.",
            Verdict.CONTRADICTED,
        ),
    ]

    pairs = [
        (
            Claim(id=f"claim_{i}", text=claim_text, span_start=0, span_end=len(claim_text)),
            [Evidence(id=f"chunk_{i}", text=evidence_text)],
        )
        for i, (claim_text, evidence_text, _) in enumerate(fixtures)
    ]

    verdicts, _ = await verifier.verify(pairs, question="")

    for verdict, (_, _, expected) in zip(verdicts, fixtures, strict=True):
        assert verdict.verdict == expected
