"""``NLIVerifier``: local cross-encoder NLI verification, no LLM calls.

Lazily imports ``transformers``/``torch`` inside ``__init__`` — nothing at this
module's top level imports them, so merely importing ``NLIVerifier`` (or
``groundcheck`` itself) never pulls in torch; only constructing one does. Requires
the ``[local]`` extra. See ``docs/SPEC.md`` §5.4.
"""

from __future__ import annotations

import os

from groundcheck.core.errors import ConfigError
from groundcheck.core.schemas import Claim, ClaimVerdict, Evidence, TokenUsage, Verdict

DEFAULT_NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
_ENTAILMENT_THRESHOLD = 0.7
_CONTRADICTION_THRESHOLD = 0.7

_Pair = tuple[Claim, list[Evidence]]
_Scores = tuple[float, float, float]  # (entailment, neutral, contradiction)


class NLIVerifier:
    """Local cross-encoder NLI verifier. Requires ``pip install groundcheck[local]``."""

    def __init__(self, model_name: str | None = None) -> None:
        """Load the NLI model (once; cached on the instance).

        Args:
            model_name: HuggingFace model id to load. Defaults to
                ``GROUNDCHECK_NLI_MODEL``, then ``DEFAULT_NLI_MODEL``.

        Raises:
            ConfigError: The ``[local]`` extra isn't installed, or the model's
                labels don't include entailment/neutral/contradiction.
        """
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ConfigError(
                "NLIVerifier requires the [local] extra. Run: pip install groundcheck[local]"
            ) from exc

        resolved_model_name = model_name or os.environ.get(
            "GROUNDCHECK_NLI_MODEL", DEFAULT_NLI_MODEL
        )
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(resolved_model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(resolved_model_name)
        self._model.eval()
        self._label_index = _label_index(dict(self._model.config.id2label))

    async def verify(
        self, pairs: list[_Pair], question: str
    ) -> tuple[list[ClaimVerdict], TokenUsage]:
        """Judge each claim against its candidate evidence via one batched NLI pass.

        For each claim, picks the candidate with the strongest entailment or
        contradiction signal and maps its probabilities to a verdict — see
        ``docs/SPEC.md`` §5.4.

        Args:
            pairs: One ``(claim, candidates)`` tuple per claim to judge.
            question: Ignored — NLI verification doesn't use question context.

        Returns:
            A tuple of one ``ClaimVerdict`` per input pair and a zero
            ``TokenUsage`` (no LLM calls are made).
        """
        if not pairs:
            return [], TokenUsage()

        flat_premises: list[str] = []
        flat_hypotheses: list[str] = []
        for claim, candidates in pairs:
            for candidate in candidates:
                flat_premises.append(candidate.text)
                flat_hypotheses.append(claim.text)

        scores = self._predict_batch(flat_premises, flat_hypotheses) if flat_premises else []

        per_claim_scores: list[list[_Scores]] = [[] for _ in pairs]
        per_claim_evidence_ids: list[list[str]] = [[] for _ in pairs]
        cursor = 0
        for claim_idx, (_, candidates) in enumerate(pairs):
            for candidate in candidates:
                per_claim_scores[claim_idx].append(scores[cursor])
                per_claim_evidence_ids[claim_idx].append(candidate.id)
                cursor += 1

        verdicts = []
        for claim_idx, (claim, _) in enumerate(pairs):
            winner = _pick_winner(per_claim_scores[claim_idx])
            if winner is None:
                verdicts.append(
                    ClaimVerdict(claim=claim, verdict=Verdict.UNSUPPORTED, confidence=0.0)
                )
                continue
            winner_idx, entail, contra = winner
            verdict, confidence = _verdict_from_scores(entail, contra)
            evidence_ids = (
                [per_claim_evidence_ids[claim_idx][winner_idx]]
                if verdict != Verdict.UNSUPPORTED
                else []
            )
            verdicts.append(
                ClaimVerdict(
                    claim=claim, verdict=verdict, confidence=confidence, evidence_ids=evidence_ids
                )
            )
        return verdicts, TokenUsage()

    def _predict_batch(self, premises: list[str], hypotheses: list[str]) -> list[_Scores]:
        inputs = self._tokenizer(
            premises, hypotheses, return_tensors="pt", padding=True, truncation=True
        )
        with self._torch.no_grad():
            logits = self._model(**inputs).logits
            probs = logits.softmax(dim=-1)
        entail_idx, neutral_idx, contra_idx = self._label_index
        return [
            (row[entail_idx].item(), row[neutral_idx].item(), row[contra_idx].item())
            for row in probs
        ]


def _label_index(id2label: dict[int, str]) -> tuple[int, int, int]:
    lookup = {name.lower(): idx for idx, name in id2label.items()}
    try:
        return lookup["entailment"], lookup["neutral"], lookup["contradiction"]
    except KeyError as exc:
        raise ConfigError(
            f"NLI model labels {list(id2label.values())} don't include "
            "entailment/neutral/contradiction."
        ) from exc


def _pick_winner(scores: list[_Scores]) -> tuple[int, float, float] | None:
    """Pick the candidate with the strongest signal in either direction.

    Returns ``(winning_index, entail_prob, contra_prob)``, or ``None`` if there are
    no candidates at all.
    """
    if not scores:
        return None
    best_idx = max(range(len(scores)), key=lambda i: max(scores[i][0], scores[i][2]))
    entail, _neutral, contra = scores[best_idx]
    return best_idx, entail, contra


def _verdict_from_scores(entail: float, contra: float) -> tuple[Verdict, float]:
    if entail >= _ENTAILMENT_THRESHOLD:
        return Verdict.SUPPORTED, entail
    if contra >= _CONTRADICTION_THRESHOLD:
        return Verdict.CONTRADICTED, contra
    return Verdict.UNSUPPORTED, max(entail, contra)
