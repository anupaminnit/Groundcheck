"""Tests for cli.main: JSON/pretty output and the CI-gate exit code.

``guard=`` injection is the seam that keeps this deterministic — no real provider or
network call, same as everywhere else in the suite.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from fakes import FakeProvider
from groundcheck import cli
from groundcheck.cli import main
from groundcheck.config import GuardConfig
from groundcheck.core.guard import Guard
from groundcheck.core.schemas import Action, GuardReport, TokenUsage


def _extractor_response() -> str:
    return json.dumps(
        [
            {
                "text": "Paris is the capital of France.",
                "source_sentence": "Paris is the capital of France.",
                "type": "claim",
            }
        ]
    )


def _judge_response(verdict: str) -> str:
    return json.dumps(
        [
            {
                "claim_id": "claim_0",
                "verdict": verdict,
                "confidence": 0.9,
                "evidence_ids": [],
                "rationale": "ok",
            }
        ]
    )


def _write_inputs(tmp_path: Path) -> tuple[str, str]:
    answer_path = tmp_path / "answer.txt"
    answer_path.write_text("Paris is the capital of France.", encoding="utf-8")
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(json.dumps(["Paris is the capital of France."]), encoding="utf-8")
    return str(answer_path), str(evidence_path)


def test_cli_check_exits_0_when_score_at_or_above_threshold(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    answer_path, evidence_path = _write_inputs(tmp_path)
    provider = FakeProvider(json_responses=[_extractor_response(), _judge_response("SUPPORTED")])
    guard = Guard(verifier="llm", policy="log", provider=provider)

    exit_code = main(["check", answer_path, evidence_path, "--threshold", "0.5"], guard=guard)

    assert exit_code == 0
    assert "grounded_score: 1.00" in capsys.readouterr().out


def test_cli_check_exits_1_when_score_below_threshold(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    answer_path, evidence_path = _write_inputs(tmp_path)
    provider = FakeProvider(json_responses=[_extractor_response(), _judge_response("UNSUPPORTED")])
    guard = Guard(verifier="llm", policy="log", provider=provider)

    exit_code = main(["check", answer_path, evidence_path, "--threshold", "0.7"], guard=guard)

    assert exit_code == 1


def test_cli_check_json_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    answer_path, evidence_path = _write_inputs(tmp_path)
    provider = FakeProvider(json_responses=[_extractor_response(), _judge_response("SUPPORTED")])
    guard = Guard(verifier="llm", policy="log", provider=provider)

    main(["check", answer_path, evidence_path, "--format", "json"], guard=guard)

    payload = json.loads(capsys.readouterr().out)
    assert payload["action"] == "pass"
    assert payload["grounded_score"] == 1.0


class _CapturingGuard:
    """Stands in for ``Guard`` so no NLI model or provider is ever loaded."""

    configs: list[GuardConfig] = []

    def __init__(self, config: GuardConfig) -> None:
        self.configs.append(config)

    def check(self, answer: str, evidence: list[Any], question: str = "") -> GuardReport:
        return GuardReport(
            grounded_score=1.0,
            action=Action.PASS,
            safe_answer=answer,
            claims=[],
            verifier="stub",
            latency_ms=0,
            tokens=TokenUsage(),
        )


@pytest.mark.parametrize("verifier", ["local", "hybrid"])
def test_cli_accepts_local_and_hybrid_verifier(
    verifier: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer_path, evidence_path = _write_inputs(tmp_path)
    _CapturingGuard.configs = []
    monkeypatch.setattr(cli, "Guard", _CapturingGuard)

    exit_code = main(["check", answer_path, evidence_path, "--verifier", verifier])

    assert exit_code == 0
    assert _CapturingGuard.configs[0].verifier == verifier
    assert _CapturingGuard.configs[0].provider is None


def test_cli_accepts_litellm_provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    answer_path, evidence_path = _write_inputs(tmp_path)
    _CapturingGuard.configs = []
    monkeypatch.setattr(cli, "Guard", _CapturingGuard)

    main(["check", answer_path, evidence_path, "--provider", "litellm", "--model", "ollama/x"])

    assert _CapturingGuard.configs[0].provider == "litellm"
    assert _CapturingGuard.configs[0].model == "ollama/x"


def test_cli_rejects_unknown_verifier(tmp_path: Path) -> None:
    answer_path, evidence_path = _write_inputs(tmp_path)

    with pytest.raises(SystemExit):
        main(["check", answer_path, evidence_path, "--verifier", "bogus"])


def test_cli_check_reads_dict_evidence(tmp_path: Path) -> None:
    answer_path = tmp_path / "answer.txt"
    answer_path.write_text("Paris is the capital of France.", encoding="utf-8")
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(
        json.dumps([{"id": "doc_1", "text": "Paris is the capital of France."}]), encoding="utf-8"
    )
    provider = FakeProvider(json_responses=[_extractor_response(), _judge_response("SUPPORTED")])
    guard = Guard(verifier="llm", policy="log", provider=provider)

    exit_code = main(
        ["check", str(answer_path), str(evidence_path), "--threshold", "0.5"], guard=guard
    )

    assert exit_code == 0
