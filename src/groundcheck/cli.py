"""``groundcheck`` command-line interface.

``groundcheck check ANSWER_FILE EVIDENCE_FILE`` runs a single ``Guard.check()``,
prints the report as JSON or a human-readable summary, and exits 1 if
``grounded_score`` is below ``--threshold`` — usable as a CI gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any

from groundcheck.config import GuardConfig
from groundcheck.core.guard import Guard
from groundcheck.core.schemas import GuardReport


def main(argv: Sequence[str] | None = None, *, guard: Guard | None = None) -> int:
    """Entry point for the ``groundcheck`` console script.

    Args:
        argv: Command-line arguments, excluding the program name. Defaults to
            ``sys.argv[1:]`` (via ``argparse``) when ``None``.
        guard: Used instead of building one from the parsed args — the seam
            tests use to inject a ``Guard`` backed by a fake provider.

    Returns:
        Process exit code: ``1`` if ``grounded_score`` is below ``--threshold``,
        ``0`` otherwise.
    """
    args = _parse_args(argv)

    answer = _read_text(args.answer_file)
    evidence = _load_evidence(args.evidence_file)

    active_guard = guard or Guard(
        config=GuardConfig(
            verifier=args.verifier,
            policy=args.policy,
            provider=args.provider,
            model=args.model,
            base_url=args.base_url,
            block_threshold=args.threshold,
        )
    )

    report = active_guard.check(answer, evidence, question=args.question)
    _print_report(report, fmt=args.format)

    return 1 if report.grounded_score < args.threshold else 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="groundcheck")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check", help="Verify an answer against its evidence.")
    check.add_argument(
        "answer_file", help="Path to a text file containing the answer ('-' for stdin)."
    )
    check.add_argument(
        "evidence_file", help="Path to a JSON file: a list of strings or {id, text} objects."
    )
    check.add_argument("--question", default="", help="The question the answer responds to.")
    check.add_argument("--verifier", default="llm", choices=["llm"])
    check.add_argument("--policy", default="log", choices=["log", "annotate", "redact", "block"])
    check.add_argument("--provider", default=None, choices=["azure", "openai", "anthropic"])
    check.add_argument("--model", default=None)
    check.add_argument("--base-url", dest="base_url", default=None)
    check.add_argument("--threshold", type=float, default=0.7)
    check.add_argument("--format", choices=["json", "pretty"], default="pretty")

    return parser.parse_args(argv)


def _read_text(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    with open(path, encoding="utf-8") as f:
        return f.read()


def _load_evidence(path: str) -> list[Any]:
    with open(path, encoding="utf-8") as f:
        data: Any = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Evidence file must contain a JSON array.")
    for item in data:
        if not isinstance(item, (str, dict)):
            raise ValueError("Each evidence item must be a string or an object.")
    return data


def _print_report(report: GuardReport, fmt: str) -> None:
    if fmt == "json":
        print(report.model_dump_json())
        return

    print(f"grounded_score: {report.grounded_score:.2f}")
    print(f"action:         {report.action.value}")
    print(f"verifier:       {report.verifier}")
    print(f"latency_ms:     {report.latency_ms}")
    if report.error:
        print(f"error:          {report.error}")
    print(f"claims:         {len(report.claims)}")
    for claim_verdict in report.claims:
        print(f"  - [{claim_verdict.verdict.value}] {claim_verdict.claim.text}")
    print()
    print("safe_answer:")
    print(report.safe_answer)


if __name__ == "__main__":
    sys.exit(main())
