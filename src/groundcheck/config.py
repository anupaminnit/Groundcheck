"""``GuardConfig``: runtime configuration for a ``Guard`` instance.

Fields are settable via constructor kwargs or ``GROUNDCHECK_*`` environment variables
(e.g. ``GROUNDCHECK_PROVIDER=openai``). Provider credentials (``OPENAI_API_KEY``,
``AZURE_OPENAI_*``, ``ANTHROPIC_API_KEY``) are read directly by the provider classes
in ``providers/``, not by this config. See ``docs/SPEC.md`` §4.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from groundcheck.core.schemas import Verdict


class GuardConfig(BaseSettings):
    """Configuration for a ``Guard``. See ``docs/SPEC.md`` §4.

    Attributes:
        verifier: Which verifier backend to use.
        policy: Which policy to apply to flagged claims.
        provider: Which LLM provider to use for ``llm``/``hybrid`` verification.
            Ignored (and unneeded) for ``verifier="local"``.
        model: Model name/deployment passed to the provider. Provider-specific;
            ``None`` uses that provider's default.
        base_url: Custom base URL for ``OpenAIProvider`` (e.g. a LiteLLM proxy,
            Ollama, or vLLM endpoint). Also settable via ``GROUNDCHECK_BASE_URL``.
        block_threshold: For the ``block`` policy, the ``grounded_score`` below
            which the answer is replaced with a fallback message.
        redact_verdicts: Which verdicts count as "flagged" for the ``annotate``
            and ``redact`` policies.
        top_k_evidence: How many candidate evidence chunks to keep per claim.
        hybrid_escalation_band: For ``verifier="hybrid"``, the NLI confidence
            range (inclusive) that gets escalated to the LLM judge.
        timeout_s: Per-call timeout, in seconds, for LLM provider calls.
        fail_open: If ``True`` (the default), verification failures return a
            ``GuardReport(action="error", ...)`` instead of raising.
    """

    model_config = SettingsConfigDict(env_prefix="GROUNDCHECK_", extra="ignore")

    verifier: Literal["llm", "local", "hybrid"] = "llm"
    policy: Literal["log", "annotate", "redact", "block"] = "log"
    provider: Literal["azure", "openai", "anthropic", "litellm"] | None = None
    model: str | None = None
    base_url: str | None = None
    block_threshold: float = 0.7
    redact_verdicts: set[Verdict] = Field(
        default_factory=lambda: {Verdict.UNSUPPORTED, Verdict.CONTRADICTED}
    )
    top_k_evidence: int = 3
    hybrid_escalation_band: tuple[float, float] = (0.35, 0.75)
    timeout_s: float = 30.0
    fail_open: bool = True
