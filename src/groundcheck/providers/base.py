"""``LLMProvider`` protocol: the common interface every provider backend implements.

See ``docs/SPEC.md`` §5.8.
"""

from __future__ import annotations

from typing import Protocol

from groundcheck.core.schemas import TokenUsage


class LLMProvider(Protocol):
    """A backend capable of JSON-mode completions and (optionally) embeddings."""

    async def complete_json(
        self, system: str, user: str, timeout: float
    ) -> tuple[str, TokenUsage]:
        """Run a completion expected to return JSON.

        Args:
            system: The system prompt.
            user: The user message.
            timeout: Timeout for the call, in seconds.

        Returns:
            A tuple of the raw completion text (not yet parsed/validated as JSON)
            and the token usage for that call.
        """
        ...

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts.

        Args:
            texts: The texts to embed.

        Returns:
            One embedding vector per input text, in the same order.

        Raises:
            ConfigError: This provider has no embedding support.
        """
        ...
