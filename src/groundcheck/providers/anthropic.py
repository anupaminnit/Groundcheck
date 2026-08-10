"""Anthropic provider backend (async, official SDK).

Reads ``ANTHROPIC_API_KEY`` from the environment unless an explicit key is passed.
Anthropic has no embeddings endpoint, so ``embed()`` raises ``ConfigError`` — use the
lexical evidence matcher fallback or a different provider for embeddings.
See ``docs/SPEC.md`` §5.8.
"""

from __future__ import annotations

import os

from anthropic import AsyncAnthropic

from groundcheck.core.errors import ConfigError
from groundcheck.core.schemas import TokenUsage

DEFAULT_MODEL = "claude-3-5-sonnet-latest"
DEFAULT_MAX_TOKENS = 4096


class AnthropicProvider:
    """LLMProvider backed by the Anthropic API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        """Initialize the provider.

        Args:
            api_key: Anthropic API key. Defaults to ``ANTHROPIC_API_KEY``.
            model: Completion model.
            max_tokens: Max output tokens per completion.

        Raises:
            ConfigError: No API key was given or found in the environment.
        """
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ConfigError("ANTHROPIC_API_KEY is not set and no api_key was provided.")
        self._client = AsyncAnthropic(api_key=key)
        self._model = model
        self._max_tokens = max_tokens

    async def complete_json(
        self, system: str, user: str, timeout: float
    ) -> tuple[str, TokenUsage]:
        """See ``LLMProvider.complete_json``."""
        response = await self._client.messages.create(
            model=self._model,
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=self._max_tokens,
            timeout=timeout,
        )
        content = "".join(block.text for block in response.content if block.type == "text")
        tokens = TokenUsage(input=response.usage.input_tokens, output=response.usage.output_tokens)
        return content, tokens

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Always raises — Anthropic has no embeddings endpoint.

        Raises:
            ConfigError: Always. Use the lexical evidence matcher fallback, or
                configure a different provider for embeddings.
        """
        raise ConfigError(
            "Anthropic has no embeddings endpoint. Use the lexical evidence matcher "
            "fallback, or configure a different provider for embeddings."
        )
