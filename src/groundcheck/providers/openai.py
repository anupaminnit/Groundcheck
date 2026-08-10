"""OpenAI provider backend (async, official SDK).

Reads ``OPENAI_API_KEY`` from the environment unless an explicit key is passed. Also
accepts an optional ``base_url`` (or ``GROUNDCHECK_BASE_URL``) so any OpenAI-shaped
server works unchanged: a LiteLLM proxy, Ollama, vLLM, LM Studio. See
``docs/SPEC.md`` §5.8.
"""

from __future__ import annotations

import os

from openai import AsyncOpenAI

from groundcheck.core.errors import ConfigError
from groundcheck.core.schemas import TokenUsage

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"


class OpenAIProvider:
    """LLMProvider backed by the OpenAI API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        embed_model: str = DEFAULT_EMBED_MODEL,
        base_url: str | None = None,
    ) -> None:
        """Initialize the provider.

        Args:
            api_key: OpenAI API key. Defaults to ``OPENAI_API_KEY``.
            model: Chat completion model.
            embed_model: Embedding model.
            base_url: Custom base URL for an OpenAI-compatible endpoint (LiteLLM
                proxy, Ollama, vLLM, LM Studio). Defaults to
                ``GROUNDCHECK_BASE_URL``, then the official OpenAI API.

        Raises:
            ConfigError: No API key was given or found in the environment.
        """
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ConfigError("OPENAI_API_KEY is not set and no api_key was provided.")
        resolved_base_url = base_url or os.environ.get("GROUNDCHECK_BASE_URL")
        self._client = AsyncOpenAI(api_key=key, base_url=resolved_base_url)
        self._model = model
        self._embed_model = embed_model

    async def complete_json(
        self, system: str, user: str, timeout: float
    ) -> tuple[str, TokenUsage]:
        """See ``LLMProvider.complete_json``."""
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            timeout=timeout,
        )
        content = response.choices[0].message.content or ""
        usage = response.usage
        tokens = TokenUsage(
            input=usage.prompt_tokens if usage else 0,
            output=usage.completion_tokens if usage else 0,
        )
        return content, tokens

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """See ``LLMProvider.embed``."""
        response = await self._client.embeddings.create(model=self._embed_model, input=texts)
        return [item.embedding for item in response.data]
