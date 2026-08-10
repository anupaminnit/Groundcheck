"""Azure OpenAI provider backend (async, official SDK).

Reads ``AZURE_OPENAI_ENDPOINT``, ``AZURE_OPENAI_API_KEY``, and
``AZURE_OPENAI_DEPLOYMENT`` from the environment unless explicit values are passed.
See ``docs/SPEC.md`` §5.8.
"""

from __future__ import annotations

import os

from openai import AsyncAzureOpenAI

from groundcheck.core.errors import ConfigError
from groundcheck.core.schemas import TokenUsage

DEFAULT_API_VERSION = "2024-10-21"


class AzureOpenAIProvider:
    """LLMProvider backed by an Azure OpenAI deployment."""

    def __init__(
        self,
        endpoint: str | None = None,
        api_key: str | None = None,
        deployment: str | None = None,
        embed_deployment: str | None = None,
        api_version: str = DEFAULT_API_VERSION,
    ) -> None:
        """Initialize the provider.

        Args:
            endpoint: Azure OpenAI resource endpoint. Defaults to
                ``AZURE_OPENAI_ENDPOINT``.
            api_key: Azure OpenAI API key. Defaults to ``AZURE_OPENAI_API_KEY``.
            deployment: Chat completion deployment name. Defaults to
                ``AZURE_OPENAI_DEPLOYMENT``.
            embed_deployment: Embedding deployment name. Defaults to the same
                deployment as ``deployment``.
            api_version: Azure OpenAI API version.

        Raises:
            ConfigError: The endpoint, key, or deployment weren't given or found
                in the environment.
        """
        resolved_endpoint = endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        resolved_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        resolved_deployment = deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        if not resolved_endpoint or not resolved_key or not resolved_deployment:
            raise ConfigError(
                "AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, and AZURE_OPENAI_DEPLOYMENT "
                "must all be set (or passed explicitly)."
            )
        self._client = AsyncAzureOpenAI(
            azure_endpoint=resolved_endpoint,
            api_key=resolved_key,
            api_version=api_version,
        )
        self._deployment = resolved_deployment
        self._embed_deployment = embed_deployment or resolved_deployment

    async def complete_json(
        self, system: str, user: str, timeout: float
    ) -> tuple[str, TokenUsage]:
        """See ``LLMProvider.complete_json``."""
        response = await self._client.chat.completions.create(
            model=self._deployment,
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
        response = await self._client.embeddings.create(model=self._embed_deployment, input=texts)
        return [item.embedding for item in response.data]
