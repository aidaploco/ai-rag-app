import asyncio
from typing import Any

from openai import AsyncOpenAI, RateLimitError

from rag_sdk.core.resilience import with_retry
from rag_sdk.core.tracing import trace
from rag_sdk.providers.base import (
    Chatter,
    Embedder,
    GeneratedResponse,
    Prompt,
    TokenUsage,
)


class OpenAIEmbedder(Embedder):
    def __init__(
        self,
        client: AsyncOpenAI,
        model: str = "text-embedding-3-small",
        dimensions: int = 1536,
        call_options: dict[str, Any] | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._dimensions = dimensions
        self._call_options = call_options or {}

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @trace()
    @with_retry(RateLimitError)
    async def embed(self, text: str) -> list[float]:
        response = await self._client.embeddings.create(
            model=self._model,
            input=text,
            **self._call_options,
        )
        return response.data[0].embedding

    @trace()
    @with_retry(RateLimitError)
    async def embed_chunks(
        self, texts: list[str], batch_size: int = 2048
    ) -> list[list[float]]:
        async def _embed_batch(batch: list[str]) -> list[list[float]]:
            response = await self._client.embeddings.create(
                model=self._model,
                input=batch,
                **self._call_options,
            )
            return [item.embedding for item in response.data]

        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
        results = await asyncio.gather(*[_embed_batch(b) for b in batches])
        return [emb for batch_result in results for emb in batch_result]


class OpenAIChatter(Chatter):
    def __init__(
        self,
        client: AsyncOpenAI,
        model: str = "gpt-4o-mini",
        call_options: dict[str, Any] | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._call_options = call_options or {}

    @trace()
    @with_retry(RateLimitError)
    async def generate(self, prompt: Prompt) -> GeneratedResponse:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ],
            **self._call_options,
        )
        usage = response.usage
        return GeneratedResponse(
            content=response.choices[0].message.content or "",
            token_usage=TokenUsage(
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
            ),
        )
