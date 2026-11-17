from typing import Any

from rag_sdk.core.tracing import trace
from rag_sdk.providers.base import Reranker, Retriever, SearchResults


class RerankingRetriever(Retriever):
    """Over-fetches candidates from ``retriever`` and keeps the best ones as judged by ``reranker``.

    It is itself a ``Retriever``, so it drops into ``RagWorkflow`` or the eval harness unchanged.
    ``top_k`` is the *final* result count (default ``top_n``); candidates fetched are
    ``max(retrieve_k, top_k)``. Results keep their retrieval ``score``, so the returned list is
    not sorted by it.
    """

    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker,
        retrieve_k: int = 20,
        top_n: int = 5,
    ) -> None:
        if retrieve_k < 1 or top_n < 1:
            raise ValueError("retrieve_k and top_n must be >= 1")
        self._retriever = retriever
        self._reranker = reranker
        self._retrieve_k = retrieve_k
        self._top_n = top_n

    @trace()
    async def retrieve(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> SearchResults:
        n = top_k if top_k is not None else self._top_n
        candidates = await self._retriever.retrieve(query, filters, top_k=max(self._retrieve_k, n))
        if len(candidates) <= 1:
            return candidates[:n]
        return await self._reranker.rerank(query, candidates, n)
