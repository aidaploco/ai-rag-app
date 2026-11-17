import asyncio
from collections.abc import Sequence
from typing import Any

from rag_sdk.core.tracing import trace
from rag_sdk.providers.base import Retriever, SearchResult, SearchResults


class HybridRetriever(Retriever):
    """Reciprocal rank fusion (RRF) over several retrievers, e.g. dense + lexical (BM25).

    Each leg is queried concurrently for ``depth`` results with the same filters; a document's
    fused score is the sum over legs of ``1 / (rrf_k + rank)`` (rank is 1-based). Legs are fused
    by ``content_id``, so every result must carry one.

    The returned ``score`` is the fused RRF score, not any leg's own score (cosine similarity and
    BM25 are on different scales and are not comparable). Ties break by best rank in any leg, then
    by leg order, so results are deterministic. The ``SearchResult`` content comes from the first
    leg that returned the document.
    """

    def __init__(
        self,
        retrievers: Sequence[Retriever],
        rrf_k: int = 60,
        depth: int = 20,
        top_n: int = 5,
    ) -> None:
        if not retrievers:
            raise ValueError("at least one retriever is required")
        if rrf_k < 1 or depth < 1 or top_n < 1:
            raise ValueError("rrf_k, depth and top_n must be >= 1")
        self._retrievers = list(retrievers)
        self._rrf_k = rrf_k
        self._depth = depth
        self._top_n = top_n

    @trace()
    async def retrieve(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> SearchResults:
        n = top_k if top_k is not None else self._top_n
        depth = max(self._depth, n)
        legs = await asyncio.gather(
            *(r.retrieve(query, filters, top_k=depth) for r in self._retrievers)
        )

        fused: dict[str, float] = {}
        best_rank: dict[str, int] = {}
        first_seen: dict[str, tuple[int, SearchResult]] = {}  # leg index, result
        for leg_index, results in enumerate(legs):
            for rank, result in enumerate(results, start=1):
                if result.content_id is None:
                    raise ValueError("HybridRetriever needs every result to have a content_id")
                cid = result.content_id
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (self._rrf_k + rank)
                best_rank[cid] = min(best_rank.get(cid, rank), rank)
                first_seen.setdefault(cid, (leg_index, result))

        ordered = sorted(fused, key=lambda c: (-fused[c], best_rank[c], first_seen[c][0]))
        return [
            SearchResult(
                content=first_seen[cid][1].content,
                score=fused[cid],
                metadata=first_seen[cid][1].metadata,
                content_id=cid,
            )
            for cid in ordered[:n]
        ]
