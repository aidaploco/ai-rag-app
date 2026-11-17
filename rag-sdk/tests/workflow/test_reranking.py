from typing import Any

import pytest
from rag_sdk.providers.base import Reranker, Retriever, SearchResult, SearchResults
from rag_sdk.workflow.reranking import RerankingRetriever


class FixedRetriever(Retriever):
    def __init__(self, n: int) -> None:
        self._all = [
            SearchResult(f"c{i}", 1.0 - i / 100, {}, content_id=f"id{i}") for i in range(n)
        ]
        self.calls: list[tuple[str, dict[str, Any] | None, int | None]] = []

    async def retrieve(
        self, query: str, filters: dict[str, Any] | None = None, top_k: int | None = None
    ) -> SearchResults:
        self.calls.append((query, filters, top_k))
        return self._all[:top_k]


class ReverseReranker(Reranker):
    def __init__(self) -> None:
        self.seen: list[tuple[int, int]] = []

    async def rerank(self, query: str, results: SearchResults, top_n: int) -> SearchResults:
        self.seen.append((len(results), top_n))
        return list(reversed(results))[:top_n]


async def test_over_fetches_then_returns_reranked_top_n():
    inner, reranker = FixedRetriever(50), ReverseReranker()
    out = await RerankingRetriever(inner, reranker, retrieve_k=20, top_n=5).retrieve("q")
    assert inner.calls == [("q", None, 20)]
    assert reranker.seen == [(20, 5)]
    assert [r.content for r in out] == ["c19", "c18", "c17", "c16", "c15"]


async def test_top_k_is_the_final_count_and_candidates_never_fewer():
    inner, reranker = FixedRetriever(50), ReverseReranker()
    rr = RerankingRetriever(inner, reranker, retrieve_k=20, top_n=5)
    assert len(await rr.retrieve("q", top_k=10)) == 10
    assert inner.calls[-1][2] == 20
    assert len(await rr.retrieve("q", top_k=30)) == 30
    assert inner.calls[-1][2] == 30  # asked for more than retrieve_k: fetch that many


async def test_filters_pass_through():
    inner = FixedRetriever(5)
    await RerankingRetriever(inner, ReverseReranker()).retrieve("q", {"filename": "a"})
    assert inner.calls[0][1] == {"filename": "a"}


async def test_skips_reranker_for_zero_or_one_candidate():
    reranker = ReverseReranker()
    assert await RerankingRetriever(FixedRetriever(0), reranker).retrieve("q") == []
    assert len(await RerankingRetriever(FixedRetriever(1), reranker).retrieve("q")) == 1
    assert reranker.seen == []


async def test_results_keep_their_retrieval_score_and_content_id():
    out = await RerankingRetriever(FixedRetriever(3), ReverseReranker()).retrieve("q")
    assert [(r.content_id, r.score) for r in out] == [("id2", 0.98), ("id1", 0.99), ("id0", 1.0)]


def test_rejects_non_positive_sizes():
    with pytest.raises(ValueError):
        RerankingRetriever(FixedRetriever(1), ReverseReranker(), retrieve_k=0)
    with pytest.raises(ValueError):
        RerankingRetriever(FixedRetriever(1), ReverseReranker(), top_n=0)
