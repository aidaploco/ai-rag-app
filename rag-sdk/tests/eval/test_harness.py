from typing import Any

import pytest
from rag_sdk.eval.harness import GoldenQuery, evaluate
from rag_sdk.providers.base import Retriever, SearchResult, SearchResults


class RankedRetriever(Retriever):
    """Returns a fixed ranking per query, honouring top_k like a real retriever."""

    def __init__(self, rankings: dict[str, list[str | None]]) -> None:
        self._rankings = rankings
        self.calls: list[tuple[str, int | None]] = []

    async def retrieve(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> SearchResults:
        self.calls.append((query, top_k))
        ids = self._rankings[query][:top_k]
        return [SearchResult(content="", score=1.0, metadata={}, content_id=i) for i in ids]


def _golden() -> list[GoldenQuery]:
    return [
        GoldenQuery("q1", frozenset({"a"}), kind="paraphrase"),
        GoldenQuery("q2", frozenset({"b", "c"}), kind="rare"),
    ]


async def test_scores_and_macro_average():
    retriever = RankedRetriever({"q1": ["a", "x", "y"], "q2": ["x", "b", "y"]})
    report = await evaluate(retriever, _golden(), ks=(1, 3))

    q1, q2 = report.queries
    assert q1.scores["recall"] == {1: 1.0, 3: 1.0}
    assert q2.scores["recall"] == {1: 0.0, 3: 0.5}
    assert q2.scores["mrr"] == {1: 0.0, 3: 0.5}
    assert report.mean("recall", 3) == pytest.approx(0.75)
    assert report.mean("mrr", 1) == pytest.approx(0.5)


async def test_first_relevant_rank_and_per_query_rows():
    retriever = RankedRetriever({"q1": ["x", "a"], "q2": ["x", "y"]})
    report = await evaluate(retriever, _golden(), ks=(2,))
    assert [q.first_relevant_rank for q in report.queries] == [2, None]
    rows = report.per_query()
    assert rows[0]["query"] == "q1" and rows[0]["first_relevant_rank"] == 2
    assert rows[0]["mrr"] == {"2": 0.5}
    assert rows[1]["first_relevant_rank"] is None and rows[1]["recall"] == {"2": 0.0}


async def test_retrieves_once_per_query_at_max_k():
    retriever = RankedRetriever({"q1": ["a"], "q2": ["b"]})
    await evaluate(retriever, _golden(), ks=(5, 1, 3))
    assert sorted(retriever.calls) == [("q1", 5), ("q2", 5)]


async def test_per_kind_breakdown_in_summary():
    retriever = RankedRetriever({"q1": ["a"], "q2": ["x", "x2"]})
    summary = (await evaluate(retriever, _golden(), ks=(1,))).summary()
    assert summary["queries"] == 2
    assert summary["by_kind"]["paraphrase"]["recall"]["1"] == 1.0
    assert summary["by_kind"]["rare"]["recall"]["1"] == 0.0
    assert summary["overall"]["recall"]["1"] == 0.5


async def test_result_without_content_id_raises_instead_of_scoring_a_miss():
    retriever = RankedRetriever({"q1": [None], "q2": ["b"]})
    with pytest.raises(ValueError, match="content_id"):
        await evaluate(retriever, _golden(), ks=(1,))


async def test_rejects_bad_inputs():
    retriever = RankedRetriever({})
    with pytest.raises(ValueError):
        await evaluate(retriever, [], ks=(1,))
    with pytest.raises(ValueError):
        await evaluate(retriever, _golden(), ks=(0,))
    with pytest.raises(ValueError):
        await evaluate(retriever, _golden(), ks=(1,), concurrency=0)
    with pytest.raises(ValueError):
        (await evaluate(RankedRetriever({"q1": ["a"], "q2": ["b"]}), _golden(), ks=(1,))).mean(
            "recall", 1, kind="nope"
        )
