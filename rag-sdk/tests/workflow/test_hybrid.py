from typing import Any

import pytest
from rag_sdk.providers.base import Retriever, SearchResult, SearchResults
from rag_sdk.workflow.hybrid import HybridRetriever


def _r(cid: str | None, score: float = 1.0, content: str | None = None) -> SearchResult:
    return SearchResult(
        content=content or f"text-{cid}", score=score, metadata={"from": cid}, content_id=cid
    )


class Leg(Retriever):
    def __init__(self, ids: list[str | None], scores: float = 1.0) -> None:
        self._ids = ids
        self._score = scores
        self.calls: list[tuple[str, dict[str, Any] | None, int | None]] = []

    async def retrieve(
        self, query: str, filters: dict[str, Any] | None = None, top_k: int | None = None
    ) -> SearchResults:
        self.calls.append((query, filters, top_k))
        return [_r(i, self._score) for i in self._ids[:top_k]]


async def test_rrf_scores_are_hand_computed():
    # leg A ranks a,b,c ; leg B ranks c,a,d ; k = 60
    hybrid = HybridRetriever([Leg(["a", "b", "c"]), Leg(["c", "a", "d"])], rrf_k=60, top_n=10)
    out = await hybrid.retrieve("q")
    scores = {r.content_id: r.score for r in out}
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["c"] == pytest.approx(1 / 63 + 1 / 61)
    assert scores["b"] == pytest.approx(1 / 62)
    assert scores["d"] == pytest.approx(1 / 63)
    assert [r.content_id for r in out] == ["a", "c", "b", "d"]


async def test_score_is_the_fused_score_not_a_legs_score():
    out = await HybridRetriever([Leg(["a"], scores=99.0)]).retrieve("q")
    assert out[0].score == pytest.approx(1 / 61)


async def test_ties_break_by_best_rank_then_leg_order():
    # a is rank 1 in leg A only; b is rank 1 in leg B only: equal scores, equal best rank,
    # so the earlier leg's document comes first. c and d tie below them the same way.
    out = await HybridRetriever([Leg(["a", "c"]), Leg(["b", "d"])], top_n=10).retrieve("q")
    assert [r.content_id for r in out] == ["a", "b", "c", "d"]


async def test_each_leg_gets_query_filters_and_depth_but_final_count_is_top_k():
    a, b = Leg(["a", "b", "c", "d"]), Leg(["d", "c", "b", "a"])
    hybrid = HybridRetriever([a, b], depth=3, top_n=2)
    out = await hybrid.retrieve("my query", {"filename": "x"})
    assert a.calls == [("my query", {"filename": "x"}, 3)]
    assert b.calls == [("my query", {"filename": "x"}, 3)]
    assert len(out) == 2


async def test_top_k_above_depth_widens_the_legs():
    leg = Leg(["a", "b", "c", "d", "e"])
    out = await HybridRetriever([leg], depth=2, top_n=2).retrieve("q", top_k=4)
    assert leg.calls[0][2] == 4 and len(out) == 4


async def test_content_and_metadata_come_from_the_first_leg_that_returned_it():
    first = Leg([])
    first.retrieve = lambda *a, **k: _async([_r("x", content="from-first")])  # type: ignore[method-assign]
    second = Leg([])
    second.retrieve = lambda *a, **k: _async([_r("x", content="from-second")])  # type: ignore[method-assign]
    out = await HybridRetriever([first, second]).retrieve("q")
    assert out[0].content == "from-first"


async def _async(value):
    return value


async def test_result_without_content_id_raises():
    with pytest.raises(ValueError, match="content_id"):
        await HybridRetriever([Leg([None])]).retrieve("q")


async def test_empty_legs_give_empty_result():
    assert await HybridRetriever([Leg([]), Leg([])]).retrieve("q") == []


def test_rejects_bad_configuration():
    with pytest.raises(ValueError):
        HybridRetriever([])
    for kwargs in ({"rrf_k": 0}, {"depth": 0}, {"top_n": 0}):
        with pytest.raises(ValueError):
            HybridRetriever([Leg([])], **kwargs)
