import asyncio
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from rag_sdk.eval.metrics import ndcg_at_k, recall_at_k, reciprocal_rank_at_k
from rag_sdk.providers.base import Retriever

METRICS = ("recall", "mrr", "ndcg")


@dataclass(frozen=True)
class GoldenQuery:
    """A query and the content IDs a good retriever should surface for it."""

    query: str
    relevant: frozenset[str]
    kind: str = "default"  # free-form tag (e.g. "paraphrase") so results can be split by type


@dataclass(frozen=True)
class QueryScore:
    query: str
    kind: str
    retrieved: tuple[str, ...]
    first_relevant_rank: int | None  # 1-based within the retrieved list; None if none retrieved
    scores: dict[str, dict[int, float]]  # metric -> k -> value


@dataclass(frozen=True)
class EvalReport:
    ks: tuple[int, ...]
    queries: list[QueryScore] = field(default_factory=list)

    def mean(self, metric: str, k: int, kind: str | None = None) -> float:
        """Macro-average of ``metric`` at ``k`` over all queries, or only those of ``kind``."""
        values = [q.scores[metric][k] for q in self.queries if kind is None or q.kind == kind]
        if not values:
            raise ValueError(f"no queries of kind {kind!r}")
        return sum(values) / len(values)

    def per_query(self) -> list[dict[str, Any]]:
        """One plain row per query, so a change in the aggregates can be traced to queries."""
        return [
            {
                "query": q.query,
                "kind": q.kind,
                "first_relevant_rank": q.first_relevant_rank,
                **{m: {str(k): round(v, 4) for k, v in q.scores[m].items()} for m in METRICS},
            }
            for q in self.queries
        ]

    def summary(self) -> dict[str, Any]:
        """Aggregate scores overall and per kind, as plain JSON-serialisable data."""
        by_kind: dict[str, int] = defaultdict(int)
        for q in self.queries:
            by_kind[q.kind] += 1

        def block(kind: str | None) -> dict[str, dict[str, float]]:
            return {m: {str(k): round(self.mean(m, k, kind), 4) for k in self.ks} for m in METRICS}

        return {
            "ks": list(self.ks),
            "queries": len(self.queries),
            "overall": block(None),
            "by_kind": {kind: {"queries": n, **block(kind)} for kind, n in sorted(by_kind.items())},
        }


async def evaluate(
    retriever: Retriever,
    golden: Sequence[GoldenQuery],
    ks: Sequence[int] = (1, 3, 5, 10),
    filters: dict[str, Any] | None = None,
    concurrency: int = 5,
) -> EvalReport:
    """Run every golden query through ``retriever`` and score it at each ``k``.

    Retrieves once per query at ``max(ks)`` and scores prefixes of that ranking. A result without
    a ``content_id`` raises: it cannot be matched to a label, and scoring it as a miss would
    silently understate the retriever.
    """
    if not golden:
        raise ValueError("golden set is empty")
    if not ks or any(k < 1 for k in ks):
        raise ValueError("ks must be non-empty and >= 1")
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    ks_sorted = tuple(sorted(set(ks)))
    semaphore = asyncio.Semaphore(concurrency)

    async def run(item: GoldenQuery) -> QueryScore:
        async with semaphore:
            results = await retriever.retrieve(item.query, filters, top_k=ks_sorted[-1])
        if any(r.content_id is None for r in results):
            raise ValueError(f"retriever returned a result with no content_id for {item.query!r}")
        ranked = [r.content_id for r in results if r.content_id is not None]
        return QueryScore(
            query=item.query,
            kind=item.kind,
            retrieved=tuple(ranked),
            first_relevant_rank=next(
                (rank for rank, i in enumerate(ranked, start=1) if i in item.relevant), None
            ),
            scores={
                "recall": {k: recall_at_k(ranked, item.relevant, k) for k in ks_sorted},
                "mrr": {k: reciprocal_rank_at_k(ranked, item.relevant, k) for k in ks_sorted},
                "ndcg": {k: ndcg_at_k(ranked, item.relevant, k) for k in ks_sorted},
            },
        )

    return EvalReport(ks=ks_sorted, queries=list(await asyncio.gather(*(run(g) for g in golden))))
