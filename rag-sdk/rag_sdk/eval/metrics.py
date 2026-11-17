"""Retrieval metrics as pure functions over a ranked list of content IDs.

Conventions (shared by every function here):

- ``ranked`` is the retrieved IDs, best first. A repeated ID counts once, at its first position.
- ``relevant`` is the labelled set of relevant IDs; it must not be empty (a query with no
  relevant documents has undefined recall, so it is rejected rather than scored as 0 or 1).
- ``k`` must be >= 1 and only the first ``k`` ranked IDs are considered.
"""

import math
from collections.abc import Collection, Mapping, Sequence


def _top_k(ranked: Sequence[str], k: int) -> list[str]:
    if k < 1:
        raise ValueError("k must be >= 1")
    return list(dict.fromkeys(ranked))[:k]


def _require_relevant(relevant: Collection[str]) -> None:
    if not relevant:
        raise ValueError("relevant must not be empty")


def recall_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Fraction of the relevant IDs found in the top ``k``."""
    _require_relevant(relevant)
    relevant_set = set(relevant)
    hits = sum(1 for doc_id in _top_k(ranked, k) if doc_id in relevant_set)
    return hits / len(relevant_set)


def reciprocal_rank_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    """1 / rank of the first relevant ID within the top ``k``, else 0. Its mean is MRR@k."""
    _require_relevant(relevant)
    relevant_set = set(relevant)
    for rank, doc_id in enumerate(_top_k(ranked, k), start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    ranked: Sequence[str], relevant: Collection[str] | Mapping[str, int], k: int
) -> float:
    """Normalised discounted cumulative gain over the top ``k``.

    ``relevant`` is either a collection of IDs (binary relevance, gain 1) or a mapping of ID to
    a graded gain. The ideal ranking is built from *all* labelled relevant IDs, not just the
    retrieved ones, so missing a relevant document lowers the score.
    """
    _require_relevant(relevant)
    gains: Mapping[str, int] = (
        relevant if isinstance(relevant, Mapping) else dict.fromkeys(relevant, 1)
    )
    if any(g < 0 for g in gains.values()):
        raise ValueError("gains must be non-negative")

    def dcg(values: Sequence[int]) -> float:
        return sum(v / math.log2(rank + 1) for rank, v in enumerate(values, start=1))

    actual = dcg([gains.get(doc_id, 0) for doc_id in _top_k(ranked, k)])
    ideal = dcg(sorted(gains.values(), reverse=True)[:k])
    return actual / ideal if ideal > 0 else 0.0
