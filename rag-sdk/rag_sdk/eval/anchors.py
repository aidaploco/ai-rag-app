"""Scoring by anchor containment, for comparing chunking strategies.

Chunking changes chunk text and therefore content IDs, so ID-based labels cannot compare
strategies. Instead a query is labelled by *anchors*: phrases in the source that answer it. A
retrieved chunk "finds" an anchor if it contains the anchor text, and an anchor that a chunk
boundary cuts in two is found by no chunk (a miss, and counted by ``split_anchors``).

Retrieval is compared at an equal *token budget*: a strategy with huge chunks cannot win just by
returning a whole document, because only whole chunks that fit within the budget count.
"""

from collections.abc import Callable, Sequence


def anchors_found(chunks: Sequence[str], anchors: Sequence[str]) -> set[str]:
    return {a for a in anchors if any(a in c for c in chunks)}


def anchor_recall_at_budget(
    ranked_chunks: Sequence[str],
    anchors: Sequence[str],
    count_tokens: Callable[[str], int],
    budget: int,
) -> float:
    """Fraction of ``anchors`` contained in the best-ranked chunks that fit in ``budget`` tokens.

    Walks ``ranked_chunks`` in order, adding each while the running token total stays within
    ``budget``, and stops at the first chunk that would exceed it (rank order is not skipped, and
    a chunk larger than the whole budget counts for nothing).
    """
    if not anchors:
        raise ValueError("anchors must not be empty")
    shown: list[str] = []
    used = 0
    for chunk in ranked_chunks:
        used += count_tokens(chunk)
        if used > budget:
            break
        shown.append(chunk)
    return len(anchors_found(shown, anchors)) / len(set(anchors))


def first_hit_rank(ranked_chunks: Sequence[str], anchors: Sequence[str]) -> int | None:
    """1-based rank of the first chunk containing any anchor, or None (mean reciprocal = MRR)."""
    for rank, chunk in enumerate(ranked_chunks, start=1):
        if any(a in chunk for a in anchors):
            return rank
    return None


def split_anchors(source: str, chunks: Sequence[str], anchors: Sequence[str]) -> list[str]:
    """Anchors present in ``source`` that no single chunk contains (a boundary cut them)."""
    return [a for a in anchors if a in source and not any(a in c for c in chunks)]
