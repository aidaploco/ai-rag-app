import pytest
from rag_sdk.eval.anchors import (
    anchor_recall_at_budget,
    anchors_found,
    first_hit_rank,
    split_anchors,
)


def words(text: str) -> int:
    return len(text.split())


def test_anchors_found_is_containment_not_equality():
    assert anchors_found(["the quick brown fox", "lazy dog"], ["brown fox", "cat"]) == {"brown fox"}


def test_recall_counts_only_chunks_that_fit_the_budget_in_rank_order():
    ranked = ["aa bb", "cc dd ee", "target phrase here", "ff"]  # tokens: 2, 3, 3, 1
    anchors = ["target phrase", "ff"]
    assert anchor_recall_at_budget(ranked, anchors, words, budget=5) == 0.0  # stops before #3
    assert anchor_recall_at_budget(ranked, anchors, words, budget=8) == 0.5  # #1-3 fit, "ff" not
    assert anchor_recall_at_budget(ranked, anchors, words, budget=9) == 1.0


def test_a_chunk_bigger_than_the_whole_budget_counts_for_nothing():
    huge = "target " + "pad " * 50
    assert anchor_recall_at_budget([huge], ["target"], words, budget=10) == 0.0
    assert anchor_recall_at_budget([huge], ["target"], words, budget=100) == 1.0


def test_budget_walk_does_not_skip_to_a_smaller_later_chunk():
    ranked = ["x " * 20, "target"]  # first chunk exceeds the budget; the small one is not shown
    assert anchor_recall_at_budget(ranked, ["target"], words, budget=5) == 0.0


def test_repeated_anchors_are_counted_once():
    assert anchor_recall_at_budget(["a b"], ["a", "a"], words, budget=5) == 1.0


def test_recall_rejects_empty_anchors():
    with pytest.raises(ValueError):
        anchor_recall_at_budget(["x"], [], words, budget=5)


def test_first_hit_rank_is_one_based_and_none_when_absent():
    assert first_hit_rank(["no", "no", "yes here", "yes"], ["yes"]) == 3
    assert first_hit_rank(["no"], ["yes"]) is None
    assert first_hit_rank([], ["yes"]) is None


def test_split_anchors_finds_those_cut_by_a_boundary():
    source = "alpha beta gamma delta"
    assert split_anchors(source, ["alpha beta", "gamma delta"], ["beta gamma", "alpha"]) == [
        "beta gamma"
    ]
    assert split_anchors(source, ["alpha beta gamma delta"], ["beta gamma"]) == []
    assert (
        split_anchors(source, ["alpha"], ["not in source"]) == []
    )  # absent anchors are not "split"
