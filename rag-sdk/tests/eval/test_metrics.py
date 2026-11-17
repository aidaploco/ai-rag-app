import math

import pytest
from rag_sdk.eval.metrics import ndcg_at_k, recall_at_k, reciprocal_rank_at_k

# ranked: x a y b z   relevant: {a, b, c}   (c is never retrieved)
RANKED = ["x", "a", "y", "b", "z"]
RELEVANT = {"a", "b", "c"}


def test_recall_counts_relevant_found_over_all_relevant():
    assert recall_at_k(RANKED, RELEVANT, k=5) == pytest.approx(2 / 3)
    assert recall_at_k(RANKED, RELEVANT, k=2) == pytest.approx(1 / 3)  # only "a" in top 2
    assert recall_at_k(RANKED, RELEVANT, k=1) == 0.0


def test_reciprocal_rank_uses_first_relevant_within_k():
    assert reciprocal_rank_at_k(RANKED, RELEVANT, k=5) == pytest.approx(1 / 2)
    assert reciprocal_rank_at_k(RANKED, RELEVANT, k=1) == 0.0  # first hit is past the cut
    assert reciprocal_rank_at_k(["a"], {"a"}, k=3) == 1.0


def test_ndcg_binary_hand_computed():
    # DCG  = 1/log2(3) + 1/log2(5)           (a at rank 2, b at rank 4)
    # IDCG = 1/log2(2) + 1/log2(3) + 1/log2(4)   (three relevant, ideal ranks 1-3)
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(RANKED, RELEVANT, k=5) == pytest.approx(dcg / idcg)


def test_ndcg_ideal_uses_all_labelled_relevant_not_just_retrieved():
    # Only "a" retrieved, at rank 1; "b" is relevant but missing. Must be < 1.
    assert ndcg_at_k(["a", "z"], {"a", "b"}, k=2) == pytest.approx(1.0 / (1.0 + 1 / math.log2(3)))


def test_ndcg_graded_gains():
    # ranked b (gain 1) then a (gain 3); ideal is a then b.
    dcg = 1 / math.log2(2) + 3 / math.log2(3)
    idcg = 3 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(["b", "a"], {"a": 3, "b": 1}, k=2) == pytest.approx(dcg / idcg)


def test_perfect_ranking_scores_one_everywhere():
    ranked = ["a", "b", "z"]
    rel = {"a", "b"}
    assert recall_at_k(ranked, rel, 3) == 1.0
    assert reciprocal_rank_at_k(ranked, rel, 3) == 1.0
    assert ndcg_at_k(ranked, rel, 3) == pytest.approx(1.0)


def test_total_miss_scores_zero_everywhere():
    assert recall_at_k(["x", "y"], {"a"}, 2) == 0.0
    assert reciprocal_rank_at_k(["x", "y"], {"a"}, 2) == 0.0
    assert ndcg_at_k(["x", "y"], {"a"}, 2) == 0.0


def test_metrics_differ_for_the_same_ranking():
    # recall=1/2, rr=1/3, ndcg strictly between: one of two relevant docs, found at rank 3.
    ranked, rel = ["x", "y", "a"], {"a", "b"}
    recall = recall_at_k(ranked, rel, 3)
    rr = reciprocal_rank_at_k(ranked, rel, 3)
    ndcg = ndcg_at_k(ranked, rel, 3)
    assert recall == pytest.approx(1 / 2)
    assert rr == pytest.approx(1 / 3)
    assert ndcg == pytest.approx((1 / 2) / (1 + 1 / math.log2(3)))
    assert len({round(recall, 6), round(rr, 6), round(ndcg, 6)}) == 3


def test_repeated_ids_count_once():
    assert recall_at_k(["a", "a", "a"], {"a", "b"}, k=3) == pytest.approx(1 / 2)
    assert ndcg_at_k(["a", "a", "b"], {"a", "b"}, k=3) == pytest.approx(
        (1 + 1 / math.log2(3)) / (1 + 1 / math.log2(3))
    )


def test_empty_relevant_set_is_rejected():
    for fn in (recall_at_k, reciprocal_rank_at_k, ndcg_at_k):
        with pytest.raises(ValueError):
            fn(["a"], set(), 1)


def test_k_must_be_positive():
    for fn in (recall_at_k, reciprocal_rank_at_k, ndcg_at_k):
        with pytest.raises(ValueError):
            fn(["a"], {"a"}, 0)


def test_negative_gain_rejected():
    with pytest.raises(ValueError):
        ndcg_at_k(["a"], {"a": -1}, 1)
