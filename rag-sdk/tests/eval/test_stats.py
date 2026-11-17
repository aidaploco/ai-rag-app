import pytest
from rag_sdk.eval.stats import compare_paired


def test_identical_arms_have_zero_difference_and_all_ties():
    r = compare_paired([0.5, 0.2, 1.0], [0.5, 0.2, 1.0])
    assert (r.mean_difference, r.ci_low, r.ci_high) == (0.0, 0.0, 0.0)
    assert (r.a_better, r.b_better, r.ties) == (0, 0, 3)
    assert not r.significant


def test_counts_and_mean_are_hand_computed():
    a = [0.0, 0.5, 1.0, 0.4]
    b = [1.0, 0.5, 0.5, 0.4]
    r = compare_paired(a, b)
    assert r.n == 4 and r.mean_difference == pytest.approx((1.0 + 0.0 - 0.5 + 0.0) / 4)
    assert (r.b_better, r.a_better, r.ties) == (1, 1, 2)


def test_a_consistent_improvement_is_significant_and_a_mixed_one_is_not():
    a = [0.5] * 40
    assert compare_paired(a, [0.6] * 40).significant
    mixed = compare_paired([0.5] * 40, [0.9, 0.1] * 20)
    assert not mixed.significant and mixed.ci_low < 0 < mixed.ci_high


def test_is_deterministic_for_a_seed_and_the_interval_contains_the_mean():
    a, b = [0.1, 0.9, 0.4, 0.4, 0.7], [0.2, 0.8, 0.9, 0.3, 0.7]
    r1, r2 = compare_paired(a, b, seed=3), compare_paired(a, b, seed=3)
    assert r1 == r2
    assert r1.ci_low <= r1.mean_difference <= r1.ci_high


@pytest.mark.parametrize(
    ("a", "b", "kwargs"),
    [
        ([1.0], [1.0, 2.0], {}),
        ([], [], {}),
        ([1.0], [1.0], {"confidence": 1.0}),
        ([1.0], [1.0], {"resamples": 0}),
    ],
)
def test_rejects_bad_input(a, b, kwargs):
    with pytest.raises(ValueError):
        compare_paired(a, b, **kwargs)
