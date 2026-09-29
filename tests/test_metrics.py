import numpy as np
from sklearn.manifold import trustworthiness
from sklearn.metrics import pairwise_distances

from agent1_dr.evaluation import (
    Reference,
    collapse_ratio,
    evaluate,
    knn,
    overlap,
    seed_invariant,
    seed_stability,
    subsample_stability,
)


def test_identity_and_scramble():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(100, 2))
    ref = Reference(X, k_max=10)
    identity = evaluate(ref, X, None, 5, 10)
    scrambled = evaluate(ref, rng.permutation(X), None, 5, 10)
    assert identity["trustworthiness"] == 1 and identity["neighbor_recall"] == 1
    assert identity["global_structure"] > 0.999
    assert scrambled["neighbor_recall"] < 0.3 and scrambled["global_structure"] < 0.2
    assert seed_stability([X, X * 3 + 2], 5) == 1


def test_trustworthiness_matches_sklearn_and_recall_matches_v1():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(150, 12))
    Y = X[:, :2] + 0.4 * rng.normal(size=(150, 2))
    ref = Reference(X, k_max=20)
    for k in (5, 15):
        assert abs(ref.trustworthiness(Y, [k])[k] - trustworthiness(X, Y, n_neighbors=k)) < 1e-12
        d = pairwise_distances(X)
        np.fill_diagonal(d, np.inf)
        e = pairwise_distances(Y)
        np.fill_diagonal(e, np.inf)
        v1 = overlap(np.argsort(d, axis=1, kind="stable")[:, :k], np.argsort(e, axis=1, kind="stable")[:, :k])
        assert ref.recall(Y, [k])[k] == v1


def test_duplicates_exclude_self_and_keep_stable_order():
    n = knn(np.zeros((8, 2)), 3)
    assert all(i not in row for i, row in enumerate(n))
    Z = np.repeat(np.random.default_rng(1).normal(size=(10, 2)), 3, axis=0)
    d = pairwise_distances(Z)
    np.fill_diagonal(d, np.inf)
    assert np.array_equal(knn(Z, 4), np.argsort(d, axis=1, kind="stable")[:, :4])


def test_subsample_stability_known_answers():
    rng = np.random.default_rng(0)
    n = 400
    subs = [np.sort(rng.choice(n, 320, replace=False)) for _ in range(3)]
    fixed = rng.normal(size=(n, 2))
    same, _ = subsample_stability([(s, fixed[s]) for s in subs], 10)
    random, _ = subsample_stability([(s, rng.normal(size=(len(s), 2))) for s in subs], 10)
    assert same == 1 and random < 0.1


def test_collapse_ratio_separates_healthy_from_collapsed():
    rng = np.random.default_rng(2)
    assert collapse_ratio(rng.normal(size=(500, 2))) > 0.3
    collapsed = np.vstack([rng.normal(scale=0.01, size=(450, 2)), np.c_[rng.uniform(0.1, 1, 50), np.zeros(50)]])
    assert collapse_ratio(collapsed) < 0.05
    assert collapse_ratio(np.zeros((20, 2))) == 0


def test_seed_invariance_detection():
    y = np.random.default_rng(4).normal(size=(30, 2))
    assert seed_invariant([y, y.copy()]) and not seed_invariant([y, y + 1e-9])
