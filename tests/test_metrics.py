import numpy as np

from agent1_dr.evaluation import evaluate, neighbors, stability


def test_identity_and_scramble():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(100, 2))
    identity = evaluate(X, X, None, 5)
    scrambled = evaluate(X, rng.permutation(X), None, 5)
    assert identity["trustworthiness"] == 1 and identity["neighbor_recall"] == 1
    assert scrambled["neighbor_recall"] < 0.3
    assert stability([X, X * 3 + 2], 5) == 1


def test_duplicates_exclude_self():
    n = neighbors(np.zeros((8, 2)), 3)
    assert all(i not in row for i, row in enumerate(n))
