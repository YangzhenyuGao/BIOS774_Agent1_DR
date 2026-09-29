from dataclasses import asdict
from itertools import combinations

import numpy as np
from sklearn.manifold import trustworthiness
from sklearn.metrics import pairwise_distances, silhouette_score

from .schemas import EvaluationResult


def neighbors(X, k):
    # Stable ordering and explicit diagonal exclusion remain correct with duplicate coordinates.
    d = pairwise_distances(X)
    np.fill_diagonal(d, np.inf)
    return np.argsort(d, axis=1, kind="stable")[:, :k]


def overlap(a, b):
    return float(np.mean([len(set(x) & set(y)) / len(x) for x, y in zip(a, b)]))


def evaluate(X, Y, labels, k):
    k = min(k, (len(X) - 1) // 2)
    sil, note = None, "labels unavailable or fewer than two classes"
    if labels is not None:
        mask = np.asarray(labels) != "unannotated"
        lbl = np.asarray(labels)[mask]
        if 1 < len(np.unique(lbl)) < len(lbl):
            sil = float(silhouette_score(Y[mask], lbl, sample_size=min(len(lbl), 2000), random_state=774))
            note = "secondary only; unannotated observations excluded; at most 2000 samples"
    return asdict(
        EvaluationResult(
            float(trustworthiness(X, Y, n_neighbors=k)),
            overlap(neighbors(X, k), neighbors(Y, k)),
            sil,
            note,
            k,
            metric_sample_size=len(X),
        )
    )


def stability(embeddings, k):
    if len(embeddings) < 2:
        return None
    graphs = [neighbors(y, k) for y in embeddings]
    return float(np.mean([overlap(a, b) for a, b in combinations(graphs, 2)]))


def quality(trust, recall, stable):
    # All component metrics already share [0,1]; no dataset-dependent normalization.
    return float(0.45 * trust + 0.45 * recall + 0.10 * stable)
