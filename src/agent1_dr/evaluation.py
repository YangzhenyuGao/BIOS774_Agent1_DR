"""Label-free embedding diagnostics against one fixed high-dimensional reference.

Every neighborhood metric uses Euclidean distance on the common input representation.
Reference neighborhoods are computed once per cohort and shared by every method.
Distances are processed in row chunks, so memory grows linearly with cohort size.
"""

from itertools import combinations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.neighbors import NearestNeighbors

CHUNK = 256


def _chunks(n):
    for start in range(0, n, CHUNK):
        yield np.arange(start, min(start + CHUNK, n))


def _row_knn(d, k):
    """First k columns of a stable argsort of each row (ties ordered by index).

    Equivalent to np.argsort(d, kind="stable")[:, :k] but linear per row.
    """
    kth = np.partition(d, k - 1, axis=1)[:, k - 1]
    out = np.empty((len(d), k), dtype=np.int64)
    for i, (row, cut) in enumerate(zip(d, kth)):
        candidates = np.flatnonzero(row <= cut)
        out[i] = candidates[np.lexsort((candidates, row[candidates]))][:k]
    return out


def knn(Y, k):
    """k nearest neighbors of every row of Y, self excluded, stable tie order."""
    Y = np.asarray(Y, dtype=np.float64)
    out = np.empty((len(Y), k), dtype=np.int64)
    for rows in _chunks(len(Y)):
        d = pairwise_distances(Y[rows], Y)
        d[np.arange(len(rows)), rows] = np.inf
        out[rows] = _row_knn(d, k)
    return out


def overlap(a, b):
    """Mean |N_a(i) intersection N_b(i)| / k over observations."""
    return float(np.mean([len(set(x) & set(y)) / len(x) for x, y in zip(a, b)]))


def _pairs(n, max_pairs, seed):
    if n * (n - 1) // 2 <= max_pairs:
        return np.triu_indices(n, 1)
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, max_pairs)
    j = (i + rng.integers(1, n, max_pairs)) % n
    return i, j


def _spearman(a, b):
    rho = spearmanr(a, b).statistic
    return float(rho) if np.isfinite(rho) else 0.0


class Reference:
    """Neighborhood and pairwise-distance structure of the common input representation."""

    def __init__(self, X, k_max, max_pairs=500_000, seed=774):
        self.X = np.asarray(X, dtype=np.float64)
        self.n = len(self.X)
        self.k_max = min(k_max, self.n - 1)
        self.knn = knn(self.X, self.k_max)
        self.pairs = _pairs(self.n, max_pairs, seed)
        self.pair_distances = np.linalg.norm(self.X[self.pairs[0]] - self.X[self.pairs[1]], axis=1)
        self.seed = seed
        self.max_pairs = max_pairs

    def subset(self, index):
        return Reference(self.X[index], self.k_max, self.max_pairs, self.seed)

    def clip_k(self, k):
        # Trustworthiness requires k < n/2; the same k is used for recall.
        return max(1, min(k, (self.n - 1) // 2, self.k_max))

    def trustworthiness(self, Y, ks):
        """sklearn.manifold.trustworthiness for several k in one pass (identical without distance ties)."""
        ks = [self.clip_k(k) for k in ks]
        neighbors = NearestNeighbors(n_neighbors=max(ks)).fit(Y).kneighbors(return_distance=False)
        penalty = dict.fromkeys(ks, 0.0)
        for rows in _chunks(self.n):
            d = pairwise_distances(self.X[rows], self.X)
            d[np.arange(len(rows)), rows] = np.inf
            for k in ks:
                v = np.take_along_axis(d, neighbors[rows, :k], axis=1)
                ranks = (d[:, None, :] < v[:, :, None]).sum(axis=2) + 1
                excess = ranks - k
                penalty[k] += float(excess[excess > 0].sum())
        n = self.n
        return {k: 1.0 - penalty[k] * (2.0 / (n * k * (2.0 * n - 3.0 * k - 1.0))) for k in ks}

    def recall(self, Y, ks):
        ks = [self.clip_k(k) for k in ks]
        embedded = knn(Y, max(ks))
        return {k: overlap(self.knn[:, :k], embedded[:, :k]) for k in ks}

    def global_structure(self, Y):
        """Spearman correlation of reference and embedding pairwise distances, clipped at 0."""
        Y = np.asarray(Y, dtype=np.float64)
        embedded = np.linalg.norm(Y[self.pairs[0]] - Y[self.pairs[1]], axis=1)
        return max(_spearman(self.pair_distances, embedded), 0.0)


def collapse_ratio(Y):
    """Median distance to the coordinate-wise median divided by the 99th-percentile distance.

    Near 0.4 for a Gaussian cloud; near 0 when most observations are compressed into a
    negligible part of the plotted extent while a minority defines the axes.
    """
    Y = np.asarray(Y, dtype=np.float64)
    r = np.linalg.norm(Y - np.median(Y, axis=0), axis=1)
    outer = float(np.quantile(r, 0.99))
    return float(np.median(r) / outer) if outer > 0 else 0.0


def evaluate(ref, Y, labels, k, k2):
    """All label-free metrics for one embedding plus the secondary label silhouette."""
    trust = ref.trustworthiness(Y, [k, k2])
    recall = ref.recall(Y, [k, k2])
    k, k2 = ref.clip_k(k), ref.clip_k(k2)
    sil, note = None, "labels unavailable or fewer than two classes"
    if labels is not None:
        mask = np.asarray(labels) != "unannotated"
        lbl = np.asarray(labels)[mask]
        if 1 < len(np.unique(lbl)) < len(lbl):
            sil = float(silhouette_score(Y[mask], lbl, sample_size=min(len(lbl), 2000), random_state=774))
            note = "secondary only; unannotated observations excluded; at most 2000 samples"
    return {
        "trustworthiness": trust[k],
        "neighbor_recall": recall[k],
        "global_structure": ref.global_structure(Y),
        "trustworthiness_k2": trust[k2],
        "neighbor_recall_k2": recall[k2],
        "collapse_ratio": collapse_ratio(Y),
        "silhouette": sil,
        "silhouette_note": note,
        "k": k,
        "k2": k2,
        "distance_metric": "euclidean",
        "metric_sample_size": ref.n,
        "global_pairs": len(ref.pairs[0]),
    }


def seed_stability(embeddings, k):
    """Mean pairwise kNN overlap between repeated-seed embeddings of the same cohort."""
    if len(embeddings) < 2:
        return None
    graphs = [knn(y, k) for y in embeddings]
    return float(np.mean([overlap(a, b) for a, b in combinations(graphs, 2)]))


def seed_invariant(embeddings):
    """True when every repeated seed produced an identical embedding."""
    return len(embeddings) > 1 and all(np.array_equal(embeddings[0], y) for y in embeddings[1:])


def subsample_stability(fits, k, max_pairs=200_000, seed=774):
    """Stability under observation resampling, comparable across all methods.

    fits: list of (cohort indices, embedding) from refits on overlapping subsamples.
    Returns mean kNN overlap on shared observations (primary) and mean Spearman
    correlation of shared pairwise distances (diagnostic).
    """
    local, global_ = [], []
    for (ia, ya), (ib, yb) in combinations(fits, 2):
        _, pa, pb = np.intersect1d(ia, ib, assume_unique=True, return_indices=True)
        kk = min(k, (len(pa) - 1) // 2)
        if kk < 1:
            continue
        local.append(overlap(knn(ya[pa], kk), knn(yb[pb], kk)))
        i, j = _pairs(len(pa), max_pairs, seed)
        da = np.linalg.norm(ya[pa][i] - ya[pa][j], axis=1)
        db = np.linalg.norm(yb[pb][i] - yb[pb][j], axis=1)
        global_.append(max(_spearman(da, db), 0.0))
    if not local:
        return None, None
    return float(np.mean(local)), float(np.mean(global_))


def quality(metrics, weights):
    """Weighted sum of label-free components that all lie in [0, 1]."""
    return float(sum(weights[name] * metrics[name] for name in weights))
