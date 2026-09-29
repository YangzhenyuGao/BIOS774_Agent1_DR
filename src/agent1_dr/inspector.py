import hashlib

import numpy as np
from scipy import sparse

from .schemas import DatasetProfile


def inspect_data(name, X, labels, provenance, feature_names=None):
    n, d = X.shape
    # Profile all values without densifying sparse matrices; feature duplicates use sparse column hashes.
    values = X.data if sparse.issparse(X) else X.ravel()
    finite = np.isfinite(values)
    if not finite.all():
        raise ValueError("Input has non-finite values; explicit imputation policy required")
    if sparse.issparse(X):
        means = np.asarray(X.mean(axis=0)).ravel()
        variance = np.maximum(0, np.asarray(X.power(2).mean(axis=0)).ravel() - means**2)
        csc = X.tocsc(copy=True)
        csc.sort_indices()
        fingerprints = [
            hashlib.sha256(csc[:, j].indices.tobytes() + csc[:, j].data.tobytes()).hexdigest()
            for j in range(d)
        ]
        nonzero = X.count_nonzero()
    else:
        means, variance = X.mean(axis=0), X.var(axis=0)
        fingerprints = [hashlib.sha256(np.ascontiguousarray(X[:, j]).tobytes()).hexdigest() for j in range(d)]
        nonzero = np.count_nonzero(X)
    classes, counts = np.unique(labels, return_counts=True) if labels is not None else ([], [])
    implicit_zeros = nonzero < n * d
    integer_valued = bool(np.issubdtype(values.dtype, np.integer) or np.all(np.mod(values, 1) == 0))
    row_sums = np.asarray(X.sum(axis=1), dtype=np.float64).ravel()
    names = np.asarray(feature_names, dtype=str) if feature_names is not None else None
    mito = int(np.char.startswith(np.char.upper(names), "MT-").sum()) if names is not None else 0
    stats = {
        "dtype": str(X.dtype),
        "numeric_features": d,
        "non_numeric_features": 0,
        "missing_values": int((~finite).sum()),
        "sparsity": 1 - float(nonzero) / (n * d),
        "negative_stored_values": int((values < 0).sum()),
        "constant_features": int((variance == 0).sum()),
        "duplicate_features": d - len(set(fingerprints)),
        "feature_mean_quantiles": np.quantile(means, [0, 0.5, 1]),
        "feature_variance_quantiles": np.quantile(variance, [0, 0.5, 1]),
        "dense_float64_mb": n * d * 8 / 2**20,
        "pairwise_float64_mb": n * n * 8 / 2**20,
        "label_counts": dict(zip(map(str, classes), map(int, counts))),
        # Evidence consumed by the preprocessing planner (planner.py).
        "nonnegative": bool(values.min() >= 0) if len(values) else True,
        "integer_valued": integer_valued,
        "min_value": float(min(values.min(), 0) if implicit_zeros else values.min()),
        "max_value": float(values.max()),
        "row_sum_quantiles": np.quantile(row_sums, [0, 0.5, 1]),
        "feature_names_available": names is not None,
        "mitochondrial_features": mito,
    }
    return DatasetProfile(name, n, d, stats, provenance)
