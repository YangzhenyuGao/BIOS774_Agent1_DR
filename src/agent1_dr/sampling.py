import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def cohort_indices(n, size, seed, labels=None):
    """Stratification controls cohort composition only; labels never enter fitting."""
    size = min(n, size)
    if size == n:
        return np.arange(n)
    if labels is None:
        return np.sort(np.random.default_rng(seed).choice(n, size, replace=False))
    # Deterministic proportional allocation also handles rare singleton classes.
    classes, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    if counts.min() >= 2 and size >= len(classes) and n - size >= len(classes):
        _, idx = next(
            StratifiedShuffleSplit(n_splits=1, test_size=size, random_state=seed).split(
                np.zeros((n, 1)), labels
            )
        )
        return np.sort(idx)
    desired = counts * size / n
    allocated = np.floor(desired).astype(int)
    order = np.argsort(-(desired - allocated), kind="stable")
    allocated[order[: size - allocated.sum()]] += 1
    rng = np.random.default_rng(seed)
    return np.sort(
        np.concatenate(
            [rng.choice(np.flatnonzero(inverse == j), k, replace=False) for j, k in enumerate(allocated)]
        )
    )
