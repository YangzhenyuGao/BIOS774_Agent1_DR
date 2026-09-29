from abc import ABC, abstractmethod

import numpy as np
from scipy.spatial.distance import pdist


class DRMethod(ABC):
    method_id: str
    stochastic = False

    def validate_input(self, X, params):
        if X.ndim != 2 or min(X.shape) < 3 or not np.isfinite(X).all():
            raise ValueError("Expected finite matrix with >=3 samples and features")
        result = dict(params)
        changes = []
        for key in ("n_neighbors", "inducing_points", "batch_size"):
            if key in result and result[key] >= len(X):
                result[key] = len(X) - 1
                changes.append(f"{key} clipped to {len(X) - 1} for cohort size")
        if "perplexity" in result and result["perplexity"] >= len(X):
            result["perplexity"] = max(2, (len(X) - 1) / 3)
            changes.append(f"perplexity clipped to {result['perplexity']}")
        return result, changes

    def default_params(self, X):
        return {}

    def tuning_grid(self, X):
        return [self.default_params(X)]

    @abstractmethod
    def fit_transform(self, X, seed, params):
        """Return coordinates and method-specific diagnostics; never receive labels."""


def median_gamma(X):
    d = pdist(X[: min(len(X), 500)], metric="sqeuclidean")
    return 1 / max(float(np.median(d[d > 0])), 1e-12)


class SklearnMethod(DRMethod):
    estimator: type

    def fit_transform(self, X, seed, params):
        import inspect

        kwargs = dict(n_components=2, **params)
        if "random_state" in inspect.signature(self.estimator).parameters:
            kwargs["random_state"] = seed
        model = self.estimator(**kwargs)
        Y = model.fit_transform(X)
        diag = {}
        for attribute in ("explained_variance_ratio_", "n_iter_", "stress_", "kl_divergence_"):
            if hasattr(model, attribute):
                value = getattr(model, attribute)
                diag[attribute] = value.tolist() if isinstance(value, np.ndarray) else value
        return Y, diag
