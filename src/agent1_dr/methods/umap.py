from umap import UMAP

from .base import SklearnMethod


class Method(SklearnMethod):
    method_id = "umap"
    estimator = UMAP
    stochastic = True

    def default_params(self, X):
        return {"n_neighbors": 15, "min_dist": 0.1, "n_jobs": 1, "n_epochs": 300}

    def tuning_grid(self, X):
        # The README grid: n_neighbors 15/30/50 x min_dist 0.1/0.5 (default first).
        return [
            dict(self.default_params(X), n_neighbors=k, min_dist=d)
            for d in [0.1, 0.5]
            for k in [15, 30, 50]
        ]
