from umap import UMAP

from .base import SklearnMethod


class Method(SklearnMethod):
    method_id = "umap"
    estimator = UMAP
    stochastic = True

    def default_params(self, X):
        return {"n_neighbors": 15, "min_dist": 0.1, "n_jobs": 1, "n_epochs": 300}

    def tuning_grid(self, X):
        return [
            dict(self.default_params(X), **p)
            for p in [{}, {"n_neighbors": 30}, {"n_neighbors": 50}, {"min_dist": 0.5}]
        ]
