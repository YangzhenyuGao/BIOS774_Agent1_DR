from sklearn.manifold import SpectralEmbedding

from .base import SklearnMethod


class Method(SklearnMethod):
    method_id = "laplacian"
    estimator = SpectralEmbedding

    def default_params(self, X):
        return {"n_neighbors": 30, "affinity": "nearest_neighbors", "eigen_solver": "arpack", "n_jobs": 1}

    def tuning_grid(self, X):
        return [dict(self.default_params(X), n_neighbors=k) for k in [30, 10, 50]]
