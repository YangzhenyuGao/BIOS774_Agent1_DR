from sklearn.manifold import Isomap

from .base import SklearnMethod


class Method(SklearnMethod):
    method_id = "isomap"
    estimator = Isomap

    def default_params(self, X):
        return {"n_neighbors": 15, "eigen_solver": "arpack", "n_jobs": 1}

    def tuning_grid(self, X):
        return [dict(self.default_params(X), n_neighbors=k) for k in [15, 5, 30]]
