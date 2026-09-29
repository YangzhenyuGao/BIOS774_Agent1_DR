from sklearn.manifold import LocallyLinearEmbedding

from .base import SklearnMethod


class Method(SklearnMethod):
    method_id = "lle"
    estimator = LocallyLinearEmbedding

    def default_params(self, X):
        return {"n_neighbors": 20, "reg": 0.001, "eigen_solver": "arpack", "n_jobs": 1}

    def tuning_grid(self, X):
        default = self.default_params(X)
        return [dict(default, n_neighbors=k) for k in [20, 10, 30]] + [dict(default, reg=0.01)]
