from sklearn.decomposition import KernelPCA

from .base import SklearnMethod, median_gamma


class Method(SklearnMethod):
    method_id = "kernel_pca"
    estimator = KernelPCA

    def default_params(self, X):
        return {"kernel": "rbf", "gamma": median_gamma(X), "eigen_solver": "arpack"}

    def tuning_grid(self, X):
        p = self.default_params(X)
        return [dict(p, gamma=p["gamma"] * f) for f in [1, 0.5, 2]]
