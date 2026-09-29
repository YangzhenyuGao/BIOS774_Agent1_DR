from sklearn.decomposition import PCA

from .base import SklearnMethod


class Method(SklearnMethod):
    method_id = "pca"
    estimator = PCA

    def default_params(self, X):
        return {"svd_solver": "full"}
