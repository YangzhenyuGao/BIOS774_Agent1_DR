from sklearn.manifold import TSNE

from .base import SklearnMethod


class Method(SklearnMethod):
    method_id = "tsne"
    estimator = TSNE
    stochastic = True

    def default_params(self, X):
        return {"perplexity": 30, "learning_rate": "auto", "init": "pca", "max_iter": 750, "n_jobs": 1}

    def tuning_grid(self, X):
        return [dict(self.default_params(X), perplexity=k) for k in [30, 15, 50]]
