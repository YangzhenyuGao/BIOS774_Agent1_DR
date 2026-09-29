import inspect

from sklearn.manifold import MDS

from .base import SklearnMethod


class Method(SklearnMethod):
    method_id = "mds"
    estimator = MDS
    stochastic = True

    def default_params(self, X):
        params = {"n_init": 1, "max_iter": 300, "normalized_stress": "auto", "n_jobs": 1}
        if "metric_mds" in inspect.signature(MDS).parameters:
            params.update(metric_mds=True, metric="euclidean", init="random")
        else:
            params["metric"] = True
        return params

    def tuning_grid(self, X):
        return [self.default_params(X), dict(self.default_params(X), n_init=2)]
