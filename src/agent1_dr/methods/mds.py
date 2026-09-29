import inspect

from sklearn.manifold import MDS

from .base import SklearnMethod


class Method(SklearnMethod):
    method_id = "mds"
    estimator = MDS
    stochastic = True

    def default_params(self, X):
        # Metric SMACOF from the classical-MDS solution (the scikit-learn 1.10 default);
        # a random start with n_init=1 stopped at max_iter on PBMC3k in v1.
        params = {"n_init": 1, "max_iter": 1000, "normalized_stress": "auto", "n_jobs": 1}
        if "metric_mds" in inspect.signature(MDS).parameters:
            params.update(metric_mds=True, metric="euclidean", init="classical_mds")
        else:
            params["metric"] = True
        return params

    def tuning_grid(self, X):
        default = self.default_params(X)
        if "init" not in default:
            return [default, dict(default, n_init=4)]
        return [default, dict(default, init="random", n_init=4)]

    def fit_transform(self, X, seed, params):
        Y, diagnostics = super().fit_transform(X, seed, params)
        diagnostics["converged"] = diagnostics.get("n_iter_", 0) < params["max_iter"]
        return Y, diagnostics
