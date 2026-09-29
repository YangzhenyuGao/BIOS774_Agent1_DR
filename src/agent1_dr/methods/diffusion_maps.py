from pydiffmap.diffusion_map import DiffusionMap

from .base import DRMethod, median_gamma


class Method(DRMethod):
    method_id = "diffusion_maps"

    def default_params(self, X):
        return {"alpha": 0.5, "k": min(64, len(X) - 1), "epsilon": 1 / (4 * median_gamma(X))}

    def tuning_grid(self, X):
        return [dict(self.default_params(X), alpha=a) for a in [0.5, 0, 1]]

    def fit_transform(self, X, seed, params):
        model = DiffusionMap.from_sklearn(n_evecs=2, **params)
        return model.fit_transform(X), {
            "eigenvalues": model.evals.tolist(),
            "implementation": "pydiffmap diffusion generator eigenvectors",
        }
