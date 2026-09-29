from pydiffmap.diffusion_map import DiffusionMap

from .base import DRMethod, median_gamma


class Method(DRMethod):
    method_id = "diffusion_maps"

    def default_params(self, X):
        return {"alpha": 0.5, "k": min(64, len(X) - 1), "epsilon": 1 / (4 * median_gamma(X))}

    def tuning_grid(self, X):
        return [dict(self.default_params(X), alpha=a) for a in [0.5, 0, 1]]

    def validate_input(self, X, params):
        result, changes = super().validate_input(X, params)
        if result.get("k", 0) >= len(X):
            result["k"] = len(X) - 1
            changes.append(f"k clipped to {len(X) - 1} for cohort size")
        return result, changes

    def fit_transform(self, X, seed, params):
        model = DiffusionMap.from_sklearn(n_evecs=2, **params)
        Y = model.fit_transform(X)
        # pydiffmap's generator is L = (P - I) / epsilon, so 1 - mu = -lambda * epsilon for the
        # Markov matrix P; a gap near zero means a nearly disconnected neighborhood graph.
        gaps = [float(-v * model.epsilon_fitted) for v in model.evals]
        return Y, {
            "eigenvalues": model.evals.tolist(),
            "markov_spectral_gaps": gaps,
            "implementation": "pydiffmap diffusion generator eigenvectors",
        }
