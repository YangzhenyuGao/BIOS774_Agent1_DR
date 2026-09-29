import numpy as np
from scipy.sparse.csgraph import connected_components, laplacian
from scipy.sparse.linalg import eigsh
from sklearn.manifold import SpectralEmbedding

from .base import SklearnMethod


class Method(SklearnMethod):
    method_id = "laplacian"
    estimator = SpectralEmbedding

    def default_params(self, X):
        return {"n_neighbors": 30, "affinity": "nearest_neighbors", "eigen_solver": "arpack", "n_jobs": 1}

    def tuning_grid(self, X):
        return [dict(self.default_params(X), n_neighbors=k) for k in [30, 10, 50]]

    def after_fit(self):
        # Small normalized-Laplacian eigenvalues indicate weakly linked graph components.
        graph, self._model = self._model.affinity_matrix_, None
        out = {"graph_components": int(connected_components(graph, directed=False)[0])}
        if graph.shape[0] > 5000:
            # Pilot-scale explanation only; avoid a sparse factorization on large final cohorts.
            return out
        try:
            values = eigsh(laplacian(graph, normed=True), k=4, sigma=-1e-3, which="LM",
                           return_eigenvectors=False)
            out["normalized_laplacian_eigenvalues"] = sorted(float(v) for v in np.abs(values))
        except Exception as exc:  # noqa: BLE001 -- diagnostic only; the embedding is unaffected
            out["eigenvalue_diagnostic_error"] = f"{type(exc).__name__}: {exc}"
        return out
