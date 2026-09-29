import numpy as np
from sklearn.datasets import make_blobs


def load_data(cfg):
    if cfg.dataset.name == "numeric":
        from .numeric import load

        return load(cfg)
    if cfg.dataset.name == "pbmc3k":
        from .pbmc3k import load

        return load(cfg)
    if cfg.dataset.name == "pathmnist":
        from .pathmnist import load

        return load(cfg)
    X, y = make_blobs(
        n_samples=cfg.dataset.synthetic_samples, n_features=8, centers=3, random_state=cfg.project.seed
    )
    return X, y.astype(str), np.array([f"synthetic_{i}" for i in range(len(X))]), {}, None
