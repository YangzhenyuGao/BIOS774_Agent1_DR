"""User-supplied dense numerical CSV or NPZ; diagnostic metadata is separated."""

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ROOT
from ..utils import checksum


def load(cfg):
    path = ROOT / Path(cfg.dataset.path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            X = data["X"]
            labels = data["labels"].astype(str) if "labels" in data else None
            ids = data["ids"].astype(str) if "ids" in data else np.arange(len(X)).astype(str)
            names = data["feature_names"].astype(str) if "feature_names" in data else None
    elif path.suffix == ".csv":
        frame = pd.read_csv(path)
        labels = (
            frame.pop(cfg.dataset.label_column).astype(str).to_numpy() if cfg.dataset.label_column else None
        )
        ids = (
            frame.pop(cfg.dataset.id_column).astype(str).to_numpy()
            if cfg.dataset.id_column
            else np.arange(len(frame)).astype(str)
        )
        names = frame.columns.astype(str).to_numpy()
        X = frame.to_numpy(dtype=float)
    else:
        raise ValueError("Numerical input must be .npz (X, optional labels/ids) or numerical .csv")
    if X.ndim != 2 or not np.issubdtype(X.dtype, np.number):
        raise ValueError("X must be a numerical matrix")
    if len(set(ids)) != len(X) or len(ids) != len(X) or (labels is not None and len(labels) != len(X)):
        raise ValueError("Observation IDs must be unique and align with X and labels")
    provenance = {
        "source": "User-supplied numerical matrix",
        "path": str(path),
        "sha256": checksum(path),
        "labels_available": labels is not None,
    }
    return X, labels, ids, provenance, names
