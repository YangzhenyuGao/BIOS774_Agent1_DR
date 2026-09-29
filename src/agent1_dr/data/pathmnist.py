from datetime import UTC, datetime

import medmnist
import numpy as np

from ..config import ROOT
from ..utils import checksum


def load(cfg):
    root = ROOT / "data/raw/pathmnist"
    root.mkdir(parents=True, exist_ok=True)
    ds = medmnist.PathMNIST(root=str(root), split=cfg.dataset.split, download=True, size=28)
    labels = np.array([ds.info["label"][str(int(x))] for x in ds.labels.ravel()])
    ids = np.array([f"{cfg.dataset.split}_{i:06d}" for i in range(len(ds))])
    provenance = {
        "source": "Official medmnist.PathMNIST",
        "dataset_version": "MedMNIST v2, size 28",
        "url": ds.info["url"],
        "split": cfg.dataset.split,
        "image_shape": list(ds.imgs.shape[1:]),
        "label_mapping": ds.info["label"],
        "license": ds.info.get("license", "See official source"),
        "package_version": medmnist.__version__,
        "access_date": datetime.now(UTC).date().isoformat(),
        "files": {p.name: checksum(p) for p in root.glob("*.npz")},
    }
    return ds.imgs.reshape(len(ds), -1), labels, ids, provenance, None
