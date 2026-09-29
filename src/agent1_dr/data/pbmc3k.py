from datetime import UTC, datetime

import numpy as np
import scanpy as sc

from ..config import ROOT
from ..utils import checksum, versions


def align_labels(raw_ids, processed):
    if not processed.obs_names.is_unique or len(set(raw_ids)) != len(raw_ids):
        raise ValueError("Barcodes must be unique for label transfer")
    if not set(processed.obs_names).issubset(set(raw_ids)):
        raise ValueError("Processed barcodes are not a subset of raw barcodes; transfer stopped")
    return processed.obs["louvain"].astype(str).reindex(raw_ids).fillna("unannotated").to_numpy()


def load(cfg):
    root = ROOT / "data/raw/pbmc3k"
    root.mkdir(parents=True, exist_ok=True)
    sc.settings.datasetdir = root
    raw = sc.datasets.pbmc3k()
    processed = sc.datasets.pbmc3k_processed()
    labels = align_labels(raw.obs_names, processed)
    provenance = {
        "source": "Scanpy datasets.pbmc3k and pbmc3k_processed; 10x PBMC3k",
        "url": "https://scanpy.readthedocs.io/en/latest/api/generated/scanpy.datasets.pbmc3k.html",
        "access_date": datetime.now(UTC).date().isoformat(),
        "files": {p.name: checksum(p) for p in root.glob("*.h5ad")},
        "versions": versions(),
        "annotation": "processed louvain annotation aligned by exact barcode",
        "unannotated": int((labels == "unannotated").sum()),
    }
    return raw.X.tocsr(), labels, np.asarray(raw.obs_names, dtype=str), provenance, np.asarray(raw.var_names, dtype=str)
