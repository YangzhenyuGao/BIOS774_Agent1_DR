from dataclasses import asdict

import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .sampling import cohort_indices
from .schemas import PreprocessingPlan


def prepare(cfg, data, size):
    X, labels, ids, _provenance, adata = data
    actions = []

    def action(name, params, reason, scope="analysis cohort only"):
        actions.append({"action": name, "parameters": params, "reason": reason, "fit_scope": scope})

    if cfg.dataset.name == "pbmc3k":
        p = cfg.preprocessing
        detected = np.asarray((X > 0).sum(axis=1)).ravel()
        total = np.asarray(X.sum(axis=1)).ravel()
        mito = np.asarray(X[:, adata.var_names.str.startswith("MT-")].sum(axis=1)).ravel()
        keep = (
            (detected >= p.min_genes)
            & (detected < p.max_genes)
            & (mito / np.maximum(total, 1) < p.max_mito_fraction)
        )
        X, labels, ids = X[keep], labels[keep], ids[keep]
        adata = adata[keep].copy()
        action(
            "cell_qc",
            {
                "min_genes": p.min_genes,
                "max_genes": p.max_genes,
                "max_mito_fraction": p.max_mito_fraction,
                "retained": int(keep.sum()),
            },
            "Remove low-information cells and extreme gene/mitochondrial counts",
            "per-cell fixed thresholds",
        )
    available = len(ids)
    idx = cohort_indices(available, size, cfg.project.seed, labels)
    labels_present = labels is not None
    X, ids = X[idx], ids[idx]
    labels = labels[idx] if labels_present else np.full(len(idx), "unannotated")
    action(
        "sample_cohort",
        {"size": len(idx), "seed": cfg.project.seed, "stratified": labels_present},
        "Common representative cohort; labels used only to preserve class proportions",
        "cohort design",
    )
    if cfg.dataset.name == "pbmc3k":
        a = adata[idx].copy()
        sc.pp.filter_genes(a, min_cells=cfg.preprocessing.min_cells)
        action(
            "filter_genes",
            {"min_cells": cfg.preprocessing.min_cells, "retained": a.n_vars},
            "Remove genes observed in too few cohort cells",
        )
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
        action("normalize_log1p", {"target_sum": 1e4}, "Adjust library size and compress count range")
        sc.pp.highly_variable_genes(
            a, n_top_genes=min(cfg.preprocessing.n_top_genes, a.n_vars), flavor="seurat"
        )
        a = a[:, a.var.highly_variable].copy()
        action(
            "select_highly_variable_genes",
            {"n_genes": a.n_vars, "flavor": "seurat"},
            "Focus on variable expression features in sparse high-dimensional count data",
        )
        sc.pp.scale(a, max_value=cfg.preprocessing.clip)
        X = np.asarray(a.X)
        action("scale_clip", {"max_value": cfg.preprocessing.clip}, "Balance gene scales and limit outliers")
    elif cfg.dataset.name == "pathmnist":
        X = X.astype(np.float64) / 255
        action(
            "scale_flatten_pixels",
            {"divisor": 255, "channels": "RGB", "size": 28},
            "Preserve original pixel-based scientific question with consistent pixel scale",
        )
    else:
        X = StandardScaler().fit_transform(X)
        action("standardize", {}, "Balance numerical feature variances")
    model = PCA(
        n_components=min(cfg.preprocessing.pca_components, len(X) - 1, X.shape[1]),
        svd_solver="full",
        random_state=cfg.project.seed,
    )
    representation = model.fit_transform(X)
    action(
        "common_pca_representation",
        {
            "n_components": representation.shape[1],
            "explained_variance_ratio": model.explained_variance_ratio_.tolist(),
        },
        "Common denoising and computational input for all ten methods; final PCA is two-dimensional",
    )
    return (
        representation,
        labels,
        ids,
        asdict(PreprocessingPlan(actions)),
        {
            "available_post_qc": available,
            "source_samples": len(data[2]),
            "selected": len(ids),
            "observation_ids": ids.tolist(),
            "labels": labels.tolist(),
            "pca_explained_variance_ratio": model.explained_variance_ratio_.tolist(),
        },
    )
