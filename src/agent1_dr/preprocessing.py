"""Execute the planner's steps on one cohort; every fitted transformation sees that cohort only."""

from dataclasses import asdict

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .inspector import inspect_data
from .planner import plan as make_plan
from .sampling import cohort_indices
from .schemas import PreprocessingPlan


def _mito_mask(feature_names):
    return np.char.startswith(np.char.upper(np.asarray(feature_names, dtype=str)), "MT-")


def prepare(cfg, data, size, profile=None):
    X, labels, ids, provenance, feature_names = data
    if profile is None:
        profile = asdict(inspect_data(cfg.dataset.name, X, labels, provenance, feature_names))
    proposal = make_plan(profile, cfg)
    actions, adata, model, available = [], None, None, len(ids)
    for step in proposal["steps"]:
        params = dict(step["parameters"])
        action = step["action"]
        if action == "observation_qc":
            detected = np.asarray((X > 0).sum(axis=1)).ravel()
            keep = (detected >= params["min_features"]) & (detected < params["max_features"])
            if "max_mito_fraction" in params:
                total = np.asarray(X.sum(axis=1)).ravel()
                mito = np.asarray(X[:, _mito_mask(feature_names)].sum(axis=1)).ravel()
                keep &= mito / np.maximum(total, 1) < params["max_mito_fraction"]
            X, ids = X[keep], ids[keep]
            labels = labels[keep] if labels is not None else None
            available = len(ids)
            params["retained"] = int(keep.sum())
        elif action == "sample_cohort":
            idx = cohort_indices(available, size, cfg.project.seed, labels)
            stratified = labels is not None
            X, ids = X[idx], ids[idx]
            labels = labels[idx] if stratified else np.full(len(idx), "unannotated")
            params.update(size=len(idx), seed=cfg.project.seed, stratified=stratified)
        elif action == "filter_features":
            names = feature_names if feature_names is not None else [f"feature_{j}" for j in range(X.shape[1])]
            adata = anndata.AnnData(X=X, var=pd.DataFrame(index=pd.Index(names, dtype=str)))
            sc.pp.filter_genes(adata, min_cells=params["min_cells"])
            params["retained"] = adata.n_vars
        elif action == "normalize_log1p":
            sc.pp.normalize_total(adata, target_sum=params["target_sum"])
            sc.pp.log1p(adata)
        elif action == "select_highly_variable_features":
            sc.pp.highly_variable_genes(adata, n_top_genes=min(params["n_top"], adata.n_vars), flavor=params["flavor"])
            adata = adata[:, adata.var.highly_variable].copy()
            params["n_selected"] = adata.n_vars
        elif action == "scale_clip":
            sc.pp.scale(adata, max_value=params["max_value"])
            X = np.asarray(adata.X)
        elif action == "scale_intensity":
            X = X.astype(np.float64) / params["divisor"]
        elif action == "standardize":
            X = StandardScaler().fit_transform(X)
        elif action == "common_pca_representation":
            model = PCA(
                n_components=min(params["n_components"], len(X) - 1, X.shape[1]),
                svd_solver="full",
                random_state=cfg.project.seed,
            )
            X = model.fit_transform(X)
            params.update(n_components=X.shape[1], explained_variance_ratio=model.explained_variance_ratio_.tolist())
        else:
            raise ValueError(f"Unknown preprocessing action {action}")
        actions.append({**step, "parameters": params})
    record = asdict(PreprocessingPlan(actions, branch=proposal["branch"], rule_evidence=proposal["rule_evidence"]))
    return (
        X,
        labels,
        ids,
        record,
        {
            "available_post_qc": available,
            "source_samples": len(data[2]),
            "selected": len(ids),
            "observation_ids": ids.tolist(),
            "labels": labels.tolist(),
            "pca_explained_variance_ratio": model.explained_variance_ratio_.tolist(),
        },
    )
