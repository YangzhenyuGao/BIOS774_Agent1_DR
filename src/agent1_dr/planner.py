"""Profile-driven preprocessing planner.

Rules read the inspected data profile, never the dataset name, and every step records the
profile values that triggered it. PBMC3k and PathMNIST reach different branches because
their data differ (sparse integer counts versus dense 8-bit intensities).
"""


def _step(action, parameters, reason, scope="analysis cohort only"):
    return {"action": action, "parameters": parameters, "reason": reason, "fit_scope": scope}


def plan(profile, cfg):
    s, p = profile["statistics"], cfg.preprocessing
    d = profile["n_features"]
    evidence = {
        "sparsity": s["sparsity"],
        "nonnegative": s["nonnegative"],
        "integer_valued": s["integer_valued"],
        "min_value": s["min_value"],
        "max_value": s["max_value"],
        "n_features": d,
        "mitochondrial_features": s["mitochondrial_features"],
    }
    sample = _step(
        "sample_cohort",
        {},
        "Common representative cohort; labels, when present, only preserve class proportions",
        "cohort design",
    )
    pca = _step(
        "common_pca_representation",
        {"n_components": p.pca_components},
        "Common denoising and computational input for all ten methods; the displayed PCA is two-dimensional",
    )
    is_count = (
        s["sparsity"] >= p.count_min_sparsity
        and s["nonnegative"]
        and s["integer_valued"]
        and d >= p.count_min_features
    )
    if is_count:
        why = (
            f"sparsity {s['sparsity']:.3f} >= {p.count_min_sparsity}, non-negative integer values, "
            f"{d:,} features >= {p.count_min_features}"
        )
        qc = {"min_features": p.min_genes, "max_features": p.max_genes}
        mito_note = "no mitochondrial feature names, so no mitochondrial filter"
        if s["mitochondrial_features"]:
            qc["max_mito_fraction"] = p.max_mito_fraction
            mito_note = f"{s['mitochondrial_features']} MT- features enable a mitochondrial-fraction filter"
        low, mid, high = s["row_sum_quantiles"]
        steps = [
            _step(
                "observation_qc",
                qc,
                f"Count matrix ({why}): remove low-information and extreme observations with configured "
                f"thresholds; {mito_note}",
                "per-observation fixed thresholds",
            ),
            sample,
            _step(
                "filter_features",
                {"min_cells": p.min_cells},
                "Remove features detected in too few cohort observations",
            ),
            _step(
                "normalize_log1p",
                {"target_sum": 1e4},
                f"Library sizes vary from {low:,.0f} to {high:,.0f} (median {mid:,.0f}): normalize totals "
                "and log-compress counts",
            ),
            _step(
                "select_highly_variable_features",
                {"n_top": p.n_top_genes, "flavor": "seurat"},
                f"{d:,} sparse features: keep the most variable after normalization",
            ),
            _step("scale_clip", {"max_value": p.clip}, "Balance feature scales and limit outliers"),
            pca,
        ]
        branch = "count_matrix"
    elif s["nonnegative"] and s["integer_valued"] and s["max_value"] <= 255:
        steps = [
            sample,
            _step(
                "scale_intensity",
                {"divisor": 255},
                f"Dense (sparsity {s['sparsity']:.3f}) non-negative integers in [{s['min_value']:.0f}, "
                f"{s['max_value']:.0f}] behave as 8-bit intensities on one known scale: divide by 255 "
                "without per-feature standardization, preserving the pixel-space question",
                "fixed constant",
            ),
            pca,
        ]
        branch = "bounded_intensity"
    else:
        low, _, high = s["feature_variance_quantiles"]
        steps = [
            sample,
            _step(
                "standardize",
                {},
                f"Continuous features with variances from {low:.3g} to {high:.3g}: z-score each feature "
                "within the cohort",
            ),
            pca,
        ]
        branch = "continuous"
    return {"branch": branch, "rule_evidence": evidence, "steps": steps}
