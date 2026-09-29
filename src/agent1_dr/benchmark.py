"""Shared evidence summaries for pilot and tuning: metrics, stability, severe diagnostics, quality."""

import numpy as np

from .evaluation import evaluate, quality, seed_invariant, seed_stability, subsample_stability
from .registry import STOCHASTIC
from .selector import SUBSAMPLE_COMPONENTS, weights
from .utils import write_json

V1_SEVERE = ("not fully connected", "singular")
PER_FIT = SUBSAMPLE_COMPONENTS + ("trustworthiness_k2", "neighbor_recall_k2", "collapse_ratio")


def make_subsamples(cfg, directory, X):
    """Fixed random subsamples of the pilot input shared by every method and grid point."""
    n = len(X)
    size = round(cfg.pilot.subsample_fraction * n)
    rng = np.random.default_rng(cfg.project.seed + 1000)
    out = []
    for b in range(cfg.pilot.n_subsamples):
        index = np.sort(rng.choice(n, size, replace=False))
        d = directory / "subsamples" / str(b)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "indices.npy", index, allow_pickle=False)
        np.save(d / "input.npy", X[index], allow_pickle=False)
        out.append((index, d / "input.npy"))
    return out


def load_subsamples(cfg, directory):
    return [
        (np.load(directory / f"subsamples/{b}/indices.npy"), directory / f"subsamples/{b}/input.npy")
        for b in range(cfg.pilot.n_subsamples)
    ]


def fit_jobs(cfg, mid, params, input_path, subsamples, root):
    """Full-cohort fits (repeated seeds for stochastic methods) plus one refit per subsample."""
    seeds = [cfg.project.seed + i for i in range(cfg.pilot.n_repeats_stochastic if mid in STOCHASTIC else 1)]
    jobs = [{"input_path": input_path, "seed": s, "params": params, "directory": root / str(s)} for s in seeds]
    jobs += [
        {"input_path": path, "seed": cfg.project.seed + b, "params": params, "directory": root / "subsamples" / str(b)}
        for b, (_, path) in enumerate(subsamples)
    ]
    return jobs, len(seeds)


def _mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def _method_diagnostics(mid, results, params):
    """Convergence and graph diagnostics that can make a method ineligible."""
    reasons, summary = [], {}
    ok = [r for r in results if r["status"] == "success"]
    if mid == "mds" and ok:
        n_iter = [r["diagnostics"].get("n_iter_") for r in ok]
        summary["smacof_iterations"] = n_iter
        if any(not r["diagnostics"].get("converged", True) for r in ok):
            reasons.append(f"SMACOF stopped at max_iter={params.get('max_iter')} (not converged)")
    if mid == "diffusion_maps" and ok:
        gaps = [r["diagnostics"]["markov_spectral_gaps"][0] for r in ok if "markov_spectral_gaps" in r["diagnostics"]]
        summary["min_markov_spectral_gap"] = min(gaps) if gaps else None
    if mid == "laplacian" and ok:
        values = [r["diagnostics"].get("normalized_laplacian_eigenvalues") for r in ok]
        summary["min_fiedler_value"] = min((v[1] for v in values if v and len(v) > 1), default=None)
        summary["graph_components"] = max(r["diagnostics"].get("graph_components", 1) for r in ok)
    if mid == "gplvm" and ok:
        summary["loss_plateau_relative_change"] = max(
            r["diagnostics"].get("loss_plateau_relative_change", 0) for r in ok
        )
        summary["optimization_steps"] = [r["diagnostics"].get("optimization_steps") for r in ok]
    return reasons, summary


def summarize(cfg, mid, jobs, results, n_full, ref, sub_refs, subsamples, labels):
    """One benchmark row plus per-subsample metrics for paired equivalence margins."""
    k, k2 = cfg.pilot.n_neighbors_eval, cfg.pilot.n_neighbors_secondary
    full = list(zip(jobs[:n_full], results[:n_full]))
    subs = list(zip(jobs[n_full:], results[n_full:]))
    embeddings, metrics = [], []
    for job, r in full:
        if r["status"] == "success":
            y = np.load(job["directory"] / "embedding.npy")
            e = evaluate(ref, y, labels, k, k2)
            write_json(job["directory"] / "metrics.json", e)
            embeddings.append(y)
            metrics.append(e)
    per_subsample, fits, collapse = [None] * len(subs), [], [m["collapse_ratio"] for m in metrics]
    for b, (job, r) in enumerate(subs):
        if r["status"] == "success":
            y = np.load(job["directory"] / "embedding.npy")
            index = subsamples[b][0]
            e = evaluate(sub_refs[b], y, labels[index], k, k2)
            write_json(job["directory"] / "metrics.json", e)
            per_subsample[b] = {c: e[c] for c in PER_FIT}
            fits.append((index, y))
            collapse.append(e["collapse_ratio"])
    stability, stability_global = subsample_stability(fits, ref.clip_k(k)) if len(fits) == len(subs) else (None, None)
    seed = seed_stability(embeddings, ref.clip_k(k)) if mid in STOCHASTIC else None
    ok = [r for r in results if r["status"] == "success"]
    full_ok = [r for _, r in full if r["status"] == "success"]
    status = (
        "success"
        if len(ok) == len(results)
        else ("timeout" if any(r["status"] == "timeout" for r in results) else "failed")
    )
    warnings = [w for r in results for w in r["warnings"]]
    reasons, diagnostics = _method_diagnostics(mid, full_ok, jobs[0]["params"])
    v1_severe = any(s in w.lower() for w in warnings for s in V1_SEVERE)
    if v1_severe:
        reasons.insert(0, "graph not fully connected or singular matrix warning")
    row = {
        "method": mid,
        "status": status,
        "attempts": len(results),
        "successes": len(ok),
        "repeats": n_full,
        "subsamples": len(subs),
        **{key: _mean([m[key] for m in metrics]) for key in PER_FIT if key != "collapse_ratio"},
        "silhouette": _mean([m["silhouette"] for m in metrics]),
        "collapse_ratio": float(np.median(collapse)) if collapse else None,
        "stability": stability,
        "stability_global": stability_global,
        "seed_stability": seed,
        "seed_invariant": seed_invariant(embeddings) if mid in STOCHASTIC else None,
        # v1 definition, retained for the ablation: deterministic methods fixed at 1.
        "stability_v1": seed if mid in STOCHASTIC else 1.0,
        "runtime_seconds": _mean([r["runtime_seconds"] for r in full_ok]) or _mean(
            [r["runtime_seconds"] for _, r in full]
        ),
        "runtime_cpu_seconds": _mean([r.get("timing", {}).get("fit_cpu_seconds") for r in full_ok]),
        "runtime_subsample_seconds": _mean([r["runtime_seconds"] for _, r in subs if r["status"] == "success"]),
        "overhead_seconds": _mean(
            [
                r.get("timing", {}).get("batch_import_seconds", 0) + r.get("timing", {}).get("batch_warmup_seconds", 0)
                for r in ok
            ]
        ),
        "peak_memory_mb": max([r["peak_memory_mb"] or 0 for r in results], default=0),
        "retry_count": sum(r["retry_count"] for r in results),
        "warnings": warnings,
        "severe_v1": v1_severe,
        "severe_reasons": reasons,
        "diagnostics": diagnostics,
    }
    return row, per_subsample


def finalize(rows, cfg, pca_collapse):
    """Collapse check against the PCA baseline, severe flags, and the pre-registered quality."""
    s, w = cfg.selection, weights(cfg)
    for r in rows:
        c = r["collapse_ratio"]
        if (
            r["status"] == "success"
            and c is not None
            and c < s.collapse_absolute
            and (pca_collapse is None or c < s.collapse_relative * pca_collapse)
        ):
            r["severe_reasons"].append(
                f"collapsed layout (median collapse ratio {c:.3f} < {s.collapse_absolute:g} and "
                f"< {s.collapse_relative:g} x PCA {pca_collapse:.3f})"
            )
        if r["method"] == "diffusion_maps":
            gap = r["diagnostics"].get("min_markov_spectral_gap")
            if gap is not None and gap < s.spectral_gap_min:
                r["severe_reasons"].append(f"near-disconnected diffusion graph (Markov spectral gap {gap:.1e})")
        r["severe_warning"] = bool(r["severe_reasons"])
        parts = {name: r[name] for name in w}
        valid = r["status"] == "success" and all(v is not None for v in parts.values())
        r["quality"] = quality(parts, w) if valid else None
    return rows
