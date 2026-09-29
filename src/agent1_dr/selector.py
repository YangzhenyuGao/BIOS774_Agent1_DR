from dataclasses import asdict

from .schemas import SelectionDecision
from .utils import complete, read_json, valid_stage, write_json

PRIORS = {
    "pca": "Linear variance baseline.",
    "kernel_pca": "Nonlinear kernel structure; quadratic memory.",
    "mds": "Global pairwise geometry; iterative quadratic distance work.",
    "isomap": "Geodesic geometry conditional on a connected neighborhood graph.",
    "lle": "Local affine reconstruction conditional on well-conditioned neighborhoods.",
    "laplacian": "Local graph geometry conditional on connectivity.",
    "diffusion_maps": "Density-corrected diffusion geometry.",
    "gplvm": "Probabilistic nonlinear latent-variable model under a fixed optimization budget.",
    "tsne": "Stochastic local-neighborhood visualization.",
    "umap": "Stochastic neighborhood-graph embedding.",
}


def select(table, cfg):
    valid = [
        r for r in table if r["status"] == "success" and not r["severe_warning"] and r["quality"] is not None
    ]
    best = max((r["quality"] for r in valid), default=0)
    threshold = best * cfg.selection.quality_relative_to_best
    frontier = {
        r["method"]
        for r in valid
        if not any(
            s["quality"] >= r["quality"]
            and s["runtime_seconds"] <= r["runtime_seconds"]
            and (s["quality"] > r["quality"] or s["runtime_seconds"] < r["runtime_seconds"])
            for s in valid
        )
    }
    ordered = sorted(valid, key=lambda r: (-r["quality"], r["runtime_seconds"], r["method"]))
    selected = []
    if cfg.selection.retain_pca_baseline and any(r["method"] == "pca" for r in valid):
        selected.append("pca")
    for r in ordered:
        if (
            r["method"] not in selected
            and (r["quality"] >= threshold or r["method"] in frontier)
            and len(selected) < cfg.selection.max_final_methods
        ):
            selected.append(r["method"])
    low_confidence = len(selected) < 2
    if low_confidence:
        for r in ordered:
            if r["method"] not in selected:
                selected.append(r["method"])
                break
    decisions = []
    for r in table:
        retain = r["method"] in selected
        if r["status"] != "success":
            reason = f"Terminal status {r['status']}; {r['successes']}/{r['repeats']} successful runs."
        elif r["severe_warning"]:
            reason = "Excluded: unresolved graph connectivity or singularity warning; see recorded warnings."
        else:
            reason = (
                f"Quality={r['quality']:.6f}; threshold={threshold:.6f}; "
                f"runtime={r['runtime_seconds']:.3f}s; Pareto={r['method'] in frontier}. "
            )
            if retain:
                reason += (
                    "Retained linear baseline."
                    if r["method"] == "pca"
                    else "Retained by quality/Pareto rank and capacity."
                )
            else:
                reason += "Below quality/Pareto eligibility or outside configured candidate capacity."
        decisions.append(
            asdict(
                SelectionDecision(
                    r["method"], PRIORS[r["method"]], r, "retain" if retain else "exclude", reason
                )
            )
        )
    return {
        "selected": selected,
        "threshold": threshold,
        "best_quality": best,
        "pareto_frontier": sorted(frontier),
        "low_confidence": low_confidence,
        "labels_used": False,
        "quality_formula": "0.45*trustworthiness+0.45*neighbor_recall+0.10*stability",
        "decisions": decisions,
    }


def selection_stage(cfg, force=False):
    directory = cfg.output / "selection"
    if not force and valid_stage(cfg, directory):
        return
    if not valid_stage(cfg, cfg.output / "pilot"):
        raise ValueError("Pilot missing or stale; run pilot first")
    decision = select(read_json(cfg.output / "pilot/pilot_metrics.json"), cfg)
    write_json(directory / "decisions.json", decision)
    write_json(cfg.output / "decisions.json", decision)
    complete("select", cfg, directory)
