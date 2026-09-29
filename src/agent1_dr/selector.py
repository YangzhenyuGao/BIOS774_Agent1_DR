"""Evidence-based candidate selection (pre-registered v2 rules; see docs/METHOD_NOTES.md)."""

from dataclasses import asdict

import numpy as np

from .registry import METHOD_IDS
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
# Per-fit components that vary across subsamples; stability is one number per method.
SUBSAMPLE_COMPONENTS = ("trustworthiness", "neighbor_recall", "global_structure")
PRIMARY_RULES = {
    "stability_key": "stability",
    "local_suffix": "",
    "severe_key": "severe_warning",
    "pareto": "epsilon",
    "equivalence": True,
}


def weights(cfg):
    s = cfg.selection
    return {
        "trustworthiness": s.weight_trustworthiness,
        "neighbor_recall": s.weight_recall,
        "global_structure": s.weight_global,
        "stability": s.weight_stability,
    }


def components(row, rules):
    """Map the quality components onto the row fields selected by a rule variant."""
    suffix = rules["local_suffix"]
    return {
        "trustworthiness": row.get("trustworthiness" + suffix),
        "neighbor_recall": row.get("neighbor_recall" + suffix),
        "global_structure": row.get("global_structure"),
        "stability": row.get(rules["stability_key"]),
    }


def score(row, w, rules):
    values = components(row, rules)
    if row["status"] != "success" or any(values[k] is None for k, v in w.items() if v):
        return None
    return float(sum(v * values[k] for k, v in w.items() if v))


class Margins:
    """Practical-equivalence margins from paired per-subsample quality differences."""

    def __init__(self, subsample, w, floor, rules):
        self.floor = floor
        suffix = rules["local_suffix"]
        self.q = {}
        for method, fits in (subsample or {}).items():
            if fits and all(f is not None for f in fits):
                self.q[method] = np.array(
                    [
                        sum(
                            w[c] * f[c + (suffix if c != "global_structure" else "")]
                            for c in SUBSAMPLE_COMPONENTS
                            if w[c]
                        )
                        for f in fits
                    ]
                )

    def se(self, a, b):
        qa, qb = self.q.get(a), self.q.get(b)
        if qa is None or qb is None or len(qa) != len(qb) or len(qa) < 2:
            return None
        return float(np.std(qa - qb, ddof=1) / np.sqrt(len(qa)))

    def margin(self, a, b):
        se = self.se(a, b)
        return max(2 * se, self.floor) if se is not None else self.floor


def dominates(s, r, margins, cfg, rules):
    """Does s make r redundant (at least as good, and materially better or faster)?"""
    ts, tr = s["runtime_seconds"], r["runtime_seconds"]
    if rules["pareto"] == "off":
        ts = tr = 1.0
    if rules["pareto"] == "strict":
        return (
            s["quality"] >= r["quality"]
            and ts <= tr
            and (s["quality"] > r["quality"] or ts < tr)
        )
    ratio, extra = cfg.selection.runtime_material_ratio, cfg.selection.runtime_material_seconds
    not_slower = ts <= max(ratio * tr, tr + extra)
    faster = tr > max(ratio * ts, ts + extra)
    better = s["quality"] - r["quality"] > margins.margin(s["method"], r["method"])
    return not_slower and s["quality"] >= r["quality"] and (better or faster)


def select(table, cfg, subsample=None, w=None, rules=None):
    rules = dict(PRIMARY_RULES, **(rules or {}))
    w = w or weights(cfg)
    rows = [dict(r) for r in table]
    for r in rows:
        r["quality"] = score(r, w, rules)
    valid = [r for r in rows if r["quality"] is not None and not r[rules["severe_key"]]]
    margins = Margins(subsample, w, cfg.selection.equivalence_floor, rules)
    best = max(valid, key=lambda r: (r["quality"], -r["runtime_seconds"]), default=None)
    threshold = best["quality"] * cfg.selection.quality_relative_to_best if best else None
    dominators = {
        r["method"]: [s["method"] for s in valid if s is not r and dominates(s, r, margins, cfg, rules)]
        for r in valid
    }
    frontier = {m for m, by in dominators.items() if not by}
    equivalent = {
        r["method"]
        for r in valid
        if rules["equivalence"] and best["quality"] - r["quality"] <= margins.margin(best["method"], r["method"])
    }
    eligible = {r["method"] for r in valid if r["quality"] >= threshold} | equivalent | frontier
    ordered = sorted(valid, key=lambda r: (-r["quality"], r["runtime_seconds"], METHOD_IDS.index(r["method"])))
    selected = []
    if cfg.selection.retain_pca_baseline and any(r["method"] == "pca" for r in valid):
        selected.append("pca")
    for r in ordered:
        if r["method"] not in selected and r["method"] in eligible and len(selected) < cfg.selection.max_final_methods:
            selected.append(r["method"])
    low_confidence = len(selected) < 2
    if low_confidence:
        for r in ordered:
            if r["method"] not in selected:
                selected.append(r["method"])
                break
    decisions = []
    for r in rows:
        decisions.append(asdict(SelectionDecision(
            r["method"], PRIORS[r["method"]], r, "retain" if r["method"] in selected else "exclude",
            _reason(r, best, threshold, margins, dominators, eligible, selected, rules),
        )))
    return {
        "selected": selected,
        "best_method": best["method"] if best else None,
        "best_quality": best["quality"] if best else None,
        "threshold": threshold,
        "equivalent_to_best": sorted(equivalent),
        "pareto_frontier": sorted(frontier),
        "eligible": sorted(eligible),
        "low_confidence": low_confidence,
        "labels_used": False,
        "weights": w,
        "rules": rules,
        "quality_formula": " + ".join(f"{v:g}*{k}" for k, v in w.items() if v),
        "decisions": decisions,
    }


def _reason(r, best, threshold, margins, dominators, eligible, selected, rules):
    if r["status"] != "success":
        return f"Excluded: terminal status {r['status']}; {r.get('successes')}/{r.get('attempts')} fits succeeded."
    if r["quality"] is None:
        return "Excluded: incomplete quality components."
    if r[rules["severe_key"]]:
        return "Excluded: severe diagnostic — " + "; ".join(r.get("severe_reasons") or ["see warnings"]) + "."
    margin = margins.margin(best["method"], r["method"])
    text = (
        f"Q={r['quality']:.4f} vs best {best['method']} {best['quality']:.4f}; "
        f"90% threshold {threshold:.4f}; equivalence margin {margin:.4f}; "
        f"fit runtime {r['runtime_seconds']:.2f}s. "
    )
    if r["method"] in selected:
        if r["method"] == "pca":
            return text + "Retained as the linear baseline."
        paths = []
        if r["quality"] >= threshold:
            paths.append("meets the 90% threshold")
        if rules["equivalence"] and best["quality"] - r["quality"] <= margin:
            paths.append("practically equivalent to the best")
        if not dominators[r["method"]]:
            paths.append("on the quality-runtime frontier")
        return text + "Retained: " + ", ".join(paths) + "."
    if r["method"] in eligible:
        return text + "Eligible but outside the configured candidate capacity (lower quality rank)."
    by = ", ".join(dominators[r["method"]][:3])
    return text + f"Excluded: below threshold, outside the equivalence margin, and made redundant by {by}."


# Rule variants re-run the same selection mechanics on the same saved evidence.
def sensitivity(table, cfg, subsample):
    primary = weights(cfg)
    variants = [
        ("primary (pre-registered)", primary, {}),
        ("local metrics only (0.5 T + 0.5 R)", {**_zero(primary), "trustworthiness": 0.5, "neighbor_recall": 0.5}, {}),
        ("v1 weights (0.45 T + 0.45 R + 0.10 S)", {**_zero(primary), "trustworthiness": 0.45,
                                                     "neighbor_recall": 0.45, "stability": 0.10}, {}),
        ("no stability term", {**primary, "stability": 0.0}, {}),
        ("global-heavy (0.25/0.25/0.40/0.10)", {"trustworthiness": 0.25, "neighbor_recall": 0.25,
                                                "global_structure": 0.40, "stability": 0.10}, {}),
        ("neighborhood k=50", primary, {"local_suffix": "_k2"}),
        ("seed stability, deterministic=1 (v1 definition)", primary, {"stability_key": "stability_v1"}),
        ("v1 severe checks only (no collapse/gap/convergence)", primary, {"severe_key": "severe_v1"}),
        ("runtime ignored", primary, {"pareto": "off"}),
        ("strict Pareto without materiality (v1)", primary, {"pareto": "strict"}),
    ]
    out = []
    for name, w, rules in variants:
        result = select(table, cfg, subsample, w, rules)
        out.append({"variant": name, "selected": result["selected"], "best": result["best_method"]})
    counts = {m: sum(m in v["selected"] for v in out) for m in METHOD_IDS}
    return {"variants": out, "selection_frequency": {m: c / len(out) for m, c in counts.items() if c}}


def ablation(table, cfg, subsample):
    """Cumulative path from the v1 rules to the pre-registered v2 rules on the same evidence."""
    primary = weights(cfg)
    v1w = {**_zero(primary), "trustworthiness": 0.45, "neighbor_recall": 0.45, "stability": 0.10}
    v1r = {"stability_key": "stability_v1", "severe_key": "severe_v1", "pareto": "strict", "equivalence": False}
    steps = [
        ("A0 v1 rules on v2 evidence (fit-only timing)", v1w, v1r),
        ("A1 + runtime materiality (epsilon-Pareto)", v1w, {**v1r, "pareto": "epsilon"}),
        ("A2 + collapse / spectral-gap / convergence checks", v1w, {**v1r, "pareto": "epsilon",
                                                                    "severe_key": "severe_warning"}),
        ("A3 + global-structure component", primary, {**v1r, "pareto": "epsilon", "severe_key": "severe_warning"}),
        ("A4 + subsample stability replaces seed stability", primary, {**v1r, "pareto": "epsilon",
                                                                       "severe_key": "severe_warning",
                                                                       "stability_key": "stability"}),
        ("A5 + equivalence-to-best eligibility (= primary)", primary, {}),
    ]
    return [
        {"step": name, "selected": select(table, cfg, subsample, w, rules)["selected"]}
        for name, w, rules in steps
    ]


def _zero(w):
    return dict.fromkeys(w, 0.0)


def selection_stage(cfg, force=False):
    directory = cfg.output / "selection"
    if not force and valid_stage(cfg, directory):
        return
    if not valid_stage(cfg, cfg.output / "pilot"):
        raise ValueError("Pilot missing or stale; run pilot first")
    table = read_json(cfg.output / "pilot/pilot_metrics.json")
    subsample = read_json(cfg.output / "pilot/subsample_metrics.json")
    decision = select(table, cfg, subsample)
    decision["sensitivity"] = sensitivity(table, cfg, subsample)
    decision["ablation"] = ablation(table, cfg, subsample)
    write_json(directory / "decisions.json", decision)
    write_json(cfg.output / "decisions.json", decision)
    complete("select", cfg, directory)
