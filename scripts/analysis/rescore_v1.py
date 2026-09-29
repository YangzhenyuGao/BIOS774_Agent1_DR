"""Offline re-scoring of the archived v1 pilot embeddings with the v2 metrics (no refitting).

Usage: python scripts/analysis/rescore_v1.py <v1_output_root> <result.json>
Each step changes one rule so that its effect on the v1 shortlist is isolated.
v1 runtimes include worker imports; fit-only timing requires refits and is not assessed here.
"""

import json
import sys
from pathlib import Path

import numpy as np

from agent1_dr.config import ROOT, load_config
from agent1_dr.evaluation import Reference, collapse_ratio, evaluate, seed_invariant, seed_stability
from agent1_dr.registry import METHOD_IDS, STOCHASTIC
from agent1_dr.selector import select, weights

root, target = Path(sys.argv[1]), Path(sys.argv[2])
summary = {}
for name in ["pbmc3k", "pathmnist"]:
    cfg = load_config(ROOT / f"config/{name}.yaml")
    pilot = root / name / "pilot"
    X = np.load(pilot / "input.npy")
    labels = np.load(pilot / "labels.npy", allow_pickle=False)
    k, k2 = 15, 50
    ref = Reference(X, k_max=k2, seed=774)
    v1 = {r["method"]: r for r in json.loads((pilot / "pilot_metrics.json").read_text())}
    delivered = json.loads((root / name / "selection/decisions.json").read_text())["selected"]
    rows, pca_collapse = [], None
    for mid in METHOD_IDS:
        seeds = [774 + i for i in range(3 if mid in STOCHASTIC else 1)]
        runs = [json.loads((pilot / f"method_runs/{mid}/{s}/result.json").read_text()) for s in seeds]
        ys = [np.load(pilot / f"method_runs/{mid}/{s}/embedding.npy") for s in seeds]
        ms = [evaluate(ref, y, labels, k, k2) for y in ys]
        mean = {key: float(np.mean([m[key] for m in ms])) for key in
                ("trustworthiness", "neighbor_recall", "global_structure", "trustworthiness_k2",
                 "neighbor_recall_k2")}
        ratio = float(np.median([collapse_ratio(y) for y in ys]))
        if mid == "pca":
            pca_collapse = ratio
        reasons = []
        if mid == "diffusion_maps":
            gap = -runs[0]["diagnostics"]["eigenvalues"][0] * runs[0]["parameters"]["epsilon"]
            if gap < cfg.selection.spectral_gap_min:
                reasons.append(f"near-disconnected diffusion graph (Markov spectral gap {gap:.1e})")
        if mid == "mds":
            n_iter = [r["diagnostics"]["n_iter_"] for r in runs]
            if max(n_iter) >= runs[0]["parameters"]["max_iter"]:
                reasons.append(f"SMACOF stopped at max_iter={runs[0]['parameters']['max_iter']}")
        rows.append({
            "method": mid, "status": v1[mid]["status"], "attempts": len(runs), "successes": len(runs),
            **mean, "collapse_ratio": ratio, "runtime_seconds": v1[mid]["runtime_seconds"],
            "stability_v1": v1[mid]["stability"], "stability": None,
            "seed_stability": seed_stability(ys, k) if mid in STOCHASTIC else None,
            "seed_invariant": seed_invariant(ys) if mid in STOCHASTIC else None,
            "severe_v1": v1[mid]["severe_warning"], "severe_reasons": reasons,
            "v1_trust": v1[mid]["trustworthiness"], "v1_recall": v1[mid]["neighbor_recall"],
        })
    for r in rows:
        s = cfg.selection
        if r["collapse_ratio"] < s.collapse_absolute and r["collapse_ratio"] < s.collapse_relative * pca_collapse:
            r["severe_reasons"].append(
                f"collapsed embedding (ratio {r['collapse_ratio']:.3f}; PCA {pca_collapse:.3f})")
        r["severe_warning"] = r["severe_v1"] or bool(r["severe_reasons"])
    w = weights(cfg)
    v1w = {**dict.fromkeys(w, 0.0), "trustworthiness": 0.45, "neighbor_recall": 0.45, "stability": 0.10}
    v1r = {"stability_key": "stability_v1", "severe_key": "severe_v1", "pareto": "strict", "equivalence": False}
    steps = [
        ("O0 v1 rules as delivered (sanity check)", v1w, v1r),
        ("O1 + runtime materiality (epsilon-Pareto)", v1w, {**v1r, "pareto": "epsilon"}),
        ("O2 + collapse / spectral-gap / convergence checks", v1w,
         {**v1r, "pareto": "epsilon", "severe_key": "severe_warning"}),
        ("O3 + global-structure component (v2 weights, v1 stability)", w,
         {**v1r, "pareto": "epsilon", "severe_key": "severe_warning"}),
        ("O4 runtime ignored entirely", w,
         {**v1r, "pareto": "off", "severe_key": "severe_warning"}),
    ]
    shortlists = [{"step": s, "selected": select(rows, cfg, None, ww, rr)["selected"]} for s, ww, rr in steps]
    # Audit counterfactual: v1 rules on the delivered v1 table, one fast method's runtime moved by +/-0.2 s.
    jitter = []
    for mid in ["pca", "kernel_pca", "isomap", "lle", "laplacian"]:
        for delta in (-0.2, 0.2):
            moved = [dict(r, runtime_seconds=r["runtime_seconds"] + (delta if r["method"] == mid else 0.0))
                     for r in rows]
            jitter.append({"method": mid, "delta_seconds": delta,
                           "selected": select(moved, cfg, None, v1w, v1r)["selected"]})
    summary[name] = {
        "delivered_v1": delivered,
        "sanity_reproduces_v1": shortlists[0]["selected"] == delivered,
        "max_abs_diff_trust_vs_v1": max(abs(r["trustworthiness"] - r["v1_trust"]) for r in rows),
        "max_abs_diff_recall_vs_v1": max(abs(r["neighbor_recall"] - r["v1_recall"]) for r in rows),
        "steps": shortlists,
        "runtime_jitter": jitter,
        "distinct_shortlists_under_jitter": len({tuple(j["selected"]) for j in jitter}
                                               | {tuple(shortlists[0]["selected"])}),
        "methods": {r["method"]: {key: r[key] for key in (
            "trustworthiness", "neighbor_recall", "global_structure", "neighbor_recall_k2",
            "collapse_ratio", "stability_v1", "seed_stability", "seed_invariant", "severe_reasons")}
            for r in rows},
    }
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(summary, indent=2))
print(json.dumps({n: {k: v for k, v in s.items() if k not in ("methods", "runtime_jitter")}
                  for n, s in summary.items()}, indent=2))
for n, s in summary.items():
    print(n)
    for m, r in s["methods"].items():
        print(f"  {m:15s} T={r['trustworthiness']:.3f} R={r['neighbor_recall']:.3f} G={r['global_structure']:.3f} "
              f"R50={r['neighbor_recall_k2']:.3f} collapse={r['collapse_ratio']:.3f} seedinv={r['seed_invariant']} "
              f"severe={r['severe_reasons']}")
