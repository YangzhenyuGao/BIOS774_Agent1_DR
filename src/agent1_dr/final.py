"""Evidence-based final cohort size, then one declared final run of every shortlisted method.

The size rule is dataset-agnostic: shortlisted methods are timed on stratified probe cohorts,
log-log slopes are fitted, and the largest candidate size whose conservatively projected cost
fits the per-run timeout, the total final-stage budget and the memory budget is declared.
"""

import numpy as np
import pandas as pd

from .data import load_data
from .evaluation import Reference, evaluate
from .execution import run_fits
from .pilot import load_profile, prepare_cohort
from .utils import complete, read_json, valid_stage, write_json


def _slope(n, y):
    n, y = np.log(np.asarray(n, dtype=float)), np.log(np.asarray(y, dtype=float))
    return float(np.polyfit(n, y, 1)[0]) if len(n) >= 2 and np.ptp(n) > 0 else 1.0


def scaling_model(records):
    """Per-method exponents; memory uses increments above the smallest probe to drop the fixed baseline."""
    model = {}
    for mid in sorted({r["method"] for r in records}):
        pts = sorted((r["n"], max(r["seconds"], 1e-3), r["peak_memory_mb"]) for r in records
                     if r["method"] == mid and r["status"] == "success")
        if not pts:
            model[mid] = None
            continue
        large = [p for p in pts if p[0] >= 1000]
        use = large if len(large) >= 2 else pts
        base = pts[0][2]
        growth = [(p[0], p[2] - base) for p in pts[1:] if p[2] - base > 1]
        model[mid] = {
            "time_exponent": _slope([p[0] for p in use], [p[1] for p in use]),
            "memory_exponent": _slope(*zip(*growth)) if len(growth) >= 2 else 1.0,
            "anchor_n": pts[-1][0],
            "anchor_seconds": pts[-1][1],
            "anchor_memory_mb": pts[-1][2],
            "baseline_memory_mb": base,
        }
    return model


def project(model, n, cfg):
    per = {}
    for mid, m in model.items():
        ratio = n / m["anchor_n"]
        per[mid] = {
            "seconds": m["anchor_seconds"] * ratio ** max(m["time_exponent"], 1.0),
            "memory_mb": m["baseline_memory_mb"]
            + (m["anchor_memory_mb"] - m["baseline_memory_mb"]) * ratio ** max(m["memory_exponent"], 1.0),
        }
    f = cfg.final
    worst = max(v["seconds"] for v in per.values()) * f.safety_factor
    total = sum(v["seconds"] for v in per.values()) * f.safety_factor
    memory = max(v["memory_mb"] for v in per.values()) * f.safety_factor
    limits = {
        "per-run timeout": worst <= cfg.pilot.timeout_minutes_per_run * 60,
        "final-stage time budget": total <= f.time_budget_minutes * 60,
        "memory budget": memory <= f.memory_budget_gb * 1024,
    }
    return {
        "n": n,
        "per_method": per,
        "max_fit_seconds_with_safety": worst,
        "total_seconds_with_safety": total,
        "max_memory_mb_with_safety": memory,
        "feasible": all(limits.values()),
        "violations": [name for name, ok in limits.items() if not ok],
    }


def plan_size(cfg, directory, chosen, data, profile, pilot_n, available, force):
    cap = min(cfg.dataset.max_final_samples, available)
    sizes = sorted({s for s in cfg.final.probe_sizes if s <= cap})
    records = []
    for s in sizes:
        prepare_cohort(cfg, directory / "probe" / f"n{s}", s, data, profile)
    timeout = cfg.pilot.timeout_minutes_per_run * 60
    for mid, params in chosen.items():
        jobs = [
            {"input_path": directory / f"probe/n{s}/input.npy", "seed": cfg.project.seed, "params": params,
             "directory": directory / f"probe/n{s}/runs/{mid}"}
            for s in sizes
        ]
        for s, r in zip(sizes, run_fits(mid, jobs, timeout, force, log_dir=directory / "probe" / mid)):
            records.append({"method": mid, "n": s, "status": r["status"], "seconds": r["runtime_seconds"],
                            "peak_memory_mb": r["peak_memory_mb"] or 0.0})
    model = scaling_model(records)
    failed = [m for m, v in model.items() if v is None]
    if failed:
        raise ValueError(f"Scaling probe failed for {failed}; saved evidence under final/probe")
    ladder = sorted({n for n in cfg.final.size_ladder if pilot_n <= n <= cap} | {cap})
    projections = [project(model, n, cfg) for n in ladder]
    feasible = [p["n"] for p in projections if p["feasible"]]
    declared = max(feasible) if feasible else min(cap, pilot_n)
    if declared == cap:
        binding = "all available observations" if cap == available else "configured cap (dataset.max_final_samples)"
    else:
        nxt = next(p for p in projections if p["n"] > declared)
        binding = f"{', '.join(nxt['violations'])} at n={nxt['n']}"
    return {
        "configured_cap": cfg.dataset.max_final_samples,
        "available_observations": available,
        "pilot_samples": pilot_n,
        "probe_sizes": sizes,
        "probe": records,
        "scaling_model": model,
        "projections": projections,
        "declared_final_size": declared,
        "binding_constraint": binding,
        "rule": (
            f"largest ladder size with safety {cfg.final.safety_factor:g}x projected cost within the per-run "
            f"timeout ({cfg.pilot.timeout_minutes_per_run:g} min), the final-stage budget "
            f"({cfg.final.time_budget_minutes:g} min) and memory ({cfg.final.memory_budget_gb:g} GB); "
            "exponents below 1 are raised to 1"
        ),
    }


def final_stage(cfg, force=False):
    directory = cfg.output / "final"
    if not force and valid_stage(cfg, directory):
        return
    if not valid_stage(cfg, cfg.output / "tuning"):
        raise ValueError("Tuning missing or stale")
    chosen = read_json(cfg.output / "tuning/chosen_parameters.json")
    # Agent-generated exports only: stale embeddings or figures must not survive a shortlist change.
    for sub in ("embeddings", "figures"):
        for old in (directory / sub).glob("*"):
            if old.is_file():
                old.unlink()
    pilot_cohort = read_json(cfg.output / "pilot/cohort_manifest.json")
    data, profile = load_data(cfg), load_profile(cfg)
    plan = plan_size(cfg, directory, chosen, data, profile, pilot_cohort["selected"],
                     pilot_cohort["available_post_qc"], force)
    X, labels, ids = prepare_cohort(cfg, directory, plan["declared_final_size"], data, profile)
    cohort = read_json(directory / "cohort_manifest.json")
    cohort["size_justification"] = plan
    write_json(directory / "cohort_manifest.json", cohort)
    write_json(directory / "size_plan.json", plan)
    ref = Reference(X, k_max=cfg.pilot.n_neighbors_secondary, seed=cfg.project.seed)
    (directory / "embeddings").mkdir(exist_ok=True)
    rows = []
    timeout = cfg.pilot.timeout_minutes_per_run * 60
    for mid, params in chosen.items():
        job = {"input_path": directory / "input.npy", "seed": cfg.project.seed, "params": params,
               "directory": directory / "runs" / mid}
        r = run_fits(mid, [job], timeout, force, log_dir=directory / "runs" / mid)[0]
        if r["status"] != "success":
            raise ValueError(f"Final {mid} {r['status']}; saved evidence, no success flag")
        y = np.load(directory / "runs" / mid / "embedding.npy")
        np.save(directory / "embeddings" / f"{mid}.npy", y)
        frame = pd.DataFrame({"observation_id": ids, "label": labels, "dim1": y[:, 0], "dim2": y[:, 1]})
        frame.to_csv(directory / "embeddings" / f"{mid}.csv", index=False)
        e = evaluate(ref, y, labels, cfg.pilot.n_neighbors_eval, cfg.pilot.n_neighbors_secondary)
        rows.append(dict(method=mid, **e, runtime_seconds=r["runtime_seconds"],
                         peak_memory_mb=r["peak_memory_mb"], status=r["status"]))
    pd.DataFrame(rows).to_csv(directory / "final_metrics.csv", index=False)
    write_json(directory / "final_metrics.json", rows)
    complete("final", cfg, directory)
