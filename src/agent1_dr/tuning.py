import json

import numpy as np
import pandas as pd

from .evaluation import evaluate, quality, stability
from .execution import parameters, run_method
from .pilot import prepare_cohort
from .registry import STOCHASTIC, get_method
from .utils import complete, read_json, valid_stage, write_json


def tuning_stage(cfg, force=False):
    directory = cfg.output / "tuning"
    if not force and valid_stage(cfg, directory):
        return
    if not valid_stage(cfg, cfg.output / "selection"):
        raise ValueError("Selection missing or stale")
    selected = read_json(cfg.output / "selection/decisions.json")["selected"]
    input_path = cfg.output / "pilot/input.npy"
    X, labels = np.load(input_path), np.load(cfg.output / "pilot/labels.npy")
    rows, chosen = [], {}
    for mid in selected:
        method = get_method(mid)
        default = parameters(cfg, mid, X)
        # Configured fit budget applies to every grid point (especially smoke GPLVM epochs).
        grid = [default]
        for proposed in method.tuning_grid(X):
            p = dict(default)
            p.update({k: v for k, v in proposed.items() if v != method.default_params(X).get(k)})
            if p not in grid:
                grid.append(p)
        candidates = []
        for gi, p in enumerate(grid):
            embeddings, metrics, runs = [], [], []
            for i in range(cfg.pilot.n_repeats_stochastic if mid in STOCHASTIC else 1):
                run_dir = directory / "runs" / mid / str(gi) / str(cfg.project.seed + i)
                r = run_method(
                    mid,
                    input_path,
                    cfg.project.seed + i,
                    p,
                    run_dir,
                    cfg.pilot.timeout_minutes_per_run * 60,
                    force,
                )
                runs.append(r)
                if r["status"] == "success":
                    y = np.load(run_dir / "embedding.npy")
                    embeddings.append(y)
                    metrics.append(evaluate(X, y, labels, cfg.pilot.n_neighbors_eval))
            stable = stability(embeddings, cfg.pilot.n_neighbors_eval) if mid in STOCHASTIC else 1.0
            ok = len(metrics) == len(runs) and not any(
                "not fully connected" in w.lower() or "singular" in w.lower()
                for r in runs
                for w in r["warnings"]
            )
            trust = float(np.mean([m["trustworthiness"] for m in metrics])) if metrics else None
            recall = float(np.mean([m["neighbor_recall"] for m in metrics])) if metrics else None
            row = {
                "method": mid,
                "grid_index": gi,
                "parameters": json.dumps(p, sort_keys=True),
                "status": "success" if ok else "failed",
                "trustworthiness": trust,
                "neighbor_recall": recall,
                "stability": stable,
                "quality": quality(trust, recall, stable) if ok else None,
                "runtime_seconds": float(np.mean([r["runtime_seconds"] for r in runs])),
            }
            rows.append(row)
            if ok:
                candidates.append(row)
        if not candidates:
            raise ValueError(f"No valid tuning run for shortlisted method {mid}")
        # Quantize quality to 1e-6 so meaningless floating-point noise does not break ties.
        winner = min(
            candidates, key=lambda r: (-round(r["quality"], 6), r["runtime_seconds"], r["grid_index"])
        )
        chosen[mid] = json.loads(winner["parameters"])
    pd.DataFrame(rows).to_csv(directory / "tuning_results.csv", index=False)
    write_json(directory / "tuning_results.json", rows)
    write_json(directory / "chosen_parameters.json", chosen)
    complete("tune", cfg, directory)


def final_stage(cfg, force=False):
    directory = cfg.output / "final"
    if not force and valid_stage(cfg, directory):
        return
    if not valid_stage(cfg, cfg.output / "tuning"):
        raise ValueError("Tuning missing or stale")
    chosen = read_json(cfg.output / "tuning/chosen_parameters.json")
    # Remove only obsolete, agent-generated method exports when the shortlist changes.
    for old in (directory / "embeddings").glob("*"):
        if old.suffix in {".csv", ".npy"} and old.stem not in chosen:
            old.unlink()
    table = read_json(cfg.output / "pilot/pilot_metrics.json")
    pilot_n = read_json(cfg.output / "pilot/cohort_manifest.json")["selected"]
    size = cfg.dataset.max_final_samples
    estimates = {}
    for row in table:
        if row["method"] in chosen:
            # Conservative cubic upper-envelope for cohort planning, not a measured scaling law.
            safe_ratio = (cfg.pilot.timeout_minutes_per_run * 60 / max(row["runtime_seconds"] * 2, 1)) ** (
                1 / 3
            )
            safe_n = max(pilot_n, int(pilot_n * safe_ratio))
            if cfg.dataset.name == "pathmnist":
                size = min(size, safe_n)
            estimates[row["method"]] = {
                "observed_pilot_seconds": row["runtime_seconds"],
                "conservative_sample_limit": safe_n,
            }
    X, labels, ids = prepare_cohort(cfg, directory, size)
    cohort = read_json(directory / "cohort_manifest.json")
    cohort["size_justification"] = {
        "configured_cap": cfg.dataset.max_final_samples,
        "declared_final_size": len(X),
        "pilot_samples": pilot_n,
        "estimates": estimates,
        "rule": "2x pilot runtime * (N_final/N_pilot)^3 <= per-run timeout; cap; "
        "conservative planning bound, not measured scaling. PBMC normally full post-QC.",
    }
    write_json(directory / "cohort_manifest.json", cohort)
    rows = []
    (directory / "embeddings").mkdir(exist_ok=True)
    for mid, p in chosen.items():
        run_dir = directory / "runs" / mid
        r = run_method(
            mid,
            directory / "input.npy",
            cfg.project.seed,
            p,
            run_dir,
            cfg.pilot.timeout_minutes_per_run * 60,
            force,
        )
        if r["status"] != "success":
            raise ValueError(f"Final {mid} {r['status']}; saved evidence, no success flag")
        y = np.load(run_dir / "embedding.npy")
        np.save(directory / "embeddings" / f"{mid}.npy", y)
        frame = pd.DataFrame({"observation_id": ids, "label": labels, "dim1": y[:, 0], "dim2": y[:, 1]})
        frame.to_csv(directory / "embeddings" / f"{mid}.csv", index=False)
        row = dict(
            method=mid,
            **evaluate(X, y, labels, cfg.pilot.n_neighbors_eval),
            runtime_seconds=r["runtime_seconds"],
            status=r["status"],
        )
        rows.append(row)
    pd.DataFrame(rows).to_csv(directory / "final_metrics.csv", index=False)
    write_json(directory / "final_metrics.json", rows)
    complete("final", cfg, directory)
