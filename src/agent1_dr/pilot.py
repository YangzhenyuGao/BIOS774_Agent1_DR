from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_data
from .evaluation import evaluate, quality, stability
from .execution import parameters, run_method
from .inspector import inspect_data
from .preprocessing import prepare
from .registry import METHOD_IDS, STOCHASTIC
from .utils import checksum, complete, manifest, read_json, valid_stage, write_json


def inspect_stage(cfg, force=False):
    directory = cfg.output / "inspection"
    if not force and valid_stage(cfg, directory):
        return
    write_json(cfg.output / "effective_config.json", asdict(cfg))
    data = load_data(cfg)
    profile = asdict(inspect_data(cfg.dataset.name, data[0], data[1], data[3]))
    write_json(cfg.output / "dataset_profile.json", profile)
    write_json(directory / "dataset_profile.json", profile)
    complete("inspect", cfg, directory)


def prepare_cohort(cfg, directory, size):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    X, labels, ids, plan, cohort = prepare(cfg, load_data(cfg), size)
    # Pandas annotations can produce object arrays even when every value is text.
    # Persist plain Unicode so downstream stages can keep pickle loading disabled.
    labels, ids = np.asarray(labels, dtype=str), np.asarray(ids, dtype=str)
    np.save(directory / "input.npy", X, allow_pickle=False)
    np.save(directory / "labels.npy", labels, allow_pickle=False)
    np.save(directory / "ids.npy", ids, allow_pickle=False)
    cohort.update(
        input_sha256=checksum(directory / "input.npy"),
        seed=cfg.project.seed,
        config_hash=cfg.hash,
        labels_used_for_fit=False,
        input_shape=list(X.shape),
    )
    write_json(directory / "cohort_manifest.json", cohort)
    write_json(directory / "preprocessing_plan.json", plan)
    return X, labels, ids


def pilot_stage(cfg, force=False):
    directory = cfg.output / "pilot"
    if not force and valid_stage(cfg, directory):
        return
    (directory / "pilot_complete.flag").unlink(missing_ok=True)
    X, labels, _ = prepare_cohort(cfg, directory, cfg.pilot.max_samples)
    write_json(cfg.output / "preprocessing_plan.json", read_json(directory / "preprocessing_plan.json"))
    table, failures = [], []
    for mid in METHOD_IDS:
        results, metrics, embeddings = [], [], []
        seeds = [
            cfg.project.seed + i for i in range(cfg.pilot.n_repeats_stochastic if mid in STOCHASTIC else 1)
        ]
        params = parameters(cfg, mid, X)
        for seed in seeds:
            run_dir = directory / "method_runs" / mid / str(seed)
            r = run_method(
                mid,
                directory / "input.npy",
                seed,
                params,
                run_dir,
                cfg.pilot.timeout_minutes_per_run * 60,
                force,
            )
            results.append(r)
            if r["status"] == "success":
                y = np.load(run_dir / "embedding.npy")
                e = evaluate(X, y, labels, cfg.pilot.n_neighbors_eval)
                embeddings.append(y)
                metrics.append(e)
                write_json(run_dir / "metrics.json", e)
            else:
                failures.append(r)
        stable = stability(embeddings, min(cfg.pilot.n_neighbors_eval, (len(X) - 1) // 2))
        # Deterministic methods have seed stability 1 by definition; stochastic partial failures are ineligible.
        if mid not in STOCHASTIC:
            stable = 1.0
        status = (
            "success"
            if all(r["status"] == "success" for r in results)
            else ("timeout" if any(r["status"] == "timeout" for r in results) else "failed")
        )
        severe = any(
            "not fully connected" in w.lower() or "singular" in w.lower()
            for r in results
            for w in r["warnings"]
        )
        row = {
            "method": mid,
            "status": status,
            "repeats": len(seeds),
            "successes": len(embeddings),
            "stability": stable,
            "stability_note": "measured across seeds" if mid in STOCHASTIC else "deterministic: defined as 1",
            "runtime_seconds": float(np.mean([r["runtime_seconds"] for r in results])),
            "peak_memory_mb": max([r["peak_memory_mb"] or 0 for r in results]),
            "severe_warning": severe,
            "warnings": [w for r in results for w in r["warnings"]],
            "retry_count": sum(r["retry_count"] for r in results),
            "input_sha256": checksum(directory / "input.npy"),
            "trustworthiness": None,
            "neighbor_recall": None,
            "silhouette": None,
            "quality": None,
        }
        if metrics:
            for key in ("trustworthiness", "neighbor_recall", "silhouette"):
                vals = [e[key] for e in metrics if e[key] is not None]
                row[key] = float(np.mean(vals)) if vals else None
            if stable is not None:
                row["quality"] = quality(row["trustworthiness"], row["neighbor_recall"], stable)
        table.append(row)
    write_json(directory / "pilot_metrics.json", table)
    pd.DataFrame(table).to_csv(directory / "pilot_metrics.csv", index=False)
    write_json(directory / "failures.json", failures)
    assert {r["method"] for r in table} == set(METHOD_IDS)
    assert all(r["status"] in {"success", "failed", "timeout"} for r in table)
    write_json(directory / "run_manifest.json", manifest("pilot", cfg, [directory / "input.npy"]))
    complete("pilot", cfg, directory)
    (directory / "pilot_complete.flag").write_text("All ten methods reached terminal statuses.\n")
