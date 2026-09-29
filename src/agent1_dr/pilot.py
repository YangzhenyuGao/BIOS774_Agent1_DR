"""Inspection and the comparable pilot benchmark (all ten methods, one cohort, one input)."""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .benchmark import finalize, fit_jobs, make_subsamples, summarize
from .data import load_data
from .evaluation import Reference
from .execution import parameters, run_fits
from .inspector import inspect_data
from .preprocessing import prepare
from .registry import METHOD_IDS
from .utils import checksum, complete, manifest, read_json, valid_stage, write_json


def inspect_stage(cfg, force=False):
    directory = cfg.output / "inspection"
    if not force and valid_stage(cfg, directory):
        return
    write_json(cfg.output / "effective_config.json", asdict(cfg))
    data = load_data(cfg)
    profile = asdict(inspect_data(cfg.dataset.name, data[0], data[1], data[3], data[4]))
    write_json(cfg.output / "dataset_profile.json", profile)
    write_json(directory / "dataset_profile.json", profile)
    complete("inspect", cfg, directory)


def load_profile(cfg):
    if not valid_stage(cfg, cfg.output / "inspection"):
        raise ValueError("Inspection missing or stale; run inspect first")
    return read_json(cfg.output / "inspection/dataset_profile.json")


def prepare_cohort(cfg, directory, size, data=None, profile=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    data = load_data(cfg) if data is None else data
    profile = load_profile(cfg) if profile is None else profile
    X, labels, ids, plan, cohort = prepare(cfg, data, size, profile)
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
    subsamples = make_subsamples(cfg, directory, X)
    ref = Reference(X, k_max=cfg.pilot.n_neighbors_secondary, seed=cfg.project.seed)
    sub_refs = [ref.subset(index) for index, _ in subsamples]
    timeout = cfg.pilot.timeout_minutes_per_run * 60
    rows, per_subsample, failures = [], {}, []
    for mid in METHOD_IDS:
        params = parameters(cfg, mid, X)
        root = directory / "method_runs" / mid
        jobs, n_full = fit_jobs(cfg, mid, params, directory / "input.npy", subsamples, root)
        results = run_fits(mid, jobs, timeout, force, log_dir=root)
        row, subs = summarize(cfg, mid, jobs, results, n_full, ref, sub_refs, subsamples, labels)
        row["parameters"] = results[0]["parameters"]
        row["input_sha256"] = checksum(directory / "input.npy")
        rows.append(row)
        per_subsample[mid] = subs
        failures += [r for r in results if r["status"] != "success"]
    pca = next(r for r in rows if r["method"] == "pca")
    finalize(rows, cfg, pca["collapse_ratio"] if pca["status"] == "success" else None)
    write_json(directory / "pilot_metrics.json", rows)
    pd.DataFrame(rows).to_csv(directory / "pilot_metrics.csv", index=False)
    write_json(directory / "subsample_metrics.json", per_subsample)
    write_json(directory / "failures.json", failures)
    assert {r["method"] for r in rows} == set(METHOD_IDS)
    assert all(r["status"] in {"success", "failed", "timeout"} for r in rows)
    write_json(directory / "run_manifest.json", manifest("pilot", cfg, [directory / "input.npy"]))
    complete("pilot", cfg, directory)
    (directory / "pilot_complete.flag").write_text("All ten methods reached terminal statuses.\n")
