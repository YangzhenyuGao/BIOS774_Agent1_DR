"""Shortlist-only refinement on the pilot cohort with the same label-free evidence."""

import json

import numpy as np
import pandas as pd

from .benchmark import finalize, fit_jobs, load_subsamples, summarize
from .evaluation import Reference
from .execution import parameters, run_fits
from .registry import get_method
from .selector import PRIMARY_RULES, Margins, weights
from .utils import complete, read_json, valid_stage, write_json


def grid_for(cfg, mid, X):
    """Default first; configured overrides (for example the GPLVM budget) apply to every point."""
    method = get_method(mid)
    default = parameters(cfg, mid, X)
    grid = [default]
    for proposed in method.tuning_grid(X):
        p = dict(default)
        p.update({k: v for k, v in proposed.items() if v != method.default_params(X).get(k)})
        if p not in grid:
            grid.append(p)
    return grid


def choose(rows, subs, cfg):
    """Keep the default unless another grid point beats it by more than the equivalence margin."""
    valid = [r for r in rows if r["quality"] is not None and not r["severe_warning"]]
    if not valid:
        return None, "no valid grid point"
    default = rows[0]
    best = max(valid, key=lambda r: (r["quality"], -r["runtime_seconds"], -r["grid_index"]))
    if default not in valid:
        return best, f"default ineligible ({default['status']}, {'; '.join(default['severe_reasons'])}); best valid point"
    if best is default:
        return default, "default has the highest quality"
    margins = Margins({str(k): v for k, v in subs.items()}, weights(cfg), cfg.selection.equivalence_floor,
                      PRIMARY_RULES)
    margin = margins.margin(str(best["grid_index"]), "0")
    gain = best["quality"] - default["quality"]
    if gain <= margin:
        return default, f"best grid point {best['grid_index']} gains {gain:.4f} <= margin {margin:.4f}; default kept"
    return best, f"grid point {best['grid_index']} gains {gain:.4f} > margin {margin:.4f}"


def tuning_stage(cfg, force=False):
    directory = cfg.output / "tuning"
    if not force and valid_stage(cfg, directory):
        return
    if not valid_stage(cfg, cfg.output / "selection"):
        raise ValueError("Selection missing or stale")
    selected = read_json(cfg.output / "selection/decisions.json")["selected"]
    pilot = cfg.output / "pilot"
    X, labels = np.load(pilot / "input.npy"), np.load(pilot / "labels.npy")
    subsamples = load_subsamples(cfg, pilot)
    ref = Reference(X, k_max=cfg.pilot.n_neighbors_secondary, seed=cfg.project.seed)
    sub_refs = [ref.subset(index) for index, _ in subsamples]
    pca_collapse = next(r for r in read_json(pilot / "pilot_metrics.json") if r["method"] == "pca")["collapse_ratio"]
    timeout = cfg.pilot.timeout_minutes_per_run * 60
    rows, chosen, decisions = [], {}, {}
    for mid in selected:
        grid = grid_for(cfg, mid, X)
        jobs, spans = [], []
        for gi, p in enumerate(grid):
            part, n_full = fit_jobs(cfg, mid, p, pilot / "input.npy", subsamples, directory / "runs" / mid / str(gi))
            spans.append((len(jobs), len(part), n_full))
            jobs += part
        results = run_fits(mid, jobs, timeout, force, log_dir=directory / "runs" / mid)
        method_rows, subs = [], {}
        for gi, (start, size, n_full) in enumerate(spans):
            row, per = summarize(cfg, mid, jobs[start : start + size], results[start : start + size], n_full,
                                 ref, sub_refs, subsamples, labels)
            row.update(grid_index=gi, parameters=json.dumps(grid[gi], sort_keys=True))
            method_rows.append(row)
            subs[gi] = per
        finalize(method_rows, cfg, pca_collapse)
        winner, reason = choose(method_rows, subs, cfg)
        if winner is None:
            raise ValueError(f"No valid tuning run for shortlisted method {mid}")
        chosen[mid] = json.loads(winner["parameters"])
        decisions[mid] = {"grid_index": winner["grid_index"], "quality": winner["quality"], "reason": reason}
        rows += method_rows
    columns = ["method", "grid_index", "parameters", "status", "trustworthiness", "neighbor_recall",
               "global_structure", "stability", "quality", "runtime_seconds", "collapse_ratio", "severe_reasons"]
    pd.DataFrame(rows)[columns].to_csv(directory / "tuning_results.csv", index=False)
    write_json(directory / "tuning_results.json", rows)
    write_json(directory / "chosen_parameters.json", chosen)
    write_json(directory / "tuning_decisions.json", decisions)
    complete("tune", cfg, directory)
