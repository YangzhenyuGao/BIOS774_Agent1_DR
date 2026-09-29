"""Audit actual artifacts rather than trusting completion flags."""

import json

import numpy as np
import pandas as pd
from pypdf import PdfReader

from .config import ROOT
from .registry import METHOD_IDS, STOCHASTIC, get_method
from .selector import select
from .utils import checksum, read_json, valid_stage, write_json


def dataset_pdf(cfg):
    name = cfg.reporting.report_name
    return ROOT / "reports" / f"{name}.pdf" if name else cfg.output / "generated_report.pdf"


def pdf_pages(path):
    return [p.extract_text() or "" for p in PdfReader(path).pages]


def main_body_pages(path):
    """Pages before the first page that starts with a References or Appendix heading."""
    pages = pdf_pages(path)
    for i, text in enumerate(pages):
        head = text.strip().splitlines()[0].strip().lower() if text.strip() else ""
        if head.startswith(("references", "appendix")):
            return i
    return len(pages)


def pdf_ok(path, max_pages=None):
    pages = pdf_pages(path)
    return (
        len(pages) > 0
        and all(bool(t.strip()) for t in pages)
        and (max_pages is None or main_body_pages(path) <= max_pages)
    )


def validate(cfg, require_reports=True, all_reports=False, write=True):
    checks, warnings = [], []

    def check(name, fn):
        try:
            value = fn()
            if value is False:
                raise ValueError("condition is false")
            checks.append({"check": name, "passed": True})
        except Exception as exc:  # noqa: BLE001 -- intentional isolation/audit boundary
            checks.append({"check": name, "passed": False, "detail": f"{type(exc).__name__}: {exc}"})

    out = cfg.output
    check("registry contains ten callable real methods", lambda: all(get_method(m).method_id == m for m in METHOD_IDS))
    for stage in ["inspection", "pilot", "selection", "tuning", "final"]:
        check(f"{stage} manifest checksums and configuration/code provenance",
              lambda s=stage: valid_stage(cfg, out / s))
    table, subsample = [], {}
    try:
        table = read_json(out / "pilot/pilot_metrics.json")
        subsample = read_json(out / "pilot/subsample_metrics.json")
    except (OSError, ValueError):
        pass
    check(
        "all ten methods have terminal pilot status",
        lambda: len(table) == 10
        and {r["method"] for r in table} == set(METHOD_IDS)
        and all(r["status"] in {"success", "failed", "timeout"} for r in table),
    )
    root = out / "pilot"

    def common_input():
        sha = checksum(root / "input.npy")
        X = np.load(root / "input.npy")
        for b in range(cfg.pilot.n_subsamples):
            index = np.load(root / f"subsamples/{b}/indices.npy")
            assert np.array_equal(np.load(root / f"subsamples/{b}/input.npy"), X[index])
        return all(r["input_sha256"] == sha for r in table)

    check("pilot input is common to all methods and subsamples are exact subsets", common_input)
    for row in table:
        if row["status"] != "success" or row["warnings"] or row["severe_reasons"]:
            warnings.append({"method": row["method"], "status": row["status"], "warnings": row["warnings"],
                             "severe": row["severe_reasons"]})
        mid = row["method"]
        repeats = cfg.pilot.n_repeats_stochastic if mid in STOCHASTIC else 1
        runs = [(root / f"method_runs/{mid}/{s}", root / "input.npy", s)
                for s in range(cfg.project.seed, cfg.project.seed + repeats)]
        runs += [(root / f"method_runs/{mid}/subsamples/{b}", root / f"subsamples/{b}/input.npy", cfg.project.seed + b)
                 for b in range(cfg.pilot.n_subsamples)]

        def inspect_runs(m=mid, runs=runs):
            for run_dir, input_path, seed in runs:
                r = read_json(run_dir / "result.json")
                assert r["method_id"] == m and r["seed"] == seed
                assert r["input_sha256"] == checksum(input_path)
                assert r["package_versions"] and r["status"] in {"success", "failed", "timeout"}
                if r["status"] == "success":
                    y = np.load(run_dir / "embedding.npy")
                    assert y.shape == (len(np.load(input_path)), 2) and np.isfinite(y).all()
                    e = read_json(run_dir / "metrics.json")
                    for key in ["trustworthiness", "neighbor_recall", "global_structure", "collapse_ratio"]:
                        assert np.isfinite(e[key]) and 0 <= e[key] <= 1, key
                    assert e["silhouette"] is not None or e["silhouette_note"]
                    assert "fit_wall_seconds" in r["timing"]
                    if m == "gplvm":
                        assert r["diagnostics"]["optimization_steps"] > 0
                        assert r["diagnostics"]["latent_displacement"] > 0
                else:
                    assert r["error_type"] and r["error_message"]

        check(f"pilot {mid}: {len(runs)} fits with input, output, metrics and status", inspect_runs)

    def metrics_complete():
        for r in table:
            if r["status"] == "success":
                for key in ("trustworthiness", "neighbor_recall", "global_structure", "stability"):
                    assert r[key] is not None, (r["method"], key)
                assert (r["quality"] is None) == (r["status"] != "success")
        return True

    check("metrics contain no unexplained missing values", metrics_complete)

    def selection_check():
        saved = read_json(out / "selection/decisions.json")
        again = select(table, cfg, subsample)
        assert saved["selected"] == again["selected"] and saved["decisions"] == again["decisions"]
        assert saved["labels_used"] is False and not cfg.selection.use_labels_for_selection
        return True

    check("selection matches the pre-registered label-free rules", selection_check)

    def tuning_check():
        from .tuning import choose

        selected = read_json(out / "selection/decisions.json")["selected"]
        t = pd.read_csv(out / "tuning/tuning_results.csv")
        assert set(t.method) == set(selected)
        chosen = read_json(out / "tuning/chosen_parameters.json")
        assert set(chosen) == set(selected)
        rows = read_json(out / "tuning/tuning_results.json")
        for mid in selected:
            mrows = [r for r in rows if r["method"] == mid]
            subs = {r["grid_index"]: None for r in mrows}
            for r in mrows:
                d = out / f"tuning/runs/{mid}/{r['grid_index']}/subsamples"
                subs[r["grid_index"]] = [
                    {c: read_json(d / f"{b}/metrics.json")[c] for c in
                     ("trustworthiness", "neighbor_recall", "global_structure", "trustworthiness_k2",
                      "neighbor_recall_k2", "collapse_ratio")}
                    if (d / f"{b}/metrics.json").exists() else None
                    for b in range(cfg.pilot.n_subsamples)
                ]
            winner, _ = choose(mrows, subs, cfg)
            assert chosen[mid] == json.loads(winner["parameters"]), mid
        return True

    check("tuning restricted to shortlist and choices follow the equivalence rule", tuning_check)

    def final_check():
        final = out / "final"
        ids, labels = np.load(final / "ids.npy"), np.load(final / "labels.npy")
        cohort = read_json(final / "cohort_manifest.json")
        assert ids.tolist() == cohort["observation_ids"] and labels.tolist() == cohort["labels"]
        assert len(set(ids)) == len(ids)
        assert cohort["input_sha256"] == checksum(final / "input.npy")
        assert cohort["labels_used_for_fit"] is False
        selected = read_json(out / "selection/decisions.json")["selected"]
        exported = {p.stem for p in (final / "embeddings").glob("*.npy")}
        assert exported == set(selected), f"stale or missing exports: {exported ^ set(selected)}"
        for mid in selected:
            y = np.load(final / "embeddings" / f"{mid}.npy")
            assert y.shape == (len(ids), 2) and np.isfinite(y).all()
            df = pd.read_csv(final / "embeddings" / f"{mid}.csv", dtype={"observation_id": str, "label": str})
            assert df.observation_id.tolist() == ids.tolist() and df.label.tolist() == labels.tolist()
        assert not read_json(final / "preprocessing_plan.json")["labels_used_for_fit"]
        return True

    check("final embeddings, IDs, labels, and label-free preprocessing align", final_check)

    def size_check():
        from .final import project, scaling_model

        plan = read_json(out / "final/size_plan.json")
        model = scaling_model(plan["probe"])
        again = [project(model, p["n"], cfg)["feasible"] for p in plan["projections"]]
        assert again == [p["feasible"] for p in plan["projections"]]
        assert plan["declared_final_size"] == read_json(out / "final/cohort_manifest.json")["selected"]
        return True

    check("final cohort size follows the recorded scaling rule", size_check)
    if require_reports:

        def figure_check():
            figures = read_json(out / "figure_manifest.json")
            assert len(figures) >= 9
            assert all(checksum(f["path"]) == f["sha256"] for f in figures)
            listed = {f["path"] for f in figures}
            on_disk = {str(p) for d in ("pilot/figures", "selection/figures", "final/figures")
                       for p in (out / d).glob("*.png")}
            assert on_disk <= listed, f"unlisted figures: {sorted(on_disk - listed)}"
            return True

        check("expected figures exist with valid checksums and no stale figures", figure_check)
        check("dataset report opens and has text on every page", lambda: pdf_ok(dataset_pdf(cfg)))

        def evidence_check():
            m = read_json(out / "evidence_bundle/manifest.json")
            assert m["validated"]
            assert all(checksum(out / "evidence_bundle" / name) == sha for name, sha in m["files"].items())
            p = read_json(out / "evidence_bundle/pdf_manifest.json")
            assert checksum(dataset_pdf(cfg)) == p["sha256"]
            return True

        check("report evidence and PDF checksums", evidence_check)
    if all_reports:
        for name in ["generated_report_1.pdf", "generated_report_2.pdf", "report.pdf"]:
            check(f"{name} opens" + (" and its main body is at most 4 pages" if name == "report.pdf" else ""),
                  lambda n=name: pdf_ok(ROOT / "reports" / n, max_pages=4 if n == "report.pdf" else None))
    result = {
        "status": "PASSED" if all(c["passed"] for c in checks) else "FAILED",
        "scope": "submission" if all_reports else ("dataset" if require_reports else "analysis"),
        "checks": checks,
        "warnings": warnings,
    }
    if write:
        write_json(out / "validation_results.json", result)
    return result
