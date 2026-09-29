"""Audit actual artifacts rather than trusting completion flags."""

import numpy as np
import pandas as pd
from pypdf import PdfReader

from .config import ROOT
from .registry import METHOD_IDS, STOCHASTIC, get_method
from .selector import select
from .utils import checksum, read_json, valid_stage, write_json


def validate(cfg, require_reports=True, all_reports=False):
    checks, warnings = [], []

    def check(name, fn):
        try:
            value = fn()
            if value is False:
                raise ValueError("condition is false")
            checks.append({"check": name, "passed": True})
        except Exception as exc:  # noqa: BLE001 -- intentional isolation/audit boundary
            checks.append({"check": name, "passed": False, "detail": f"{type(exc).__name__}: {exc}"})

    check(
        "registry contains ten callable real methods",
        lambda: all(get_method(m).method_id == m for m in METHOD_IDS),
    )
    for stage in ["inspection", "pilot", "selection", "tuning", "final"]:
        check(
            f"{stage} manifest checksums and configuration/code provenance",
            lambda s=stage: valid_stage(cfg, cfg.output / s),
        )
    table = []
    try:
        table = read_json(cfg.output / "pilot/pilot_metrics.json")
    except (OSError, ValueError):
        pass
    check(
        "all ten methods have terminal pilot status",
        lambda: (
            len(table) == 10
            and {r["method"] for r in table} == set(METHOD_IDS)
            and all(r["status"] in {"success", "failed", "timeout"} for r in table)
        ),
    )
    for row in table:
        if row["status"] != "success" or row["warnings"]:
            warnings.append({"method": row["method"], "status": row["status"], "warnings": row["warnings"]})
        mid = row["method"]
        repeats = cfg.pilot.n_repeats_stochastic if mid in STOCHASTIC else 1
        for seed in range(cfg.project.seed, cfg.project.seed + repeats):

            def inspect_run(m=mid, s=seed):
                root = cfg.output / "pilot"
                run_dir = root / "method_runs" / m / str(s)
                r = read_json(run_dir / "result.json")
                assert r["method_id"] == m and r["seed"] == s
                assert r["input_sha256"] == checksum(root / "input.npy")
                assert r["package_versions"] and r["status"] in {"success", "failed", "timeout"}
                if r["status"] == "success":
                    y = np.load(run_dir / "embedding.npy")
                    assert y.shape == (len(np.load(root / "input.npy")), 2) and np.isfinite(y).all()
                    e = read_json(run_dir / "metrics.json")
                    assert all(
                        np.isfinite(e[k]) and 0 <= e[k] <= 1 for k in ["trustworthiness", "neighbor_recall"]
                    )
                    assert e["silhouette"] is not None or e["silhouette_note"]
                    if m == "gplvm":
                        assert r["diagnostics"]["optimization_steps"] > 0
                        assert r["diagnostics"]["latent_displacement"] > 0
                else:
                    assert r["error_type"] and r["error_message"]

            check(f"pilot {mid}/{seed}: input, output, metrics, status", inspect_run)
    check(
        "selection matches label-free configured rules",
        lambda: read_json(cfg.output / "selection/decisions.json") == select(table, cfg),
    )

    def tuning_check():
        selected = read_json(cfg.output / "selection/decisions.json")["selected"]
        t = pd.read_csv(cfg.output / "tuning/tuning_results.csv")
        assert set(t.method) == set(selected)
        chosen = read_json(cfg.output / "tuning/chosen_parameters.json")
        assert set(chosen) == set(selected)
        import json

        rows = read_json(cfg.output / "tuning/tuning_results.json")
        for mid in selected:
            candidates = [r for r in rows if r["method"] == mid and r["status"] == "success"]
            best = min(
                candidates, key=lambda r: (-round(r["quality"], 6), r["runtime_seconds"], r["grid_index"])
            )
            assert chosen[mid] == json.loads(best["parameters"])

    check("tuning restricted to shortlist", tuning_check)

    def final_check():
        root = cfg.output / "final"
        ids, labels = np.load(root / "ids.npy"), np.load(root / "labels.npy")
        cohort = read_json(root / "cohort_manifest.json")
        assert ids.tolist() == cohort["observation_ids"] and labels.tolist() == cohort["labels"]
        assert len(set(ids)) == len(ids)
        assert cohort["input_sha256"] == checksum(root / "input.npy")
        assert cohort["labels_used_for_fit"] is False
        selected = read_json(cfg.output / "selection/decisions.json")["selected"]
        assert {p.stem for p in (root / "embeddings").glob("*.npy")} == set(selected)
        for mid in selected:
            y = np.load(root / "embeddings" / f"{mid}.npy")
            assert y.shape == (len(ids), 2) and np.isfinite(y).all()
            df = pd.read_csv(root / "embeddings" / f"{mid}.csv", dtype={"observation_id": str, "label": str})
            assert df.observation_id.tolist() == ids.tolist() and df.label.tolist() == labels.tolist()
        assert not read_json(root / "preprocessing_plan.json")["labels_used_for_fit"]

    check("final embeddings, IDs, labels, and label-free preprocessing align", final_check)
    if require_reports:

        def figure_check():
            figures = read_json(cfg.output / "figure_manifest.json")
            assert len(figures) >= 7
            assert all(checksum(f["path"]) == f["sha256"] for f in figures)

        check("expected figures exist with valid checksums", figure_check)
        check("dataset report opens and has text", lambda: pdf_ok(dataset_pdf(cfg)))

        def evidence_check():
            m = read_json(cfg.output / "evidence_bundle/manifest.json")
            assert m["validated"]
            assert all(
                checksum(cfg.output / "evidence_bundle" / name) == sha for name, sha in m["files"].items()
            )
            p = read_json(cfg.output / "evidence_bundle/pdf_manifest.json")
            assert checksum(dataset_pdf(cfg)) == p["sha256"]

        check("report evidence and PDF checksums", evidence_check)
    if all_reports:
        for name in ["generated_report_1.pdf", "generated_report_2.pdf", "report.pdf"]:
            check(
                name, lambda n=name: pdf_ok(ROOT / "reports" / n, max_pages=4 if n == "report.pdf" else None)
            )
    result = {
        "status": "PASSED" if all(c["passed"] for c in checks) else "FAILED",
        "scope": "submission" if all_reports else ("dataset" if require_reports else "analysis"),
        "checks": checks,
        "warnings": warnings,
    }
    write_json(cfg.output / "validation_results.json", result)
    return result


def dataset_pdf(cfg):
    if cfg.dataset.name not in {"pbmc3k", "pathmnist"}:
        return cfg.output / "generated_report.pdf"
    return (
        ROOT
        / "reports"
        / ("generated_report_1.pdf" if cfg.dataset.name == "pbmc3k" else "generated_report_2.pdf")
    )


def pdf_ok(path, max_pages=None):
    pdf = PdfReader(path)
    return (
        len(pdf.pages) > 0
        and (max_pages is None or len(pdf.pages) <= max_pages)
        and all(bool(p.extract_text().strip()) for p in pdf.pages)
    )
