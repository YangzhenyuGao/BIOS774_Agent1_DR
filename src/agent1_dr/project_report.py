"""Final project report rendered from frozen evidence (the narrative lives in the template)."""

import shutil

from .config import ROOT
from .freeze import verify_frozen
from .reporting import context, render
from .utils import checksum, write_json
from .validation import pdf_ok, validate


def project_report(configs):
    frozen = verify_frozen()
    contexts = []
    for cfg in configs:
        if validate(cfg, write=False)["status"] != "PASSED":
            raise ValueError(f"{cfg.dataset.name} must validate before the project report")
        contexts.append(context(cfg))
    target = ROOT / "reports/report.pdf"
    render("project_report.html.j2", target, html_copy=ROOT / "reports/source/report.html",
           datasets={c["name"]: c for c in contexts}, frozen=frozen)
    if not pdf_ok(target, max_pages=4):
        raise ValueError("Project report main body exceeds four pages")
    records = []
    for folder in ["src", "config", "templates", "scripts", "tests", "docs", "reports", "examples"]:
        for p in sorted((ROOT / folder).rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc":
                records.append({"path": str(p.relative_to(ROOT)), "sha256": checksum(p), "bytes": p.stat().st_size})
    for name in ["README.md", "environment.yml", "environment.lock.txt", "environment.conda-explicit.txt",
                 "pyproject.toml", "Makefile", ".gitignore"]:
        p = ROOT / name
        records.append({"path": name, "sha256": checksum(p), "bytes": p.stat().st_size})
    write_json(ROOT / "submission_manifest.json", {"frozen_evidence_commit": frozen["git_commit"],
                                                   "files": records})
    for cfg in configs:
        result = validate(cfg, all_reports=True, write=False)
        if result["status"] != "PASSED":
            raise ValueError(f"Submission validation failed: {[c for c in result['checks'] if not c['passed']]}")
    return records


def export_examples(configs):
    """Small, Git-friendly snapshots of the frozen evidence."""
    for cfg in configs:
        dest = ROOT / "examples" / cfg.dataset.name
        dest.mkdir(parents=True, exist_ok=True)
        for file in ["pilot/pilot_metrics.csv", "final/final_metrics.csv", "selection/decisions.json",
                     "tuning/tuning_results.csv", "tuning/tuning_decisions.json", "final/size_plan.json",
                     "preprocessing_plan.json", "dataset_profile.json", "validation_results.json"]:
            shutil.copy2(cfg.output / file, dest / file.split("/")[-1])
            (dest / file.split("/")[-1]).chmod(0o644)
