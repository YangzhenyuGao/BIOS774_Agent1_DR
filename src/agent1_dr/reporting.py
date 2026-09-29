import json
import os
import shutil
from dataclasses import asdict

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape
from weasyprint import HTML

from .config import ROOT
from .schemas import ReportEvidenceBundle
from .utils import checksum, complete, manifest, read_json, valid_stage, write_json
from .validation import dataset_pdf, pdf_ok, validate
from .visualization import figures


def extractive_llm(facts, model, client=None):
    """LLM selects/organizes exact evidence sentences; arbitrary claims are rejected."""
    if client is None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "OpenAI reporting requested but OPENAI_API_KEY is unavailable; no fallback performed"
            )
        from openai import OpenAI

        client = OpenAI(timeout=90, max_retries=1)
    schema = {
        "type": "object",
        "properties": {"sentences": {"type": "array", "items": {"type": "string"}}},
        "required": ["sentences"],
        "additionalProperties": False,
    }
    response = client.responses.create(
        model=model,
        store=False,
        instructions="Organize a concise evidence summary. Return only exact unmodified sentences from the "
        "supplied validated evidence list; never invent or change a number, fact or citation.",
        input=json.dumps(facts),
        text={
            "format": {"type": "json_schema", "name": "evidence_summary", "schema": schema, "strict": True}
        },
    )
    sentences = json.loads(response.output_text)["sentences"]
    if not sentences or any(s not in facts for s in sentences):
        raise ValueError("LLM claim validation failed: returned text is not in validated evidence")
    return {
        "sentences": sentences,
        "model": model,
        "response_id": response.id,
        "mode": "openai: extractive evidence organization, exact-claim validation",
    }


def context(cfg):
    out = cfg.output
    return {
        "name": cfg.dataset.name,
        "profile": read_json(out / "dataset_profile.json"),
        "plan": read_json(out / "final/preprocessing_plan.json"),
        "pilot": read_json(out / "pilot/pilot_metrics.json"),
        "cohort": read_json(out / "final/cohort_manifest.json"),
        "pilot_cohort": read_json(out / "pilot/cohort_manifest.json"),
        "decisions": read_json(out / "selection/decisions.json"),
        "tuning": read_json(out / "tuning/tuning_results.json"),
        "final": read_json(out / "final/final_metrics.json"),
        "figures": read_json(out / "figure_manifest.json"),
        "manifest": read_json(out / "run_manifest.json"),
        "mode": cfg.reporting.mode,
        "k": cfg.pilot.n_neighbors_eval,
    }


def render(template, output, **ctx):
    env = Environment(loader=FileSystemLoader(ROOT / "templates"), autoescape=select_autoescape(["j2"]))
    html = env.get_template(template).render(css=(ROOT / "templates/report.css").read_text(), **ctx)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".html").write_text(html)
    HTML(string=html, base_url=str(ROOT)).write_pdf(output)
    if not pdf_ok(output):
        raise ValueError(f"Invalid PDF {output}")


def report_stage(cfg, force=False):
    initial = validate(cfg, require_reports=False)
    if initial["status"] != "PASSED":
        raise ValueError("Analysis evidence audit failed before report generation")
    if not force and valid_stage(cfg, cfg.output / "evidence_bundle"):
        try:
            saved_pdf = read_json(cfg.output / "evidence_bundle/pdf_manifest.json")
            if (
                checksum(dataset_pdf(cfg)) == saved_pdf["sha256"]
                and pdf_ok(dataset_pdf(cfg))
                and validate(cfg)["status"] == "PASSED"
            ):
                return
        except (OSError, ValueError, KeyError):
            pass
    # Figures are separate report artifacts; refresh final manifest after adding them.
    figures(cfg)
    complete("final", cfg, cfg.output / "final")
    validation = validate(cfg, require_reports=False)
    if validation["status"] != "PASSED":
        raise ValueError("Analysis evidence audit failed; inspect validation_results.json")
    write_json(
        cfg.output / "run_manifest.json",
        manifest(
            "report",
            cfg,
            [
                cfg.output / "pilot/pilot_metrics.json",
                cfg.output / "selection/decisions.json",
                cfg.output / "final/final_metrics.csv",
            ],
        ),
    )
    bundle = cfg.output / "evidence_bundle"
    bundle.mkdir(exist_ok=True)
    paths = [
        "dataset_profile.json",
        "preprocessing_plan.json",
        "pilot/pilot_metrics.csv",
        "tuning/tuning_results.csv",
        "final/final_metrics.csv",
        "selection/decisions.json",
        "run_manifest.json",
        "figure_manifest.json",
        "validation_results.json",
    ]
    checksums = {}
    for relative in paths:
        src = cfg.output / relative
        target = bundle / src.name
        shutil.copy2(src, target)
        checksums[target.name] = checksum(target)
    write_json(
        bundle / "manifest.json",
        asdict(ReportEvidenceBundle(cfg.dataset.name, checksums, cfg.reporting.mode, True)),
    )
    ctx = context(cfg)
    ctx["summary"] = None
    if cfg.reporting.mode == "openai":
        facts = [d["method"] + ": " + d["reason"] for d in ctx["decisions"]["decisions"]]
        ctx["summary"] = extractive_llm(facts, cfg.reporting.model)
        write_json(bundle / "llm_summary.json", ctx["summary"])
    render("generated_report.html.j2", dataset_pdf(cfg), **ctx)
    write_json(
        bundle / "pdf_manifest.json", {"path": str(dataset_pdf(cfg)), "sha256": checksum(dataset_pdf(cfg))}
    )
    validation = validate(cfg)
    if validation["status"] != "PASSED":
        raise ValueError("Dataset validation failed")
    complete("report", cfg, bundle)


def project_report(configs):
    contexts = []
    for cfg in configs:
        result = validate(cfg)
        if result["status"] != "PASSED":
            raise ValueError(f"{cfg.dataset.name} must validate before project report")
        contexts.append(context(cfg))
    render("project_report.html.j2", ROOT / "reports/report.pdf", datasets=contexts)
    if not pdf_ok(ROOT / "reports/report.pdf", max_pages=4):
        raise ValueError("Project report exceeds four-page limit")
    records = []
    for folder in ["src", "config", "templates", "scripts", "tests", "docs", "reports"]:
        for p in sorted((ROOT / folder).rglob("*")):
            if p.is_file() and "__pycache__" not in str(p) and p.suffix != ".pyc":
                records.append(
                    {"path": str(p.relative_to(ROOT)), "sha256": checksum(p), "bytes": p.stat().st_size}
                )
    for name in [
        "README.md",
        "environment.yml",
        "environment.lock.txt",
        "environment.conda-explicit.txt",
        "pyproject.toml",
        "Makefile",
        ".gitignore",
    ]:
        p = ROOT / name
        records.append({"path": name, "sha256": checksum(p), "bytes": p.stat().st_size})
    write_json(ROOT / "submission_manifest.json", records)
    small = ROOT / "examples"
    small.mkdir(exist_ok=True)
    for cfg in configs:
        dest = small / cfg.dataset.name
        dest.mkdir(exist_ok=True)
        for file in ["pilot/pilot_metrics.csv", "final/final_metrics.csv", "selection/decisions.json"]:
            shutil.copy2(cfg.output / file, dest / file.split("/")[-1])
        result = validate(cfg, all_reports=True)
        if result["status"] != "PASSED":
            raise ValueError("Submission validation failed")
    return pd.DataFrame(records)
