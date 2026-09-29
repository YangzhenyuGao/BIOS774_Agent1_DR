import json
import os
import shutil
from dataclasses import asdict

from jinja2 import Environment, FileSystemLoader, select_autoescape
from weasyprint import HTML

from .config import ROOT
from .registry import METHOD_IDS
from .schemas import ReportEvidenceBundle
from .utils import checksum, complete, manifest, read_json, valid_stage, write_json
from .validation import dataset_pdf, pdf_ok, validate
from .visualization import NAMES, figures


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


def _best(rows, key):
    valid = [r for r in rows if r.get(key) is not None]
    return max(valid, key=lambda r: r[key]) if valid else None


def _names(methods):
    return ", ".join(NAMES[m] for m in methods) if methods else "none"


def interpretation(ctx):
    """Evidence sentences computed from saved artifacts; the template only lays them out."""
    pilot, final, decision = ctx["pilot"], ctx["final"], ctx["decisions"]
    valid = [r for r in pilot if r["quality"] is not None and not r["severe_warning"]]
    facts = []
    best = decision["best_method"]
    if best:
        equiv = [m for m in decision["equivalent_to_best"] if m != best]
        facts.append(
            f"On the pilot cohort, {NAMES[best]} has the highest pre-registered quality "
            f"(Q = {decision['best_quality']:.3f}); "
            + (f"{_names(equiv)} fall within its equivalence margin." if equiv
               else "no other valid method falls within its equivalence margin.")
        )
    for key, label in (("neighbor_recall", "neighbor recall"), ("global_structure", "global structure"),
                       ("stability", "subsample stability")):
        top = _best(valid, key)
        if top:
            facts.append(f"Among valid pilot methods, {NAMES[top['method']]} has the highest {label} "
                         f"({top[key]:.3f}).")
    severe = [r for r in pilot if r["severe_warning"]]
    if severe:
        facts.append("Severe diagnostics made these methods ineligible: " + "; ".join(
            f"{NAMES[r['method']]} ({', '.join(r['severe_reasons'])})" for r in severe) + ".")
    invariant = [r["method"] for r in pilot if r.get("seed_invariant")]
    if invariant:
        facts.append(f"{_names(invariant)} returned identical embeddings for every seed, so its seed "
                     "stability is trivially 1; the subsample stability used in Q is not affected.")
    freq = decision["sensitivity"]["selection_frequency"]
    n_var = len(decision["sensitivity"]["variants"])
    robust = [m for m in METHOD_IDS if freq.get(m, 0) == 1]
    fragile = [m for m in decision["selected"] if freq.get(m, 0) < 0.5]
    facts.append(f"Across {n_var} rule variants, {_names(robust)} {'is' if len(robust) == 1 else 'are'} "
                 "retained in every variant"
                 + (f"; {_names(fragile)} {'is' if len(fragile) == 1 else 'are'} retained in fewer than half."
                    if fragile else "."))
    if final:
        size = ctx["cohort"]["selected"]
        t, g = _best(final, "trustworthiness"), _best(final, "global_structure")
        r = _best(final, "neighbor_recall")
        facts.append(
            f"On the final cohort (n = {size:,}), {NAMES[t['method']]} has the highest trustworthiness "
            f"({t['trustworthiness']:.3f}), {NAMES[r['method']]} the highest neighbor recall "
            f"({r['neighbor_recall']:.3f}) and {NAMES[g['method']]} the best global structure "
            f"({g['global_structure']:.3f})."
        )
        sil = [x for x in final if x["silhouette"] is not None]
        if sil and all(x["silhouette"] < 0 for x in sil):
            facts.append("Every final embedding has a negative secondary label silhouette: the annotated "
                         "classes are not separated in this input representation.")
        elif sil:
            top = max(sil, key=lambda x: x["silhouette"])
            facts.append(f"The secondary label silhouette is highest for {NAMES[top['method']]} "
                         f"({top['silhouette']:.3f}); labels were not used for any fit or decision.")
    ratios = ctx["cohort"]["pca_explained_variance_ratio"]
    facts.append(f"The first two principal components explain {100 * sum(ratios[:2]):.1f}% of the preprocessed "
                 f"feature variance; {len(ratios)} components explain {100 * sum(ratios):.1f}%.")
    return facts


def context(cfg):
    out = cfg.output
    figures_ = {f["name"]: f for f in read_json(out / "figure_manifest.json")}
    ctx = {
        "name": cfg.dataset.name,
        "profile": read_json(out / "dataset_profile.json"),
        "plan": read_json(out / "final/preprocessing_plan.json"),
        "pilot_plan": read_json(out / "pilot/preprocessing_plan.json"),
        "pilot": read_json(out / "pilot/pilot_metrics.json"),
        "cohort": read_json(out / "final/cohort_manifest.json"),
        "pilot_cohort": read_json(out / "pilot/cohort_manifest.json"),
        "decisions": read_json(out / "selection/decisions.json"),
        "tuning": read_json(out / "tuning/tuning_results.json"),
        "tuning_decisions": read_json(out / "tuning/tuning_decisions.json"),
        "chosen": read_json(out / "tuning/chosen_parameters.json"),
        "final": read_json(out / "final/final_metrics.json"),
        "size_plan": read_json(out / "final/size_plan.json"),
        "figures": figures_,
        "manifest": read_json(out / "run_manifest.json"),
        "mode": cfg.reporting.mode,
        "cfg": cfg,
        "names": NAMES,
        "method_ids": METHOD_IDS,
    }
    defaults = {}
    for r in ctx["tuning"]:
        current = json.loads(r["parameters"])
        base = defaults.setdefault(r["method"], current)
        r["changed"] = {k: v for k, v in current.items() if base.get(k) != v}
    ctx["facts"] = interpretation(ctx)
    return ctx


def _fmt(value):
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, list):
        return f"[{len(value)} values]" if len(value) > 4 else ", ".join(_fmt(v) for v in value)
    return str(value)


def params(d, skip=("explained_variance_ratio",)):
    return "; ".join(f"{k}={_fmt(v)}" for k, v in d.items() if k not in skip) or "none"


def render(template, output, html_copy=None, **ctx):
    env = Environment(loader=FileSystemLoader(ROOT / "templates"), autoescape=select_autoescape(["j2"]))
    env.filters["params"] = params
    env.filters["fmt"] = _fmt
    html = env.get_template(template).render(css=(ROOT / "templates/report.css").read_text(), **ctx)
    output.parent.mkdir(parents=True, exist_ok=True)
    if html_copy is not None:
        html_copy.parent.mkdir(parents=True, exist_ok=True)
        html_copy.write_text(html)
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
    for old in (cfg.output / "pilot/figures").glob("*.png"):
        old.unlink()
    figures(cfg)
    # Figures are report artifacts; refresh the final manifest after adding them.
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
        "pilot/subsample_metrics.json",
        "tuning/tuning_results.csv",
        "tuning/tuning_decisions.json",
        "final/final_metrics.csv",
        "final/size_plan.json",
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
        ctx["summary"] = extractive_llm(ctx["facts"], cfg.reporting.model)
        write_json(bundle / "llm_summary.json", ctx["summary"])
    render("generated_report.html.j2", dataset_pdf(cfg), html_copy=bundle / "report.html", **ctx)
    write_json(
        bundle / "pdf_manifest.json", {"path": str(dataset_pdf(cfg)), "sha256": checksum(dataset_pdf(cfg))}
    )
    validation = validate(cfg)
    if validation["status"] != "PASSED":
        raise ValueError("Dataset validation failed")
    complete("report", cfg, bundle)
