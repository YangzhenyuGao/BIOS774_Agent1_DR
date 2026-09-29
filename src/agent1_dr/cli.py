import argparse
import json

from .config import ROOT, load_config

DATASET_CONFIGS = ("config/pbmc3k.yaml", "config/pathmnist.yaml")


def main():
    parser = argparse.ArgumentParser(description="Evidence-driven dimensionality reduction agent")
    parser.add_argument(
        "stage",
        choices=["inspect", "pilot", "select", "tune", "final", "report", "validate", "run-all", "download",
                 "freeze", "project-report", "examples"],
    )
    parser.add_argument("--config", default=str(ROOT / "config/pbmc3k.yaml"))
    parser.add_argument("--resume", action="store_true", help="Reuse valid results (also the default)")
    parser.add_argument("--force", action="store_true", help="Recompute requested stage")
    parser.add_argument("--all-reports", action="store_true", help="Also check all three deliverable PDFs")
    args = parser.parse_args()
    cfg = load_config(args.config)
    both = [load_config(ROOT / c) for c in DATASET_CONFIGS]
    if args.stage == "download":
        from .data import load_data

        data = load_data(cfg)
        print(json.dumps({"shape": data[0].shape, "provenance": data[3]}, indent=2))
    elif args.stage == "freeze":
        from .freeze import freeze

        record = freeze(both)
        print(json.dumps({k: v for k, v in record.items() if k != "files"}, indent=2))
    elif args.stage == "project-report":
        from .project_report import project_report

        project_report(both)
    elif args.stage == "examples":
        from .project_report import export_examples

        export_examples(both)
    elif args.stage == "validate":
        from .validation import validate

        result = validate(cfg, all_reports=args.all_reports)
        print(json.dumps(result, indent=2))
        if result["status"] != "PASSED":
            raise SystemExit(1)
    else:
        from .orchestrator import run_all, run_stage

        if args.stage == "run-all":
            run_all(cfg, args.force)
        else:
            run_stage(args.stage, cfg, args.force)
