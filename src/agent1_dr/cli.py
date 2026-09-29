import argparse
import json

from .config import ROOT, load_config


def main():
    parser = argparse.ArgumentParser(description="Evidence-driven dimensionality reduction agent")
    parser.add_argument(
        "stage",
        choices=[
            "inspect",
            "pilot",
            "select",
            "tune",
            "final",
            "report",
            "validate",
            "run-all",
            "download",
            "project-report",
        ],
    )
    parser.add_argument("--config", default=str(ROOT / "config/pbmc3k.yaml"))
    parser.add_argument("--resume", action="store_true", help="Reuse valid results (also the default)")
    parser.add_argument("--force", action="store_true", help="Recompute requested stage")
    parser.add_argument(
        "--all-reports", action="store_true", help="Include both dataset PDFs and project page limit"
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.stage == "download":
        from .data import load_data

        data = load_data(cfg)
        print(json.dumps({"shape": data[0].shape, "provenance": data[3]}, indent=2))
    elif args.stage == "project-report":
        from .reporting import project_report

        project_report(
            [load_config(ROOT / "config/pbmc3k.yaml"), load_config(ROOT / "config/pathmnist.yaml")]
        )
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
