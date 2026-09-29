import os
from datetime import UTC, datetime

from .config import ROOT
from .pilot import inspect_stage, pilot_stage
from .reporting import report_stage
from .selector import selection_stage
from .tuning import final_stage, tuning_stage
from .validation import validate

STAGES = {
    "inspect": inspect_stage,
    "pilot": pilot_stage,
    "select": selection_stage,
    "tune": tuning_stage,
    "final": final_stage,
    "report": report_stage,
}


def run_stage(stage, cfg, force=False):
    if stage != "inspect" and not os.environ.get("SLURM_JOB_ID") and cfg.pilot.max_samples > 100:
        raise RuntimeError("Benchmark computation requires a SLURM allocation; use scripts/slurm")
    print(f"STAGE {stage} dataset={cfg.dataset.name}", flush=True)
    STAGES[stage](cfg, force)
    with open(ROOT / "docs/PROGRESS.md", "a") as f:
        f.write(
            f"\n- {datetime.now(UTC).isoformat()}: {cfg.dataset.name} {stage} complete; "
            f"job={os.environ.get('SLURM_JOB_ID', 'local-small-test')}; outputs={cfg.output}; "
            "stage errors, if any, are preserved in method result files.\n"
        )


def run_all(cfg, force=False):
    for stage in STAGES:
        run_stage(stage, cfg, force)
    result = validate(cfg)
    if result["status"] != "PASSED":
        raise RuntimeError("Validation failed; see validation_results.json")
