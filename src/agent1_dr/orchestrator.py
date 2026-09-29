import json
import os
from datetime import UTC, datetime

from .final import final_stage
from .pilot import inspect_stage, pilot_stage
from .reporting import report_stage
from .selector import selection_stage
from .tuning import tuning_stage
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
    started = datetime.now(UTC)
    STAGES[stage](cfg, force)
    # Machine-readable stage history lives with the outputs, not in the hand-written docs.
    cfg.output.mkdir(parents=True, exist_ok=True)
    with open(cfg.output / "stage_history.jsonl", "a") as f:
        f.write(json.dumps({
            "stage": stage,
            "dataset": cfg.dataset.name,
            "started": started.isoformat(),
            "finished": datetime.now(UTC).isoformat(),
            "job": os.environ.get("SLURM_JOB_ID"),
            "host": os.uname().nodename,
        }) + "\n")


def run_all(cfg, force=False):
    for stage in STAGES:
        run_stage(stage, cfg, force)
    result = validate(cfg)
    if result["status"] != "PASSED":
        raise RuntimeError("Validation failed; see validation_results.json")
