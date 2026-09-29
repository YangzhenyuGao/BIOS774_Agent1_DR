from agent1_dr.config import ROOT, load_config
from agent1_dr.orchestrator import run_all
from agent1_dr.utils import checksum, read_json
from agent1_dr.validation import dataset_pdf, validate


def test_fresh_pipeline(tmp_path):
    cfg = load_config(ROOT / "config/smoke.yaml")
    cfg.project.output_root = str(tmp_path)
    run_all(cfg)
    result = validate(cfg)
    assert result["status"] == "PASSED", [c for c in result["checks"] if not c["passed"]]
    assert dataset_pdf(cfg).is_file()
    decision = read_json(cfg.output / "selection/decisions.json")
    assert decision["sensitivity"]["variants"] and decision["ablation"]
    path = cfg.output / "pilot/method_runs/pca/774/result.json"
    digest, mtime = checksum(path), path.stat().st_mtime_ns
    run_all(cfg)
    assert checksum(path) == digest and path.stat().st_mtime_ns == mtime
    # Integrity failures must be detected rather than trusting a success flag.
    (cfg.output / "final/ids.npy").write_bytes(b"corrupted")
    assert validate(cfg, require_reports=False)["status"] == "FAILED"
