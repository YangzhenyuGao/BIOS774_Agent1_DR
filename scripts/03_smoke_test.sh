#!/usr/bin/env bash
# Compute-node acceptance gate: environment, tests, lint, and a fresh synthetic end-to-end run.
set -euo pipefail
source /proj/yunligrp/users/ygao/BIOS774Agent/Agent1_DR/scripts/slurm/common.sh
python scripts/check_environment.py
python -m agent1_dr --help > /dev/null
pytest -q -p no:cacheprovider
ruff check src tests scripts/analysis
python -m agent1_dr run-all --config config/smoke.yaml --resume
python -m agent1_dr validate --config config/smoke.yaml > /dev/null
python - <<'PY'
from agent1_dr.utils import write_json, code_hash, versions
write_json('logs/smoke_acceptance.json', dict(passed=True, code_hash=code_hash(), package_versions=versions()))
PY
echo "SMOKE ACCEPTANCE PASSED"
