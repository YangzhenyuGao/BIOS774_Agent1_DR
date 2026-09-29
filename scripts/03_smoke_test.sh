#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=/proj/yunligrp/users/ygao/conda/envs/agent/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
cd /proj/yunligrp/users/ygao/BIOS774Agent/Agent1_DR
export PATH=/proj/yunligrp/users/ygao/conda/envs/agent/bin:$PATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export MPLCONFIGDIR="$PWD/.cache/matplotlib" NUMBA_CACHE_DIR="$PWD/.cache/numba"
python scripts/check_environment.py
python -m agent1_dr --help
pytest -q
ruff check src tests
python -m agent1_dr run-all --config config/smoke.yaml --resume
python -m agent1_dr validate --config config/smoke.yaml
python - <<'PY'
from agent1_dr.utils import write_json, code_hash, versions
write_json('logs/smoke_acceptance.json', dict(passed=True, code_hash=code_hash(), package_versions=versions()))
PY
