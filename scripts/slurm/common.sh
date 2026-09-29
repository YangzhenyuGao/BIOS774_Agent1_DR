# Shared execution environment for every Agent 1 SLURM script (sourced, not submitted).
set -euo pipefail
PROJECT=/proj/yunligrp/users/ygao/BIOS774Agent/Agent1_DR
AGENT_ENV=/proj/yunligrp/users/ygao/conda/envs/agent
cd "$PROJECT"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="$AGENT_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$AGENT_ENV/bin:$PATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export MPLCONFIGDIR="$PROJECT/.cache/matplotlib" NUMBA_CACHE_DIR="$PROJECT/.cache/numba"

start_log() {  # start_log <dataset-or-scope> <stage>
  mkdir -p "logs/$1/$2"
  exec > >(tee "logs/$1/$2/${SLURM_JOB_ID}.log") 2>&1
  hostname
  date -Is
  printf 'Job=%s stage=%s scope=%s\n' "$SLURM_JOB_ID" "$2" "$1"
  git rev-parse HEAD 2>/dev/null || echo "no git commit"
  python -c 'from agent1_dr.utils import versions; print(versions())'
}

require_smoke() {  # full computation only after a passing smoke run of the current code
  python - <<'PY'
from agent1_dr.utils import read_json, code_hash
gate = read_json('logs/smoke_acceptance.json')
assert gate['passed'] and gate['code_hash'] == code_hash(), 'Pass current-code smoke and pilot dry run first'
PY
}
