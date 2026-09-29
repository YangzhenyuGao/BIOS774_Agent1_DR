#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=/proj/yunligrp/users/ygao/conda/envs/agent/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
cd /proj/yunligrp/users/ygao/BIOS774Agent/Agent1_DR
PY=/proj/yunligrp/users/ygao/conda/envs/agent/bin/python
for DATASET in pbmc3k pathmnist; do
  "$PY" -m agent1_dr download --config "config/$DATASET.yaml"
done
