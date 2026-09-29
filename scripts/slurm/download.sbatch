#!/usr/bin/env bash
#SBATCH --job-name=agent1_download
#SBATCH --partition=general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/proj/yunligrp/users/ygao/BIOS774Agent/Agent1_DR/logs/%x_%j.log
set -euo pipefail
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=/proj/yunligrp/users/ygao/conda/envs/agent/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
cd /proj/yunligrp/users/ygao/BIOS774Agent/Agent1_DR
export PATH=/proj/yunligrp/users/ygao/conda/envs/agent/bin:$PATH
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export MPLCONFIGDIR="$PWD/.cache/matplotlib" NUMBA_CACHE_DIR="$PWD/.cache/numba"
CONFIG=${1:-config/pbmc3k.yaml}
DATASET=$(basename "$CONFIG" .yaml)
mkdir -p "logs/$DATASET/download"
exec > >(tee "logs/$DATASET/download/${SLURM_JOB_ID}.log") 2>&1
hostname
date -Is
printf 'Job=%s config=%s\n' "$SLURM_JOB_ID" "$CONFIG"
git rev-parse HEAD || true
python -c 'from agent1_dr.utils import versions; print(versions())'
python scripts/check_data.py
