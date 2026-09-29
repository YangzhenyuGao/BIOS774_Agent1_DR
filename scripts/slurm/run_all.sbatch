#!/usr/bin/env bash
#SBATCH --job-name=agent1_run_all
#SBATCH --partition=general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/proj/yunligrp/users/ygao/BIOS774Agent/Agent1_DR/logs/slurm/%x_%j.log
source /proj/yunligrp/users/ygao/BIOS774Agent/Agent1_DR/scripts/slurm/common.sh
CONFIG=${1:-config/pbmc3k.yaml}
start_log "$(basename "$CONFIG" .yaml)" run-all
require_smoke
python -m agent1_dr run-all --config "$CONFIG" --resume
