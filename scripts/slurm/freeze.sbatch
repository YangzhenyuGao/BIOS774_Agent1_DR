#!/usr/bin/env bash
#SBATCH --job-name=agent1_freeze
#SBATCH --partition=general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/proj/yunligrp/users/ygao/BIOS774Agent/Agent1_DR/logs/slurm/%x_%j.log
source /proj/yunligrp/users/ygao/BIOS774Agent/Agent1_DR/scripts/slurm/common.sh
start_log submission freeze
python -m agent1_dr validate --config config/pbmc3k.yaml
python -m agent1_dr validate --config config/pathmnist.yaml
python -m agent1_dr freeze
python -m agent1_dr examples
