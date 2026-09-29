#!/usr/bin/env bash
# Read-only inspection except its own log (log parent must exist).
set -uo pipefail
cd /proj/yunligrp/users/ygao/BIOS774Agent/Agent1_DR
exec > >(tee logs/preflight.log) 2>&1
date -Is
hostname
pwd
ls -la
df -h .
git status --short || true
ls -ld /proj/yunligrp/users/ygao/conda{,/envs/agent} || true
module load anaconda/2024.02
conda --version
conda info --base
timeout 20 sinfo -o '%P %a %l %D %c %m %G'
timeout 20 scontrol show partition general
module load r/4.4.0
Rscript -e '.libPaths()'
