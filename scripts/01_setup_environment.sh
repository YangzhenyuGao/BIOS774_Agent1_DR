#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1
cd /proj/yunligrp/users/ygao/BIOS774Agent/Agent1_DR
module load anaconda/2024.02
AGENT_ENV=/proj/yunligrp/users/ygao/conda/envs/agent
export CONDA_PKGS_DIRS=/proj/yunligrp/users/ygao/conda/pkgs
export PIP_CACHE_DIR="$PWD/.cache/pip"
if [[ ! -d "$AGENT_ENV" ]]; then
  conda create -y -p "$AGENT_ENV" --override-channels -c conda-forge python=3.11 pip numpy scipy 'pandas<3' scikit-learn matplotlib h5py pango
fi
"$AGENT_ENV/bin/python" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
"$AGENT_ENV/bin/python" -m pip install anndata scanpy umap-learn pydiffmap medmnist gpytorch seaborn pyyaml joblib tqdm psutil jinja2 weasyprint pypdf openai pytest pytest-cov ruff
"$AGENT_ENV/bin/python" -m pip install -e .
"$AGENT_ENV/bin/python" scripts/export_environment.py
conda list -p "$AGENT_ENV" --explicit > environment.conda-explicit.txt

conda env config vars set -p "$AGENT_ENV" PYTHONNOUSERSITE=1 LD_LIBRARY_PATH="$AGENT_ENV/lib"
