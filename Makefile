export PYTHONNOUSERSITE := 1
PY := /proj/yunligrp/users/ygao/conda/envs/agent/bin/python
CONFIG ?= config/pbmc3k.yaml
.PHONY: setup test smoke run freeze report validate
setup:
	bash scripts/01_setup_environment.sh
test:
	$(PY) -m pytest -q
	$(PY) -m ruff check src tests scripts/analysis
smoke:
	sbatch scripts/slurm/smoke.sbatch
run:
	sbatch scripts/slurm/run_all.sbatch $(CONFIG)
freeze:
	sbatch scripts/slurm/freeze.sbatch
report:
	sbatch scripts/slurm/project_report.sbatch
validate:
	$(PY) -m agent1_dr validate --config $(CONFIG) --all-reports
