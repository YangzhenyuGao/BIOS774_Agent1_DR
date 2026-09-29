export PYTHONNOUSERSITE := 1
PY := /proj/yunligrp/users/ygao/conda/envs/agent/bin/python
CONFIG ?= config/pbmc3k.yaml
.PHONY: setup test pilot run validate reports
setup:
	bash scripts/01_setup_environment.sh
test:
	$(PY) -m pytest -q
	$(PY) -m ruff check src tests
pilot:
	sbatch scripts/slurm/pilot.sbatch $(CONFIG)
run:
	sbatch scripts/slurm/run_all.sbatch $(CONFIG)
validate:
	$(PY) -m agent1_dr validate --config $(CONFIG) --all-reports
reports:
	$(PY) -m agent1_dr project-report
