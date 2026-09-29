# Requirement traceability

No separate official assignment document was present at preflight. The unchanged user-supplied README is the implementation contract. If an official rubric is later supplied it must be reconciled without replacing its source.

| README section | Requirement | Implementation/evidence |
|---|---|---|
| 5 | Read-only preflight, environment/resource inspection | scripts/00_preflight.sh; logs/preflight.log; PROGRESS.md |
| 6 | Exact agent environment, real dependencies, CPU GP/PDF checks | scripts/01_setup_environment.sh; scripts/check_environment.py; environment files |
| 7 | Typed config, schemas, resumable stages | config.py; schemas.py; utils.py; CLI |
| 8–9 | Two real datasets, provenance, explicit label-free preprocessing | data/; inspector.py; preprocessing.py; sampling.py |
| 10 | Ten genuine methods, failures/timeouts and optimization tests | methods/; registry.py; worker.py; execution.py; tests/test_methods_smoke.py |
| 11 | Common cohort, all methods attempted, comparable metrics | pilot.py; evaluation.py; pilot manifests and tables |
| 12 | Evidence-backed PCA-plus-shortlist selection | selector.py; tests/test_selector.py; decisions.json |
| 13 | Shortlist tuning; justified final cohorts | tuning.py; tuning_results.csv; final cohort manifests |
| 14 | Figures, evidence-grounded dataset PDFs, <=4-page project PDF | visualization.py; reporting.py; templates/; reports/ |
| 15 | Validation, tests, clean smoke, no secrets, reproducibility | validation.py; tests/; lock files; submission_manifest.json |
| 16 | Heavy work only through SLURM with recorded resources | scripts/slurm/; job-specific logs |
| 17–18 | Complete handoff of artifacts and limitations | PROGRESS.md; final validation_results.json |

This table maps requirements to implementation locations; acceptance status is recorded from actual executions in PROGRESS.md, not assumed from file existence.
