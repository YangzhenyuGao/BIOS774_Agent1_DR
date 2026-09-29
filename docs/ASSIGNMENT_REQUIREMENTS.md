# Requirement traceability

No separate official assignment document was present at preflight. The unchanged user-supplied README (Sections 1–19) is the implementation contract. If an official rubric is supplied later, it must be reconciled without replacing its source.

| README section | Requirement | Implementation / evidence |
|---|---|---|
| 5 | Read-only preflight; environment and resource inspection | scripts/00_preflight.sh; logs/preflight.log; PROGRESS.md |
| 6 | Exact agent environment, real dependencies, CPU GP and PDF checks | scripts/01_setup_environment.sh; scripts/check_environment.py; environment files |
| 7 | Typed configuration, schemas, resumable stages | config.py (validated weights, thresholds, budgets); schemas.py; utils.py (stage manifests); cli.py |
| 8–9 | Two real datasets with provenance; explicit label-free preprocessing inferred from the data | data/; inspector.py (profile); planner.py (profile-driven branch, reasons cite profile values); preprocessing.py; sampling.py |
| 10 | Ten genuine methods; failure, timeout and retry isolation; optimization tests | methods/; registry.py; worker.py (warm batch, fit-only timing, per-fit peak RSS); execution.py (per-fit timeouts, crash isolation, cache); tests/test_methods_smoke.py |
| 11 | Common cohort; all methods attempted; comparable metrics including stability | pilot.py; benchmark.py (seeds + subsample refits for every method); evaluation.py (T, R, G, S, collapse); pilot/figures |
| 12 | Transparent, tested selection with PCA baseline; label-free | selector.py (pre-registered rules, equivalence margins, ε-Pareto, sensitivity, ablation); tests/test_selector.py; selection/decisions.json |
| 13 | Shortlist-only tuning; final cohort justified from scaling evidence | tuning.py (one-SE-style rule); final.py (scaling probe, projections, binding constraint); final/size_plan.json |
| 14 | Figures; evidence-grounded dataset PDFs; project PDF of at most 4 pages | visualization.py; reporting.py; templates/; reports/ |
| 15 | Validation, tests, clean smoke run, no secrets, reproducibility | validation.py (recomputes decisions, tuning choices, size plan; checks for stale files); tests/; lock files; freeze.py; submission_manifest.json |
| 16 | Heavy work only through SLURM, with recorded resources | scripts/slurm/ (common.sh plus stage scripts); job-specific logs |
| 17–18 | Complete handoff of artifacts and limitations | PROGRESS.md; METHOD_NOTES.md; validation_results.json; outputs/FROZEN_MANIFEST.json |

This table maps requirements to where they are implemented. Acceptance status comes from actual executions recorded in PROGRESS.md, not from the fact that a file exists.
