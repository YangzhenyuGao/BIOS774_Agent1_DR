# Progress

## Phase 0 — 2026-09-22
- Read the entire supplied README; it is the only project file and requirement source. No AGENTS.md or official assignment/rubric found in semester workspace. Original README preserved.
- Host: longleaf-login2.its.unc.edu. Project: /proj/yunligrp/users/ygao/BIOS774Agent/Agent1_DR. Writable; project filesystem reports 2.3 PiB available (not a personal quota guarantee). No Git repository yet.
- Conda root requested is an environment/package storage root, not an installation. Existing group scripts use `module load anaconda/2024.02`; executable base /nas/longleaf/rhel9/apps/anaconda/2024.02, Conda 24.1.2. Requested agent environment absent; no existing environments will be replaced.
- Live SLURM inspection: general up, MaxTime=11 days, MaxMemPerNode=232448 MB. CPU baseline, 4 CPUs/16 GB, phase-specific walltime. Sandbox socket restriction resolved through approved cluster inspection.
- `module load r/4.4.0` succeeds. Libraries: /nas/longleaf/home/gaoyang/R/x86_64-pc-linux-gnu-library/4.4 and /nas/longleaf/rhel9/apps/r/4.4.0/lib64/R/library. R unnecessary for this implementation.
- Next: create Python 3.11 environment, implement pipeline, pass smoke and dry run before full pilots. No scientific results yet.

## Implementation phases 2–9 (initial code, not yet accepted)
- Created installable package, typed config/schema validation, two real loaders, explicit preprocessing plans, stratified common cohorts, ten real method wrappers, isolated workers and timeout logging.
- GPLVM follows the genuine Bayesian variational-latent/inducing-GP formulation, with loss and coordinate displacement recorded. Diffusion maps uses pydiffmap, separate from SpectralEmbedding.
- Implemented metric-based selection, shortlist-only tuning, conservative runtime-based final cohort cap, figures, evidence bundles, deterministic PDFs, optional extractive OpenAI reports, and artifact audits.
- Tests cover all ten methods on blobs/Swiss roll, fixed-seed geometry, GPLVM optimization, failure/timeout isolation, labels, selection, and a fresh synthetic end-to-end workflow.
- Implementation acceptance remains pending environment setup, tests, and data/pilot execution. No final benchmark has been submitted.

## Environment and implementation follow-up
- Conda foundation created successfully with Python 3.11 and compiled conda-forge libraries; installation of CPU PyTorch and remaining pip dependencies is in progress (logs/environment_setup.log). Shared-filesystem transaction and wheel extraction are slow but making progress; no environment was deleted/recreated.
- Source compilation passed (`python3 -m compileall -q src tests`). This is not the scientific smoke acceptance.
- Added numerical CSV/NPZ ingestion so a new user dataset can traverse the same public interface, including unlabeled data. Added label-invariance test on a fixed cohort.
- Initialized local Git repository on main; no commit/push and no license selected. Original README contract retained with appended operational commands.

- Initial full source lint now PASSED: `agent/bin/ruff check src tests`. Formatting/import cleanup applied before any benchmark.
- Environment compatibility diagnosis: anndata requires `pandas!=2.1.2,<3,>=2.1.0`; initial conda-forge solve chose pandas 3.0.6. Pip is replacing pandas only within the newly created agent environment. Human-maintained environment.yml now constrains pandas<3; final pip lock captures the actual tested version. Existing user environments are untouched.

## Environment isolation correction and validation jobs
- Initial small local checks: 6 configuration/metric/selection tests PASSED. Pip audit exposed unrelated TensorFlow/Keras user-site distributions, not packages intentionally installed for this project.
- Canceled pending jobs 2104040/2104041 before execution. Confirmed Python included ~/.local/lib/python3.11/site-packages by default. Disabled user site with PYTHONNOUSERSITE=1 in all execution/setup scripts, worker subprocesses and Conda activation variables. No user-site packages were modified.
- Isolated dependency audit identified only missing `termcolor` (fire dependency); repaired within the agent environment.
- Added scripts/export_environment.py to create portable exact version pins without conda build-host file:// URLs or user-site packages. Compiled foundation remains in environment.conda-explicit.txt.
- Resubmitted CPU smoke+dry run job 2104260 and real-data/cache validation job 2104261. Full pilots remain gated on smoke acceptance.

## First numerical/environment diagnostics
- Job 2104260 started on c151412, then found an import-order issue: Torch loaded the system libstdc++ before SciPy 1.17.1, which requires CXXABI_1.3.15. The newer runtime is already installed in agent/lib. Execution scripts and workers now explicitly prioritize that library directory; no system libraries changed.
- Tiny Diffusion Maps test diagnosed pydiffmap's BGH automatic epsilon producing non-finite sqrt coordinates on well-separated synthetic blobs. Changed the documented default to a median squared-distance bandwidth (epsilon=median(d²)/4) while retaining real pydiffmap diffusion maps; this is a numerical bandwidth choice, not an algorithm substitution. Retest pending.
- SLURM automatically routed general requests to spill; the user's association is capped at 1,000 CPUs. Existing unrelated jobs occupy almost all of that quota. Smoke had started; canceled only the still-pending data check 2104261 and resubmitted as 2104590 with one CPU. Future scripts request one CPU because all numerical workers use one thread. No unrelated jobs were modified.

- Isolated `pip check`: PASSED, no broken requirements.
- Tiny local suite completed: 3 PASSED, 1 FAILED (the original BGH Diffusion Maps blobs case); both GPLVM datasets passed actual optimization and fixed-seed reproducibility checks. Corrected bandwidth will be tested in the full smoke job.
- Data-check job 2104590 started on c141601; PBMC3k downloads, exact annotation alignment, cache reread, and 60-observation preprocessing PASSED. PathMNIST acquisition still running.
- Submitted corrected single-CPU smoke job 2104905. Original compute import-failure job 2104260 retained in logs for diagnosis.

## Data acceptance and corrected smoke
- Job 2104590 COMPLETED (2m17s; peak batch RSS ~1.59 GB). Both actual dataset loaders passed repeat/cache checks and 60-observation preprocessing. Evidence: logs/data_checks.json and data/raw/.
- Direct 45-point bandwidth diagnosis saved in logs/diffusion_bandwidth_diagnosis.json: BGH eigenvalues [3.12e-16, -3.59e-15] yielded non-finite coordinates; median-distance epsilon 2.4142936634 yielded negative generator eigenvalues and finite coordinates. This confirms the recorded numerical adjustment.
- Corrected smoke job 2104905 started on c0317. No full benchmark submitted yet.

- 2026-09-22T19:44:57.036195+00:00: synthetic inspect complete; job=2104905; outputs=/tmp/gaoyang/2104905/c0317.ll.unc.edu/pytest-of-gaoyang/pytest-0/test_fresh_pipeline0/synthetic; stage errors, if any, are preserved in method result files.

- 2026-09-22T19:47:39.789550+00:00: synthetic pilot complete; job=2104905; outputs=/tmp/gaoyang/2104905/c0317.ll.unc.edu/pytest-of-gaoyang/pytest-0/test_fresh_pipeline0/synthetic; stage errors, if any, are preserved in method result files.

- 2026-09-22T19:47:39.961349+00:00: synthetic select complete; job=2104905; outputs=/tmp/gaoyang/2104905/c0317.ll.unc.edu/pytest-of-gaoyang/pytest-0/test_fresh_pipeline0/synthetic; stage errors, if any, are preserved in method result files.

- 2026-09-22T19:52:18.927922+00:00: synthetic tune complete; job=2104905; outputs=/tmp/gaoyang/2104905/c0317.ll.unc.edu/pytest-of-gaoyang/pytest-0/test_fresh_pipeline0/synthetic; stage errors, if any, are preserved in method result files.

- Read-only inspection inside job 2104905 confirmed end-to-end tuning was progressing (18/19 successful fits; PCA, t-SNE, MDS, UMAP shortlisted). The initial fresh-output test uses the original four-candidate smoke configuration.
- Based on this measured orchestration overhead, reduced only config/smoke.yaml max_final_methods to 2 for the subsequent public-CLI dry run. All ten methods are still attempted in its pilot. Production candidate cap remains 4; no production cohorts/defaults were reduced.
- PDF environment-test page rendered with pdftoppm and visually inspected; text and page boundaries are correct.

- 2026-09-22T19:53:09.006085+00:00: synthetic final complete; job=2104905; outputs=/tmp/gaoyang/2104905/c0317.ll.unc.edu/pytest-of-gaoyang/pytest-0/test_fresh_pipeline0/synthetic; stage errors, if any, are preserved in method result files.

- 2026-09-22T19:53:16.717341+00:00: synthetic report complete; job=2104905; outputs=/tmp/gaoyang/2104905/c0317.ll.unc.edu/pytest-of-gaoyang/pytest-0/test_fresh_pipeline0/synthetic; stage errors, if any, are preserved in method result files.

- 2026-09-22T19:53:17.799533+00:00: synthetic inspect complete; job=2104905; outputs=/tmp/gaoyang/2104905/c0317.ll.unc.edu/pytest-of-gaoyang/pytest-0/test_fresh_pipeline0/synthetic; stage errors, if any, are preserved in method result files.

- 2026-09-22T19:53:17.869345+00:00: synthetic pilot complete; job=2104905; outputs=/tmp/gaoyang/2104905/c0317.ll.unc.edu/pytest-of-gaoyang/pytest-0/test_fresh_pipeline0/synthetic; stage errors, if any, are preserved in method result files.

- 2026-09-22T19:53:17.942131+00:00: synthetic select complete; job=2104905; outputs=/tmp/gaoyang/2104905/c0317.ll.unc.edu/pytest-of-gaoyang/pytest-0/test_fresh_pipeline0/synthetic; stage errors, if any, are preserved in method result files.

- 2026-09-22T19:53:18.014874+00:00: synthetic tune complete; job=2104905; outputs=/tmp/gaoyang/2104905/c0317.ll.unc.edu/pytest-of-gaoyang/pytest-0/test_fresh_pipeline0/synthetic; stage errors, if any, are preserved in method result files.

- 2026-09-22T19:53:18.099434+00:00: synthetic final complete; job=2104905; outputs=/tmp/gaoyang/2104905/c0317.ll.unc.edu/pytest-of-gaoyang/pytest-0/test_fresh_pipeline0/synthetic; stage errors, if any, are preserved in method result files.

- 2026-09-22T19:53:19.424562+00:00: synthetic report complete; job=2104905; outputs=/tmp/gaoyang/2104905/c0317.ll.unc.edu/pytest-of-gaoyang/pytest-0/test_fresh_pipeline0/synthetic; stage errors, if any, are preserved in method result files.

- Compiled-environment reconstruction dry run PASSED: `conda create --dry-run --offline -p /tmp/gaoyang/agent1_dr_lock_check --file environment.conda-explicit.txt`; log in logs/environment_reconstruction_dry_run.log. This validates the explicit cached package plan; a second full environment was not physically built. The pip lock supplies the required pandas 2.3.3 replacement over the initial compiled foundation.

- 2026-09-22T19:59:44.590222+00:00: synthetic inspect complete; job=2104905; outputs=/proj/yunligrp/users/ygao/BIOS774Agent/Agent1_DR/outputs/smoke/synthetic; stage errors, if any, are preserved in method result files.

## Synthetic acceptance and production queue
- In job 2104905, `pytest -q` PASSED: 36 tests, 1 visible PyTorch JIT deprecation warning, 882.17 seconds. This includes all ten genuine methods on blobs/Swiss roll, GPLVM optimization, fixed-seed geometry, isolated failures/timeouts, corruption recovery, metrics, selection, label invariance and fresh end-to-end/resume/report validation.
- `ruff check src tests` PASSED. All required imports, CPU GP calculation and one-page PDF rendering PASSED on compute node.
- Public CLI smoke run is still completing with the two-candidate smoke cap; the all-ten-method pilot is unchanged.
- Queued PBMC3k job 2107227 and PathMNIST job 2107228 with `--dependency=afterok:2104905 --time=02:00:00`, one CPU and 16 GB each. The script also requires a matching current-code smoke_acceptance.json before doing any full computation. No full benchmark can start if the smoke/CLI job fails.

- 2026-09-22T20:02:40.086504+00:00: synthetic pilot complete; job=2104905; outputs=/proj/yunligrp/users/ygao/BIOS774Agent/Agent1_DR/outputs/smoke/synthetic; stage errors, if any, are preserved in method result files.

- 2026-09-22T20:02:40.479532+00:00: synthetic select complete; job=2104905; outputs=/proj/yunligrp/users/ygao/BIOS774Agent/Agent1_DR/outputs/smoke/synthetic; stage errors, if any, are preserved in method result files.

- 2026-09-22T20:03:16.708871+00:00: synthetic tune complete; job=2104905; outputs=/proj/yunligrp/users/ygao/BIOS774Agent/Agent1_DR/outputs/smoke/synthetic; stage errors, if any, are preserved in method result files.

- 2026-09-22T20:03:26.235613+00:00: synthetic final complete; job=2104905; outputs=/proj/yunligrp/users/ygao/BIOS774Agent/Agent1_DR/outputs/smoke/synthetic; stage errors, if any, are preserved in method result files.

- 2026-09-22T20:03:33.545090+00:00: synthetic report complete; job=2104905; outputs=/proj/yunligrp/users/ygao/BIOS774Agent/Agent1_DR/outputs/smoke/synthetic; stage errors, if any, are preserved in method result files.

## Pilot dry-run acceptance
- Public CLI `run-all --config config/smoke.yaml --resume` and `validate` PASSED in job 2104905; logs/smoke_acceptance.json is present with the current code hash. Every one of the ten methods has a success status on the common synthetic pilot. Method warnings (including disconnected graph and clipped tiny-cohort parameters) remain visible.
- Production dependencies released only after this successful completion. The two dataset run-all jobs retain max_samples=1000, stochastic repeats=3, candidate cap=4 and GPLVM 300 optimization steps.
- Queued final report/audit job 2107412 with afterok dependency on both 2107227 and 2107228. It will create reports/report.pdf and run both submission-level validators after the dataset PDFs exist.

## User priority instruction — production waiting for CPU allocation
- User explicitly requested continued queueing because acrobat_he is more important and has higher priority. Do not change array 1921007 concurrency, priority, or running tasks. Its original ArrayTaskThrottle=100000 was only inspected and has NOT been modified.
- At inspection, the user's 1,000-CPU association limit was occupied by 125 existing eight-CPU jobs. Production jobs 2107227 (PBMC3k) and 2107228 (PathMNIST) remain pending with AssocGrpCpuLimit; report/audit job 2107412 depends on both succeeding.
- Source, environment, data acquisition, 36 tests, lint, all-ten-method synthetic pilot, public CLI end-to-end run, resume/corruption checks, PDF renderer and reconstruction dry run are completed. Real-data pilot/tuning/final evidence and the three required production PDFs are NOT yet generated. Do not call the whole project complete until those jobs finish and production outputs/PDF pages are audited.
- Smoke PDF inspected visually. Refined templates to repeat table headings across pages and omit long PCA ratio arrays from prose (full arrays remain in artifacts and variance plots). Refined PDF layout re-rendered successfully, seven pages for the synthetic example.
- Next action: retain queue order; monitor 2107227/2107228/2107412. If a job fails, diagnose its saved log and resume the relevant stage, preserving the other project's priority. After successful completion, visually inspect representative pages of all three production PDFs and refresh submission_manifest.json after final progress documentation.

- 2026-09-22T20:26:40.053030+00:00: pbmc3k inspect complete; job=2107227; outputs=/proj/yunligrp/users/ygao/BIOS774Agent/Agent1_DR/outputs/pbmc3k; stage errors, if any, are preserved in method result files.

- 2026-09-22T20:28:27.624630+00:00: pathmnist inspect complete; job=2107228; outputs=/proj/yunligrp/users/ygao/BIOS774Agent/Agent1_DR/outputs/pathmnist; stage errors, if any, are preserved in method result files.

- 2026-09-22T20:31:19.758660+00:00: pathmnist pilot complete; job=2107228; outputs=/proj/yunligrp/users/ygao/BIOS774Agent/Agent1_DR/outputs/pathmnist; stage errors, if any, are preserved in method result files.

- 2026-09-22T20:31:19.855100+00:00: pathmnist select complete; job=2107228; outputs=/proj/yunligrp/users/ygao/BIOS774Agent/Agent1_DR/outputs/pathmnist; stage errors, if any, are preserved in method result files.

- 2026-09-22T20:31:27.583337+00:00: pbmc3k pilot complete; job=2107227; outputs=/proj/yunligrp/users/ygao/BIOS774Agent/Agent1_DR/outputs/pbmc3k; stage errors, if any, are preserved in method result files.

- 2026-09-22T20:31:27.944584+00:00: pbmc3k select complete; job=2107227; outputs=/proj/yunligrp/users/ygao/BIOS774Agent/Agent1_DR/outputs/pbmc3k; stage errors, if any, are preserved in method result files.

- 2026-09-22T20:32:28.923040+00:00: pathmnist tune complete; job=2107228; outputs=/proj/yunligrp/users/ygao/BIOS774Agent/Agent1_DR/outputs/pathmnist; stage errors, if any, are preserved in method result files.

- 2026-09-22T20:33:16.581273+00:00: pathmnist final complete; job=2107228; outputs=/proj/yunligrp/users/ygao/BIOS774Agent/Agent1_DR/outputs/pathmnist; stage errors, if any, are preserved in method result files.

- 2026-09-22T20:33:23.700147+00:00: pathmnist report complete; job=2107228; outputs=/proj/yunligrp/users/ygao/BIOS774Agent/Agent1_DR/outputs/pathmnist; stage errors, if any, are preserved in method result files.

## Production recovery: annotation serialization
- Status inspection found 2107227 FAILED at tuning after all ten PBMC pilot methods succeeded. NumPy rejected the saved pandas-derived object label array with `allow_pickle=False`. Job 2107228 completed PathMNIST and produced generated_report_2.pdf; dependent report job 2107412 was cancelled without running.
- Fixed the shared cohort persistence boundary to convert labels and observation IDs to plain Unicode and explicitly disallow pickle for all saved cohort arrays. This also covers pandas CSV metadata. Added a regression test exercising object metadata through preparation, persistence and default-safe loading.
- Original dataset outputs and PathMNIST report preserved under outputs/archive/pre_metadata_fix and reports/archive/pre_metadata_fix. Code provenance invalidates previous stages, so both datasets will regenerate after current-code smoke acceptance. No unrelated SLURM jobs or priorities are changed.
- Recovery jobs submitted with normal queue priority: smoke/regression 2133668; PBMC3k 2133669 and PathMNIST 2133716 depend on its success; combined report/audit 2133784 depends on both dataset jobs succeeding. Initial smoke status PENDING (Priority).
- Focused regression validation PASSED: tests/test_data_loaders.py, 5 tests (101.13 s including imports), and ruff check src tests. Full compute-node smoke/regression job 2133668 is now RUNNING. Archived earlier PDF companions are under outputs/archive/pre_metadata_fix/reports, outside the active deliverable directory.

- 2026-09-23T00:07:53.367686+00:00: synthetic inspect complete; job=2133668; outputs=/tmp/gaoyang/2133668/c151416.ll.unc.edu/pytest-of-gaoyang/pytest-0/test_fresh_pipeline0/synthetic; stage errors, if any, are preserved in method result files.

- 2026-09-23T00:09:45.234071+00:00: synthetic pilot complete; job=2133668; outputs=/tmp/gaoyang/2133668/c151416.ll.unc.edu/pytest-of-gaoyang/pytest-0/test_fresh_pipeline0/synthetic; stage errors, if any, are preserved in method result files.

- 2026-09-23T00:09:45.301972+00:00: synthetic select complete; job=2133668; outputs=/tmp/gaoyang/2133668/c151416.ll.unc.edu/pytest-of-gaoyang/pytest-0/test_fresh_pipeline0/synthetic; stage errors, if any, are preserved in method result files.

- 2026-09-23T00:09:59.230290+00:00: synthetic tune complete; job=2133668; outputs=/tmp/gaoyang/2133668/c151416.ll.unc.edu/pytest-of-gaoyang/pytest-0/test_fresh_pipeline0/synthetic; stage errors, if any, are preserved in method result files.

- 2026-09-23T00:10:03.059204+00:00: synthetic final complete; job=2133668; outputs=/tmp/gaoyang/2133668/c151416.ll.unc.edu/pytest-of-gaoyang/pytest-0/test_fresh_pipeline0/synthetic; stage errors, if any, are preserved in method result files.

- 2026-09-23T00:10:06.455895+00:00: synthetic report complete; job=2133668; outputs=/tmp/gaoyang/2133668/c151416.ll.unc.edu/pytest-of-gaoyang/pytest-0/test_fresh_pipeline0/synthetic; stage errors, if any, are preserved in method result files.

- 2026-09-23T00:10:06.871752+00:00: synthetic inspect complete; job=2133668; outputs=/tmp/gaoyang/2133668/c151416.ll.unc.edu/pytest-of-gaoyang/pytest-0/test_fresh_pipeline0/synthetic; stage errors, if any, are preserved in method result files.

- 2026-09-23T00:10:06.887666+00:00: synthetic pilot complete; job=2133668; outputs=/tmp/gaoyang/2133668/c151416.ll.unc.edu/pytest-of-gaoyang/pytest-0/test_fresh_pipeline0/synthetic; stage errors, if any, are preserved in method result files.

- 2026-09-23T00:10:06.900526+00:00: synthetic select complete; job=2133668; outputs=/tmp/gaoyang/2133668/c151416.ll.unc.edu/pytest-of-gaoyang/pytest-0/test_fresh_pipeline0/synthetic; stage errors, if any, are preserved in method result files.

- 2026-09-23T00:10:06.914038+00:00: synthetic tune complete; job=2133668; outputs=/tmp/gaoyang/2133668/c151416.ll.unc.edu/pytest-of-gaoyang/pytest-0/test_fresh_pipeline0/synthetic; stage errors, if any, are preserved in method result files.

- 2026-09-23T00:10:06.927854+00:00: synthetic final complete; job=2133668; outputs=/tmp/gaoyang/2133668/c151416.ll.unc.edu/pytest-of-gaoyang/pytest-0/test_fresh_pipeline0/synthetic; stage errors, if any, are preserved in method result files.

- 2026-09-23T00:10:07.348198+00:00: synthetic report complete; job=2133668; outputs=/tmp/gaoyang/2133668/c151416.ll.unc.edu/pytest-of-gaoyang/pytest-0/test_fresh_pipeline0/synthetic; stage errors, if any, are preserved in method result files.

- 2026-09-23T00:13:16.463710+00:00: synthetic inspect complete; job=2133668; outputs=/proj/yunligrp/users/ygao/BIOS774Agent/Agent1_DR/outputs/smoke/synthetic; stage errors, if any, are preserved in method result files.

- 2026-09-23T00:14:50.342294+00:00: synthetic pilot complete; job=2133668; outputs=/proj/yunligrp/users/ygao/BIOS774Agent/Agent1_DR/outputs/smoke/synthetic; stage errors, if any, are preserved in method result files.

- 2026-09-23T00:14:50.484268+00:00: synthetic select complete; job=2133668; outputs=/proj/yunligrp/users/ygao/BIOS774Agent/Agent1_DR/outputs/smoke/synthetic; stage errors, if any, are preserved in method result files.

- 2026-09-23T00:15:08.146679+00:00: synthetic tune complete; job=2133668; outputs=/proj/yunligrp/users/ygao/BIOS774Agent/Agent1_DR/outputs/smoke/synthetic; stage errors, if any, are preserved in method result files.

- 2026-09-23T00:15:13.540748+00:00: synthetic final complete; job=2133668; outputs=/proj/yunligrp/users/ygao/BIOS774Agent/Agent1_DR/outputs/smoke/synthetic; stage errors, if any, are preserved in method result files.

- 2026-09-23T00:15:17.016340+00:00: synthetic report complete; job=2133668; outputs=/proj/yunligrp/users/ygao/BIOS774Agent/Agent1_DR/outputs/smoke/synthetic; stage errors, if any, are preserved in method result files.

- 2026-09-23T00:18:07.318712+00:00: pbmc3k inspect complete; job=2133669; outputs=/proj/yunligrp/users/ygao/BIOS774Agent/Agent1_DR/outputs/pbmc3k; stage errors, if any, are preserved in method result files.

- 2026-09-23T00:19:17.487026+00:00: pathmnist inspect complete; job=2133716; outputs=/proj/yunligrp/users/ygao/BIOS774Agent/Agent1_DR/outputs/pathmnist; stage errors, if any, are preserved in method result files.

- 2026-09-23T00:22:05.429847+00:00: pathmnist pilot complete; job=2133716; outputs=/proj/yunligrp/users/ygao/BIOS774Agent/Agent1_DR/outputs/pathmnist; stage errors, if any, are preserved in method result files.

- 2026-09-23T00:22:05.631194+00:00: pathmnist select complete; job=2133716; outputs=/proj/yunligrp/users/ygao/BIOS774Agent/Agent1_DR/outputs/pathmnist; stage errors, if any, are preserved in method result files.

- 2026-09-23T00:22:09.879611+00:00: pbmc3k pilot complete; job=2133669; outputs=/proj/yunligrp/users/ygao/BIOS774Agent/Agent1_DR/outputs/pbmc3k; stage errors, if any, are preserved in method result files.

- 2026-09-23T00:22:10.076382+00:00: pbmc3k select complete; job=2133669; outputs=/proj/yunligrp/users/ygao/BIOS774Agent/Agent1_DR/outputs/pbmc3k; stage errors, if any, are preserved in method result files.

- 2026-09-23T00:23:22.483654+00:00: pathmnist tune complete; job=2133716; outputs=/proj/yunligrp/users/ygao/BIOS774Agent/Agent1_DR/outputs/pathmnist; stage errors, if any, are preserved in method result files.

- 2026-09-23T00:23:42.252926+00:00: pbmc3k tune complete; job=2133669; outputs=/proj/yunligrp/users/ygao/BIOS774Agent/Agent1_DR/outputs/pbmc3k; stage errors, if any, are preserved in method result files.

- 2026-09-23T00:24:06.471623+00:00: pathmnist final complete; job=2133716; outputs=/proj/yunligrp/users/ygao/BIOS774Agent/Agent1_DR/outputs/pathmnist; stage errors, if any, are preserved in method result files.

- 2026-09-23T00:24:13.146997+00:00: pathmnist report complete; job=2133716; outputs=/proj/yunligrp/users/ygao/BIOS774Agent/Agent1_DR/outputs/pathmnist; stage errors, if any, are preserved in method result files.

- 2026-09-23T00:24:22.710234+00:00: pbmc3k final complete; job=2133669; outputs=/proj/yunligrp/users/ygao/BIOS774Agent/Agent1_DR/outputs/pbmc3k; stage errors, if any, are preserved in method result files.

- 2026-09-23T00:24:28.407787+00:00: pbmc3k report complete; job=2133669; outputs=/proj/yunligrp/users/ygao/BIOS774Agent/Agent1_DR/outputs/pbmc3k; stage errors, if any, are preserved in method result files.


## Final delivery audit — COMPLETED
- Recovery jobs all completed with exit code 0: smoke/regression 2133668 (10m04s); PBMC3k 2133669 (8m10s); PathMNIST 2133716 (7m55s); combined report/submission audit 2133784 (3m19s). No Agent1 jobs remain queued or running. Other projects were not modified.
- Current-code compute-node validation: 37 tests PASSED (321.97 s), one visible PyTorch JIT deprecation warning; ruff PASSED; all imports, CPU GP/PDF checks, fresh synthetic workflow, resume/corruption checks, public CLI smoke and validation PASSED. Smoke acceptance hash matches current source.
- Both real pilots attempted all ten genuine methods on a shared 1,000-observation, 50-component representation; all ten achieved success on each dataset. Both submission validators PASSED all 34 checks, with no listed analysis warnings. Earlier failures and logs remain archived.
- PBMC3k final cohort: all 2,638 post-QC cells. Retained/tuned/fitted PCA, t-SNE, Isomap and Laplacian Eigenmaps. PathMNIST final cohort: stratified 3,000-observation subset of the 89,996-image training split. Retained/tuned/fitted PCA, t-SNE, Laplacian Eigenmaps and LLE. This is not the complete PathMNIST dataset.
- Final PDFs open with text on every page: reports/generated_report_1.pdf (8 pages), reports/generated_report_2.pdf (8 pages), reports/report.pdf (4 pages, within the limit). Visually inspected pages 1 and 4 of both dataset reports and all four project-report pages: text, figures, legends and tables are readable without clipping. Rendered audit pages are in logs/delivery_*.png.
- Reports are deterministic evidence-based template outputs; no live LLM API generated them. Scientific wording is provided for user review. Timing-based Pareto eligibility can change across machines/reruns; current reports and decisions consistently use the recovered run's measured evidence. Original pre-fix outputs are preserved under outputs/archive/pre_metadata_fix.
- Git-ready files reviewed; data, outputs, caches, logs and credential files are ignored. No credentials or large raw/intermediate datasets are included in the submission manifest; no license was chosen and no commit/push/upload was performed. Exact environment reconstruction received the documented dry-run check, not a second physical environment build.
- Refreshed submission_manifest.json after this final documentation, including the small examples/ metrics and decision snapshots. All listed file sizes/checksums verify. Required implementation, executions, reports and audit are complete; there are no unresolved execution blockers.
