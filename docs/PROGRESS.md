# Progress and handoff

Concise record of what was done, with job IDs and evidence locations. The full v1 log, including 57 auto-appended stage lines, is archived at `logs/archive/v1/PROGRESS_v1_full.md` (not in Git). From v2 on, machine-readable stage history is written to `outputs/<dataset>/stage_history.jsonl` rather than to this file.

## v1 (2026-09-22) — original implementation (commit c9e180b, baseline snapshot)
- Preflight on longleaf-login2: project writable, Conda 24.1.2 via `module load anaconda/2024.02`, agent environment created at `/proj/yunligrp/users/ygao/conda/envs/agent` (Python 3.11). The user site is disabled and the environment's `lib` directory is placed first on `LD_LIBRARY_PATH`, because SciPy needs a newer CXXABI than the system libstdc++. R is not needed.
- Ten genuine method wrappers, isolated workers, typed config, deterministic reports. Diffusion-map bandwidth set to median(d²)/4 after pydiffmap's BGH estimate gave non-finite coordinates on a 45-point test.
- Production runs 2133669 (PBMC3k) and 2133716 (PathMNIST), report job 2133784. These followed an earlier failure (2107227) caused by object-dtype label arrays.
- v1 shortlists:
  - PBMC: PCA, t-SNE, Isomap, Laplacian (final n = 2,638);
  - PathMNIST: PCA, t-SNE, Laplacian, LLE (final n = 3,000).
- Archived evidence: `outputs/archive/v1/`; small snapshot: `examples/v1/`.

## Audit (2026-09-28)
Read-only audit of v1. Findings (details in `docs/METHOD_NOTES.md`, "Why v2"):
- **Runtime noise decided the shortlist.** Runtimes were dominated by imports, and a strict Pareto rule let sub-second differences decide finalists. Two runs on identical input, with qualities identical to 6 decimals, retained different fourth methods.
- **Seed stability was not comparable across methods.** t-SNE with PCA initialization gave bit-identical seeds, while deterministic methods were set to 1.
- **Undetected failures.** MDS on PBMC did not converge; diffusion maps on PBMC were near-disconnected; the final PBMC Laplacian layout was collapsed.
- **Preprocessing branched on the dataset name.**
- **Only local metrics were used.**
- **The final-size cap was always binding.**
- **Report presentation issues.** The project PDF printed a draft notice, and example figures were chosen alphabetically.

## v2 implementation (2026-09-28, branch v2-evidence-selection)
Changes, each validated separately:
1. **Metrics** (evaluation.py): chunked shared reference neighborhoods; T identical to scikit-learn and R identical to v1 (difference 0, tested); new global structure G, subsample stability S and collapse ratio. Simulation checks passed before any real-data use.
2. **Offline re-scoring of the v1 embeddings** (scripts/analysis/rescore_v1.py → examples/v1/offline_rescore.json): reproduces both delivered v1 shortlists exactly, then isolates each rule's effect.
3. **Profile-driven planner** (planner.py): bit-identical pilot inputs, IDs and labels versus v1 for both datasets, verified with a v1 worktree on one node (srun job 2831...; recorded in this session).
4. **Warm batch worker** with fit-only timing, per-fit peak RSS, per-fit timeouts and crash isolation (worker.py, execution.py).
5. **Method fixes**: MDS classical initialization with a convergence flag; GPLVM true epochs (75 = v1's 300 steps at n = 1,000); diffusion Markov gap and Laplacian spectrum diagnostics; the README's full UMAP grid; LLE reg option.
6. **Pre-registered selection** (selector.py): equivalence margins, ε-Pareto, severe diagnostics, sensitivity and ablation. Tuning keeps the default unless the gain exceeds the margin. Final size comes from a scaling probe (final.py).
7. **Reports, validation, freeze**: reports/figures rebuilt; validation now recomputes decisions, tuning choices and the size plan, and checks for stale files; `freeze` locks the evidence before the project report.
8. **Tests**: 49 passing (26 fast + 23 slow; job 2834689 for the slow set); ruff clean.
9. **Cleanup**: v1 outputs, reports and logs moved to `archive/v1` folders; caches removed (the tree went from 768 MB to 359 MB).

## v2 production (pending)
