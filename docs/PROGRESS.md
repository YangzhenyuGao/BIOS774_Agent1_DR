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
3. **Profile-driven planner** (planner.py): bit-identical pilot inputs, IDs and labels versus v1 for both datasets, verified with a v1 worktree on one node.
4. **Warm batch worker** with fit-only timing, per-fit peak RSS, per-fit timeouts and crash isolation (worker.py, execution.py).
5. **Method fixes**: MDS classical initialization with a convergence flag; GPLVM true epochs (75 = v1's 300 steps at n = 1,000); diffusion Markov gap and Laplacian spectrum diagnostics; the README's full UMAP grid; LLE reg option.
6. **Pre-registered selection** (selector.py): equivalence margins, ε-Pareto, severe diagnostics, sensitivity and ablation. Tuning keeps the default unless the gain exceeds the margin. Final size comes from a scaling probe (final.py).
7. **Reports, validation, freeze**: reports/figures rebuilt; validation now recomputes decisions, tuning choices and the size plan, and checks for stale files; `freeze` locks the evidence before the project report.
8. **Tests**: 49 passing (26 fast + 23 slow; job 2834689 for the slow set); ruff clean.
9. **Cleanup**: v1 outputs, reports and logs moved to `archive/v1` folders; caches removed (the tree went from 768 MB to 359 MB).

## v2 production (2026-09-28/29, commit 766f1d1, partition `interact`)
- PBMC3k: job 2836846 on c0316, 15 min 43 s. PathMNIST: job 2836847 on c0402, 38 min 47 s. The two ran in parallel with 1 CPU and 32 GB each. Both passed the dataset validator (26 checks each).
- **PBMC3k**:
  - Branch: count matrix.
  - Shortlist: PCA, MDS, t-SNE. MDS had the best pilot quality.
  - Final cohort: all 2,638 post-QC cells.
  - Final T/R/G: PCA 0.825/0.051/0.808; MDS 0.848/0.064/0.842; t-SNE 0.897/0.208/0.654.
  - Diffusion maps ineligible: collapsed layout, Markov gap 9e-8.
- **PathMNIST**:
  - Branch: bounded intensity.
  - Shortlist: PCA, MDS, t-SNE, UMAP.
  - Final cohort: 12,500 of 89,996 training images, declared by the scaling rule. MDS was the binding method; at 15,000 it would exceed the timeout, time budget and memory limit.
  - The final MDS fit took 1,038 s against a point projection of 757 s; the 2x safety factor absorbed the difference.
  - Final T/R/G: MDS 0.850/0.073/0.955; t-SNE 0.900/0.186/0.780.
  - All label silhouettes are negative.
- Tuning kept every default. PathMNIST t-SNE perplexity 50 gained 0.0028, within the 0.005 margin.
- Robustness across 10 rule variants: PCA, MDS and t-SNE are retained in at least 80% on both datasets. UMAP is retained in 80% on PathMNIST and 20% on PBMC3k.
- Only the report template changed after the runs (dataset title and QC wording), so no analysis stage needed recomputing.
- **Frozen** 2026-09-29T02:05:21 UTC: 1,751 files checksummed in `outputs/FROZEN_MANIFEST.json`; outputs and both dataset PDFs set read-only.
- `reports/report.pdf` (4 main-body pages + references/appendix) was rendered from the verified frozen evidence. `submission_manifest.json` was refreshed.
- The v1 audit counterfactuals are saved in `examples/v1/offline_rescore.json`, including the six PathMNIST shortlists produced by ±0.2 s runtime jitter.

## Handoff notes
- The project-report narrative is `templates/project_report.html.j2`; its numbers are template variables read from the frozen evidence. Edit the template or CSS and re-run `python -m agent1_dr project-report` on a compute node. Editing any `src/` file changes the code hash and makes the frozen stages fail validation.
- Deferred on purpose so the frozen evidence would not need recomputing: larger fonts in the pilot metric panel, and splitting the provenance hash into analysis and report layers (staged, not applied).
- The author line ("Y. Gao") and the AI-assistance sentence in the report header need the author's confirmation.
- Git: branch `v2-evidence-selection` (baseline c9e180b, v2 766f1d1, deliverables commit next). `main` has no commits. Nothing has been pushed; no license has been selected.

## Public repository and report link (2026-09-28)
- At the user's request, the repository is public at https://github.com/YangzhenyuGao/BIOS774_Agent1_DR; anonymous API access was confirmed. The initial upload preserved the three existing commits and both main/v2-evidence-selection branches. Data, full outputs, logs and caches remain excluded from Git.
- Added a prominent black, bold, clickable repository link before the project-report title in templates/project_report.html.j2. No analysis source or frozen dataset evidence changed.
- Initial general/spill report job 2846458 was cancelled while pending. With explicit user authorization, rendering and built-in frozen-evidence/submission checks moved to interact job 2847015 on c0306 (1 CPU, 4 GB). The user also requested direct links to both dataset PDFs; the opening box now contains two sentences linking to the public repository, PBMC3k report and PathMNIST report. Rendering and built-in validation completed successfully (exit code 0). The refreshed PDF was visually inspected: the first content is the black, bold link box, all three first-page link annotations match the intended URLs, both dataset-report URLs return HTTP 200 anonymously, and the document remains 4 main-body pages plus 1 references/appendix page. The report, HTML source and template are prepared for publication on the public main branch.
