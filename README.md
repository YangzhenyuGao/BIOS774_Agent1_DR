# BIOS 774 Agent 1: Evidence-Driven Dimensionality Reduction Agent

This repository contains **Agent 1** for BIOS 774: a lightweight, reproducible statistical agent that inspects a dataset, plans preprocessing, pilots every required dimensionality-reduction method, quantitatively narrows the candidate set, tunes the shortlisted methods, runs the final analysis, verifies its outputs, and generates an evidence-grounded report.

The central design principle is:

> **Support all methods -> pilot all methods -> evaluate empirically -> shortlist candidates -> tune -> run final analysis -> verify -> report.**

The agent must not choose methods only because they are popular or because a language model recommends them. Prior statistical knowledge may determine how a method is piloted safely, but advancement to the final stage must be supported by dataset-specific evidence.

---

## 1. Instructions to Codex

Read this entire README before changing files or launching jobs. Treat it as the implementation contract unless a later, explicit user instruction or the official assignment document overrides it.

### Working locations

- Semester workspace: `/proj/yunligrp/users/ygao/BIOS774Agent`
- This project: `/proj/yunligrp/users/ygao/BIOS774Agent/Agent1_DR`
- Future GitHub repository name: `BIOS774_Agent1_DR`
- Conda installation root: `/proj/yunligrp/users/ygao/conda`
- Required Python environment: `/proj/yunligrp/users/ygao/conda/envs/agent`
- R, if genuinely needed: `module load r/4.4.0`

Do not develop this project in another permanent location. Temporary scratch space may be used for disposable intermediate files, but source code, configuration, logs needed for reproducibility, and final deliverables belong in the project directory.

### Required working behavior

1. Begin with the preflight checks in Section 5. Inspect the existing directory and preserve any user-created files.
2. Inspect `/proj/yunligrp/users/ygao/conda` and existing job scripts before deciding how Conda is initialized on Longleaf. Follow the established local structure rather than guessing an activation command.
3. If the `agent` environment already exists, **do not delete or recreate it**. Validate it, install missing requirements into it, and document changes. If incompatible packages are already present, diagnose the conflict before modifying shared dependencies.
4. When a required Python or R package/tool is missing, install the actual required package promptly. Do not silently omit a method, replace it with a weaker method, or relabel an approximation as the requested method.
5. Heavy computation must run through SLURM, not on a login node. Use the login node only for file inspection, environment setup, small unit tests, and job submission/status checks.
6. Before hard-coding a partition, GPU type, memory limit, or time limit, inspect the currently available Longleaf resources with the appropriate cluster commands and follow any existing Yun Li group conventions.
7. Never commit secrets, API keys, raw credentials, large downloaded datasets, caches, or generated binary intermediates to Git.
8. Do not push to GitHub unless the user explicitly asks. Make the repository Git-ready and reproducible locally.
9. Maintain `docs/PROGRESS.md`. After every phase, record:
   - what was completed;
   - commands or SLURM job IDs used;
   - output locations;
   - tests that passed or failed;
   - unresolved issues and the next action.
10. Work phase by phase. Do not launch the full benchmark before the smoke tests and the pilot dry run pass.
11. Ask the user only when blocked by an external requirement such as unavailable credentials, an ambiguous official assignment requirement, unavailable data access, or a destructive change. Routine package installation and debugging are authorized within the paths above.

### Non-negotiable scientific rules

- All ten required methods must have real, callable implementations in the registry.
- All ten must be **attempted** in the pilot benchmark on the same dataset-specific pilot cohort and input representation.
- A failure, timeout, or numerical instability is a valid pilot outcome, but it must be logged; it must never disappear silently.
- Method selection must use quantitative pilot evidence. PCA is retained as a linear baseline even if it is not among the highest-scoring methods.
- Ground-truth class/cell-type labels must not influence preprocessing, embedding fitting, or primary method selection by default. Label-based metrics are secondary diagnostics unless the configuration explicitly states otherwise.
- Reports must distinguish prior suitability, empirical evidence, and the final decision.
- Do not state that an LLM wrote a report unless an LLM API was actually used.

---

## 2. Project Goal and Scope

The agent receives a new numerical dataset and performs the following workflow:

1. Inspect the dataset and infer a data profile.
2. Select and execute an appropriate preprocessing plan.
3. Construct a standardized pilot cohort and analysis representation.
4. Attempt every supported dimensionality-reduction method.
5. Measure embedding quality, runtime, stability, and execution status.
6. Select a small, defensible candidate set from the pilot evidence.
7. Perform a small hyperparameter search only for shortlisted methods.
8. Run the selected methods on the final analysis cohort.
9. Produce figures, tables, a machine-readable decision log, and a generated analysis report.
10. Verify output completeness and reproducibility.

The implementation should remain lightweight: one orchestrating agent, modular deterministic tools, and an optional LLM reporting layer. Do not introduce LangChain, a multi-agent framework, a web application, cloud deployment, or a database for this assignment.

### Demonstration datasets

The required demonstrations are intentionally heterogeneous:

1. **PBMC 3K**: sparse, high-dimensional single-cell RNA-seq count data.
2. **PathMNIST**: dense histopathology image data with class labels.

This pairing is intended to show that the agent adapts its preprocessing and computational plan to the input data rather than replaying one fixed notebook.

---

## 3. Required Method Registry

The registry must expose a common interface for the following ten methods:

| ID | Method | Required implementation |
|---|---|---|
| `pca` | Principal Component Analysis | `sklearn.decomposition.PCA` |
| `kernel_pca` | Kernel PCA | `sklearn.decomposition.KernelPCA` |
| `mds` | Multidimensional Scaling | `sklearn.manifold.MDS` |
| `isomap` | Isomap | `sklearn.manifold.Isomap` |
| `lle` | Locally Linear Embedding | `sklearn.manifold.LocallyLinearEmbedding` |
| `laplacian` | Laplacian Eigenmaps | `sklearn.manifold.SpectralEmbedding` |
| `diffusion_maps` | Diffusion Maps | Prefer `pydiffmap`; an exact, documented diffusion-map implementation is acceptable if compatibility requires it |
| `gplvm` | Gaussian Process Latent Variable Model | A genuine GPyTorch Bayesian GPLVM implementation |
| `tsne` | t-distributed Stochastic Neighbor Embedding | `sklearn.manifold.TSNE` |
| `umap` | Uniform Manifold Approximation and Projection | `umap-learn` |

Important distinctions:

- `laplacian` and `diffusion_maps` are separate methods and must not call the same wrapper under different names.
- `gplvm` must optimize a Gaussian-process latent-variable model. A PCA initialization is allowed, but returning PCA coordinates is not a GPLVM implementation.
- If a preferred package has a genuine compatibility problem, install a compatible release or use another exact implementation and document the reason, version, and validation. Do not substitute a conceptually different method.

Every method wrapper must return a standard result object containing at least:

```text
method_id
embedding
parameters
seed
status                 # success, failed, or timeout
runtime_seconds
peak_memory_mb         # when measurable
warnings
error_type
error_message
retry_count
package_versions
```

---

## 4. Target Repository Structure

Codex should create this structure incrementally; do not generate empty complexity that is never used.

```text
Agent1_DR/
├── README.md
├── LICENSE                         # only if the user selects a license
├── .gitignore
├── pyproject.toml
├── environment.yml
├── environment.lock.txt            # generated after environment validation
├── Makefile                         # short reproducible entry points
├── config/
│   ├── base.yaml
│   ├── pbmc3k.yaml
│   └── pathmnist.yaml
├── docs/
│   ├── PROGRESS.md
│   ├── METHOD_NOTES.md
│   └── DATA_PROVENANCE.md
├── scripts/
│   ├── 00_preflight.sh
│   ├── 01_setup_environment.sh
│   ├── 02_download_data.sh
│   ├── 03_smoke_test.sh
│   └── slurm/
│       ├── pilot.sbatch
│       ├── tune.sbatch
│       ├── final.sbatch
│       └── reports.sbatch
├── src/
│   └── agent1_dr/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py
│       ├── orchestrator.py
│       ├── config.py
│       ├── schemas.py
│       ├── inspector.py
│       ├── preprocessing.py
│       ├── sampling.py
│       ├── registry.py
│       ├── pilot.py
│       ├── selector.py
│       ├── tuning.py
│       ├── evaluation.py
│       ├── visualization.py
│       ├── reporting.py
│       ├── validation.py
│       ├── data/
│       │   ├── pbmc3k.py
│       │   └── pathmnist.py
│       └── methods/
│           ├── base.py
│           ├── pca.py
│           ├── kernel_pca.py
│           ├── mds.py
│           ├── isomap.py
│           ├── lle.py
│           ├── laplacian.py
│           ├── diffusion_maps.py
│           ├── gplvm.py
│           ├── tsne.py
│           └── umap.py
├── templates/
│   ├── generated_report.html.j2
│   ├── project_report.html.j2
│   └── report.css
├── tests/
│   ├── test_registry.py
│   ├── test_methods_smoke.py
│   ├── test_metrics.py
│   ├── test_selector.py
│   ├── test_data_loaders.py
│   └── test_end_to_end_smoke.py
├── data/
│   ├── raw/                         # ignored by Git
│   ├── interim/                     # ignored by Git
│   └── processed/                   # ignored by Git
├── outputs/                         # ignored except small example/manifest files
│   ├── pbmc3k/
│   └── pathmnist/
└── reports/
    ├── generated_report_1.pdf
    ├── generated_report_2.pdf
    └── report.pdf                   # final project report, <=4 pages excluding references/appendix
```

Use a proper installable `src/` package. The expected command-line interface is:

```bash
python -m agent1_dr inspect --config config/pbmc3k.yaml
python -m agent1_dr pilot --config config/pbmc3k.yaml
python -m agent1_dr select --config config/pbmc3k.yaml
python -m agent1_dr tune --config config/pbmc3k.yaml
python -m agent1_dr final --config config/pbmc3k.yaml
python -m agent1_dr report --config config/pbmc3k.yaml
python -m agent1_dr validate --config config/pbmc3k.yaml
python -m agent1_dr run-all --config config/pbmc3k.yaml --resume
```

Each stage must be resumable. Existing valid stage outputs should not be recomputed unless `--force` is supplied.

---

## 5. Phase 0: Preflight and Requirement Capture

Before installing packages or writing the implementation:

1. Confirm the current hostname, project path, storage availability, and permissions.
2. List existing files in `Agent1_DR` and check Git status if it is already a repository.
3. Search the project directory for the official assignment prompt, rubric, starter code, or prior user notes. If found, read them and create `docs/ASSIGNMENT_REQUIREMENTS.md` containing a traceable checklist. Do not overwrite the official source document.
4. Inspect the Conda root and identify the working Conda/Mamba executable and initialization convention used in existing Yun Li group scripts.
5. Check whether `/proj/yunligrp/users/ygao/conda/envs/agent` exists.
6. Inspect cluster partitions and limits. Do not assume a partition name from another cluster or an old job script.
7. Check whether `module load r/4.4.0` succeeds and record `.libPaths()`; do not install R packages unless R becomes part of an implemented workflow.
8. Create `docs/PROGRESS.md` and record the preflight results without exposing secrets.

The preflight script must be read-only except for writing its own log. It must never remove or replace an environment.

**Phase 0 acceptance criteria**

- Project and environment paths are confirmed.
- Existing work is preserved.
- Official requirements are captured if available.
- Cluster execution assumptions are based on current inspection.

---

## 6. Phase 1: Build and Validate the Python Environment

Use Python 3.11 unless current Longleaf compatibility or an existing `agent` environment requires another supported version. Create or update the environment at the exact requested path:

```text
/proj/yunligrp/users/ygao/conda/envs/agent
```

The environment should include the real packages needed for the design, including:

```text
numpy
pandas
scipy
scikit-learn
anndata
scanpy
umap-learn
pydiffmap
medmnist
torch
gpytorch
matplotlib
seaborn
pyyaml
joblib
tqdm
psutil
jinja2
weasyprint
pypdf
openai
pytest
pytest-cov
ruff
```

Also install any package genuinely required by the final implementation. Prefer Conda/conda-forge for compiled scientific dependencies and use pip inside the same environment when a package is unavailable or better maintained there. Do not use `pip --user` for this project.

After installation:

1. Import every core dependency.
2. Print and save core package versions.
3. Confirm PyTorch and GPyTorch can run a small CPU calculation.
4. Confirm WeasyPrint can render a one-page test PDF.
5. Confirm the project installs in editable mode.
6. Save a human-maintained `environment.yml` and an exact `environment.lock.txt` or equivalent explicit package export.

Never place API keys in `environment.yml`, shell scripts, configs, logs, or Git. LLM credentials, if later used, must come from environment variables such as `OPENAI_API_KEY`.

**Phase 1 acceptance criteria**

- The `agent` environment activates or runs reliably on a compute node.
- All required imports pass.
- A real GPyTorch model and PDF renderer pass smoke tests.
- The environment can be reconstructed from committed specifications.

---

## 7. Phase 2: Configuration, Schemas, and Reproducible Skeleton

Implement configuration-driven behavior before dataset-specific analysis. `config/base.yaml` should define at least:

```yaml
project:
  seed: 774
  output_root: outputs

pilot:
  max_samples: 1000
  n_repeats_stochastic: 3
  timeout_minutes_per_run: 30
  n_neighbors_eval: 15

selection:
  retain_pca_baseline: true
  max_final_methods: 4
  quality_relative_to_best: 0.90
  use_labels_for_selection: false

reporting:
  mode: deterministic  # deterministic or openai
  fail_if_llm_requested_but_unavailable: true
```

Exact defaults may be revised after a small dry run, but changes must be documented. Use typed dataclasses or Pydantic-style validation so impossible settings fail early with clear messages.

Define schemas for:

- dataset profile;
- preprocessing plan;
- method run result;
- evaluation result;
- selection decision;
- stage manifest;
- report evidence bundle.

Set random seeds for Python, NumPy, scikit-learn-compatible methods, UMAP, and PyTorch. Record software versions, configuration hashes, and the Git commit hash when available in each run manifest.

**Phase 2 acceptance criteria**

- `python -m agent1_dr --help` works.
- Config validation tests pass.
- A synthetic dataset can traverse an empty/smoke pipeline without dataset-specific code.

---

## 8. Phase 3: Data Acquisition and Provenance

### PBMC 3K

Use Scanpy/10x-maintained PBMC 3K data. The preferred design is:

- raw count matrix for preprocessing and embeddings;
- processed PBMC 3K annotations aligned by cell barcode for optional diagnostic cell-type labels;
- no use of cell-type labels during fitting or primary selection.

Verify that barcodes align before transferring annotations. If raw and processed objects cannot be aligned safely, stop label transfer and document the limitation rather than guessing.

Expected preprocessing candidate plan:

1. basic cell/gene quality checks;
2. filter only by documented, configurable criteria;
3. library-size normalization;
4. `log1p` transformation;
5. highly variable gene selection (target approximately 2,000 genes, subject to the data);
6. scaling/clipping as appropriate;
7. a 50-component PCA representation used as the common input for computationally intensive nonlinear methods.

The final displayed PCA embedding remains a two-dimensional PCA result. The use of a higher-dimensional PCA representation as a denoising/input stage for manifold methods must be logged explicitly, not hidden.

### PathMNIST

Use the official `medmnist` package and PathMNIST split(s). Record package version, dataset version, split, sample counts, label mapping, image dimensions, and license/source citation.

Expected preprocessing candidate plan:

1. convert images to reproducible numerical features;
2. scale pixel values consistently;
3. flatten images for the classical DR methods;
4. use stratified sampling for pilot/final cohorts;
5. optionally use a 50-component PCA representation as the common input for expensive nonlinear methods, with the transformation fitted only on the relevant analysis cohort.

Do not quietly replace image pixels with pretrained neural-network embeddings in the base analysis; that would change the scientific question. Such an extension may be added only after the required pixel-based workflow is complete and must be labeled as optional.

### Data rules

- Store downloads under `data/raw/`; never commit them.
- Record checksums when feasible.
- Write `docs/DATA_PROVENANCE.md` with source URLs/package loaders, access date, versions, citations, licenses, and transformation steps.
- The loader must be deterministic and cache-aware.
- Pilot sampling must preserve labels through stratification when labels exist, but labels must not be passed into embedding algorithms.

**Phase 3 acceptance criteria**

- Both loaders run twice without corrupting or needlessly redownloading data.
- Shapes, sparsity, missingness, label counts, and checksums/manifests are recorded.
- A small data-loader test passes without running the full benchmark.

---

## 9. Phase 4: Inspector and Preprocessing Planner

The inspector should calculate, where applicable:

- number of samples and features;
- numeric/non-numeric feature types;
- missingness and non-finite values;
- sparsity;
- zero/negative-value patterns;
- per-feature scale and variance summaries;
- duplicate and constant features;
- class counts and imbalance, if labels are provided;
- approximate memory and pairwise-distance cost.

The preprocessing planner must return an explicit structured plan with an action and justification for every transformation. Example:

```json
{
  "action": "select_highly_variable_genes",
  "parameters": {"n_top_genes": 2000},
  "reason": "The input is sparse high-dimensional count data.",
  "fit_scope": "analysis cohort only"
}
```

Preprocessing must be fit without labels. Avoid data leakage between any tuning and held-out diagnostic split if a held-out split is implemented.

**Phase 4 acceptance criteria**

- PBMC 3K and PathMNIST receive different, sensible preprocessing plans.
- Plans and completed transformations are serialized to JSON.
- No method code contains hidden dataset-name-specific preprocessing.

---

## 10. Phase 5: Implement and Smoke-Test All Ten Methods

Implement a shared abstract interface such as:

```python
class DRMethod:
    method_id: str

    def validate_input(self, X, config): ...
    def fit_transform(self, X, seed, params): ...
    def default_params(self, profile): ...
    def tuning_grid(self, profile): ...
```

Each method runs inside an isolation boundary that can capture warnings, exceptions, runtime, and timeouts without crashing the entire pipeline. A subprocess or SLURM task boundary is preferred for methods that can hang or retain large amounts of memory.

### Failure recovery policy

For a failed run:

1. capture the original error and full parameters;
2. make at most one automatic retry using a valid, predefined numerical adjustment;
3. log the adjustment and its rationale;
4. if the retry fails, mark the method `failed` or `timeout` and continue;
5. never switch to a different algorithm while keeping the original method name.

Examples of valid recovery include enforcing `n_neighbors < n_samples`, choosing a compatible eigensolver, adding documented numerical regularization for LLE, or reducing an invalid perplexity. Reducing pilot sample size for just one method is not allowed in the main comparable benchmark; instead, choose a common pilot size that all methods are expected to attempt, or record that the method could not complete the common benchmark.

### Initial tuning grids

Keep grids small and configurable. Suggested starting points:

| Method | Small refinement grid |
|---|---|
| PCA | solver choice only if needed; 2 output dimensions |
| Kernel PCA | RBF gamma around a median-distance heuristic; optionally RBF vs polynomial |
| MDS | metric setting fixed by the declared analysis; limited `n_init` |
| Isomap | `n_neighbors`: 5, 15, 30 |
| LLE | `n_neighbors`: 10, 20, 30; valid regularization options |
| Laplacian | `n_neighbors`: 10, 30, 50 |
| Diffusion Maps | neighborhood/epsilon heuristic and a small alpha grid |
| GPLVM | inducing-point count, learning rate, and training epochs within a fixed compute budget |
| t-SNE | perplexity: 15, 30, 50; learning rate heuristic |
| UMAP | `n_neighbors`: 15, 30, 50; `min_dist`: 0.1, 0.5 |

Clip invalid values based on sample size and log the adjustment.

First run every method on tiny synthetic datasets (for example, blobs and Swiss roll). Tests must confirm:

- output shape is `(n_samples, 2)`;
- all coordinates are finite;
- the method identifier is correct;
- parameters and versions are recorded;
- fixed seeds reproduce deterministic/stochastic results within an appropriate tolerance;
- failures are isolated rather than aborting the suite.

**Phase 5 acceptance criteria**

- Registry test proves that all ten required IDs are present.
- All methods pass a tiny smoke test or have a diagnosed, unresolved package-level blocker documented for the user.
- GPLVM test demonstrates actual loss optimization, not only initialization.

---

## 11. Phase 6: Pilot Benchmark Stage

This is the core agentic component.

### Comparable pilot design

For each dataset:

1. Create one deterministic, representative pilot cohort, initially capped at 1,000 observations.
2. Use the same cohort and same preprocessed input representation for all ten methods.
3. Attempt every registered method with a default or heuristic parameter set.
4. Repeat stochastic methods with three seeds. Deterministic methods may run once unless numerical stability is being tested explicitly.
5. Record success/failure, quality metrics, runtime, memory, warnings, and stability.

If 1,000 observations make the complete pilot infeasible under a reasonable cluster budget, reduce the **common pilot size for all methods**, document the evidence, and rerun the comparable pilot. Do not give expensive methods an easier cohort and compare their raw metrics as if the cohorts were identical.

### Required evaluation metrics

Calculate at least:

1. **Trustworthiness** at a configured neighborhood size.
2. **Neighborhood preservation**, defined explicitly as mean high-dimensional k-nearest-neighbor recall or Jaccard overlap in the embedding.
3. **Embedding stability** across seeds for stochastic methods, preferably using neighborhood overlap rather than direct coordinate equality.
4. **Runtime** in seconds.
5. **Execution status**, warnings, retry count, and convergence information.
6. **Silhouette score** using known labels when available, reported only as a secondary diagnostic by default.
7. **Explained variance** for PCA as a method-specific diagnostic.

Optionally add a global-structure diagnostic such as Spearman correlation between sampled pairwise distances, but do not overload the assignment with redundant metrics.

Use the exact same high-dimensional reference representation when comparing neighborhood metrics across methods within a dataset. Record the metric definition, `k`, distance metric, and any subsampling used for metric calculation.

### Pilot outputs

Each dataset must produce:

```text
outputs/<dataset>/pilot/
├── cohort_manifest.json
├── run_manifest.json
├── method_runs/<method>/<seed>/...
├── pilot_metrics.csv
├── pilot_metrics.json
├── failures.json
├── figures/
└── pilot_complete.flag
```

Do not create `pilot_complete.flag` until all ten method IDs have a terminal status and required result files validate.

**Phase 6 acceptance criteria**

- Every method has `success`, `failed`, or `timeout` status.
- Successful methods have finite embeddings and complete metrics.
- Metrics are comparable and traceable to one cohort manifest.
- The benchmark can resume after interruption without corrupting successful runs.

---

## 12. Phase 7: Evidence-Based Candidate Selection

Selection must be transparent and configurable, not an unexplained single composite score.

### Default decision logic

1. Retain PCA as the linear baseline if it ran successfully.
2. Exclude methods with terminal failure, invalid embeddings, or unresolved severe numerical warnings.
3. Among successful methods, calculate a quality summary from unsupervised metrics (trustworthiness, neighborhood preservation, and stability). Do not include label silhouette in the default selection score.
4. Identify methods reaching a configured fraction of the best empirical quality (initially 90%) and methods on a quality-runtime Pareto frontier.
5. Retain no more than four final candidates, including PCA, while favoring strong empirical quality and avoiding redundant methods with materially worse runtime and no quality benefit.
6. If fewer than two methods survive, retain PCA plus the strongest valid nonlinear method and mark the low-confidence selection.
7. Do not exclude a method solely because its family was considered expensive before the pilot; use observed runtime/failure evidence.

The exact normalization and tie-breaking rules must be unit-tested and written to `docs/METHOD_NOTES.md`.

### Decision log

Create `outputs/<dataset>/decisions.json` with entries like:

```json
{
  "method": "umap",
  "prior_suitability": "Potentially suitable for high-dimensional nonlinear structure.",
  "pilot_evidence": {
    "trustworthiness": 0.95,
    "neighbor_recall": 0.81,
    "stability": 0.88,
    "runtime_seconds": 4.2,
    "status": "success"
  },
  "decision": "retain",
  "reason": "High local-structure quality, acceptable stability, and practical runtime."
}
```

Reasons must be generated from actual metrics and thresholds, not generic method descriptions.

**Phase 7 acceptance criteria**

- Re-running selection on the same pilot table gives the same shortlist.
- Each retained/excluded method has an evidence-backed reason.
- PCA baseline handling and label-free primary selection are verified by tests.

---

## 13. Phase 8: Shortlist Tuning and Final Runs

Only shortlisted methods receive a small hyperparameter refinement. Do not run a large Cartesian benchmark across all ten methods.

For each candidate:

1. evaluate the small configured grid on the pilot cohort;
2. select parameters with the same label-free primary quality criteria;
3. use deterministic tie-breaking that favors lower runtime and simpler/default parameters;
4. record every tried configuration in `tuning_results.csv`;
5. run the chosen configuration on the final analysis cohort.

### Final cohort guidance

- PBMC 3K is small enough that the full post-QC dataset should normally be used.
- PathMNIST is much larger and includes methods with quadratic or worse scaling. Define a reproducible stratified final analysis cohort that is large enough to be informative but feasible for the shortlisted methods. The cohort size must be justified from pilot runtime/memory scaling, not chosen silently.
- “Full-data analysis” in the report must mean the declared final analysis cohort. If it is a subset of the complete source dataset, state this prominently and never call it the complete PathMNIST dataset.

Keep raw pilot and tuning evidence separate from final-run metrics.

**Phase 8 acceptance criteria**

- Tuning tables show all tried configurations.
- Final embeddings are finite, correctly aligned with observation IDs/labels, and reproducible from config plus seed.
- Final cohort manifests state exactly which observations were used.

---

## 14. Phase 9: Visualizations and Reports

### Required figures

For each dataset, produce:

- two-dimensional embedding plot for each final method;
- PCA explained-variance plot;
- pilot quality comparison plot;
- runtime-versus-quality plot;
- method-selection summary figure or table;
- stability plot when repeated stochastic runs are available.

Use consistent colors for the same labels across methods. Figures must remain interpretable in print, include informative captions, and avoid presenting visual separation as proof of biological truth.

### Evidence bundle

Before report generation, construct a machine-readable evidence bundle containing only validated facts:

```text
dataset_profile.json
preprocessing_plan.json
pilot_metrics.csv
tuning_results.csv
final_metrics.csv
decisions.json
run_manifest.json
figure_manifest.json
validation_results.json
```

### Generated dataset reports

Create:

- `reports/generated_report_1.pdf` for PBMC 3K;
- `reports/generated_report_2.pdf` for PathMNIST.

Each report should include:

1. dataset inspection;
2. preprocessing and its rationale;
3. pilot benchmark design;
4. quantitative pilot results;
5. candidate decisions;
6. tuning and final embeddings;
7. interpretation grounded in actual outputs;
8. limitations and computational caveats;
9. reproducibility information.

Use Jinja2 plus HTML/CSS and WeasyPrint for a reproducible PDF path unless a better working project-local renderer is established. Validate the generated PDFs with `pypdf` and render/inspect representative pages before considering them final.

### Deterministic versus LLM reporting

The reporting layer must support:

- `deterministic`: fill a structured report template directly from the evidence bundle;
- `openai`: send only the validated evidence bundle and a constrained prompt to the OpenAI API, then validate the returned claims before rendering.

If `reporting.mode: openai` is requested but credentials/network access are unavailable, fail clearly and ask the user. Do not silently fall back and still call the output LLM-generated. Never send raw restricted data or secrets to an external model.

LLM output must not invent metrics, dataset sizes, methods, citations, or conclusions. Where practical, post-validate every numeric string against the evidence bundle. The deterministic report must always remain available as the reproducible baseline.

### Final project report

Create `reports/report.pdf`, limited to **four pages excluding references and appendix**, covering:

1. system architecture;
2. dataset-aware planning and decision logic;
3. tools/models and all ten supported methods;
4. PBMC 3K and PathMNIST results;
5. strengths, limitations, and reproducibility.

This report should emphasize the core contribution:

> Rather than selecting dimensionality-reduction methods solely from hard-coded heuristics or LLM recommendations, the agent performs a lightweight pilot benchmark across all supported methods, quantitatively evaluates their empirical behavior, and uses the resulting evidence to determine which methods advance to full-scale analysis.

Codex may draft this report from validated artifacts, but the user must have an easy opportunity to review the final scientific wording.

**Phase 9 acceptance criteria**

- All three required PDFs exist, open successfully, and have correct names.
- The final project report satisfies the page limit.
- Every reported numeric claim traces to a saved artifact.
- Report mode is labeled honestly.

---

## 15. Phase 10: Validation and Reproducibility Audit

Implement a validator that checks at least:

```text
[ ] all ten methods are registered
[ ] all ten methods were attempted in each pilot
[ ] every pilot method has a terminal status
[ ] successful embeddings have expected shape and finite values
[ ] pilot cohorts are common across methods within a dataset
[ ] metrics contain no unexplained missing values
[ ] selection decisions match configured rules
[ ] labels were not used for default fitting/selection
[ ] tuning is restricted to shortlisted methods
[ ] final observation IDs align with labels and embeddings
[ ] expected figures and tables exist
[ ] decisions.json and manifests are valid
[ ] generated_report_1.pdf exists and opens
[ ] generated_report_2.pdf exists and opens
[ ] report.pdf exists, opens, and meets the page limit
[ ] environment and software versions are recorded
```

Run:

```bash
pytest -q
ruff check src tests
python -m agent1_dr validate --config config/pbmc3k.yaml
python -m agent1_dr validate --config config/pathmnist.yaml
```

Also run a clean end-to-end smoke workflow in a fresh output directory. A final status of `PASSED` requires all mandatory checks; warnings must remain visible.

Before handoff:

1. verify `.gitignore` excludes data, caches, environments, API files, SLURM logs, and large generated intermediates;
2. ensure README commands match the actual CLI;
3. confirm the environment specification works from a clean test if resources permit;
4. generate a small submission manifest with file paths and checksums;
5. review `git status` and do not remove unrelated user files;
6. do not upload or push without the user's instruction.

---

## 16. SLURM Execution Principles

Create cluster scripts only after inspecting current Longleaf partitions and group conventions. Each script should:

- use the exact `agent` environment;
- set explicit CPU, memory, wall-time, and log paths;
- print hostname, timestamp, job ID, Git commit, config path, and package versions;
- use unbuffered output;
- fail on shell errors;
- write to a dataset/stage-specific log directory;
- resume safely when requeued or rerun.

Suggested workflow:

```text
small local/unit tests
    -> pilot dry run on a tiny cohort
    -> PBMC pilot
    -> PathMNIST pilot
    -> selection
    -> shortlist tuning
    -> final runs
    -> report rendering
    -> validation
```

Use CPU execution as the portable baseline. GPLVM may use a GPU only after GPU availability and the PyTorch build are verified; its result must remain reproducible and its hardware recorded. Do not delay the entire pipeline waiting for a GPU if a practical CPU pilot is available.

---

## 17. Output Contract

At minimum, each dataset directory must contain:

```text
outputs/<dataset>/
├── dataset_profile.json
├── preprocessing_plan.json
├── pilot/
│   ├── cohort_manifest.json
│   ├── pilot_metrics.csv
│   ├── pilot_metrics.json
│   └── failures.json
├── selection/
│   └── decisions.json
├── tuning/
│   └── tuning_results.csv
├── final/
│   ├── cohort_manifest.json
│   ├── final_metrics.csv
│   ├── embeddings/
│   └── figures/
├── evidence_bundle/
├── run_manifest.json
└── validation_results.json
```

Final submission-facing deliverables are:

```text
source code and configuration
README.md
environment specification
reports/generated_report_1.pdf
reports/generated_report_2.pdf
reports/report.pdf
```

Keep bulky reproducible outputs out of Git when necessary, but preserve small metrics, manifests, decision logs, and final figures if course submission rules allow them. Document any external or archived output location.

---

## 18. Definition of Done

The project is complete only when:

1. both datasets run through the same public agent interface;
2. all ten required methods are implemented and attempted in both pilots;
3. quantitative evidence, not preference alone, determines the shortlist;
4. failures and computational limitations remain visible;
5. shortlisted methods are tuned and run on declared final cohorts;
6. decisions and results are machine-readable and reproducible;
7. reports are automatically generated from validated evidence;
8. all tests and validation checks pass, or any unresolved blocker is clearly reported;
9. the final repository contains no credentials, hidden manual steps, or mislabeled approximations;
10. `docs/PROGRESS.md` gives the user a concise, accurate handoff.

---

## 19. Immediate Next Actions for Codex

Execute these in order:

1. Read this README completely.
2. Run the Phase 0 preflight without modifying or deleting existing work.
3. Create/update `docs/PROGRESS.md` with the preflight findings.
4. Inspect and validate the requested `agent` Conda environment.
5. Create `environment.yml`, `pyproject.toml`, `.gitignore`, the package skeleton, and configuration schemas.
6. Install missing real dependencies and run import/PDF/GPLVM smoke checks.
7. Implement loaders and tiny data tests.
8. Implement the registry and all ten method wrappers.
9. Pass the synthetic smoke suite before submitting any full pilot job.
10. Submit the pilot jobs, monitor them, and continue through Sections 11–15 using saved evidence at every decision point.

Do not jump directly to attractive UMAP/t-SNE figures. The main scientific deliverable is the auditable path from data inspection to empirical method selection.

---

## 20. Implemented execution entry points (v2)

Sections 1–19 above are the original implementation contract and are preserved unchanged. The implemented decision rules, weights and thresholds are in `docs/METHOD_NOTES.md`; the handoff history is in `docs/PROGRESS.md`.

```bash
cd /proj/yunligrp/users/ygao/BIOS774Agent/Agent1_DR
bash scripts/01_setup_environment.sh              # once; never deletes the agent environment
sbatch scripts/slurm/smoke.sbatch                 # tests + lint + fresh synthetic run; writes the smoke gate
sbatch scripts/slurm/run_all.sbatch config/pbmc3k.yaml      # requires the current-code smoke gate
sbatch scripts/slurm/run_all.sbatch config/pathmnist.yaml
sbatch scripts/slurm/freeze.sbatch                # validate both datasets, checksum and lock the evidence
sbatch scripts/slurm/project_report.sbatch        # render reports/report.pdf from the frozen evidence
```

Stage commands (`inspect`, `pilot`, `select`, `tune`, `final`, `report`, `validate`, `run-all`) accept `--config`. Valid outputs are reused by default; `--force` recomputes a stage. Method fits are cached by method, seed, parameters, input checksum and the hash of the fit code, so report-only changes do not refit methods. Do not run two writers on one dataset output directory.

Outputs: `outputs/<dataset>/` holds inspection, pilot (all ten methods, repeated seeds, subsample refits, figures), selection (decisions, sensitivity, ablation), tuning, final (scaling probe, declared cohort, embeddings, figures), the evidence bundle and `validation_results.json`. Deliverables are `reports/generated_report_1.pdf` (PBMC3k), `reports/generated_report_2.pdf` (PathMNIST) and `reports/report.pdf` (project report; editable narrative in `templates/project_report.html.j2`). Small Git-friendly snapshots of the frozen evidence live in `examples/`. Archived v1 outputs are in `outputs/archive/v1/` (not in Git).

New numerical datasets: set `dataset.name: numeric` and `dataset.path` to a CSV (optional `label_column`, `id_column`) or NPZ (`X`, optional `labels`, `ids`, `feature_names`). The planner chooses the preprocessing branch from the inspected profile (count matrix, bounded intensity, or continuous). Labels are optional and never used for fitting or selection.

Environment reconstruction on Linux x86-64: `conda create -p <prefix> --file environment.conda-explicit.txt`, then install `-r environment.lock.txt` with that environment's Python and `PYTHONNOUSERSITE=1`; put `<prefix>/lib` first in `LD_LIBRARY_PATH`. The SLURM scripts apply these settings through `scripts/slurm/common.sh`. No environment, data, outputs or credentials are committed; no license has been selected and nothing has been pushed.
