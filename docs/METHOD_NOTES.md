# Method and decision notes

## Implementations

Seven wrappers call the named scikit-learn estimators in the supplied README; Diffusion Maps calls `pydiffmap.diffusion_map.DiffusionMap.from_sklearn`, GPLVM uses GPyTorch `BayesianGPLVM` with `VariationalLatentVariable` and inducing-point `VariationalStrategy`, and UMAP calls `umap.UMAP`.

GPLVM fits independent output GPs with batch kernels, a standard Normal latent prior, Gaussian observation likelihood and variational ELBO including latent KL. PCA initializes the two latent dimensions; the returned variational means are optimized with Adam. A global scalar rescales the common input to stabilize the likelihood without changing relative feature distances. Loss history, steps and displacement from initialization are recorded. Default 300 minibatch steps is a computational budget, not a convergence guarantee. All methods use CPU; one numerical thread per isolated worker, four allocated CPUs available to orchestration. Parameters, package versions and max RSS are saved. Successful-run timing starts inside worker.main before method-specific imports, input loading and fitting. Python subprocess startup and common imports (such as NumPy) are excluded. Timeout timing covers the full subprocess wait. Peak RSS describes the worker, not the orchestrator.

The common cohort is prepared independently for pilot and final. All ten pilot methods get the exact same saved NumPy representation, whose SHA256 is copied to every result. No model receives labels. Label stratification affects cohort composition and is explicitly disclosed. PBMC post-QC counts are normalized/logged, variable genes selected within the cohort, scaled/clipped, then projected to up to 50 PCs. PathMNIST uses flattened RGB pixels divided by 255, then up to 50 PCs. The final two-dimensional PCA on that representation is the linear baseline. PCA explained variance in plots is relative to the preprocessed feature space, not the reduced representation.

## Metrics and selection

Euclidean trustworthiness uses the full cohort; neighbor recall is mean |N_high(i) intersection N_embed(i)|/k with self excluded explicitly, stable distance tie sorting. k=15 normally, clipped below n/2. Stability averages this neighbor overlap over all pairs of successful seed embeddings. MDS, GPLVM, t-SNE and UMAP use three seeds. Deterministic stability is defined as 1, not estimated by resampling.

Quality = 0.45 trustworthiness + 0.45 recall + 0.10 stability. Components already lie in [0,1], so there is no min-max normalization. Silhouette is excluded, and labels are never passed to the score. Silhouette excludes `unannotated`, uses at most 2,000 observations and has a reason string when undefined. Failed methods have null quality, with explicit status and exceptions. A stochastic method is eligible only if all configured seeds succeed. Disconnected graph or singularity warnings are treated as severe and exclude a method; original warnings remain visible.

Selection retains successful valid PCA; computes the maximum quality and 90% threshold among eligible methods; identifies the quality-runtime Pareto frontier; and ranks eligible threshold/frontier methods by decreasing quality, increasing observed runtime, then method ID. At most four candidates including PCA survive. If fewer than two survive, the strongest additional eligible method is added and low confidence marked. If no valid nonlinear method exists the failure remains visible rather than fabricating a second candidate. Every method has a prior, measured evidence and a threshold-dependent decision reason.

Tuning only considers selected methods. Each grid point uses the same pilot input and stochastic repeat count. Quality ties rounded to six decimals favor runtime, then grid order (default first). Run results are cached by input, code, seed and parameters. LLE can retry once with dense eigensolver and reg=0.01 after failure; both attempts and total runtime are retained. Timeouts never shrink one method's cohort.

Final PathMNIST size is capped at 3,000 and further bounded for each shortlisted method by `2 * pilot_seconds * (N_final/N_pilot)^3 <= timeout_seconds`, with no extrapolated cap below the already completed pilot size. This conservative planning envelope is not a measured scaling law. PBMC normally uses all post-QC cells. Cohort sizes and exact IDs are saved. Final metrics are separate from pilot/tuning evidence; all are in-cohort descriptive results, not held-out estimates.

## Reporting and reproducibility

Deterministic templates generate the default PDFs. OpenAI mode can select and organize exact validated evidence sentences through Responses structured JSON output; every sentence must match the provided evidence exactly. Arbitrary generated prose is rejected. There is no live API claim unless a response ID is recorded; absent credentials or network cause explicit failure. Mock API tests validate the contract, not live network operation.

Stages save configuration/code hashes, Git commit when available, dependency versions, host, job ID and artifact checksums. Reuse requires matching provenance and valid files. `--force` recomputes; no data/env deletion is performed. Deterministic runs can still differ across dependency versions/hardware; lock files and manifests define the tested environment.

## Primary references (accessed 2026-09-22)

- Scikit-learn estimators: https://scikit-learn.org/stable/modules/manifold.html and https://scikit-learn.org/stable/modules/decomposition.html
- Diffusion maps implementation: https://pydiffmap.readthedocs.io/
- Bayesian GPLVM implementation reference: https://docs.gpytorch.ai/en/latest/examples/045_GPLVM/Gaussian_Process_Latent_Variable_Models_with_Stochastic_Variational_Inference.html
- Titsias & Lawrence (2010): https://proceedings.mlr.press/v9/titsias10a.html
- UMAP: https://umap-learn.readthedocs.io/
- Optional Responses schema: https://developers.openai.com/api/docs/guides/structured-outputs

## Validated numerical adjustments

The initial pydiffmap BGH automatic bandwidth produced a tiny positive generator eigenvalue and non-finite `sqrt(-1/eigenvalue)` coordinates on the 45-point separated-blobs smoke dataset. The default now uses epsilon=median(nonzero squared pairwise distances)/4, computed on at most the first 500 observations of the common input; the real pydiffmap implementation remains unchanged. Alpha is refined over 0.5, 0 and 1 if shortlisted. This adjustment is supported by the saved failing smoke result and retest; pilot statuses remain empirical.

The tested conda-forge SciPy C++ extension needs a newer CXXABI than Longleaf's system libstdc++. CPU Torch can load the system library first. Scripts and workers put the agent environment's lib directory first in LD_LIBRARY_PATH, and disable Python user-site imports. These are required execution settings; they do not alter system libraries or user-installed packages. Future jobs request one CPU because numerical execution is deliberately single-threaded; the initial failed environment-check job had requested four.
