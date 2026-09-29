# Method and decision notes (v2)

This file states every rule, weight and threshold used by the agent. The v2 rules below were written and committed **before** any v2 real-data run; the only real-data evidence consulted while designing them was the v1 audit (2026-09-28), summarized in "Why v2" at the end. Thresholds for the new diagnostics were calibrated on simulated data only.

## Implementations

Seven wrappers call the scikit-learn estimators named in the README. Diffusion maps call `pydiffmap.diffusion_map.DiffusionMap.from_sklearn`. GPLVM uses GPyTorch `BayesianGPLVM` with `VariationalLatentVariable` and an inducing-point `VariationalStrategy`. UMAP calls `umap.UMAP`. All fits are CPU and single-threaded.

- **MDS** is metric SMACOF started from the classical-MDS solution (`init="classical_mds"`, `max_iter=1000`; this becomes the scikit-learn default in 1.10). In v1 the random start with `n_init=1` stopped at `max_iter=300` on every PBMC seed. Tuning compares this start with a random start that uses `n_init=4`.
- **GPLVM** optimizes the variational ELBO including the latent KL term, starting from PCA. One *epoch* is `ceil(n / batch_size)` minibatch steps. The default of 75 epochs equals v1's 300 steps at n = 1,000, and the budget now scales with cohort size, so larger cohorts are not under-trained. The loss history, latent displacement and a loss-plateau statistic (relative change between the last two tenths of training) are saved. The fixed budget is not a convergence guarantee.
- **t-SNE** keeps `init="pca"`, which is recommended for global structure. With PCA initialization the seed has no effect: v1's three seeds were bit-identical. The pipeline now detects and reports this as `seed_invariant`.
- **Diffusion maps** use ε = median(squared distance)/4 on at most 500 observations. The Markov spectral gap 1 − μ₁ = −λ₁ε is recorded, because pydiffmap's generator is L = (P − I)/ε.
- **Laplacian eigenmaps** record the connected components and the smallest normalized-Laplacian eigenvalues of the affinity graph. This is an untimed diagnostic.

## Execution and timing

One warm worker subprocess runs a batch of fits for one method. Imports and a 64-row warm-up fit (numba JIT, torch initialization) are recorded as overhead and excluded from `runtime_seconds`, which is the wall time of `fit_transform` only. v1's runtimes were dominated by imports: PCA took 0.01 s to fit but was recorded as 1.6 s, and UMAP about 2 s but was recorded as 20–24 s.

Every fit writes an atomic result. The parent enforces the per-fit timeout through a progress file. A timed-out or crashing fit becomes terminal, and the remaining fits restart in a fresh worker. Peak memory is the per-fit resident high-water mark: `/proc/self/clear_refs` is reset before each fit. Only LLE has a predefined numerical retry (`reg=0.01`, dense eigensolver).

Fit caches are keyed by method, seed, parameters, input SHA-256, and a hash of the code that can change an embedding (method wrappers, worker, registry, schemas).

## Preprocessing planner (profile-driven)

The planner reads the inspected profile, never the dataset name:

1. **Count matrix**: sparsity ≥ 0.5, non-negative, integer-valued and ≥ 1,000 features.
   - Steps: observation QC (configured thresholds; a mitochondrial filter only when MT- feature names exist), sampling, feature filtering within the cohort, library-size normalization + log1p, Seurat HVG selection, scaling with clipping, then PCA.
2. **Bounded intensity**: otherwise, non-negative integers with maximum ≤ 255.
   - Steps: sampling, division by 255 (a fixed known scale, no per-feature standardization), then PCA.
3. **Continuous**: everything else.
   - Steps: sampling, z-score within the cohort, then PCA.

Every step records the profile values that triggered it. For PBMC3k and PathMNIST, v2 reproduces v1's pilot inputs bit for bit (verified on one node: identical SHA-256, IDs and labels).

## Metrics (all label-free, Euclidean, in the common input space)

- **T**: trustworthiness at k = 15, identical to `sklearn.manifold.trustworthiness` (tested, difference 0).
- **R**: mean kNN recall with self excluded and stable tie order, identical to v1.
- **G**: Spearman correlation of reference and embedding pairwise distances, clipped at 0. All pairs are used up to 500,000; beyond that, a fixed random sample of 500,000 pairs.
- **S (subsample stability)**: every method is refitted on the same 5 random 80% subsamples of the pilot. Each pair of refits is compared by the mean kNN overlap (k = 15) on the observations they share.
  - This applies to deterministic and stochastic methods alike. It replaces v1's "deterministic = 1 by definition" and "seed stability", which gave seed-invariant t-SNE a measured 1.0.
  - Seed stability is still reported for stochastic methods as a diagnostic.
- Secondary diagnostics: T and R at k = 50; silhouette with labels (at most 2,000 observations, unannotated excluded); the collapse ratio.
- Distances are processed in 256-row chunks, so evaluation memory is linear in n. Reference neighborhoods are computed once per cohort.

## Pre-registered quality and severe diagnostics

**Q = 0.35 T + 0.35 R + 0.20 G + 0.10 S.** Local structure keeps the largest share, as in v1's 0.45/0.45. G is added because v1 had no global component. S keeps v1's 0.10 weight but uses the comparable definition.

A method is **ineligible** when any of the following holds:

- a warning reports a disconnected graph or a singular matrix (the v1 checks);
- **collapsed layout**: the median collapse ratio (median distance to the coordinate-wise median ÷ 99th-percentile distance) is < 0.05 **and** < 0.25 × PCA's ratio on the same cohort;
- the diffusion Markov spectral gap is < 1e-6;
- SMACOF reached `max_iter`.

A stochastic method is also ineligible unless every seed and every subsample fit succeeded.

Collapse calibration (simulation only):

| Case | Collapse ratio |
|---|---|
| Healthy layouts | 0.15–0.74 (Gaussian 0.36, unequal clusters 0.15, t-SNE of blobs 0.74, Isomap and Laplacian of a Swiss roll 0.60) |
| Artificially collapsed layouts | 0.00–0.024 |
| Faithful 90% bulk with a far minority | 0.047 |

The last case is why the rule also requires collapse relative to linear PCA.

## Selection rules (pre-registered)

1. Retain PCA as the linear baseline if it is valid.
2. Exclude ineligible methods.
3. Let best = the maximum Q among valid methods.
4. A valid method is **eligible** if any of the following holds:
   - (a) Q ≥ 0.90 × best (README rule);
   - (b) best − Q ≤ margin, where margin = max(2 × SE, 0.005) and SE is the standard error of the paired per-subsample quality differences (T, R, G on each subsample refit; S is common to the subsamples);
   - (c) it lies on the **ε-Pareto frontier**. Method s makes method r redundant when s's quality is ≥ r's, s is not materially slower (t_s ≤ max(2 t_r, t_r + 1 s)), and s is materially better (quality gain > the paired margin) or materially faster (t_r > max(2 t_s, t_s + 1 s)).
5. Rank eligible methods by Q, then fit time, then registry order. Keep at most 4 including PCA. If fewer than 2 remain, add the best valid method and mark the selection low-confidence.
6. Every decision records Q with its components, the threshold, the margin, the Pareto status and the methods that made it redundant.

Robustness is reported, never used for the decision. The same mechanics are re-run on the same evidence under these variants:

- local metrics only;
- v1 weights;
- no stability term;
- global-heavy weights;
- k = 50;
- v1 stability definition;
- v1 severe checks only;
- runtime ignored;
- strict Pareto without materiality.

A cumulative ablation (A0 v1 rules → A5 v2 rules) attributes each change in the shortlist to a single rule.

## Tuning

Only shortlisted methods are tuned, on the pilot cohort, with the same seeds, subsamples, metrics, flags and Q. Grids:

| Method | Grid |
|---|---|
| Kernel PCA | γ × {1, 0.5, 2} |
| MDS | classical vs random start with `n_init=4` |
| Isomap | k ∈ {15, 5, 30} |
| LLE | k ∈ {20, 10, 30}, plus reg = 0.01 |
| Laplacian | k ∈ {30, 10, 50} |
| Diffusion maps | α ∈ {0.5, 0, 1} |
| GPLVM | defaults, 48 inducing points, lr = 0.01 |
| t-SNE | perplexity ∈ {30, 15, 50} |
| UMAP | the README's full grid: n_neighbors ∈ {15, 30, 50} × min_dist ∈ {0.1, 0.5} |

A non-default grid point replaces the default only if its gain exceeds the paired equivalence margin (a one-SE-style rule). This replaces v1's quality ties at 1e-6, which chose perplexity 50 for a gain of 0.0001.

## Final cohort size (dataset-agnostic)

Each shortlisted method is timed on stratified probe cohorts of 500, 1,000, 2,000 and 4,000 observations (capped at the available count).

- **Time exponent**: log-log slope of fit time.
- **Memory exponent**: log-log slope of peak-RSS growth above the smallest probe, which removes the fixed interpreter baseline.
- Both exponents are floored at 1.

Candidate sizes (1,000 … 20,000, the cap, and all available observations) are projected with a 2× safety factor. The declared size is the largest candidate that satisfies all of the following:

- max fit time ≤ the per-run timeout (30 min);
- summed fit time ≤ the final-stage budget (40 min);
- max memory ≤ 24 GB.

The binding constraint is reported. PBMC3k uses all post-QC cells whenever feasible. The PathMNIST cap is 20,000 of the 89,996 training images: a declared subset, never the complete dataset. These projections are planning envelopes, not fitted complexity laws. Final metrics describe the declared final cohort and are not held-out estimates.

## Reporting and reproducibility

Dataset reports are rendered deterministically from checksummed evidence; every sentence in "Findings" is computed from saved artifacts. The optional OpenAI mode may only select exact evidence sentences, and it fails explicitly without credentials. Stage manifests record configuration and code hashes, the git commit, versions, host and job. After both dataset runs validate, `freeze` checksums every output file and the two dataset PDFs and makes them read-only. The project report is rendered only after the frozen checksums verify.

## Why v2 (v1 audit evidence, 2026-09-28)

- PathMNIST v1 was run twice on identical inputs; all ten qualities agreed to 6 decimals, yet the fourth finalist changed (kernel PCA → LLE) because kernel PCA's import-dominated runtime went from 1.27 s to 3.00 s.
- Perturbing one fast method's runtime by ±0.2 s produced 6 different PathMNIST shortlists.
- Setting S = 1 for every method, or dropping S, made UMAP the second-best method on both datasets.
- PBMC MDS never converged. PBMC diffusion maps had a Markov gap of 9e-8. The PBMC final Laplacian layout had a collapse ratio of 0.018.
- Applying runtime materiality alone to the v1 evidence removed the noise-driven PathMNIST picks (LLE, Laplacian).

## Primary references (accessed 2026-09-22 and 2026-09-28)

- Scikit-learn manifold and decomposition estimators: https://scikit-learn.org/stable/modules/manifold.html, https://scikit-learn.org/stable/modules/decomposition.html
- Diffusion maps: Coifman & Lafon (2006), Applied and Computational Harmonic Analysis 21:5–30; pydiffmap: https://pydiffmap.readthedocs.io/
- Bayesian GPLVM: Titsias & Lawrence (2010), https://proceedings.mlr.press/v9/titsias10a.html; GPyTorch tutorial: https://docs.gpytorch.ai/en/latest/examples/045_GPLVM/Gaussian_Process_Latent_Variable_Models_with_Stochastic_Variational_Inference.html
- t-SNE: van der Maaten & Hinton (2008), JMLR 9:2579–2605; PCA initialization: Kobak & Linderman (2021), Nature Biotechnology 39:156–157
- UMAP: McInnes, Healy & Melville (2018), arXiv:1802.03426; https://umap-learn.readthedocs.io/
- Trustworthiness: Venna & Kaski (2001), ICANN
- Optional Responses structured output: https://developers.openai.com/api/docs/guides/structured-outputs
