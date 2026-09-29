# Data provenance

Downloads are cached only under `data/raw/` (Git-ignored). Each dataset profile records access date, actual shapes, package versions, source URLs and SHA256 of the downloaded files. Pilot/final cohort manifests retain exact IDs and diagnostic labels. Actual download validation is recorded in PROGRESS.md after execution.

## PBMC3k

Use Scanpy's `datasets.pbmc3k()` raw count matrix and `datasets.pbmc3k_processed()` annotations. Barcode uniqueness and subset inclusion are asserted before transfer; unmatched raw barcodes are `unannotated`. Processed `louvain` cell annotations are secondary, clustering-derived diagnostics rather than independent ground truth. Data describe PBMCs from a healthy donor, originally provided publicly by 10x Genomics. No license is inferred for redistribution: see the provider's dataset terms; raw files are not included in Git/submission.

- Scanpy loader: https://scanpy.readthedocs.io/en/latest/api/generated/scanpy.datasets.pbmc3k.html
- Processed loader: https://scanpy.readthedocs.io/en/latest/api/generated/scanpy.datasets.pbmc3k_processed.html
- 10x dataset: https://www.10xgenomics.com/datasets/3-k-pbm-cs-from-a-healthy-donor-1-standard-1-1-0

PBMC transformation details and fitted scope are serialized separately for the pilot and final cohorts. Source raw dimensions and post-QC counts are both retained.

## PathMNIST

Use official `medmnist.PathMNIST(size=28, split='train')`; preserve nine-class mapping from package `INFO`. Numerical input is RGB pixels (uint8 flattened for inspection, float divided by 255 for analysis), followed by cohort-fitted PCA. No pretrained model features. The `.npz` download contains source splits; the base analysis uses only the declared training split. Pilot and final are explicitly stratified subsets.

- Official distribution and overview: https://medmnist.com/
- Package/source: https://github.com/MedMNIST/MedMNIST
- Yang et al. (2023), MedMNIST v2, Scientific Data 10:41: https://doi.org/10.1038/s41597-022-01721-8
- Original colorectal histology source: Kather et al., 100,000 histological images of human colorectal cancer and healthy tissue: https://doi.org/10.5281/zenodo.1214456

The actual URL, package version, label mapping and license advertised by the installed official loader are copied verbatim into the dataset profile. Source data license must be respected independently of this project's code, for which no license has been selected.

The official MedMNIST `INFO['pathmnist']` source inspected on 2026-09-22 identifies the PathMNIST license as **CC BY 4.0**, with training/validation/test counts 89,996 / 10,004 / 7,180 and 28-pixel download MD5 `a8b06965200029087d5bd730944a56c1`. The installed loader's actual metadata and local SHA256 are recorded independently rather than substituting these reference counts for observed counts. Reference: https://raw.githubusercontent.com/MedMNIST/MedMNIST/main/medmnist/info.py
