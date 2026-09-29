from copy import deepcopy

from agent1_dr.config import Config
from agent1_dr.selector import ablation, select, sensitivity


def row(mid, q, time, **extras):
    # One number per component; quality recomputed from the pre-registered weights.
    base = {
        "method": mid, "status": "success", "severe_warning": False, "severe_v1": False, "severe_reasons": [],
        "successes": 8, "attempts": 8, "runtime_seconds": time, "trustworthiness": q, "neighbor_recall": q,
        "global_structure": q, "stability": q, "stability_v1": 1.0, "trustworthiness_k2": q,
        "neighbor_recall_k2": q,
    }
    base.update(extras)
    return base


def test_pca_retained_and_labels_ignored():
    table = [row("pca", 0.2, 0.01, silhouette=-0.9), row("tsne", 0.9, 5, silhouette=0.5),
             row("umap", 0.95, 2, silhouette=0.1), row("gplvm", 0.1, 80, silhouette=1)]
    cfg = Config()
    decision = select(table, cfg)
    assert decision["selected"][0] == "pca" and "gplvm" not in decision["selected"]
    changed = deepcopy(table)
    for r in changed:
        r["silhouette"] = 100
    assert select(changed, cfg)["selected"] == decision["selected"]
    assert select(table, cfg) == decision


def test_runtime_noise_does_not_create_frontier_methods():
    # v1 failure mode: a slightly worse method 0.1 s faster than PCA must not be "Pareto".
    table = [row("pca", 0.58, 1.6), row("lle", 0.55, 1.5), row("laplacian", 0.56, 1.55), row("tsne", 0.65, 5.8)]
    decision = select(table, Config())
    assert decision["selected"] == ["pca", "tsne"]
    assert "lle" not in decision["pareto_frontier"] and "laplacian" not in decision["pareto_frontier"]
    strict = select(table, Config(), rules={"pareto": "strict"})
    assert "lle" in strict["pareto_frontier"]


def test_materially_faster_method_stays_on_frontier():
    table = [row("pca", 0.40, 0.01), row("tsne", 0.65, 6.0), row("umap", 0.60, 2.0)]
    decision = select(table, Config())
    assert {"pca", "umap", "tsne"} <= set(decision["pareto_frontier"])


def test_severe_and_equivalence_margin():
    table = [row("pca", 0.4, 0.01), row("umap", 0.9, 2), row("tsne", 0.9, 2), row("mds", 0.897, 3)]
    table[1]["severe_warning"] = True
    table[1]["severe_reasons"] = ["collapsed layout"]
    decision = select(table, Config())
    assert "umap" not in decision["selected"] and "mds" in decision["equivalent_to_best"]
    # A large paired SE widens the margin; identical per-subsample differences shrink it to the floor.
    subs = {m: [{"trustworthiness": v, "neighbor_recall": v, "global_structure": v} for v in vals]
            for m, vals in {"tsne": [0.9, 0.95, 0.85], "mds": [0.84, 0.99, 0.80], "pca": [0.4, 0.4, 0.4]}.items()}
    wide = select(table, Config(), subs)
    assert "mds" in wide["equivalent_to_best"]


def test_variants_and_ablation_run_on_same_evidence():
    table = [row("pca", 0.5, 0.01), row("tsne", 0.7, 5), row("umap", 0.66, 2, stability=0.3, stability_v1=0.3)]
    cfg = Config()
    s = sensitivity(table, cfg, {})
    assert s["variants"][0]["selected"] == select(table, cfg)["selected"]
    steps = ablation(table, cfg, {})
    assert steps[-1]["selected"] == select(table, cfg)["selected"]
