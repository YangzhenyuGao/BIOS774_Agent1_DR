from copy import deepcopy

from agent1_dr.config import Config
from agent1_dr.selector import select


def row(mid, q, time, **extras):
    return dict(
        method=mid,
        quality=q,
        runtime_seconds=time,
        status="success",
        severe_warning=False,
        successes=3,
        repeats=3,
        **extras,
    )


def test_pca_and_labels():
    table = [
        row("pca", 0.2, 1, silhouette=-0.9),
        row("tsne", 0.9, 2, silhouette=0.5),
        row("umap", 0.95, 1.5, silhouette=0.1),
        row("gplvm", 0.1, 80, silhouette=1),
    ]
    cfg = Config()
    decision = select(table, cfg)
    assert "pca" in decision["selected"] and "gplvm" not in decision["selected"]
    changed = deepcopy(table)
    for r in changed:
        r["silhouette"] = 100
    assert select(changed, cfg)["selected"] == decision["selected"]
    assert select(table, cfg) == decision


def test_severe_and_tie():
    table = [row("pca", 0.4, 1), row("umap", 0.9, 2), row("tsne", 0.9, 2)]
    table[1]["severe_warning"] = True
    assert select(table, Config())["selected"] == ["pca", "tsne"]
