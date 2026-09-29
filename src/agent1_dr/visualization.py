import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .utils import checksum, read_json, write_json


def figures(cfg):
    directory = cfg.output / "final/figures"
    directory.mkdir(parents=True, exist_ok=True)
    manifest = []

    def save(fig, name, caption):
        path = directory / f"{name}.png"
        fig.savefig(path, dpi=170, bbox_inches="tight")
        plt.close(fig)
        manifest.append({"path": str(path), "sha256": checksum(path), "caption": caption})

    labels = np.load(cfg.output / "final/labels.npy")
    classes = sorted(set(labels))
    palette = plt.get_cmap("tab10" if len(classes) <= 10 else "tab20").resampled(len(classes))
    colors = {label: palette(i) for i, label in enumerate(classes)}
    for path in sorted((cfg.output / "final/embeddings").glob("*.npy")):
        y = np.load(path)
        fig, ax = plt.subplots(figsize=(6, 4))
        for label in sorted(set(labels)):
            idx = labels == label
            ax.scatter(
                y[idx, 0], y[idx, 1], s=5, alpha=0.7, color=colors[label], label=label, rasterized=True
            )
        ax.set(title=f"{cfg.dataset.name}: {path.stem}", xlabel="Dimension 1", ylabel="Dimension 2")
        ax.legend(fontsize=6, loc="center left", bbox_to_anchor=(1, 0.5), markerscale=2)
        save(
            fig,
            path.stem,
            f"{path.stem} on the declared final cohort. Colors are secondary labels; "
            "visual separation is not evidence of biological truth or generalization.",
        )
    cohort = read_json(cfg.output / "final/cohort_manifest.json")
    ratios = cohort["pca_explained_variance_ratio"]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(np.arange(1, len(ratios) + 1), np.cumsum(ratios), marker=".", markersize=3)
    ax.set(
        xlabel="PCA components",
        ylabel="Cumulative explained variance",
        ylim=(0, 1),
        title="Variance relative to preprocessed features",
    )
    save(
        fig,
        "pca_variance",
        "Variance retained by the common PCA input representation; first two "
        "components correspond to the displayed linear PCA baseline.",
    )
    table = read_json(cfg.output / "pilot/pilot_metrics.json")
    valid = [r for r in table if r["quality"] is not None]
    fig, ax = plt.subplots(figsize=(8, 3))
    xx = np.arange(len(valid))
    ax.bar(xx - 0.18, [r["trustworthiness"] for r in valid], width=0.35, label="Trustworthiness")
    ax.bar(xx + 0.18, [r["neighbor_recall"] for r in valid], width=0.35, label="Neighbor recall")
    ax.set_xticks(xx, [r["method"] for r in valid], rotation=30, ha="right")
    ax.set(ylim=(0, 1), ylabel="Pilot metric")
    ax.legend(fontsize=8)
    save(
        fig,
        "pilot_quality",
        "Metrics averaged across successful seeds on the same pilot cohort and reference.",
    )
    fig, ax = plt.subplots(figsize=(7, 3))
    for r in valid:
        ax.scatter(r["runtime_seconds"], r["quality"])
        ax.annotate(r["method"], (r["runtime_seconds"], r["quality"]), fontsize=7)
    ax.set(xscale="log", xlabel="Mean fit runtime (seconds; includes worker imports)", ylabel="Pilot quality")
    save(
        fig,
        "runtime_quality",
        "Observed runtime against declared label-free quality; higher quality and "
        "lower runtime are favorable. Warnings also affect eligibility.",
    )
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar([r["method"] for r in valid], [r["stability"] for r in valid])
    ax.tick_params(axis="x", rotation=30)
    ax.set(ylim=(0, 1.05), ylabel="Neighbor overlap across seeds")
    save(
        fig,
        "stability",
        "Mean pairwise seed neighborhood overlap; deterministic methods are defined as 1, "
        "not empirically replicated.",
    )
    decision = read_json(cfg.output / "selection/decisions.json")
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")
    t = ax.table(
        cellText=[[d["method"], d["decision"], d["pilot_evidence"]["status"]] for d in decision["decisions"]],
        colLabels=["Method", "Decision", "Pilot status"],
        loc="center",
    )
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    save(fig, "selection", "Candidate decisions; numerical reasons are recorded in decisions.json.")
    write_json(cfg.output / "figure_manifest.json", manifest)
    return manifest
