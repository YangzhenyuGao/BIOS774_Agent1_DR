"""Report figures from saved evidence only.

Palette: the eight validated categorical slots in fixed order (dataviz reference palette);
classes beyond eight are drawn in neutral gray. Class identity is never color-alone: every
embedding panel carries a legend and class names at class medians. Grids and axes are
recessive hairlines; text uses ink tokens, never series colors.
"""

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt

from .registry import METHOD_IDS
from .utils import checksum, read_json, write_json

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
EXCLUDED, SEVERE = "#c3c2b7", "#e1e0d9"
NAMES = {
    "pca": "PCA", "kernel_pca": "Kernel PCA", "mds": "MDS", "isomap": "Isomap", "lle": "LLE",
    "laplacian": "Laplacian", "diffusion_maps": "Diffusion maps", "gplvm": "GPLVM", "tsne": "t-SNE",
    "umap": "UMAP",
}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK, "axes.titlesize": 8.5,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
})


def _style(ax, grid="both"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)


def class_colors(labels):
    classes = sorted(set(labels) - {"unannotated"})
    colors = {c: (SLOTS[i] if i < len(SLOTS) else MUTED) for i, c in enumerate(classes)}
    colors["unannotated"] = MUTED
    return colors


def _scatter(ax, y, labels, colors, title, annotate=True):
    order = [c for c in colors if np.any(labels == c)]
    for c in order:
        idx = labels == c
        ax.scatter(y[idx, 0], y[idx, 1], s=3, alpha=0.65, color=colors[c], linewidths=0, rasterized=True,
                   label=c)
    if annotate and len(order) > 1:
        for c in order:
            idx = labels == c
            mx, my = np.median(y[idx, 0]), np.median(y[idx, 1])
            ax.text(mx, my, c, fontsize=5.5, color=INK, ha="center", va="center",
                    path_effects=[pe.withStroke(linewidth=2, foreground="white")])
    ax.set_title(title, loc="left")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_color(GRID)


def _legend(fig, colors, labels):
    present = [c for c in colors if np.any(labels == c)]
    if len(present) < 2:
        return
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=5, color=colors[c]) for c in present]
    fig.legend(handles, present, loc="lower center", ncol=min(len(present), 5), frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, -0.01))


def _decision_color(method, decisions):
    d = decisions[method]
    if d["decision"] == "retain":
        return SLOTS[0]
    return SEVERE if d["pilot_evidence"].get("severe_warning") else EXCLUDED


def pilot_figures(cfg, manifest, save):
    pilot = cfg.output / "pilot"
    labels = np.load(pilot / "labels.npy")
    colors = class_colors(labels)
    rows = {r["method"]: r for r in read_json(pilot / "pilot_metrics.json")}
    decision = read_json(cfg.output / "selection/decisions.json")
    decisions = {d["method"]: d for d in decision["decisions"]}

    fig, axes = plt.subplots(2, 5, figsize=(11, 5.3))
    for ax, mid in zip(axes.ravel(), METHOD_IDS):
        r = rows[mid]
        path = pilot / "method_runs" / mid / str(cfg.project.seed) / "embedding.npy"
        tag = "retained" if decisions[mid]["decision"] == "retain" else (
            "excluded: severe" if r["severe_warning"] else "excluded")
        q = f"Q {r['quality']:.3f}" if r["quality"] is not None else "Q n/a"
        title = f"{NAMES[mid]} - {tag}\n{q} · collapse {r['collapse_ratio'] or 0:.2f}"
        if path.exists():
            _scatter(ax, np.load(path), labels, colors, title, annotate=False)
        else:
            ax.set_axis_off()
            ax.set_title(f"{NAMES[mid]}: {r['status']}", loc="left")
    _legend(fig, colors, labels)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, pilot / "figures/pilot_embeddings.png",
         "All ten pilot embeddings on the identical input (seed-774 fit). Titles give the pre-registered "
         "quality Q, the collapse ratio and the decision; colors are secondary labels only.", "pilot")

    components = [("trustworthiness", "Trustworthiness (k=15)"), ("neighbor_recall", "Neighbor recall (k=15)"),
                  ("global_structure", "Global structure (distance Spearman)"),
                  ("stability", "Subsample stability"), ("quality", "Quality Q")]
    fig, axes = plt.subplots(1, 5, figsize=(11, 3.3), sharey=True)
    ypos = np.arange(len(METHOD_IDS))[::-1]
    for ax, (key, title) in zip(axes, components):
        values = [rows[m][key] for m in METHOD_IDS]
        shown = [v for v in values if v is not None]
        # Dot plot: positions, not bar lengths, carry the value, so a zoomed axis is not misleading.
        lo = max(0.0, min(shown) - 0.05) if shown else 0.0
        hi = min(1.0, max(shown) + 0.05) if shown else 1.0
        for y, m, v in zip(ypos, METHOD_IDS, values):
            ax.axhline(y, color=GRID, linewidth=0.6, zorder=1)
            if v is not None:
                ax.scatter(v, y, s=34, zorder=3, color=_decision_color(m, decisions), edgecolors="white",
                           linewidths=1)
        ax.set_xlim(lo, hi)
        ax.set_title(title, loc="left")
        _style(ax, "x")
    axes[0].set_yticks(ypos, [NAMES[m] for m in METHOD_IDS])
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=6, color=c) for c in
               (SLOTS[0], EXCLUDED, SEVERE)]
    fig.legend(handles, ["retained", "excluded", "excluded: severe diagnostic"], loc="lower center", ncol=3,
               frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, pilot / "figures/pilot_metrics.png",
         "Label-free pilot components on the common cohort; each panel's axis spans the observed range. "
         "Q = 0.35 T + 0.35 R + 0.20 G + 0.10 S.", "pilot")

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    frontier = set(decision["pareto_frontier"])
    for mid in METHOD_IDS:
        r = rows[mid]
        if r["quality"] is None:
            continue
        severe = r["severe_warning"]
        ax.scatter(r["runtime_seconds"], r["quality"], s=46, zorder=3,
                   color="white" if severe else _decision_color(mid, decisions),
                   edgecolors=MUTED if severe else "white", linewidths=1.2)
        ax.annotate(NAMES[mid] + (" *" if mid in frontier else ""), (r["runtime_seconds"], r["quality"]),
                    xytext=(4, 3), textcoords="offset points", fontsize=6.5, color=INK2)
    ax.set_xscale("log")
    ax.set(xlabel="Mean fit time on the pilot cohort (s, fit_transform only; log scale)", ylabel="Quality Q")
    _style(ax)
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=6, color=c,
                          markeredgecolor=e) for c, e in ((SLOTS[0], "white"), (EXCLUDED, "white"), ("white", MUTED))]
    ax.legend(handles, ["retained", "excluded", "severe diagnostic"], frameon=False, fontsize=6.5, loc="lower right")
    fig.tight_layout()
    save(fig, pilot / "figures/runtime_quality.png",
         "Fit-only runtime against quality. * marks the epsilon-Pareto frontier (a method is redundant only "
         "if another is at least as good and materially better or faster; see the decision rules).", "pilot")

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    vals = [rows[m]["stability"] or 0.0 for m in METHOD_IDS]
    ax.barh(ypos, vals, height=0.6, color=SLOTS[0], label="subsample stability (used in Q)")
    seed = [(y, rows[m]["seed_stability"]) for y, m in zip(ypos, METHOD_IDS) if rows[m]["seed_stability"] is not None]
    ax.scatter([s for _, s in seed], [y for y, _ in seed], marker="D", s=26, color=SLOTS[1], zorder=3,
               edgecolors="white", linewidths=1, label="seed stability (stochastic methods)")
    for y, m in zip(ypos, METHOD_IDS):
        if rows[m].get("seed_invariant"):
            ax.text(1.01, y, "seeds give identical output", fontsize=6, color=INK2, va="center")
    ax.set_yticks(ypos, [NAMES[m] for m in METHOD_IDS])
    ax.set(xlim=(0, 1), xlabel="Mean k-nearest-neighbor overlap (k=15) between refits")
    _style(ax, "x")
    ax.legend(frameon=False, fontsize=6.5, loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=2)
    fig.tight_layout()
    save(fig, pilot / "figures/stability.png",
         "Subsample stability compares refits on overlapping 80% subsamples and applies to every method; "
         "seed stability compares repeated seeds on the same cohort.", "pilot")

    sens = decision["sensitivity"]["variants"]
    fig, ax = plt.subplots(figsize=(8.2, 0.32 * len(sens) + 1.0))
    for i, v in enumerate(sens):
        for j, m in enumerate(METHOD_IDS):
            chosen = m in v["selected"]
            ax.add_patch(plt.Rectangle((j + 0.08, i + 0.08), 0.84, 0.84, color=SLOTS[0] if chosen else "#f3f2ee"))
    ax.set_xlim(0, len(METHOD_IDS))
    ax.set_ylim(len(sens), 0)
    ax.set_xticks(np.arange(len(METHOD_IDS)) + 0.5, [NAMES[m] for m in METHOD_IDS], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(sens)) + 0.5, [v["variant"] for v in sens], fontsize=6.5)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    save(fig, cfg.output / "selection/figures/sensitivity.png",
         "Shortlist under alternative rule variants on the same saved pilot evidence (filled = retained). "
         "Only the first row determined the analysis; the others test robustness.", "selection")


def final_figures(cfg, manifest, save):
    final = cfg.output / "final"
    labels = np.load(final / "labels.npy")
    colors = class_colors(labels)
    metrics = {r["method"]: r for r in read_json(final / "final_metrics.json")}
    methods = list(metrics)
    cols = 2 if len(methods) > 1 else 1
    rows = int(np.ceil(len(methods) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(8.4, 3.7 * rows), squeeze=False)
    for ax in axes.ravel()[len(methods):]:
        ax.set_axis_off()
    for ax, mid in zip(axes.ravel(), methods):
        m = metrics[mid]
        y = np.load(final / "embeddings" / f"{mid}.npy")
        _scatter(ax, y, labels, colors,
                 f"{NAMES[mid]}   T {m['trustworthiness']:.3f} · R {m['neighbor_recall']:.3f} · "
                 f"G {m['global_structure']:.3f}")
    _legend(fig, colors, labels)
    fig.tight_layout(rect=(0, 0.07 if rows > 1 else 0.12, 1, 1))
    save(fig, final / "figures/final_embeddings.png",
         f"Final embeddings of the {len(methods)} shortlisted methods on the declared final cohort "
         f"(n={len(labels):,}). Class names sit at class medians; colors are secondary labels, and visual "
         "separation is not evidence of biological or histological truth.", "final")
    for mid in methods:
        fig, ax = plt.subplots(figsize=(4.6, 3.8))
        _scatter(ax, np.load(final / "embeddings" / f"{mid}.npy"), labels, colors, NAMES[mid])
        fig.tight_layout()
        save(fig, final / f"figures/{mid}.png", f"{NAMES[mid]} final embedding (n={len(labels):,}).", "final-single")

    cohort = read_json(final / "cohort_manifest.json")
    ratios = np.cumsum(cohort["pca_explained_variance_ratio"])
    fig, ax = plt.subplots(figsize=(4.8, 2.6))
    ax.plot(np.arange(1, len(ratios) + 1), ratios, color=SLOTS[0], linewidth=1.5)
    ax.scatter([2], [ratios[1]], s=30, color=SLOTS[0], edgecolors="white", zorder=3)
    ax.annotate(f"2 PCs: {ratios[1]:.1%}", (2, ratios[1]), xytext=(8, -2), textcoords="offset points",
                fontsize=6.5, color=INK2)
    ax.annotate(f"{len(ratios)} PCs: {ratios[-1]:.1%}", (len(ratios), ratios[-1]), xytext=(-40, 6),
                textcoords="offset points", fontsize=6.5, color=INK2)
    ax.set(xlabel="PCA components of the preprocessed features", ylabel="Cumulative explained variance", ylim=(0, 1))
    _style(ax)
    fig.tight_layout()
    save(fig, final / "figures/pca_variance.png",
         "Variance of the preprocessed feature space retained by the common PCA representation (final cohort).",
         "final")

    plan = read_json(final / "size_plan.json")
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 2.9))
    for i, mid in enumerate(methods):
        pts = sorted((r["n"], r["seconds"], r["peak_memory_mb"]) for r in plan["probe"]
                     if r["method"] == mid and r["status"] == "success")
        if not pts:
            continue
        color = SLOTS[i % len(SLOTS)]
        grid = np.array(sorted({p["n"] for p in plan["projections"]} | {p[0] for p in pts}), dtype=float)
        proj = [next((p for p in plan["projections"] if p["n"] == n), None) for n in grid]
        for ax, j, key in ((axes[0], 1, "seconds"), (axes[1], 2, "memory_mb")):
            ax.plot([p[0] for p in pts], [p[j] for p in pts], "o", color=color, markersize=4, zorder=3)
            ns = [n for n, p in zip(grid, proj) if p]
            ax.plot(ns, [p["per_method"][mid][key] for p in proj if p], color=color, linewidth=1.2, alpha=0.8,
                    label=NAMES[mid])
    declared = plan["declared_final_size"]
    for ax, title, unit in ((axes[0], "Fit time (projected lines, measured points)", "seconds"),
                            (axes[1], "Peak memory of the fit process", "MB")):
        ax.set(xscale="log", yscale="log", xlabel="Cohort size n", ylabel=unit)
        ax.axvline(declared, color=MUTED, linewidth=0.8)
        ax.text(declared, 0.97, f" declared n={declared:,}", transform=ax.get_xaxis_transform(),
                fontsize=6, color=INK2, va="top")
        ax.set_title(title, loc="left")
        _style(ax)
    axes[0].legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    save(fig, final / "figures/scaling.png",
         "Scaling probe of the shortlisted methods and the conservative projections used to declare the final "
         f"cohort size ({plan['binding_constraint']}).", "final")


def figures(cfg):
    manifest = []

    def save(fig, path, caption, section):
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        manifest.append({"path": str(path), "sha256": checksum(path), "caption": caption, "section": section,
                         "name": path.stem})

    pilot_figures(cfg, manifest, save)
    final_figures(cfg, manifest, save)
    write_json(cfg.output / "figure_manifest.json", manifest)
    return manifest

