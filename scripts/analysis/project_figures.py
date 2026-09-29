"""Figures for the project report, drawn read-only from the frozen dataset evidence.

Usage: python scripts/analysis/project_figures.py   (writes reports/source/figures/)
"""

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agent1_dr.config import ROOT
from agent1_dr.freeze import verify_frozen
from agent1_dr.registry import METHOD_IDS
from agent1_dr.utils import read_json
from agent1_dr.visualization import EXCLUDED, GRID, NAMES, SLOTS, _scatter, _style, class_colors

TITLES = {"pbmc3k": "PBMC3k (count matrix)", "pathmnist": "PathMNIST (8-bit pixels)"}
OUT = ROOT / "reports/source/figures"


def pilot_quality():
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9), sharey=True)
    ypos = np.arange(len(METHOD_IDS))[::-1]
    for ax, name in zip(axes, ["pbmc3k", "pathmnist"]):
        rows = {r["method"]: r for r in read_json(ROOT / f"outputs/{name}/pilot/pilot_metrics.json")}
        decisions = {d["method"]: d for d in read_json(ROOT / f"outputs/{name}/selection/decisions.json")["decisions"]}
        for y, m in zip(ypos, METHOD_IDS):
            ax.axhline(y, color=GRID, linewidth=0.6, zorder=1)
            q = rows[m]["quality"]
            if q is None:
                continue
            if rows[m]["severe_warning"]:
                ax.scatter(q, y, s=34, facecolors="white", edgecolors="#898781", linewidths=1.1, zorder=3)
            else:
                color = SLOTS[0] if decisions[m]["decision"] == "retain" else EXCLUDED
                ax.scatter(q, y, s=34, color=color, edgecolors="white", linewidths=1, zorder=3)
        ax.set_title(TITLES[name], loc="left")
        ax.set_xlabel("Pilot quality Q (pre-registered)")
        ax.set_yticks(ypos, [NAMES[m] for m in METHOD_IDS], fontsize=7)
        _style(ax, "x")
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=6, color=SLOTS[0]),
               plt.Line2D([], [], marker="o", linestyle="", markersize=6, color=EXCLUDED),
               plt.Line2D([], [], marker="o", linestyle="", markersize=6, markerfacecolor="white",
                          markeredgecolor="#898781")]
    fig.legend(handles, ["retained", "excluded", "severe diagnostic (ineligible)"], loc="lower center", ncol=3,
               frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT / "pilot_quality_both.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def final_embeddings():
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 6.2))
    for row, name in enumerate(["pbmc3k", "pathmnist"]):
        final = ROOT / f"outputs/{name}/final"
        labels = np.load(final / "labels.npy")
        colors = class_colors(labels)
        metrics = {r["method"]: r for r in read_json(final / "final_metrics.json")}
        for col, mid in enumerate(["mds", "tsne"]):
            m = metrics[mid]
            y = np.load(final / "embeddings" / f"{mid}.npy")
            _scatter(axes[row, col], y, labels, colors,
                     f"{TITLES[name].split(' (')[0]} · {NAMES[mid]} (n={len(labels):,})\n"
                     f"T {m['trustworthiness']:.3f} · R {m['neighbor_recall']:.3f} · G {m['global_structure']:.3f}")
        present = [c for c in colors if np.any(labels == c)]
        handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=4, color=colors[c]) for c in present]
        axes[row, 1].legend(handles, present, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False,
                            fontsize=5.8)
    fig.tight_layout()
    fig.savefig(OUT / "final_embeddings_both.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    verify_frozen()
    OUT.mkdir(parents=True, exist_ok=True)
    pilot_quality()
    final_embeddings()
    print("figures written to", OUT)
