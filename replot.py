#!/usr/bin/env python3
"""Standalone replotting script for GBM vs Heston comparison figures.

Run this after the main experiment to regenerate comparison plots with
different styling — no need to rerun the full pipeline.

Usage:
    python replot.py                          # default paths
    python replot.py --data outputs/comparison/comparison_data.pt
    python replot.py --out  outputs/comparison_v2

Edit the STYLE dict below to adjust figures to your liking.
"""
import argparse
import os
import json

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════════════════════════
# STYLE CONFIG — edit these to adjust every figure without rerunning
# ═══════════════════════════════════════════════════════════════════════
STYLE = {
    # ── Global ──
    "dpi": 150,
    "font_size": 11,            # matplotlib base font size
    "title_size": 13,
    "label_size": 11,
    "tick_size": 10,
    "legend_size": 10,
    "grid": True,               # show grid lines
    "grid_alpha": 0.2,
    "grid_axis": "both",        # "both", "x", "y"
    "tight_layout": True,

    # ── Histogram settings ──
    "hist_bins": 80,
    "hist_edgecolor": "white",
    "hist_edgewidth": 0.3,
    "hist_zero_line": True,     # vertical line at P&L = 0

    # ── Per-model histogram (GBM vs Heston for each model) ──
    "per_model_hist_figsize": (8, 5),
    "per_model_hist_gbm_color": "#00CED1",
    "per_model_hist_gbm_alpha": 0.55,
    "per_model_hist_heston_color": "#9B59B6",
    "per_model_hist_heston_alpha": 0.55,

    # ── Per-model violin ──
    "per_model_violin_figsize": (6, 5),
    "per_model_violin_gbm_color": "#00CED1",
    "per_model_violin_heston_color": "#9B59B6",
    "per_model_violin_alpha": 0.7,
    "per_model_violin_show_means": True,
    "per_model_violin_show_medians": True,

    # ── Cross-model histogram (all models under one regime) ──
    "cross_hist_figsize": (9, 5),
    "cross_hist_alpha": 0.45,
    "cross_hist_model_colors": {
        "FNN-5":  "#2196F3",
        "LSTM-5": "#FF9800",
        "DBSDE":  "#4CAF50",
    },

    # ── Cross-model violin ──
    "cross_violin_figsize": (8, 5),
    "cross_violin_alpha": 0.7,
    "cross_violin_model_colors": {
        "FNN-5":  "#2196F3",
        "LSTM-5": "#FF9800",
        "DBSDE":  "#4CAF50",
    },
    "cross_violin_show_means": True,
    "cross_violin_show_medians": True,

    # ── Stacked overlay histogram (old combined view) ──
    "overlay_hist_figsize": (10, 5),    # per-subplot height
    "overlay_hist_heston_color": "#7B2D8E",
    "overlay_hist_heston_alpha": 0.7,
    "overlay_hist_gbm_color": "#2EC4B6",
    "overlay_hist_gbm_alpha": 0.5,
    "overlay_hist_bins": 70,

    # ── Stacked violin (old combined view) ──
    "overlay_violin_figsize": (8, 6),   # per-subplot height
    "overlay_violin_heston_color": "#E88AED",
    "overlay_violin_gbm_color": "#A8E6E2",
    "overlay_violin_alpha": 0.75,

    # ── Price path comparison ──
    "price_figsize": (14, 5),
    "price_n_sample": 20,
    "price_path_alpha": 0.3,
    "price_path_linewidth": 0.5,

    # ── Metric bar chart ──
    "bars_figsize_per_metric": 5,       # width per metric subplot
    "bars_height": 5,
    "bars_width": 0.35,
    "bars_alpha": 0.85,
    "bars_capsize": 3,
}


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _apply_global_style():
    plt.rcParams.update({
        "font.size": STYLE["font_size"],
        "axes.titlesize": STYLE["title_size"],
        "axes.labelsize": STYLE["label_size"],
        "xtick.labelsize": STYLE["tick_size"],
        "ytick.labelsize": STYLE["tick_size"],
        "legend.fontsize": STYLE["legend_size"],
    })


def _grid(ax, axis=None):
    if STYLE["grid"]:
        ax.grid(True, alpha=STYLE["grid_alpha"],
                axis=axis or STYLE["grid_axis"])


def _finish(fig, path):
    if STYLE["tight_layout"]:
        fig.tight_layout()
    fig.savefig(path, dpi=STYLE["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def _pnl(comp, model):
    vt = comp[model]["V_T"]
    ht = comp[model]["H_tilde"]
    if torch.is_tensor(vt):
        return (vt - ht).cpu().numpy()
    return np.array(vt) - np.array(ht)


# ═══════════════════════════════════════════════════════════════════════
# Plot functions
# ═══════════════════════════════════════════════════════════════════════

def replot_per_model_hist(gbm_comp, heston_comp, out):
    """One histogram figure per model: GBM vs Heston P&L."""
    s = STYLE
    models = sorted(set(gbm_comp) & set(heston_comp))
    for model in models:
        g = _pnl(gbm_comp, model)
        h = _pnl(heston_comp, model)
        bins = np.linspace(min(g.min(), h.min()), max(g.max(), h.max()),
                           s["hist_bins"])

        fig, ax = plt.subplots(figsize=s["per_model_hist_figsize"])
        ax.hist(g, bins=bins, alpha=s["per_model_hist_gbm_alpha"],
                color=s["per_model_hist_gbm_color"],
                label="GBM (Const. $\\sigma$)",
                edgecolor=s["hist_edgecolor"], linewidth=s["hist_edgewidth"])
        ax.hist(h, bins=bins, alpha=s["per_model_hist_heston_alpha"],
                color=s["per_model_hist_heston_color"],
                label="Heston (Stoch. $\\sigma$)",
                edgecolor=s["hist_edgecolor"], linewidth=s["hist_edgewidth"])
        if s["hist_zero_line"]:
            ax.axvline(0, color="black", ls="--", lw=0.8, alpha=0.5)
        ax.set_xlabel("Profit / Loss")
        ax.set_ylabel("Frequency")
        ax.set_title(f"{model}: P&L — Constant vs Stochastic Volatility")
        ax.legend(framealpha=0.9)
        _grid(ax)
        safe = model.replace(" ", "_").replace("-", "_").lower()
        _finish(fig, os.path.join(out, f"pnl_hist_{safe}_gbm_vs_heston.png"))


def replot_per_model_violin(gbm_comp, heston_comp, out):
    """One violin figure per model: GBM vs Heston P&L."""
    s = STYLE
    models = sorted(set(gbm_comp) & set(heston_comp))
    for model in models:
        g = _pnl(gbm_comp, model)
        h = _pnl(heston_comp, model)

        fig, ax = plt.subplots(figsize=s["per_model_violin_figsize"])
        parts = ax.violinplot(
            [g, h], positions=[1, 2],
            showmeans=s["per_model_violin_show_means"],
            showmedians=s["per_model_violin_show_medians"],
            showextrema=True,
        )
        colors = [s["per_model_violin_gbm_color"],
                  s["per_model_violin_heston_color"]]
        for pc, c in zip(parts["bodies"], colors):
            pc.set_facecolor(c); pc.set_edgecolor(c)
            pc.set_alpha(s["per_model_violin_alpha"])
        for k in ("cmeans", "cmedians", "cmins", "cmaxes", "cbars"):
            if k in parts:
                parts[k].set_color("#333"); parts[k].set_linewidth(1.2)
        ax.axhline(0, color="black", ls="--", lw=0.8, alpha=0.4)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["GBM (Const. $\\sigma$)",
                            "Heston (Stoch. $\\sigma$)"])
        ax.set_ylabel("Profit / Loss")
        ax.set_title(f"{model}: P&L — Constant vs Stochastic Volatility")
        _grid(ax, axis="y")
        safe = model.replace(" ", "_").replace("-", "_").lower()
        _finish(fig, os.path.join(out, f"pnl_violin_{safe}_gbm_vs_heston.png"))


def replot_cross_model_hist(gbm_comp, heston_comp, out):
    """All models overlaid in one histogram — one figure per regime."""
    s = STYLE
    mc = s["cross_hist_model_colors"]
    for regime, comp, label in [
        ("gbm", gbm_comp, "GBM (Const. $\\sigma$)"),
        ("heston", heston_comp, "Heston (Stoch. $\\sigma$)"),
    ]:
        if not comp:
            continue
        pnls = {m: _pnl(comp, m) for m in sorted(comp)}
        if not pnls:
            continue
        lo = min(p.min() for p in pnls.values())
        hi = max(p.max() for p in pnls.values())
        bins = np.linspace(lo, hi, s["hist_bins"])

        fig, ax = plt.subplots(figsize=s["cross_hist_figsize"])
        for m, p in pnls.items():
            ax.hist(p, bins=bins, alpha=s["cross_hist_alpha"],
                    color=mc.get(m, "#888"), label=m,
                    edgecolor=s["hist_edgecolor"], linewidth=s["hist_edgewidth"])
        if s["hist_zero_line"]:
            ax.axvline(0, color="black", ls="--", lw=0.8, alpha=0.5)
        ax.set_xlabel("Profit / Loss")
        ax.set_ylabel("Frequency")
        ax.set_title(f"All Models P&L — {label}")
        ax.legend(framealpha=0.9)
        _grid(ax)
        _finish(fig, os.path.join(out, f"pnl_hist_all_models_{regime}.png"))


def replot_cross_model_violin(gbm_comp, heston_comp, out):
    """All models as violins — one figure per regime."""
    s = STYLE
    mc = s["cross_violin_model_colors"]
    for regime, comp, label in [
        ("gbm", gbm_comp, "GBM (Const. $\\sigma$)"),
        ("heston", heston_comp, "Heston (Stoch. $\\sigma$)"),
    ]:
        if not comp:
            continue
        models = sorted(comp)
        if not models:
            continue
        data = [_pnl(comp, m) for m in models]
        pos = list(range(1, len(models) + 1))

        fig, ax = plt.subplots(figsize=s["cross_violin_figsize"])
        parts = ax.violinplot(
            data, positions=pos,
            showmeans=s["cross_violin_show_means"],
            showmedians=s["cross_violin_show_medians"],
            showextrema=True,
        )
        for pc, m in zip(parts["bodies"], models):
            c = mc.get(m, "#888")
            pc.set_facecolor(c); pc.set_edgecolor(c)
            pc.set_alpha(s["cross_violin_alpha"])
        for k in ("cmeans", "cmedians", "cmins", "cmaxes", "cbars"):
            if k in parts:
                parts[k].set_color("#333"); parts[k].set_linewidth(1.2)
        ax.axhline(0, color="black", ls="--", lw=0.8, alpha=0.4)
        ax.set_xticks(pos)
        ax.set_xticklabels(models)
        ax.set_ylabel("Profit / Loss")
        ax.set_title(f"All Models P&L — {label}")
        _grid(ax, axis="y")
        _finish(fig, os.path.join(out, f"pnl_violin_all_models_{regime}.png"))


def replot_overlay_hist(gbm_comp, heston_comp, out):
    """Stacked per-model subplots: Heston vs GBM histograms."""
    s = STYLE
    models = sorted(set(gbm_comp) & set(heston_comp))
    if not models:
        return
    n = len(models)
    w, h = s["overlay_hist_figsize"]
    fig, axes = plt.subplots(n, 1, figsize=(w, h * n))
    if n == 1:
        axes = [axes]
    for i, model in enumerate(models):
        ax = axes[i]
        g = _pnl(gbm_comp, model)
        hp = _pnl(heston_comp, model)
        lo = min(g.min(), hp.min())
        hi_ = max(g.max(), hp.max())
        bins = np.linspace(lo, hi_, s["overlay_hist_bins"])
        ax.hist(hp, bins=bins, alpha=s["overlay_hist_heston_alpha"],
                color=s["overlay_hist_heston_color"],
                edgecolor="black", linewidth=0.3,
                label="Heston (Stoch. $\\sigma$)")
        ax.hist(g, bins=bins, alpha=s["overlay_hist_gbm_alpha"],
                color=s["overlay_hist_gbm_color"],
                edgecolor="black", linewidth=0.3,
                label="GBM (Const. $\\sigma$)")
        ax.set_xlabel("Profit/Loss")
        ax.set_ylabel("Frequency")
        ax.set_title(f"{model}: P&L with Different Volatility Models")
        ax.legend()
        _grid(ax)
    _finish(fig, os.path.join(out, "pnl_gbm_vs_heston.png"))


def replot_overlay_violin(gbm_comp, heston_comp, out):
    """Stacked per-model subplots: Heston vs GBM violins."""
    s = STYLE
    models = sorted(set(gbm_comp) & set(heston_comp))
    if not models:
        return
    n = len(models)
    w, h = s["overlay_violin_figsize"]
    fig, axes = plt.subplots(n, 1, figsize=(w, h * n))
    if n == 1:
        axes = [axes]
    for i, model in enumerate(models):
        ax = axes[i]
        g = _pnl(gbm_comp, model)
        hp = _pnl(heston_comp, model)
        parts = ax.violinplot([hp, g], positions=[1, 2],
                              showmeans=False, showmedians=True,
                              showextrema=True)
        colors = [s["overlay_violin_heston_color"],
                  s["overlay_violin_gbm_color"]]
        for pc, c in zip(parts["bodies"], colors):
            pc.set_facecolor(c); pc.set_edgecolor(c)
            pc.set_alpha(s["overlay_violin_alpha"])
        for k in ("cmedians", "cmins", "cmaxes", "cbars"):
            if k in parts:
                parts[k].set_color("#333"); parts[k].set_linewidth(1.5)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Heston (Stoch. $\\sigma$)",
                            "GBM (Const. $\\sigma$)"])
        ax.set_ylabel("Profit/Loss")
        ax.set_title(f"{model}: P&L with Different Volatility Models")
        _grid(ax, axis="y")
    _finish(fig, os.path.join(out, "pnl_violin_gbm_vs_heston.png"))


def replot_price_paths(data, out):
    """Price path comparison: Heston vs GBM."""
    s = STYLE
    S_h = data["S_tilde_heston_sample"][:, :, 0].numpy()
    S_g = data["S_tilde_gbm_sample"][:, :, 0].numpy()
    t = data["time_grid"].numpy()
    n = min(s["price_n_sample"], S_h.shape[0])

    fig, axes = plt.subplots(1, 2, figsize=s["price_figsize"])
    for j in range(n):
        axes[0].plot(t, S_h[j], alpha=s["price_path_alpha"],
                     lw=s["price_path_linewidth"], color="C0")
        axes[1].plot(t, S_g[j], alpha=s["price_path_alpha"],
                     lw=s["price_path_linewidth"], color="C1")
    axes[0].plot(t, S_h[:n].mean(axis=0), "k-", lw=2, label="Mean")
    axes[0].set_title("Heston: Asset 1 Price Paths")
    axes[0].set_xlabel("Time $t$"); axes[0].set_ylabel("$\\tilde{S}_t$")
    axes[0].legend(); _grid(axes[0])

    axes[1].plot(t, S_g[:n].mean(axis=0), "k-", lw=2, label="Mean")
    axes[1].set_title("GBM: Asset 1 Price Paths")
    axes[1].set_xlabel("Time $t$"); axes[1].set_ylabel("$\\tilde{S}_t$")
    axes[1].legend(); _grid(axes[1])

    fig.suptitle("Price Path Comparison: Heston vs GBM", fontsize=s["title_size"])
    _finish(fig, os.path.join(out, "heston_vs_gbm_paths.png"))


def replot_metric_bars(metrics_path, out):
    """Grouped bar chart comparing GBM and Heston metrics."""
    if not os.path.exists(metrics_path):
        print(f"  Skipping metric bars — {metrics_path} not found")
        return
    with open(metrics_path) as f:
        combined = json.load(f)
    gbm_agg = combined.get("gbm", {})
    heston_agg = combined.get("heston", {})
    models = sorted(set(gbm_agg) & set(heston_agg))
    if not models:
        return

    s = STYLE
    metric_names = ["CVaR95_shortfall", "MSE", "MAE", "mean_shortfall"]
    labels = ["CVaR95", "MSE", "MAE", "Mean Shortfall"]

    fig, axes = plt.subplots(1, len(metric_names),
                             figsize=(s["bars_figsize_per_metric"] * len(metric_names),
                                      s["bars_height"]))
    for idx, (mn, lbl) in enumerate(zip(metric_names, labels)):
        ax = axes[idx]
        x = np.arange(len(models))
        gv, ge, hv, he = [], [], [], []
        for m in models:
            g = gbm_agg[m].get(mn, {})
            h = heston_agg[m].get(mn, {})
            gv.append(g.get("mean", 0) if isinstance(g, dict) else g)
            ge.append(g.get("std", 0) if isinstance(g, dict) else 0)
            hv.append(h.get("mean", 0) if isinstance(h, dict) else h)
            he.append(h.get("std", 0) if isinstance(h, dict) else 0)
        ax.bar(x - s["bars_width"] / 2, gv, s["bars_width"], yerr=ge,
               label="GBM", capsize=s["bars_capsize"], alpha=s["bars_alpha"])
        ax.bar(x + s["bars_width"] / 2, hv, s["bars_width"], yerr=he,
               label="Heston", capsize=s["bars_capsize"], alpha=s["bars_alpha"])
        ax.set_xticks(x); ax.set_xticklabels(models)
        ax.set_title(lbl); ax.legend()
        _grid(ax, axis="y")
    fig.suptitle("GBM vs Heston: Model Comparison", fontsize=s["title_size"])
    _finish(fig, os.path.join(out, "gbm_vs_heston_bars.png"))


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Replot comparison figures")
    p.add_argument("--data", default="outputs/comparison/comparison_data.pt",
                   help="Path to saved comparison_data.pt")
    p.add_argument("--metrics", default="outputs/metrics_summary.json",
                   help="Path to metrics_summary.json")
    p.add_argument("--out", default="outputs/comparison",
                   help="Output directory for figures")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    _apply_global_style()

    print(f"Loading data from {args.data} ...")
    data = torch.load(args.data, map_location="cpu", weights_only=False)
    gbm_comp = data.get("gbm_comp", {})
    heston_comp = data.get("heston_comp", {})

    print(f"Generating plots to {args.out}/ ...\n")

    if gbm_comp and heston_comp:
        replot_per_model_hist(gbm_comp, heston_comp, args.out)
        replot_per_model_violin(gbm_comp, heston_comp, args.out)
        replot_cross_model_hist(gbm_comp, heston_comp, args.out)
        replot_cross_model_violin(gbm_comp, heston_comp, args.out)
        replot_overlay_hist(gbm_comp, heston_comp, args.out)
        replot_overlay_violin(gbm_comp, heston_comp, args.out)
    else:
        print("  No P&L comparison data found — skipping P&L plots")

    if "S_tilde_gbm_sample" in data and "S_tilde_heston_sample" in data:
        replot_price_paths(data, args.out)

    replot_metric_bars(args.metrics, args.out)

    print("\nDone. Edit STYLE dict in replot.py and rerun to adjust figures.")


if __name__ == "__main__":
    main()
