"""2D plotting functions for hedging diagnostics.

Generates PNG plots inspired by Guo-Langrene-Wu (2023):
loss curves, error histograms, CVaR curves, scatter plots, daily P/L, drawdown.
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.training.losses import terminal_error, shortfall, cvar
from src.utils.io import ensure_dir


def generate_all_plots(model_name, seed, V_T, H_tilde, V_path, Z_intrinsic,
                       train_losses, val_losses, output_dir="outputs/plots"):
    """Generate all 2D diagnostic plots for a single model run.

    Args:
        model_name: str, e.g. "FNN", "LSTM", "DBSDE"
        seed: int
        V_T: [n] terminal portfolio value
        H_tilde: [n] discounted payoff
        V_path: [n, N+1] portfolio value path
        Z_intrinsic: [n, N+1] intrinsic process
        train_losses, val_losses: lists of per-epoch values
        output_dir: directory for saving plots
    """
    ensure_dir(output_dir)
    prefix = f"{model_name}_seed{seed}"

    e = terminal_error(V_T, H_tilde).cpu().numpy()
    s = shortfall(V_T, H_tilde).cpu().numpy()
    V_T_np = V_T.cpu().numpy()
    H_np = H_tilde.cpu().numpy()

    # (1) Loss curves
    plot_loss_curves(train_losses, val_losses, prefix, output_dir)

    # (2) Terminal error histogram
    plot_histogram(e, f"{prefix} Terminal Error $e_T$",
                   "Terminal Error", f"{prefix}_error_hist.png", output_dir)

    # (3) Shortfall histogram
    plot_histogram(s[s > 0], f"{prefix} Shortfall $s$",
                   "Shortfall", f"{prefix}_shortfall_hist.png", output_dir)

    # (4) Overlay: terminal error vs intrinsic gap
    if V_path is not None and Z_intrinsic is not None:
        gap = (V_path - Z_intrinsic).cpu().numpy()
        mean_gap = gap.mean(axis=1)
        plot_overlay_hist(e, mean_gap, prefix, output_dir)

    # (5) CVaR curve
    plot_cvar_curve(V_T, H_tilde, prefix, output_dir)

    # (6) Scatter: V_T vs H_tilde
    plot_scatter(V_T_np, H_np, prefix, output_dir)

    # (7) Daily P/L
    if V_path is not None:
        dPL = (V_path[:, 1:] - V_path[:, :-1]).cpu().numpy()
        plot_daily_pnl(dPL, prefix, output_dir)

    # (8) Drawdown distribution
    if V_path is not None:
        plot_drawdown(V_path, prefix, output_dir)

    plt.close("all")


def plot_loss_curves(train_losses, val_losses, prefix, output_dir):
    """(1) Train vs validation loss curves."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_losses, label="Train Loss", linewidth=1.5)
    ax.plot(val_losses, label="Val CVaR95(shortfall)", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss / Metric")
    ax.set_title(f"{prefix}: Training Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{prefix}_loss_curves.png"), dpi=150)
    plt.close(fig)


def plot_histogram(data, title, xlabel, filename, output_dir, bins=60):
    """Generic histogram plot."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(data, bins=bins, density=True, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.axvline(np.mean(data), color="red", linestyle="--", label=f"Mean={np.mean(data):.4f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close(fig)


def plot_overlay_hist(error, gap, prefix, output_dir):
    """(4) Overlay histogram: terminal error vs mean intrinsic gap."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(error, bins=60, density=True, alpha=0.5, label="Terminal Error $e_T$")
    ax.hist(gap, bins=60, density=True, alpha=0.5, label="Mean Intrinsic Gap")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_title(f"{prefix}: Error vs Intrinsic Gap")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{prefix}_overlay_hist.png"), dpi=150)
    plt.close(fig)


def plot_cvar_curve(V_T, H_tilde, prefix, output_dir):
    """(5) CVaR curve: CVaR_q(shortfall) for q in [0.80, 0.99]."""
    s = shortfall(V_T, H_tilde)
    qs = np.linspace(0.80, 0.99, 20)
    cvars = [cvar(s, q=q).item() for q in qs]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(qs, cvars, "o-", linewidth=1.5)
    for q_mark in [0.90, 0.95, 0.99]:
        c = cvar(s, q=q_mark).item()
        ax.axvline(q_mark, color="gray", linestyle=":", alpha=0.5)
        ax.annotate(f"q={q_mark}\nCVaR={c:.4f}",
                    xy=(q_mark, c), fontsize=8)
    ax.set_xlabel("Quantile $q$")
    ax.set_ylabel("CVaR$_q$(shortfall)")
    ax.set_title(f"{prefix}: CVaR Curve")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{prefix}_cvar_curve.png"), dpi=150)
    plt.close(fig)


def plot_scatter(V_T, H_tilde, prefix, output_dir):
    """(6) Scatter: V_T vs H_tilde with 45-degree line."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(H_tilde, V_T, s=2, alpha=0.3)
    lo = min(H_tilde.min(), V_T.min())
    hi = max(H_tilde.max(), V_T.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="$V_T = H$")
    ax.set_xlabel("$\\tilde{H}$ (Discounted Payoff)")
    ax.set_ylabel("$V_T$ (Portfolio Value)")
    ax.set_title(f"{prefix}: Hedge vs Payoff")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{prefix}_scatter.png"), dpi=150)
    plt.close(fig)


def plot_daily_pnl(dPL, prefix, output_dir):
    """(7) Daily P/L: mean +/- 1 std band over time."""
    mean_pnl = dPL.mean(axis=0)
    std_pnl = dPL.std(axis=0)
    steps = np.arange(len(mean_pnl))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, mean_pnl, label="Mean dPL", linewidth=1.2)
    ax.fill_between(steps, mean_pnl - std_pnl, mean_pnl + std_pnl,
                    alpha=0.2, label="$\\pm 1$ std")
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Time Step $k$")
    ax.set_ylabel("Daily P/L")
    ax.set_title(f"{prefix}: Daily P/L")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{prefix}_daily_pnl.png"), dpi=150)
    plt.close(fig)


def plot_drawdown(V_path, prefix, output_dir):
    """(8) Max drawdown distribution."""
    V_np = V_path.cpu().numpy()
    running_max = np.maximum.accumulate(V_np, axis=1)
    drawdown = (running_max - V_np).max(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(drawdown, bins=60, density=True, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.axvline(np.mean(drawdown), color="red", linestyle="--",
               label=f"Mean={np.mean(drawdown):.4f}")
    ax.set_xlabel("Max Drawdown")
    ax.set_ylabel("Density")
    ax.set_title(f"{prefix}: Max Drawdown Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{prefix}_drawdown.png"), dpi=150)
    plt.close(fig)


def plot_substeps_convergence(substeps_results, output_dir="outputs/plots"):
    """(9) DBSDE substeps convergence plot.

    Args:
        substeps_results: list of dicts with keys 'substeps', 'val_MSE', 'val_CVaR95'
    """
    ensure_dir(output_dir)
    subs = [r["substeps"] for r in substeps_results]
    mses = [r["val_MSE"] for r in substeps_results]
    cvars = [r["val_CVaR95"] for r in substeps_results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(subs, mses, "o-", linewidth=1.5)
    axes[0].set_xlabel("Substeps")
    axes[0].set_ylabel("Validation MSE")
    axes[0].set_title("DBSDE: MSE vs Substeps")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(subs, cvars, "o-", linewidth=1.5, color="orange")
    axes[1].set_xlabel("Substeps")
    axes[1].set_ylabel("Validation CVaR95(shortfall)")
    axes[1].set_title("DBSDE: CVaR95 vs Substeps")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "substeps_convergence.png"), dpi=150)
    plt.close(fig)


def plot_summary_table(all_metrics, output_dir="outputs/plots"):
    """Generate a summary comparison table as an image."""
    ensure_dir(output_dir)
    models = list(all_metrics.keys())
    metric_names = ["MAE", "MSE", "R2", "mean_shortfall",
                    "CVaR95_shortfall", "P_negative_error"]

    cell_text = []
    for model in models:
        row = []
        m = all_metrics[model]
        for mn in metric_names:
            if isinstance(m.get(mn), dict):
                row.append(f"{m[mn]['mean']:.4f} +/- {m[mn]['std']:.4f}")
            else:
                row.append(f"{m.get(mn, 0):.4f}")
        cell_text.append(row)

    fig, ax = plt.subplots(figsize=(14, 2 + len(models) * 0.5))
    ax.axis("off")
    table = ax.table(cellText=cell_text, rowLabels=models,
                     colLabels=metric_names, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.4)
    ax.set_title("Model Comparison (Test Set)", fontsize=12, pad=20)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "summary_table.png"), dpi=150)
    plt.close(fig)
