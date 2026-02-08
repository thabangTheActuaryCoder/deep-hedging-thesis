"""Heston-specific diagnostic plots.

Includes variance path visualization, implied volatility surface,
and GBM-vs-Heston model comparison.
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.utils.io import ensure_dir


def plot_variance_paths(V_paths, time_grid, output_dir, n_sample=50):
    """Plot sample variance process paths over time.

    Args:
        V_paths: [n_paths, N+1, d_traded] variance processes
        time_grid: [N+1]
        output_dir: output directory
        n_sample: number of sample paths to plot
    """
    ensure_dir(output_dir)
    V_np = V_paths.cpu().numpy()
    t_np = time_grid.cpu().numpy()
    d_traded = V_np.shape[2]

    fig, axes = plt.subplots(1, d_traded, figsize=(7 * d_traded, 5))
    if d_traded == 1:
        axes = [axes]

    for i in range(d_traded):
        ax = axes[i]
        for j in range(min(n_sample, V_np.shape[0])):
            ax.plot(t_np, V_np[j, :, i], alpha=0.3, linewidth=0.5)
        # Mean path
        mean_v = V_np[:, :, i].mean(axis=0)
        ax.plot(t_np, mean_v, "k-", linewidth=2, label="Mean $v_t$")
        ax.set_xlabel("Time $t$")
        ax.set_ylabel(f"Variance $v_{i+1}(t)$")
        ax.set_title(f"Asset {i+1}: Variance Paths")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "heston_variance_paths.png"), dpi=150)
    plt.close(fig)


def plot_implied_vol_surface(S_tilde, V_paths, K, T, time_grid, output_dir):
    """Plot implied volatility vs moneyness at selected time slices.

    Uses the instantaneous vol sqrt(v_k) as a proxy for implied vol
    (exact BS inversion would require option prices; this shows the vol smile shape).

    Args:
        S_tilde: [n_paths, N+1, d_traded] discounted prices
        V_paths: [n_paths, N+1, d_traded] variance processes
        K: strike price
        T: terminal time
        time_grid: [N+1]
        output_dir: output directory
    """
    ensure_dir(output_dir)
    S_np = S_tilde[:, :, 0].cpu().numpy()  # asset 1
    V_np = V_paths[:, :, 0].cpu().numpy()
    t_np = time_grid.cpu().numpy()
    N = len(t_np) - 1

    time_slices = [0, N // 4, N // 2, 3 * N // 4]
    fig, ax = plt.subplots(figsize=(10, 6))

    for k in time_slices:
        moneyness = np.log(S_np[:, k] / K)
        inst_vol = np.sqrt(np.clip(V_np[:, k], 0, None))

        # Bin by moneyness for cleaner plot
        bins = np.linspace(-0.5, 0.5, 30)
        bin_idx = np.digitize(moneyness, bins)
        bin_means = []
        bin_centers = []
        for b in range(1, len(bins)):
            mask = bin_idx == b
            if mask.sum() > 10:
                bin_means.append(inst_vol[mask].mean())
                bin_centers.append((bins[b - 1] + bins[b]) / 2)

        if bin_centers:
            tau = T - t_np[k]
            ax.plot(bin_centers, bin_means, "o-", markersize=3,
                    label=f"$\\tau$={tau:.2f}")

    ax.set_xlabel("Log-Moneyness $\\ln(S/K)$")
    ax.set_ylabel("Instantaneous Vol $\\sqrt{v_t}$")
    ax.set_title("Heston: Volatility Smile (Asset 1)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "heston_implied_vol_surface.png"), dpi=150)
    plt.close(fig)


def plot_vol_distribution(V_paths, time_grid, heston_params, output_dir):
    """Plot terminal variance/vol distribution vs theoretical long-run values.

    Args:
        V_paths: [n_paths, N+1, d_traded] variance processes
        time_grid: [N+1]
        heston_params: dict with theta (long-run variance)
        output_dir: output directory
    """
    ensure_dir(output_dir)
    V_np = V_paths.cpu().numpy()
    d_traded = V_np.shape[2]
    theta = heston_params["theta"]

    fig, axes = plt.subplots(1, d_traded, figsize=(7 * d_traded, 5))
    if d_traded == 1:
        axes = [axes]

    for i in range(d_traded):
        ax = axes[i]
        v_T = V_np[:, -1, i]
        vol_T = np.sqrt(np.clip(v_T, 0, None))
        ax.hist(vol_T, bins=60, density=True, alpha=0.7, edgecolor="black",
                linewidth=0.5, label="Terminal $\\sqrt{v_T}$")
        ax.axvline(np.sqrt(theta[i]), color="red", linestyle="--", linewidth=2,
                   label=f"$\\sqrt{{\\theta}}$={np.sqrt(theta[i]):.3f}")
        ax.axvline(np.mean(vol_T), color="blue", linestyle=":", linewidth=1.5,
                   label=f"Mean={np.mean(vol_T):.3f}")
        ax.set_xlabel("Volatility $\\sqrt{v_T}$")
        ax.set_ylabel("Density")
        ax.set_title(f"Asset {i+1}: Terminal Volatility Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "heston_vol_distribution.png"), dpi=150)
    plt.close(fig)


def plot_leverage_effect(S_tilde, V_paths, time_grid, output_dir):
    """Plot correlation between returns and variance changes (leverage effect).

    Heston models have rho < 0, meaning negative returns correspond to
    increasing variance — the classic volatility leverage effect.

    Args:
        S_tilde: [n_paths, N+1, d_traded] discounted prices
        V_paths: [n_paths, N+1, d_traded] variance processes
        time_grid: [N+1]
        output_dir: output directory
    """
    ensure_dir(output_dir)
    d_traded = S_tilde.shape[2]

    fig, axes = plt.subplots(1, d_traded, figsize=(7 * d_traded, 6))
    if d_traded == 1:
        axes = [axes]

    for i in range(d_traded):
        ax = axes[i]
        S_np = S_tilde[:, :, i].cpu().numpy()
        V_np = V_paths[:, :, i].cpu().numpy()

        # Log returns and variance changes
        log_ret = np.diff(np.log(np.clip(S_np, 1e-8, None)), axis=1)
        dv = np.diff(V_np, axis=1)

        # Flatten and subsample for scatter
        n_sample = min(50000, log_ret.size)
        idx = np.random.choice(log_ret.size, n_sample, replace=False)
        ret_flat = log_ret.flatten()[idx]
        dv_flat = dv.flatten()[idx]

        ax.scatter(ret_flat, dv_flat, s=1, alpha=0.1)
        corr = np.corrcoef(ret_flat, dv_flat)[0, 1]
        ax.set_xlabel("Log Return $\\Delta \\log S$")
        ax.set_ylabel("Variance Change $\\Delta v$")
        ax.set_title(f"Asset {i+1}: Leverage Effect ($\\rho$={corr:.3f})")
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=0.5)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "heston_leverage_effect.png"), dpi=150)
    plt.close(fig)


def plot_vol_of_vol(V_paths, time_grid, output_dir, window=10):
    """Plot the realized volatility-of-volatility over time.

    Shows how the volatility process itself fluctuates, which is a key
    feature of stochastic volatility models.

    Args:
        V_paths: [n_paths, N+1, d_traded] variance processes
        time_grid: [N+1]
        output_dir: output directory
        window: rolling window size for vol-of-vol computation
    """
    ensure_dir(output_dir)
    V_np = V_paths.cpu().numpy()
    t_np = time_grid.cpu().numpy()
    d_traded = V_np.shape[2]

    fig, axes = plt.subplots(1, d_traded, figsize=(7 * d_traded, 5))
    if d_traded == 1:
        axes = [axes]

    for i in range(d_traded):
        ax = axes[i]
        vol = np.sqrt(np.clip(V_np[:, :, i], 0, None))

        # Compute rolling std of vol across time (vol-of-vol)
        mean_vol = vol.mean(axis=0)
        std_vol = vol.std(axis=0)

        ax.plot(t_np, mean_vol, "b-", linewidth=1.5, label="Mean $\\sqrt{v_t}$")
        ax.fill_between(t_np, mean_vol - std_vol, mean_vol + std_vol,
                        alpha=0.2, color="blue", label="$\\pm 1$ std")
        ax.set_xlabel("Time $t$")
        ax.set_ylabel("Volatility $\\sqrt{v_t}$")
        ax.set_title(f"Asset {i+1}: Volatility Evolution (mean $\\pm$ std)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "heston_vol_of_vol.png"), dpi=150)
    plt.close(fig)


def plot_price_vs_gbm(S_tilde_heston, S_tilde_gbm, time_grid, output_dir,
                      n_sample=20):
    """Compare sample price paths from Heston vs GBM.

    Args:
        S_tilde_heston: [n_paths, N+1, d_traded] Heston prices
        S_tilde_gbm: [n_paths, N+1, d_traded] GBM prices
        time_grid: [N+1]
        output_dir: output directory
        n_sample: number of sample paths
    """
    ensure_dir(output_dir)
    t_np = time_grid.cpu().numpy()
    S_h = S_tilde_heston[:, :, 0].cpu().numpy()
    S_g = S_tilde_gbm[:, :, 0].cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for j in range(min(n_sample, S_h.shape[0])):
        axes[0].plot(t_np, S_h[j], alpha=0.3, linewidth=0.5, color="C0")
        axes[1].plot(t_np, S_g[j], alpha=0.3, linewidth=0.5, color="C1")

    axes[0].plot(t_np, S_h.mean(axis=0), "k-", linewidth=2, label="Mean")
    axes[0].set_title("Heston: Asset 1 Price Paths")
    axes[0].set_xlabel("Time $t$")
    axes[0].set_ylabel("$\\tilde{S}_t$")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_np, S_g.mean(axis=0), "k-", linewidth=2, label="Mean")
    axes[1].set_title("GBM: Asset 1 Price Paths")
    axes[1].set_xlabel("Time $t$")
    axes[1].set_ylabel("$\\tilde{S}_t$")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Price Path Comparison: Heston vs GBM", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "heston_vs_gbm_paths.png"), dpi=150)
    plt.close(fig)


def plot_heston_summary(V_paths, S_tilde, time_grid, heston_params, output_dir):
    """Combined 2x2 summary of Heston stochastic volatility dynamics.

    Panels:
        (a) Sample variance paths
        (b) Terminal vol distribution
        (c) Leverage effect scatter
        (d) Instantaneous vol smile
    """
    ensure_dir(output_dir)
    V_np = V_paths[:, :, 0].cpu().numpy()
    S_np = S_tilde[:, :, 0].cpu().numpy()
    t_np = time_grid.cpu().numpy()
    theta = heston_params["theta"][0]
    K = 1.0
    T = t_np[-1]
    N = len(t_np) - 1

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Variance paths
    ax = axes[0, 0]
    for j in range(30):
        ax.plot(t_np, V_np[j], alpha=0.3, linewidth=0.5)
    ax.plot(t_np, V_np.mean(axis=0), "k-", linewidth=2, label="Mean")
    ax.axhline(theta, color="red", linestyle="--", label=f"$\\theta$={theta:.3f}")
    ax.set_xlabel("Time $t$")
    ax.set_ylabel("Variance $v_t$")
    ax.set_title("(a) Variance Process Paths")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (b) Terminal vol distribution
    ax = axes[0, 1]
    vol_T = np.sqrt(np.clip(V_np[:, -1], 0, None))
    ax.hist(vol_T, bins=60, density=True, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.axvline(np.sqrt(theta), color="red", linestyle="--", linewidth=2,
               label=f"$\\sqrt{{\\theta}}$={np.sqrt(theta):.3f}")
    ax.set_xlabel("$\\sqrt{v_T}$")
    ax.set_ylabel("Density")
    ax.set_title("(b) Terminal Volatility Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (c) Leverage effect
    ax = axes[1, 0]
    log_ret = np.diff(np.log(np.clip(S_np, 1e-8, None)), axis=1)
    dv = np.diff(V_np, axis=1)
    n_pts = min(30000, log_ret.size)
    idx = np.random.choice(log_ret.size, n_pts, replace=False)
    corr = np.corrcoef(log_ret.flatten()[idx], dv.flatten()[idx])[0, 1]
    ax.scatter(log_ret.flatten()[idx], dv.flatten()[idx], s=1, alpha=0.1)
    ax.set_xlabel("Log Return")
    ax.set_ylabel("$\\Delta v$")
    ax.set_title(f"(c) Leverage Effect (corr={corr:.3f})")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.grid(True, alpha=0.3)

    # (d) Vol smile at time slices
    ax = axes[1, 1]
    for k in [0, N // 4, N // 2, 3 * N // 4]:
        moneyness = np.log(S_np[:, k] / K)
        inst_vol = np.sqrt(np.clip(V_np[:, k], 0, None))
        bins = np.linspace(-0.5, 0.5, 25)
        bin_idx = np.digitize(moneyness, bins)
        centers, means = [], []
        for b in range(1, len(bins)):
            mask = bin_idx == b
            if mask.sum() > 10:
                means.append(inst_vol[mask].mean())
                centers.append((bins[b - 1] + bins[b]) / 2)
        if centers:
            tau = T - t_np[k]
            ax.plot(centers, means, "o-", markersize=3, label=f"$\\tau$={tau:.2f}")
    ax.set_xlabel("Log-Moneyness")
    ax.set_ylabel("$\\sqrt{v_t}$")
    ax.set_title("(d) Volatility Smile")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Heston Stochastic Volatility Summary", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "heston_summary.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison_gbm_vs_heston(gbm_agg, heston_agg, output_dir):
    """Grouped bar chart comparing GBM and Heston results across models.

    Args:
        gbm_agg: {model_name: aggregated metrics} for GBM market
        heston_agg: {model_name: aggregated metrics} for Heston market
        output_dir: output directory
    """
    ensure_dir(output_dir)
    models = sorted(set(gbm_agg.keys()) & set(heston_agg.keys()))
    metric_names = ["CVaR95_shortfall", "MSE", "MAE", "mean_shortfall"]
    labels = ["CVaR95", "MSE", "MAE", "Mean Shortfall"]

    fig, axes = plt.subplots(1, len(metric_names), figsize=(5 * len(metric_names), 5))

    for idx, (mn, lbl) in enumerate(zip(metric_names, labels)):
        ax = axes[idx]
        x = np.arange(len(models))
        width = 0.35

        gbm_vals = []
        gbm_errs = []
        hes_vals = []
        hes_errs = []
        for m in models:
            g = gbm_agg[m].get(mn, {})
            h = heston_agg[m].get(mn, {})
            gbm_vals.append(g.get("mean", 0) if isinstance(g, dict) else g)
            gbm_errs.append(g.get("std", 0) if isinstance(g, dict) else 0)
            hes_vals.append(h.get("mean", 0) if isinstance(h, dict) else h)
            hes_errs.append(h.get("std", 0) if isinstance(h, dict) else 0)

        ax.bar(x - width / 2, gbm_vals, width, yerr=gbm_errs,
               label="GBM", capsize=3, alpha=0.85)
        ax.bar(x + width / 2, hes_vals, width, yerr=hes_errs,
               label="Heston", capsize=3, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_title(lbl)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("GBM vs Heston: Model Comparison", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "gbm_vs_heston_bars.png"), dpi=150)
    plt.close(fig)
