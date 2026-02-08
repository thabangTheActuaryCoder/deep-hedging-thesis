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


def plot_pnl_gbm_vs_heston(gbm_comp, heston_comp, output_dir):
    """Overlay P&L histograms comparing GBM (const sigma) vs Heston (stoch sigma).

    For each model (FNN, LSTM, DBSDE), plots the terminal hedging P&L
    distribution under both market dynamics side by side, similar to
    "NN Sigma vs Const Sigma" comparisons in the literature.

    Args:
        gbm_comp: {model_name: {"V_T": tensor, "H_tilde": tensor}} from GBM pipeline
        heston_comp: {model_name: {"V_T": tensor, "H_tilde": tensor}} from Heston pipeline
        output_dir: output directory
    """
    ensure_dir(output_dir)
    models = sorted(set(gbm_comp.keys()) & set(heston_comp.keys()))
    if not models:
        return

    n_models = len(models)
    fig, axes = plt.subplots(n_models, 1, figsize=(10, 5 * n_models))
    if n_models == 1:
        axes = [axes]

    for i, model in enumerate(models):
        ax = axes[i]

        # GBM P&L
        g_vt = gbm_comp[model]["V_T"]
        g_h = gbm_comp[model]["H_tilde"]
        if torch.is_tensor(g_vt):
            g_pnl = (g_vt - g_h).cpu().numpy()
        else:
            g_pnl = np.array(g_vt) - np.array(g_h)

        # Heston P&L
        h_vt = heston_comp[model]["V_T"]
        h_h = heston_comp[model]["H_tilde"]
        if torch.is_tensor(h_vt):
            h_pnl = (h_vt - h_h).cpu().numpy()
        else:
            h_pnl = np.array(h_vt) - np.array(h_h)

        # Shared bin range
        lo = min(g_pnl.min(), h_pnl.min())
        hi = max(g_pnl.max(), h_pnl.max())
        bins = np.linspace(lo, hi, 70)

        ax.hist(h_pnl, bins=bins, alpha=0.7, color="#7B2D8E",
                edgecolor="black", linewidth=0.3,
                label="Heston (Stoch. $\\sigma$)")
        ax.hist(g_pnl, bins=bins, alpha=0.5, color="#2EC4B6",
                edgecolor="black", linewidth=0.3,
                label="GBM (Const. $\\sigma$)")

        ax.set_xlabel("Profit/Loss")
        ax.set_ylabel("Frequency")
        ax.set_title(f"{model}: P&L with Different Volatility Models")
        ax.legend()
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "pnl_gbm_vs_heston.png"), dpi=150)
    plt.close(fig)


def plot_pnl_violin_gbm_vs_heston(gbm_comp, heston_comp, output_dir):
    """Violin plots comparing P&L distributions: Heston (stoch sigma) vs GBM (const sigma).

    For each model, shows two side-by-side violins with median lines and
    whiskers — pink for Heston, cyan for GBM.

    Args:
        gbm_comp: {model_name: {"V_T": tensor, "H_tilde": tensor}} from GBM pipeline
        heston_comp: {model_name: {"V_T": tensor, "H_tilde": tensor}} from Heston pipeline
        output_dir: output directory
    """
    ensure_dir(output_dir)
    models = sorted(set(gbm_comp.keys()) & set(heston_comp.keys()))
    if not models:
        return

    n_models = len(models)
    fig, axes = plt.subplots(n_models, 1, figsize=(8, 6 * n_models))
    if n_models == 1:
        axes = [axes]

    for i, model in enumerate(models):
        ax = axes[i]

        # Compute P&L
        g_vt = gbm_comp[model]["V_T"]
        g_h = gbm_comp[model]["H_tilde"]
        if torch.is_tensor(g_vt):
            g_pnl = (g_vt - g_h).cpu().numpy()
        else:
            g_pnl = np.array(g_vt) - np.array(g_h)

        h_vt = heston_comp[model]["V_T"]
        h_h = heston_comp[model]["H_tilde"]
        if torch.is_tensor(h_vt):
            h_pnl = (h_vt - h_h).cpu().numpy()
        else:
            h_pnl = np.array(h_vt) - np.array(h_h)

        parts = ax.violinplot([h_pnl, g_pnl], positions=[1, 2],
                              showmeans=False, showmedians=True,
                              showextrema=True)

        # Color violins: pink for Heston, cyan for GBM
        colors = ["#E88AED", "#A8E6E2"]
        for pc, color in zip(parts["bodies"], colors):
            pc.set_facecolor(color)
            pc.set_edgecolor(color)
            pc.set_alpha(0.75)

        # Style median/extrema lines
        for key in ("cmedians", "cmins", "cmaxes", "cbars"):
            if key in parts:
                parts[key].set_color("#333333")
                parts[key].set_linewidth(1.5)

        ax.set_xticks([1, 2])
        ax.set_xticklabels([
            f"Heston (Stoch. $\\sigma$)",
            f"GBM (Const. $\\sigma$)",
        ])
        ax.set_ylabel("Profit/Loss")
        ax.set_title(f"{model}: P&L with Different Volatility Models")
        ax.grid(True, alpha=0.2, axis="y")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "pnl_violin_gbm_vs_heston.png"), dpi=150)
    plt.close(fig)


def plot_pnl_per_model_hist(gbm_comp, heston_comp, output_dir):
    """Per-model histogram: GBM (const sigma) vs Heston (stoch vol) P&L.

    Creates one figure per model, each with two overlaid histograms.
    """
    models = sorted(set(gbm_comp.keys()) & set(heston_comp.keys()))
    if not models:
        return

    for model in models:
        fig, ax = plt.subplots(figsize=(8, 5))

        g_vt = gbm_comp[model]["V_T"]
        g_h = gbm_comp[model]["H_tilde"]
        g_pnl = (g_vt - g_h).cpu().numpy() if torch.is_tensor(g_vt) else np.array(g_vt) - np.array(g_h)

        h_vt = heston_comp[model]["V_T"]
        h_h = heston_comp[model]["H_tilde"]
        h_pnl = (h_vt - h_h).cpu().numpy() if torch.is_tensor(h_vt) else np.array(h_vt) - np.array(h_h)

        bins = np.linspace(
            min(g_pnl.min(), h_pnl.min()),
            max(g_pnl.max(), h_pnl.max()),
            80,
        )
        ax.hist(g_pnl, bins=bins, alpha=0.55, color="#00CED1", label="GBM (Const. $\\sigma$)",
                edgecolor="white", linewidth=0.3)
        ax.hist(h_pnl, bins=bins, alpha=0.55, color="#9B59B6", label="Heston (Stoch. $\\sigma$)",
                edgecolor="white", linewidth=0.3)

        ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Profit / Loss")
        ax.set_ylabel("Frequency")
        ax.set_title(f"{model}: P&L — Constant vs Stochastic Volatility")
        ax.legend(framealpha=0.9)
        ax.grid(True, alpha=0.2)

        fig.tight_layout()
        safe = model.replace(" ", "_").replace("-", "_").lower()
        fig.savefig(os.path.join(output_dir, f"pnl_hist_{safe}_gbm_vs_heston.png"), dpi=150)
        plt.close(fig)


def plot_pnl_per_model_violin(gbm_comp, heston_comp, output_dir):
    """Per-model violin: GBM (const sigma) vs Heston (stoch vol) P&L.

    Creates one figure per model, each with two violin bodies.
    """
    models = sorted(set(gbm_comp.keys()) & set(heston_comp.keys()))
    if not models:
        return

    for model in models:
        fig, ax = plt.subplots(figsize=(6, 5))

        g_vt = gbm_comp[model]["V_T"]
        g_h = gbm_comp[model]["H_tilde"]
        g_pnl = (g_vt - g_h).cpu().numpy() if torch.is_tensor(g_vt) else np.array(g_vt) - np.array(g_h)

        h_vt = heston_comp[model]["V_T"]
        h_h = heston_comp[model]["H_tilde"]
        h_pnl = (h_vt - h_h).cpu().numpy() if torch.is_tensor(h_vt) else np.array(h_vt) - np.array(h_h)

        parts = ax.violinplot([g_pnl, h_pnl], positions=[1, 2],
                              showmeans=True, showmedians=True, showextrema=True)

        colors = ["#00CED1", "#9B59B6"]
        for pc, color in zip(parts["bodies"], colors):
            pc.set_facecolor(color)
            pc.set_edgecolor(color)
            pc.set_alpha(0.7)
        for key in ("cmeans", "cmedians", "cmins", "cmaxes", "cbars"):
            if key in parts:
                parts[key].set_color("#333333")
                parts[key].set_linewidth(1.2)

        ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.4)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["GBM (Const. $\\sigma$)", "Heston (Stoch. $\\sigma$)"])
        ax.set_ylabel("Profit / Loss")
        ax.set_title(f"{model}: P&L — Constant vs Stochastic Volatility")
        ax.grid(True, alpha=0.2, axis="y")

        fig.tight_layout()
        safe = model.replace(" ", "_").replace("-", "_").lower()
        fig.savefig(os.path.join(output_dir, f"pnl_violin_{safe}_gbm_vs_heston.png"), dpi=150)
        plt.close(fig)


def plot_pnl_all_models_by_regime_hist(gbm_comp, heston_comp, output_dir):
    """Cross-model histogram: all models overlaid under same regime.

    Two figures: (1) all models under GBM, (2) all models under Heston.
    """
    model_colors = {
        "FNN-5": "#2196F3",
        "LSTM-5": "#FF9800",
        "DBSDE": "#4CAF50",
    }

    for regime, comp, label in [
        ("gbm", gbm_comp, "GBM (Const. $\\sigma$)"),
        ("heston", heston_comp, "Heston (Stoch. $\\sigma$)"),
    ]:
        if not comp:
            continue

        fig, ax = plt.subplots(figsize=(9, 5))
        all_pnl = {}
        for model in sorted(comp.keys()):
            vt = comp[model]["V_T"]
            ht = comp[model]["H_tilde"]
            pnl = (vt - ht).cpu().numpy() if torch.is_tensor(vt) else np.array(vt) - np.array(ht)
            all_pnl[model] = pnl

        if not all_pnl:
            plt.close(fig)
            continue

        lo = min(p.min() for p in all_pnl.values())
        hi = max(p.max() for p in all_pnl.values())
        bins = np.linspace(lo, hi, 80)

        for model, pnl in all_pnl.items():
            color = model_colors.get(model, "#888888")
            ax.hist(pnl, bins=bins, alpha=0.45, color=color, label=model,
                    edgecolor="white", linewidth=0.3)

        ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Profit / Loss")
        ax.set_ylabel("Frequency")
        ax.set_title(f"All Models P&L — {label}")
        ax.legend(framealpha=0.9)
        ax.grid(True, alpha=0.2)

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"pnl_hist_all_models_{regime}.png"), dpi=150)
        plt.close(fig)


def plot_pnl_all_models_by_regime_violin(gbm_comp, heston_comp, output_dir):
    """Cross-model violin: all models compared under same regime.

    Two figures: (1) all models under GBM, (2) all models under Heston.
    """
    model_colors = {
        "FNN-5": "#2196F3",
        "LSTM-5": "#FF9800",
        "DBSDE": "#4CAF50",
    }

    for regime, comp, label in [
        ("gbm", gbm_comp, "GBM (Const. $\\sigma$)"),
        ("heston", heston_comp, "Heston (Stoch. $\\sigma$)"),
    ]:
        if not comp:
            continue

        models = sorted(comp.keys())
        if not models:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        pnl_data = []
        labels = []
        for model in models:
            vt = comp[model]["V_T"]
            ht = comp[model]["H_tilde"]
            pnl = (vt - ht).cpu().numpy() if torch.is_tensor(vt) else np.array(vt) - np.array(ht)
            pnl_data.append(pnl)
            labels.append(model)

        positions = list(range(1, len(models) + 1))
        parts = ax.violinplot(pnl_data, positions=positions,
                              showmeans=True, showmedians=True, showextrema=True)

        for pc, model in zip(parts["bodies"], models):
            color = model_colors.get(model, "#888888")
            pc.set_facecolor(color)
            pc.set_edgecolor(color)
            pc.set_alpha(0.7)
        for key in ("cmeans", "cmedians", "cmins", "cmaxes", "cbars"):
            if key in parts:
                parts[key].set_color("#333333")
                parts[key].set_linewidth(1.2)

        ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.4)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Profit / Loss")
        ax.set_title(f"All Models P&L — {label}")
        ax.grid(True, alpha=0.2, axis="y")

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"pnl_violin_all_models_{regime}.png"), dpi=150)
        plt.close(fig)
