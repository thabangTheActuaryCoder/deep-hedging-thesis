"""
Plotting module inspired by Guo-Langrene-Wu (2023).

Produces:
1. Train vs Val loss curves
2. Hedging error histogram overlay (total vs worst)
3. Scatter V_T vs H with 45-degree line
4. Discretization/substeps convergence plots
5. Function shape over time (DBSDE Z_k diagnostics)
6. Summary table figure
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Optional


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def plot_loss_curves(
    train_losses: List[float],
    val_losses: List[float],
    model_name: str,
    save_dir: str,
):
    """Plot 1: Training vs Validation loss curves."""
    ensure_dir(save_dir)
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label='Train MSE', color='steelblue')
    ax.plot(epochs, val_losses, label='Val MSE', color='coral')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title(f'{model_name}: Train vs Validation Loss')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f'{model_name}_loss_curves.png'), dpi=150)
    plt.close(fig)


def plot_error_histograms(
    total_error: torch.Tensor,
    worst_error: Optional[torch.Tensor],
    model_name: str,
    save_dir: str,
):
    """Plot 2: Hedging error histograms (total vs worst) overlaid."""
    ensure_dir(save_dir)
    fig, ax = plt.subplots(figsize=(8, 5))

    te = total_error.detach().cpu().numpy()
    ax.hist(te, bins=80, alpha=0.6, density=True, color='steelblue', label='Total error ($V_T - H$)')

    if worst_error is not None:
        we = worst_error.detach().cpu().numpy()
        ax.hist(we, bins=80, alpha=0.6, density=True, color='coral',
                label='Worst error ($\\min_k (V_k - Z_k)$)')

    ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Hedging Error')
    ax.set_ylabel('Density')
    ax.set_title(f'{model_name}: Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f'{model_name}_error_hist.png'), dpi=150)
    plt.close(fig)


def plot_scatter_vt_h(
    V_T: torch.Tensor,
    H: torch.Tensor,
    model_name: str,
    save_dir: str,
):
    """Plot 3: Scatter V_T vs H with 45-degree line."""
    ensure_dir(save_dir)
    fig, ax = plt.subplots(figsize=(6, 6))

    vt = V_T.detach().cpu().numpy()
    h = H.detach().cpu().numpy()

    ax.scatter(h, vt, alpha=0.15, s=4, color='steelblue')

    # 45-degree line
    lo = min(h.min(), vt.min())
    hi = max(h.max(), vt.max())
    ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1.5, label='$V_T = H$')

    ax.set_xlabel('Payoff $H$')
    ax.set_ylabel('Portfolio $V_T$')
    ax.set_title(f'{model_name}: $V_T$ vs $H$')
    ax.legend()
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f'{model_name}_scatter.png'), dpi=150)
    plt.close(fig)


def plot_substeps_convergence(
    substeps_list: List[int],
    mse_values: Dict[str, List[float]],
    worst_values: Dict[str, List[float]],
    save_dir: str,
):
    """Plot 4: MSE and worst error vs number of substeps (discretization convergence).

    Args:
        substeps_list: List of substep counts
        mse_values: {model_name: [mse for each substep]}
        worst_values: {model_name: [mean worst_error for each substep]}
    """
    ensure_dir(save_dir)
    colors = {'FNN-5': 'steelblue', 'LSTM-5': 'coral', 'DBSDE': 'forestgreen'}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for name, vals in mse_values.items():
        c = colors.get(name, 'gray')
        ax1.plot(substeps_list, vals, 'o-', label=name, color=c)
    ax1.set_xlabel('Substeps')
    ax1.set_ylabel('MSE')
    ax1.set_title('MSE vs Discretization Substeps')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    for name, vals in worst_values.items():
        c = colors.get(name, 'gray')
        ax2.plot(substeps_list, vals, 's-', label=name, color=c)
    ax2.set_xlabel('Substeps')
    ax2.set_ylabel('Mean Worst Error')
    ax2.set_title('Worst Error vs Discretization Substeps')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'substeps_convergence.png'), dpi=150)
    plt.close(fig)


def plot_function_shape(
    model,
    features_template: torch.Tensor,
    time: torch.Tensor,
    S_range: torch.Tensor,
    time_indices: List[int],
    model_name: str,
    save_dir: str,
    device: str = "cpu",
):
    """Plot 5: Function shape over time — how learned Delta varies with stock price.

    For DBSDE: show Z_k norm or Delta_k^1 vs standardized S for several time points.

    Args:
        model: Trained model
        features_template: Template features [1, N, feature_dim] to modify
        time: Time grid [N+1]
        S_range: Grid of stock values to evaluate [n_grid]
        time_indices: Which time steps to plot (e.g. [0, 10, 20, 30, 40, 49])
        model_name: Name for title/filename
        save_dir: Directory to save
        device: torch device
    """
    ensure_dir(save_dir)
    model.eval()

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(time_indices)))

    n_grid = S_range.shape[0]
    feat_dim = features_template.shape[-1]

    for idx, k in enumerate(time_indices):
        deltas_at_k = []

        for s_val in S_range:
            # Create a feature vector with log(s_val) in the first position
            feat = features_template[:, k, :].clone().expand(1, -1).to(device)
            feat = feat.clone()
            feat[0, 0] = torch.log(s_val.clamp(min=1e-8))  # Replace first feature (logS1)

            with torch.no_grad():
                if hasattr(model, 'z_net'):
                    # DBSDE: use z_net directly
                    t_k = time[k].unsqueeze(0).to(device)
                    Z = model.z_net(t_k, feat)
                    if model.z_to_delta is not None:
                        delta = model.z_to_delta(Z)
                    else:
                        delta = Z[:, :model.d_traded]
                    deltas_at_k.append(delta[0, 0].item())
                else:
                    # FNN: direct forward
                    delta = model(feat)
                    deltas_at_k.append(delta[0, 0].item())

        ax.plot(S_range.numpy(), deltas_at_k, color=colors[idx],
                label=f't={time[k].item():.2f}')

    ax.set_xlabel('Stock Price $S$')
    ax.set_ylabel('$\\Delta^1$ (first asset)')
    ax.set_title(f'{model_name}: Hedge Ratio Shape Over Time')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f'{model_name}_function_shape.png'), dpi=150)
    plt.close(fig)


def plot_summary_table(
    summary: Dict[str, Dict[str, str]],
    save_dir: str,
):
    """Plot 6: Summary table as figure.

    Args:
        summary: {model_name: {metric: "mean +/- std"}}
    """
    ensure_dir(save_dir)

    models = list(summary.keys())
    if not models:
        return

    metrics = list(summary[models[0]].keys())

    # Build table data
    cell_text = []
    for model in models:
        row = [summary[model].get(m, "N/A") for m in metrics]
        cell_text.append(row)

    fig, ax = plt.subplots(figsize=(max(12, len(metrics) * 2), 1 + len(models) * 0.6))
    ax.axis('off')

    table = ax.table(
        cellText=cell_text,
        rowLabels=models,
        colLabels=metrics,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)

    # Style
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_text_props(fontweight='bold')
            cell.set_facecolor('#e6e6e6')
        if j == -1:
            cell.set_text_props(fontweight='bold')

    ax.set_title('Model Comparison: Test Set Performance (mean +/- std across seeds)',
                 fontsize=12, fontweight='bold', pad=20)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'summary_table.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def generate_all_plots(
    model_name: str,
    train_losses: List[float],
    val_losses: List[float],
    V_T: torch.Tensor,
    H: torch.Tensor,
    total_error: torch.Tensor,
    worst_error: Optional[torch.Tensor],
    save_dir: str,
):
    """Generate all per-model plots (1-3)."""
    plot_loss_curves(train_losses, val_losses, model_name, save_dir)
    plot_error_histograms(total_error, worst_error, model_name, save_dir)
    plot_scatter_vt_h(V_T, H, model_name, save_dir)
