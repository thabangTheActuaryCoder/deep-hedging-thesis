"""
Evaluation metrics for hedging performance.

Metrics computed on (V_T, H) pairs:
- MAE: Mean Absolute Error
- MSE: Mean Squared Error
- R2: Coefficient of determination

Additional statistics:
- mean/std of total hedging error
- mean/std of worst hedging error
- shortfall rate: P(worst_error < 0)
"""

import torch
import numpy as np
from typing import Dict


def compute_metrics(
    V_T: torch.Tensor,
    H: torch.Tensor,
    total_error: torch.Tensor,
    worst_error: torch.Tensor = None,
) -> Dict[str, float]:
    """Compute all hedging performance metrics.

    Args:
        V_T: Terminal portfolio value [paths]
        H: Terminal payoff [paths]
        total_error: V_T - H [paths]
        worst_error: min_k(V_k - Z_k) [paths], optional

    Returns:
        dict of metric name -> value
    """
    V_T = V_T.detach().cpu()
    H = H.detach().cpu()
    total_error = total_error.detach().cpu()

    mae = torch.mean(torch.abs(V_T - H)).item()
    mse = torch.mean((V_T - H) ** 2).item()

    # R2
    ss_res = torch.sum((H - V_T) ** 2).item()
    ss_tot = torch.sum((H - torch.mean(H)) ** 2).item()
    r2 = 1.0 - ss_res / max(ss_tot, 1e-10)

    metrics = {
        'MAE': mae,
        'MSE': mse,
        'R2': r2,
        'total_error_mean': total_error.mean().item(),
        'total_error_std': total_error.std().item(),
    }

    if worst_error is not None:
        worst_error = worst_error.detach().cpu()
        metrics['worst_error_mean'] = worst_error.mean().item()
        metrics['worst_error_std'] = worst_error.std().item()
        metrics['shortfall_rate'] = (worst_error < 0).float().mean().item()

    return metrics


def format_metrics(metrics: Dict[str, float], model_name: str = "") -> str:
    """Format metrics dict as a readable string."""
    lines = []
    if model_name:
        lines.append(f"--- {model_name} ---")
    for k, v in metrics.items():
        lines.append(f"  {k:25s}: {v:.6f}")
    return "\n".join(lines)


def aggregate_seed_metrics(seed_metrics: list) -> Dict[str, str]:
    """Aggregate metrics across seeds (mean +/- std).

    Args:
        seed_metrics: List of metric dicts from different seeds

    Returns:
        dict of metric name -> "mean +/- std" string
    """
    keys = seed_metrics[0].keys()
    result = {}
    result_raw = {}

    for k in keys:
        vals = [m[k] for m in seed_metrics]
        mean = np.mean(vals)
        std = np.std(vals)
        result[k] = f"{mean:.6f} +/- {std:.6f}"
        result_raw[k + '_mean'] = mean
        result_raw[k + '_std'] = std

    return result, result_raw
