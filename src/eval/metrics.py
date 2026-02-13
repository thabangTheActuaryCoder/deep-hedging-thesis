"""Evaluation metrics for hedging performance.

Reports terminal-value metrics and super-hedging diagnostics.
"""
import numpy as np
import torch
from src.training.losses import terminal_error, shortfall, over_hedge, cvar


def compute_metrics(V_T, H_tilde, V_path=None, cvar_q=0.95):
    """Compute full suite of hedging metrics.

    Args:
        V_T: [n] terminal portfolio value
        H_tilde: [n] discounted payoff
        V_path: [n, N+1] optional portfolio path (for daily P/L metrics)

    Returns:
        metrics: dict of metric name -> value
    """
    e = terminal_error(V_T, H_tilde)
    s = shortfall(V_T, H_tilde)
    o = over_hedge(V_T, H_tilde)

    metrics = {
        "MAE": e.abs().mean().item(),
        "MSE": (e ** 2).mean().item(),
        "R2": _r2(V_T, H_tilde),
        "mean_error": e.mean().item(),
        "std_error": e.std().item(),
        "P_negative_error": (e < 0).float().mean().item(),
        "P_positive_error": (e > 0).float().mean().item(),
        "mean_shortfall": s.mean().item(),
        "mean_over_hedge": o.mean().item(),
        "CVaR90_shortfall": cvar(s, q=0.90).item(),
        "CVaR95_shortfall": cvar(s, q=0.95).item(),
        "CVaR99_shortfall": cvar(s, q=0.99).item(),
    }

    if V_path is not None:
        dPL = V_path[:, 1:] - V_path[:, :-1]
        metrics["mean_dPL"] = dPL.mean().item()
        metrics["std_dPL"] = dPL.std().item()
        worst_daily = dPL.min(dim=1)[0]
        metrics["mean_worst_daily_loss"] = worst_daily.mean().item()
        metrics["std_worst_daily_loss"] = worst_daily.std().item()
        running_max = torch.cummax(V_path, dim=1)[0]
        drawdown = (running_max - V_path).max(dim=1)[0]
        metrics["mean_max_drawdown"] = drawdown.mean().item()
        metrics["std_max_drawdown"] = drawdown.std().item()

    return metrics


def _r2(predictions, targets):
    """R-squared: 1 - SS_res/SS_tot."""
    ss_res = ((predictions - targets) ** 2).sum().item()
    ss_tot = ((targets - targets.mean()) ** 2).sum().item()
    if ss_tot < 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def aggregate_seed_metrics(seed_metrics_list):
    """Aggregate metrics across seeds: mean +/- std + 95% CI.

    Args:
        seed_metrics_list: list of dicts (one per seed)

    Returns:
        agg: dict with mean, std, ci_lo, ci_hi for each metric
    """
    keys = seed_metrics_list[0].keys()
    agg = {}
    n = len(seed_metrics_list)

    for key in keys:
        vals = [m[key] for m in seed_metrics_list]
        arr = np.array(vals)
        mean = arr.mean()
        std = arr.std(ddof=1) if n > 1 else 0.0
        se = std / np.sqrt(n) if n > 1 else 0.0
        t_val = 2.776 if n == 5 else 2.0
        agg[key] = {
            "mean": float(mean),
            "std": float(std),
            "ci_lo": float(mean - t_val * se),
            "ci_hi": float(mean + t_val * se),
            "values": vals,
        }

    return agg


def select_representative_seed(seed_results):
    """Pick representative seed = argmin validation CVaR95(shortfall).

    Args:
        seed_results: list of dicts with "seed" and "val_metrics" keys

    Returns:
        best_seed: int
    """
    best = min(seed_results, key=lambda r: r["val_metrics"]["CVaR95_shortfall"])
    return best["seed"]
