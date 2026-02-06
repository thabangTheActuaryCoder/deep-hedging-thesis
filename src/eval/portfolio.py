"""
Self-financing portfolio simulation and hedging error computation.

Portfolio dynamics:
    V_{k+1} = V_k + sum_j Delta_k^j * (S_{k+1}^j - S_k^j)

Errors:
    total_error = V_T - H  (terminal hedging error)
    worst_error = min_k (V_k - Z_k)  (worst hedging error across exercise times)
"""

import torch
from typing import Tuple, Optional


def simulate_portfolio(
    deltas: torch.Tensor,
    S: torch.Tensor,
    V0: float = 0.0,
) -> torch.Tensor:
    """Simulate self-financing portfolio terminal value.

    Args:
        deltas: Hedge ratios [paths, N, d_traded]
        S: Stock prices [paths, N+1, d_traded]
        V0: Initial portfolio value

    Returns:
        V_T: Terminal portfolio value [paths]
    """
    # Price increments: dS_k = S_{k+1} - S_k
    dS = S[:, 1:, :] - S[:, :-1, :]  # [paths, N, d_traded]

    # Portfolio gains: sum over time of Delta_k * dS_k
    gains = (deltas * dS).sum(dim=-1)  # [paths, N]
    total_gain = gains.sum(dim=-1)  # [paths]

    V_T = V0 + total_gain
    return V_T


def simulate_portfolio_path(
    deltas: torch.Tensor,
    S: torch.Tensor,
    V0: float = 0.0,
) -> torch.Tensor:
    """Simulate self-financing portfolio value path.

    Args:
        deltas: Hedge ratios [paths, N, d_traded]
        S: Stock prices [paths, N+1, d_traded]
        V0: Initial portfolio value

    Returns:
        V: Portfolio value path [paths, N+1]
    """
    n_paths, N, d_traded = deltas.shape

    dS = S[:, 1:, :] - S[:, :-1, :]  # [paths, N, d_traded]
    gains = (deltas * dS).sum(dim=-1)  # [paths, N]

    V = torch.zeros(n_paths, N + 1, device=deltas.device)
    V[:, 0] = V0
    for k in range(N):
        V[:, k + 1] = V[:, k] + gains[:, k]

    return V


def compute_hedging_errors(
    deltas: torch.Tensor,
    S: torch.Tensor,
    payoff_T: torch.Tensor,
    payoff_path: Optional[torch.Tensor] = None,
    V0: float = 0.0,
) -> dict:
    """Compute hedging error statistics.

    Args:
        deltas: [paths, N, d_traded]
        S: [paths, N+1, d_traded]
        payoff_T: Terminal payoff [paths]
        payoff_path: Bermudan payoff process [paths, N+1] (optional)
        V0: Initial portfolio value

    Returns:
        dict with:
            'total_error': V_T - H [paths]
            'worst_error': min_k(V_k - Z_k) [paths] (if payoff_path given)
            'V_T': terminal portfolio value [paths]
            'V_path': portfolio value path [paths, N+1]
    """
    V_path = simulate_portfolio_path(deltas, S, V0)
    V_T = V_path[:, -1]
    total_error = V_T - payoff_T

    result = {
        'total_error': total_error,
        'V_T': V_T,
        'V_path': V_path,
    }

    if payoff_path is not None:
        # Error at each exercise time
        error_path = V_path - payoff_path  # [paths, N+1]
        # Worst error: minimum across time (most negative = worst shortfall)
        worst_error = error_path.min(dim=1).values  # [paths]
        result['worst_error'] = worst_error
        result['error_path'] = error_path

    return result
