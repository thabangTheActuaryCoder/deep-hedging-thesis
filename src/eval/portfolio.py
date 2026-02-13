"""Portfolio simulation and hedging error computation.

Uses sigmoid allocation: V_{k+1} = V_k + phi_k^T * dS_k.
"""
import torch
from src.training.train import forward_portfolio


def simulate_portfolio_path(model, features, S_tilde, V_0, d_traded):
    """Full portfolio simulation returning path-level diagnostics.

    Args:
        model: hedger model (eval mode)
        features: [batch, N, feat_dim]
        S_tilde: [batch, N+1, d_traded]
        V_0: [batch] initial portfolio value
        d_traded: int

    Returns:
        V_path: [batch, N+1] portfolio value path
        V_T: [batch] terminal value
    """
    model.eval()
    with torch.no_grad():
        V_T, V_path = forward_portfolio(model, features, S_tilde, V_0, d_traded)
    return V_path, V_T


def compute_daily_pnl(V_path):
    """Compute daily P/L: dPL_k = V_k - V_{k-1}.

    Returns:
        dPL: [batch, N] daily P/L
    """
    return V_path[:, 1:] - V_path[:, :-1]


def compute_max_drawdown(V_path):
    """Compute max drawdown per path: max_k(max_{j<=k} V_j - V_k).

    Returns:
        max_dd: [batch] max drawdown (positive value)
    """
    running_max = torch.cummax(V_path, dim=1)[0]
    drawdown = running_max - V_path
    return drawdown.max(dim=1)[0]
