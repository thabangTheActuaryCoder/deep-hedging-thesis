"""Portfolio simulation and hedging error computation.

Uses discounted self-financing: V_{k+1} = V_k + Delta_k^T * (S_tilde_{k+1} - S_tilde_k).
"""
import torch
from src.training.train import forward_portfolio


def simulate_portfolio_path(nn1, controller, features, Z_intrinsic, dS,
                            use_controller=True):
    """Full portfolio simulation returning path-level diagnostics.

    Args:
        nn1, controller: models (eval mode)
        features: [batch, N, feat_dim]
        Z_intrinsic: [batch, N]
        dS: [batch, N, d_traded]

    Returns:
        V_path: [batch, N+1] portfolio value path
        Delta: [batch, N, d_traded] hedge ratios
        info: dict with gate, Delta0, etc.
    """
    nn1.eval()
    if controller is not None:
        controller.eval()

    with torch.no_grad():
        V_T, info = forward_portfolio(
            nn1, controller, features, Z_intrinsic, dS,
            use_controller=use_controller, tbptt=0,
        )
    V_path = info["V_path"]
    Delta = info.get("Delta", info.get("Delta0"))
    return V_path, Delta, info


def simulate_bsde_portfolio(model, features, S_tilde, dW, time_grid,
                            substeps=0):
    """Simulate portfolio using BSDE-derived deltas.

    The BSDE gives Y_path (option value) and Z (controls).
    Delta is obtained via Z-to-Delta projection.
    Portfolio is then: V_{k+1} = V_k + Delta_k^T * dS_k.

    Returns:
        V_path: [batch, N+1] hedging portfolio path
        Y_path: [batch, N+1] BSDE value process
        Delta_all: [batch, N, d_traded] hedge ratios
    """
    model.eval()
    with torch.no_grad():
        Y_T, Y_path, Z_all = model(features, dW, time_grid, substeps=substeps)
        Delta_all = model.compute_deltas(features, S_tilde, time_grid)

    batch, N_plus_1, d = S_tilde.shape
    N = N_plus_1 - 1
    dS = S_tilde[:, 1:, :] - S_tilde[:, :-1, :]

    V = torch.zeros(batch, device=S_tilde.device)
    V_path_hedge = [V]
    for k in range(N):
        V = V + (Delta_all[:, k, :] * dS[:, k, :]).sum(dim=1)
        V_path_hedge.append(V)
    V_path_hedge = torch.stack(V_path_hedge, dim=1)

    return V_path_hedge, Y_path, Delta_all


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
