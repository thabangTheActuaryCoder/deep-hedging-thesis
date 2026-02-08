"""Incomplete market simulation via Euler-Maruyama under risk-neutral measure.

Simulates d_traded assets driven by m_brownian > d_traded Brownian motions,
creating an incomplete market where perfect replication is impossible.
"""
import math
import numpy as np
import torch


def build_diffusion_matrix(d_traded, m_brownian, vols, extra_vol=0.05):
    """Build volatility loading matrix sigma: [d_traded, m_brownian].

    Each asset i has main volatility vols[i] on its own Brownian driver i,
    plus extra_vol exposure to each additional Brownian driver j >= d_traded.
    The extra drivers create market incompleteness.
    """
    sigma = torch.zeros(d_traded, m_brownian)
    for i in range(d_traded):
        sigma[i, i] = vols[i]
    for i in range(d_traded):
        for j in range(d_traded, m_brownian):
            sigma[i, j] = extra_vol
    return sigma


def simulate_market(n_paths, N, T, d_traded, m_brownian, r, vols,
                    extra_vol=0.05, seed=42, device="cpu"):
    """Simulate incomplete market under risk-neutral measure.

    Uses log-Euler scheme for discounted prices:
        log(S_tilde_{k+1}/S_tilde_k) = -0.5*||sigma_i||^2*dt + sigma_i . dW_k

    Args:
        n_paths: number of Monte Carlo paths
        N: number of time steps
        T: terminal time
        d_traded: number of tradeable assets
        m_brownian: number of Brownian drivers (>= d_traded)
        r: risk-free rate
        vols: list of per-asset volatilities
        extra_vol: loading on extra Brownian drivers
        seed: random seed for simulation
        device: torch device

    Returns:
        S_tilde: [n_paths, N+1, d_traded] discounted prices
        dW: [n_paths, N, m_brownian] Brownian increments
        time_grid: [N+1] time points
        sigma: [d_traded, m_brownian] diffusion matrix
    """
    assert m_brownian >= d_traded, "Need m_brownian >= d_traded"
    assert len(vols) == d_traded, "Need one vol per traded asset"

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    dt = T / N
    sqrt_dt = math.sqrt(dt)
    time_grid = torch.linspace(0, T, N + 1, device=device)

    sigma = build_diffusion_matrix(d_traded, m_brownian, vols, extra_vol)
    sigma = sigma.to(device)

    # Squared row norms for Ito correction
    sigma_sq = (sigma ** 2).sum(dim=1)  # [d_traded]

    # Generate Brownian increments dW ~ N(0, dt)
    dW = torch.randn(n_paths, N, m_brownian, generator=gen, device="cpu")
    dW = dW.to(device) * sqrt_dt

    # Log-Euler scheme for discounted prices
    S_tilde = torch.zeros(n_paths, N + 1, d_traded, device=device)
    S_tilde[:, 0, :] = 1.0  # S_0 = 1

    for k in range(N):
        # diffusion: [n_paths, m_brownian] @ [m_brownian, d_traded] -> [n_paths, d_traded]
        diffusion = dW[:, k, :] @ sigma.T
        log_inc = -0.5 * sigma_sq.unsqueeze(0) * dt + diffusion
        S_tilde[:, k + 1, :] = S_tilde[:, k, :] * torch.exp(log_inc)

    return S_tilde, dW, time_grid, sigma


def compute_european_put_payoff(S_tilde, K, r, T):
    """Compute discounted European put payoff on asset 1.

    H = max(K - S_T^1, 0),  H_tilde = exp(-rT) * H
    where S_T^1 = S_tilde_T^1 * exp(rT) is the undiscounted terminal price.

    Returns:
        H_tilde: [n_paths] discounted terminal payoff
    """
    S_T_1 = S_tilde[:, -1, 0] * math.exp(r * T)
    H = torch.clamp(K - S_T_1, min=0.0)
    H_tilde = math.exp(-r * T) * H
    return H_tilde


def compute_intrinsic_process(S_tilde, K, r, time_grid):
    """Compute discounted intrinsic value process (diagnostic only).

    Z_k = exp(-r*t_k) * max(K - S_{t_k}^1, 0)
    This is NOT the optimal stopping value.

    Returns:
        Z: [n_paths, N+1] discounted intrinsic value at each step
    """
    n_paths, N_plus_1, _ = S_tilde.shape
    Z = torch.zeros(n_paths, N_plus_1, device=S_tilde.device)
    for k in range(N_plus_1):
        t_k = time_grid[k].item()
        S_k_1 = S_tilde[:, k, 0] * math.exp(r * t_k)
        Z[:, k] = math.exp(-r * t_k) * torch.clamp(K - S_k_1, min=0.0)
    return Z


def split_data(n_paths, seed=42):
    """60/20/20 train/val/test split with deterministic indices.

    Returns:
        train_idx, val_idx, test_idx: numpy arrays of indices
    """
    gen = np.random.RandomState(seed)
    indices = gen.permutation(n_paths)
    n_train = int(0.6 * n_paths)
    n_val = int(0.2 * n_paths)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    return train_idx, val_idx, test_idx
