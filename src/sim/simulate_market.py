"""
Incomplete market simulation via Euler-Maruyama with correlated Brownian motion.

Supports:
- d_traded tradeable assets
- m_brownian >= d_traded Brownian drivers (incompleteness)
- Configurable correlation structure
- Bermudan payoff process computation
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class MarketData:
    """Container for simulated market data."""
    S: torch.Tensor        # [paths, N+1, d_traded]
    dW: torch.Tensor       # [paths, N, m_brownian]
    time: torch.Tensor     # [N+1]
    dt: float
    T: float
    N: int
    d_traded: int
    m_brownian: int


def build_correlation_matrix(m_brownian: int, rho: float = 0.3) -> torch.Tensor:
    """Build a valid correlation matrix for m_brownian factors.

    Uses constant off-diagonal correlation rho, ensuring positive-definiteness.
    """
    corr = torch.eye(m_brownian) * (1.0 - rho) + rho * torch.ones(m_brownian, m_brownian)
    # Cholesky for generating correlated increments
    L = torch.linalg.cholesky(corr)
    return L


def build_diffusion_matrix(d_traded: int, m_brownian: int) -> torch.Tensor:
    """Build diffusion (volatility) loading matrix sigma: [d_traded, m_brownian].

    Each traded asset loads on the first d_traded Brownian factors (diagonal)
    plus small loadings on extra factors to create incompleteness.
    """
    sigma = torch.zeros(d_traded, m_brownian)
    # Diagonal loadings
    for i in range(d_traded):
        sigma[i, i] = 0.2  # base vol
    # Small loadings on extra Brownian factors (source of incompleteness)
    for i in range(d_traded):
        for j in range(d_traded, m_brownian):
            sigma[i, j] = 0.05
    return sigma


def simulate_market(
    n_paths: int,
    N: int = 50,
    T: float = 1.0,
    d_traded: int = 2,
    m_brownian: int = 3,
    S0: Optional[torch.Tensor] = None,
    mu: Optional[torch.Tensor] = None,
    sigma_matrix: Optional[torch.Tensor] = None,
    rho: float = 0.3,
    seed: Optional[int] = None,
    device: str = "cpu",
    substeps: int = 0,
) -> MarketData:
    """Simulate incomplete market diffusion via Euler-Maruyama.

    dS_t^i = S_t^i * (mu_i dt + sum_j sigma_{ij} dW_t^j)

    Args:
        n_paths: Number of Monte Carlo paths
        N: Number of exercise/observation time steps
        T: Terminal time
        d_traded: Number of tradeable assets
        m_brownian: Number of Brownian drivers (>= d_traded for incompleteness)
        S0: Initial stock prices [d_traded], default ones * 100
        mu: Drift vector [d_traded], default 0.05
        sigma_matrix: Diffusion matrix [d_traded, m_brownian]
        rho: Correlation parameter for Brownian factors
        seed: Random seed for reproducibility
        device: torch device
        substeps: Number of extra substeps between each exercise time

    Returns:
        MarketData with S, dW, time tensors
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    assert m_brownian >= d_traded, "m_brownian must be >= d_traded for incompleteness"

    # Defaults
    if S0 is None:
        S0 = torch.ones(d_traded, device=device) * 100.0
    if mu is None:
        mu = torch.ones(d_traded, device=device) * 0.05
    if sigma_matrix is None:
        sigma_matrix = build_diffusion_matrix(d_traded, m_brownian).to(device)

    # Correlation structure
    L = build_correlation_matrix(m_brownian, rho).to(device)

    # Total fine steps
    n_fine = N * (1 + substeps)
    dt_fine = T / n_fine

    # Time grid (fine)
    time_fine = torch.linspace(0, T, n_fine + 1, device=device)

    # Generate independent increments then correlate
    # dZ ~ N(0, dt_fine) independent
    dZ = torch.randn(n_paths, n_fine, m_brownian, device=device) * np.sqrt(dt_fine)
    # Correlate: dW = dZ @ L^T
    dW_fine = torch.matmul(dZ, L.T)  # [paths, n_fine, m_brownian]

    # Euler-Maruyama on fine grid
    S_fine = torch.zeros(n_paths, n_fine + 1, d_traded, device=device)
    S_fine[:, 0, :] = S0.unsqueeze(0)

    for k in range(n_fine):
        S_k = S_fine[:, k, :]  # [paths, d_traded]
        dW_k = dW_fine[:, k, :]  # [paths, m_brownian]

        # Drift: S * mu * dt
        drift = S_k * mu.unsqueeze(0) * dt_fine
        # Diffusion: S * (sigma @ dW)
        # sigma_matrix: [d_traded, m_brownian], dW_k: [paths, m_brownian]
        diffusion = S_k * torch.matmul(dW_k, sigma_matrix.T)  # [paths, d_traded]

        S_fine[:, k + 1, :] = S_k + drift + diffusion
        # Floor at small positive to avoid negative prices
        S_fine[:, k + 1, :] = torch.clamp(S_fine[:, k + 1, :], min=1e-6)

    # Subsample to exercise grid
    indices = torch.arange(0, n_fine + 1, 1 + substeps, device=device)
    S = S_fine[:, indices, :]  # [paths, N+1, d_traded]
    time = time_fine[indices]  # [N+1]

    # Aggregate Brownian increments to exercise grid
    # dW on exercise grid: sum fine increments within each interval
    dW = torch.zeros(n_paths, N, m_brownian, device=device)
    for i in range(N):
        start = i * (1 + substeps)
        end = (i + 1) * (1 + substeps)
        dW[:, i, :] = dW_fine[:, start:end, :].sum(dim=1)

    dt = T / N

    return MarketData(
        S=S, dW=dW, time=time, dt=dt,
        T=T, N=N, d_traded=d_traded, m_brownian=m_brownian
    )


def compute_payoffs(
    S: torch.Tensor,
    K: float = 100.0,
    payoff_type: str = "put",
    asset_idx: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute terminal and Bermudan payoff process.

    Args:
        S: Stock prices [paths, N+1, d_traded]
        K: Strike price
        payoff_type: 'put' or 'call'
        asset_idx: Which asset to use for payoff

    Returns:
        payoff_T: Terminal payoff [paths]
        payoff_path: Bermudan payoff process [paths, N+1]
    """
    S_asset = S[:, :, asset_idx]  # [paths, N+1]

    if payoff_type == "put":
        payoff_path = torch.clamp(K - S_asset, min=0.0)
    elif payoff_type == "call":
        payoff_path = torch.clamp(S_asset - K, min=0.0)
    else:
        raise ValueError(f"Unknown payoff_type: {payoff_type}")

    payoff_T = payoff_path[:, -1]  # Terminal payoff

    return payoff_T, payoff_path


def split_data(
    *tensors: torch.Tensor,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    seed: Optional[int] = None,
) -> Tuple:
    """Split tensors along first dimension into train/val/test.

    Returns:
        Tuple of (train_tensors, val_tensors, test_tensors)
        Each is a tuple matching the input tensors.
    """
    n = tensors[0].shape[0]

    if seed is not None:
        gen = torch.Generator()
        gen.manual_seed(seed)
        perm = torch.randperm(n, generator=gen)
    else:
        perm = torch.randperm(n)

    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    idx_train = perm[:n_train]
    idx_val = perm[n_train:n_train + n_val]
    idx_test = perm[n_train + n_val:]

    train_out = tuple(t[idx_train] for t in tensors)
    val_out = tuple(t[idx_val] for t in tensors)
    test_out = tuple(t[idx_test] for t in tensors)

    return train_out, val_out, test_out
