"""Heston stochastic volatility market simulation via Euler-Maruyama.

Each of d_traded assets has its own variance process:
    dS_i = r * S_i * dt + sqrt(v_i) * S_i * dW_i^S
    dv_i = kappa_i * (theta_i - v_i) * dt + xi_i * sqrt(v_i) * dW_i^v
    corr(dW_i^S, dW_i^v) = rho_i

Incompleteness: variance processes are not directly tradeable, plus optional
extra Brownian driver(s) for additional unhedgeable risk.
"""
import math
import torch


def simulate_heston_market(n_paths, N, T, d_traded, heston_params, r=0.043,
                           extra_vol=0.0, seed=42, device="cpu"):
    """Simulate Heston stochastic volatility market.

    Brownian structure:
        - 2 * d_traded correlated drivers (W_i^S, W_i^v for each asset)
        - 1 extra independent driver for additional incompleteness (if extra_vol > 0)
    Total m_brownian = 2 * d_traded + (1 if extra_vol > 0 else 0)

    Args:
        n_paths: number of Monte Carlo paths
        N: number of time steps
        T: terminal time
        d_traded: number of tradeable assets
        heston_params: dict with keys kappa, theta, xi, rho, v0 (lists of length d_traded)
        r: risk-free rate
        extra_vol: loading on extra Brownian driver (0 = no extra driver)
        seed: random seed
        device: torch device

    Returns:
        S_tilde: [n_paths, N+1, d_traded] discounted prices
        dW: [n_paths, N, m_brownian] Brownian increments (all drivers)
        time_grid: [N+1]
        sigma_avg: [d_traded, m_brownian] effective diffusion matrix (sqrt(theta) based)
        V_paths: [n_paths, N+1, d_traded] variance process paths
    """
    kappa = heston_params["kappa"]
    theta = heston_params["theta"]
    xi = heston_params["xi"]
    rho = heston_params["rho"]
    v0 = heston_params["v0"]

    assert len(kappa) == d_traded
    assert len(theta) == d_traded

    has_extra = extra_vol > 0
    m_brownian = 2 * d_traded + (1 if has_extra else 0)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    dt = T / N
    sqrt_dt = math.sqrt(dt)
    time_grid = torch.linspace(0, T, N + 1, device=device)

    # Generate independent standard normals
    Z_raw = torch.randn(n_paths, N, m_brownian, generator=gen, device="cpu").to(device)

    # Build correlated increments via Cholesky for each asset pair
    # For asset i: (W_i^S, W_i^v) are correlated with rho_i
    # Indices: W_i^S = column 2*i, W_i^v = column 2*i+1
    dW = torch.zeros_like(Z_raw)
    for i in range(d_traded):
        rho_i = rho[i]
        # Cholesky: W_S = Z1, W_v = rho*Z1 + sqrt(1-rho^2)*Z2
        dW[:, :, 2 * i] = Z_raw[:, :, 2 * i] * sqrt_dt
        dW[:, :, 2 * i + 1] = (
            rho_i * Z_raw[:, :, 2 * i]
            + math.sqrt(1.0 - rho_i ** 2) * Z_raw[:, :, 2 * i + 1]
        ) * sqrt_dt

    # Extra driver (independent)
    if has_extra:
        dW[:, :, -1] = Z_raw[:, :, -1] * sqrt_dt

    # Euler-Maruyama for coupled (S, v) system
    S_tilde = torch.zeros(n_paths, N + 1, d_traded, device=device)
    S_tilde[:, 0, :] = 1.0
    V_paths = torch.zeros(n_paths, N + 1, d_traded, device=device)
    for i in range(d_traded):
        V_paths[:, 0, i] = v0[i]

    for k in range(N):
        for i in range(d_traded):
            v_k = V_paths[:, k, i]
            sqrt_v = torch.sqrt(v_k.clamp(min=0.0))

            # Variance update: dv = kappa*(theta - v)*dt + xi*sqrt(v)*dW^v
            dv = (kappa[i] * (theta[i] - v_k) * dt
                  + xi[i] * sqrt_v * dW[:, k, 2 * i + 1])
            v_new = v_k + dv
            # Reflection scheme to keep variance non-negative
            V_paths[:, k + 1, i] = v_new.abs()

            # Price update (log-Euler under risk-neutral measure):
            # log(S̃_{k+1}/S̃_k) = -0.5*v_k*dt + sqrt(v_k)*dW^S + extra_vol*dW^extra
            diffusion = sqrt_v * dW[:, k, 2 * i]
            if has_extra:
                diffusion = diffusion + extra_vol * dW[:, k, -1]
            log_inc = -0.5 * v_k * dt + diffusion
            S_tilde[:, k + 1, i] = S_tilde[:, k, i] * torch.exp(log_inc)

    # Build effective constant sigma matrix (sqrt(theta) approximation) for BSDE
    # This gives the Z-to-Delta projection a reasonable constant sigma
    sigma_avg = torch.zeros(d_traded, m_brownian, device=device)
    for i in range(d_traded):
        sigma_avg[i, 2 * i] = math.sqrt(theta[i])
        if has_extra:
            sigma_avg[i, -1] = extra_vol

    return S_tilde, dW, time_grid, sigma_avg, V_paths
