"""OLS regression hedger.

Closed-form fit: beta = (X'X)^{-1} X'y using Black-Scholes put delta as target.
nn.Module wrapper for interface compatibility with FNN/GRU.
Output: d_traded scalars per time step (direct hedge positions).
"""
import math
import torch
import torch.nn as nn


class RegressionHedger(nn.Module):
    """OLS regression hedger with nn.Module interface.

    At each time step, predicts h_t = X_t @ beta.
    Beta is fit via closed-form OLS on training data.
    """

    def __init__(self, input_dim, d_traded=2):
        super().__init__()
        self.input_dim = input_dim
        self.d_traded = d_traded
        # Store beta as a buffer (not a learnable parameter)
        self.register_buffer("beta", torch.zeros(input_dim, d_traded))
        self._fitted = False

    def fit(self, X, y):
        """Fit OLS regression: beta = (X'X)^{-1} X'y.

        Args:
            X: [n_samples, input_dim] feature matrix
            y: [n_samples, d_traded] target (e.g. BS put delta per asset)
        """
        if y.dim() == 1:
            y = y.unsqueeze(1)
        # Add numerical stability via ridge (tiny lambda)
        XtX = X.T @ X
        reg = 1e-6 * torch.eye(self.input_dim, device=X.device)
        Xty = X.T @ y
        self.beta = torch.linalg.solve(XtX + reg, Xty)
        self._fitted = True

    def forward(self, x):
        """Forward pass: h = X @ beta.

        Args:
            x: [batch, feat_dim] or [batch, N, feat_dim]

        Returns:
            h: [batch, d_traded] or [batch, N, d_traded]
        """
        if x.dim() == 3:
            batch, N, feat = x.shape
            out = x.reshape(-1, feat) @ self.beta
            return out.reshape(batch, N, self.d_traded)
        return x @ self.beta

    @property
    def is_fitted(self):
        return self._fitted


def bs_put_delta(S, K, r, T, sigma, tau):
    """Black-Scholes put delta for use as OLS regression target.

    Args:
        S: [n] spot prices
        K: strike
        r: risk-free rate
        T: not used (tau is time-to-maturity)
        sigma: volatility
        tau: [n] time to maturity

    Returns:
        delta: [n] put delta in (-1, 0)
    """
    tau = tau.clamp(min=1e-8)
    d1 = (torch.log(S / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * torch.sqrt(tau))
    return _norm_cdf(d1) - 1.0


def _norm_cdf(x):
    """Standard normal CDF using error function."""
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
