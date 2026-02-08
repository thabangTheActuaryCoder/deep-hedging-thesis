"""Deep BSDE solver/hedger.

Learns initial portfolio value Y0 and control process Z via a neural network.
Propagation: Y_{k+1} = Y_k + <Z_k, dW_k>  (driver f=0)
Loss: MSE(Y_N - H_tilde)

Delta is recovered from Z via pseudo-inverse projection of the diffusion matrix.
Separate from the NN2 gating system.
"""
import torch
import torch.nn as nn
from src.models.fnn import get_activation


class DeepBSDE(nn.Module):
    """Deep BSDE solver for option pricing and hedging.

    Components:
        Y0: learnable scalar (initial portfolio value)
        z_net: neural network mapping (features, time_embedding) -> Z in R^m_brownian
        sigma: diffusion matrix for Z-to-Delta projection
    """

    def __init__(self, input_dim, d_traded, m_brownian, sigma_matrix,
                 depth=5, width=128, act_schedule="relu_all", dropout=0.1,
                 sigma_avg=None):
        """
        Args:
            input_dim: feature dimension
            d_traded: number of traded assets
            m_brownian: number of Brownian drivers
            sigma_matrix: [d_traded, m_brownian] diffusion matrix
            depth, width, act_schedule, dropout: architecture params
            sigma_avg: optional [d_traded, m_brownian] effective sigma for
                       Z-to-Delta projection (e.g. sqrt(theta) for Heston).
                       If None, uses sigma_matrix for projection.
        """
        super().__init__()
        self.d_traded = d_traded
        self.m_brownian = m_brownian
        self.time_embed_dim = 32

        # Learnable initial value
        self.Y0 = nn.Parameter(torch.tensor(0.05))

        # Z network
        z_input_dim = input_dim + self.time_embed_dim
        layers = []
        in_dim = z_input_dim
        for i in range(depth):
            layers.extend([
                nn.Linear(in_dim, width),
                nn.LayerNorm(width),
                get_activation(act_schedule, i),
                nn.Dropout(dropout),
            ])
            in_dim = width
        self.z_net = nn.Sequential(*layers)
        self.z_head = nn.Linear(width, m_brownian)

        # Store sigma for Z-to-Delta projection
        # Use sigma_avg if provided (e.g. Heston effective sigma), else sigma_matrix
        proj_sigma = sigma_avg if sigma_avg is not None else sigma_matrix
        self.register_buffer("sigma", sigma_matrix.clone())
        # Pre-compute pseudo-inverse of sigma^T
        sigma_T_pinv = torch.linalg.pinv(proj_sigma.T)  # [d_traded, m_brownian]
        self.register_buffer("sigma_T_pinv", sigma_T_pinv)

    def time_embedding(self, t):
        """Sinusoidal time embedding: [batch] -> [batch, time_embed_dim]."""
        dim = self.time_embed_dim
        freqs = torch.exp(
            torch.linspace(0, -4, dim // 2, device=t.device)
        )
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def compute_z(self, features_k, t_k):
        """Compute Z_k given features and time.

        Args:
            features_k: [batch, feat_dim]
            t_k: [batch] time values

        Returns:
            Z_k: [batch, m_brownian]
        """
        t_emb = self.time_embedding(t_k)
        z_input = torch.cat([features_k, t_emb], dim=-1)
        h = self.z_net(z_input)
        return self.z_head(h)

    def z_to_delta(self, Z, S_tilde_k):
        """Project Z (m_brownian) to Delta (d_traded).

        Relationship: Z = sigma^T @ diag(S_tilde) @ Delta
        => Delta = pinv(sigma^T) @ Z / S_tilde

        Args:
            Z: [batch, m_brownian]
            S_tilde_k: [batch, d_traded]

        Returns:
            Delta: [batch, d_traded]
        """
        S_Delta = Z @ self.sigma_T_pinv.T  # [batch, d_traded]
        return S_Delta / S_tilde_k.clamp(min=1e-8)

    def forward(self, features, dW, time_grid, substeps=0):
        """Full BSDE forward propagation.

        Args:
            features: [batch, N+1, feat_dim]
            dW: [batch, N, m_brownian]
            time_grid: [N+1]
            substeps: extra sub-steps per interval (0 = standard)

        Returns:
            Y_T: [batch] terminal value
            Y_path: [batch, N+1]
            Z_all: [batch, N, m_brownian]
        """
        batch = features.shape[0]
        N = dW.shape[1]
        device = features.device

        Y = self.Y0.expand(batch)
        Y_path = [Y]
        Z_all = []

        n_sub = substeps + 1

        for k in range(N):
            feat_k = features[:, k, :]

            if n_sub == 1:
                t_k = time_grid[k].expand(batch)
                Z_k = self.compute_z(feat_k, t_k)
                Z_all.append(Z_k)
                Y = Y + (Z_k * dW[:, k, :]).sum(dim=1)
            else:
                dt = (time_grid[k + 1] - time_grid[k]).item()
                dt_sub = dt / n_sub
                # Split Brownian increment into sub-increments summing to dW_k
                dW_subs = _split_brownian(dW[:, k, :], n_sub, dt_sub, device)
                Z_k_main = None
                for s in range(n_sub):
                    t_sub = time_grid[k].item() + s * dt_sub
                    t_batch = torch.full((batch,), t_sub, device=device)
                    Z_sub = self.compute_z(feat_k, t_batch)
                    if s == 0:
                        Z_k_main = Z_sub
                    Y = Y + (Z_sub * dW_subs[:, s, :]).sum(dim=1)
                Z_all.append(Z_k_main)

            Y_path.append(Y)

        Y_path = torch.stack(Y_path, dim=1)
        Z_all = torch.stack(Z_all, dim=1)
        return Y, Y_path, Z_all

    def compute_deltas(self, features, S_tilde, time_grid, substeps=0):
        """Compute hedge ratios from Z using the sigma projection.

        Args:
            features: [batch, N+1, feat_dim]
            S_tilde: [batch, N+1, d_traded]
            time_grid: [N+1]

        Returns:
            Delta_all: [batch, N, d_traded]
        """
        batch = features.shape[0]
        N = S_tilde.shape[1] - 1
        device = features.device
        Delta_all = []

        for k in range(N):
            t_k = time_grid[k].expand(batch)
            Z_k = self.compute_z(features[:, k, :], t_k)
            Delta_k = self.z_to_delta(Z_k, S_tilde[:, k, :])
            Delta_all.append(Delta_k)

        return torch.stack(Delta_all, dim=1)


def _split_brownian(dW_k, n_sub, dt_sub, device):
    """Split a Brownian increment into n_sub sub-increments that sum to dW_k.

    Uses additive adjustment: generate independent N(0, dt_sub) increments,
    then shift so they sum exactly to dW_k.

    Args:
        dW_k: [batch, m_brownian]
        n_sub: number of sub-increments
        dt_sub: sub-interval length

    Returns:
        dW_subs: [batch, n_sub, m_brownian]
    """
    if n_sub == 1:
        return dW_k.unsqueeze(1)
    batch, m = dW_k.shape
    import math
    raw = torch.randn(batch, n_sub, m, device=device) * math.sqrt(dt_sub)
    raw_sum = raw.sum(dim=1, keepdim=True)
    adjustment = (dW_k.unsqueeze(1) - raw_sum) / n_sub
    return raw + adjustment
