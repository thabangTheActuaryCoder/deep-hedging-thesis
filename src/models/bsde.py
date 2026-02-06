"""
Deep BSDE (DBSDE) Solver/Hedger.

Learns:
- Y0: Initial value of the hedging portfolio (scalar parameter)
- Z_net: Network mapping (t_k, state_k) -> Z_k in R^{m_brownian}

Propagation (driver f=0):
    Y_{k+1} = Y_k + <Z_k, dW_k>

Terminal loss:
    MSE(Y_N - H)

Hedge ratios Delta_k derived from Z_k via linear projection when
m_brownian != d_traded.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding."""

    def __init__(self, embed_dim: int = 32):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed scalar time values.

        Args:
            t: [batch] or [batch, 1] time values

        Returns:
            emb: [batch, embed_dim]
        """
        if t.dim() == 2:
            t = t.squeeze(-1)
        half = self.embed_dim // 2
        freqs = torch.exp(
            -np.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / half
        )
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb


class ZNet(nn.Module):
    """Network for Z_k = z_net(t_k, state_k).

    Maps time embedding + state features to Z in R^{m_brownian}.
    """

    def __init__(
        self,
        state_dim: int,
        m_brownian: int,
        hidden_dim: int = 128,
        n_layers: int = 4,
        time_embed_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.time_embed = TimeEmbedding(time_embed_dim)
        self.m_brownian = m_brownian

        input_dim = state_dim + time_embed_dim
        layers = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        self.hidden = nn.Sequential(*layers)
        self.output_head = nn.Linear(hidden_dim, m_brownian)

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Compute Z_k.

        Args:
            t: Time values [batch]
            state: State features [batch, state_dim]

        Returns:
            Z: [batch, m_brownian]
        """
        t_emb = self.time_embed(t)  # [batch, time_embed_dim]
        x = torch.cat([t_emb, state], dim=-1)
        h = self.hidden(x)
        Z = self.output_head(h)
        return Z


class DeepBSDE(nn.Module):
    """Deep BSDE solver/hedger.

    Learns Y0 and Z_net to solve the BSDE:
        Y_{k+1} = Y_k + <Z_k, dW_k>
    with terminal condition Y_N ≈ H.
    """

    def __init__(
        self,
        state_dim: int,
        d_traded: int,
        m_brownian: int,
        hidden_dim: int = 128,
        n_layers: int = 4,
        time_embed_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_traded = d_traded
        self.m_brownian = m_brownian

        # Learnable initial value
        self.Y0 = nn.Parameter(torch.tensor(5.0))

        # Z network
        self.z_net = ZNet(
            state_dim=state_dim,
            m_brownian=m_brownian,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            time_embed_dim=time_embed_dim,
            dropout=dropout,
        )

        # Linear projection from Z (m_brownian) to Delta (d_traded)
        # Used when m_brownian != d_traded
        if m_brownian != d_traded:
            self.z_to_delta = nn.Linear(m_brownian, d_traded, bias=False)
        else:
            self.z_to_delta = None

    def forward(
        self,
        features: torch.Tensor,
        dW: torch.Tensor,
        time: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward propagation of BSDE.

        Args:
            features: State features [paths, N, state_dim]
            dW: Brownian increments [paths, N, m_brownian]
            time: Time grid [N+1]

        Returns:
            Y_T: Terminal portfolio value [paths]
            Y_path: Portfolio value path [paths, N+1]
            Z_all: Z values at each step [paths, N, m_brownian]
        """
        n_paths, N, state_dim = features.shape

        Y = self.Y0.expand(n_paths)  # [paths]
        Y_path = [Y]
        Z_all = []

        for k in range(N):
            t_k = time[k].expand(n_paths)  # [paths]
            state_k = features[:, k, :]  # [paths, state_dim]

            Z_k = self.z_net(t_k, state_k)  # [paths, m_brownian]
            Z_all.append(Z_k)

            # BSDE propagation: Y_{k+1} = Y_k + <Z_k, dW_k>
            dW_k = dW[:, k, :]  # [paths, m_brownian]
            Y = Y + (Z_k * dW_k).sum(dim=-1)  # [paths]
            Y_path.append(Y)

        Y_path = torch.stack(Y_path, dim=1)  # [paths, N+1]
        Z_all = torch.stack(Z_all, dim=1)  # [paths, N, m_brownian]

        return Y[:], Y_path, Z_all

    def compute_deltas(
        self,
        features: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        """Extract hedge ratios Delta from Z.

        Args:
            features: [paths, N, state_dim]
            time: [N+1]

        Returns:
            deltas: [paths, N, d_traded]
        """
        n_paths, N, state_dim = features.shape
        deltas = []

        for k in range(N):
            t_k = time[k].expand(n_paths)
            state_k = features[:, k, :]
            Z_k = self.z_net(t_k, state_k)  # [paths, m_brownian]

            if self.z_to_delta is not None:
                delta_k = self.z_to_delta(Z_k)  # [paths, d_traded]
            else:
                delta_k = Z_k[:, :self.d_traded]  # [paths, d_traded]

            deltas.append(delta_k)

        deltas = torch.stack(deltas, dim=1)  # [paths, N, d_traded]
        return deltas

    def compute_deltas_batch(
        self,
        features: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        """Batch-compute hedge ratios (more efficient).

        Args:
            features: [paths, N, state_dim]
            time: [N+1]

        Returns:
            deltas: [paths, N, d_traded]
        """
        n_paths, N, state_dim = features.shape

        # Expand time to match batch
        t_all = time[:N].unsqueeze(0).expand(n_paths, -1)  # [paths, N]
        t_flat = t_all.reshape(-1)  # [paths*N]
        feat_flat = features.reshape(-1, state_dim)  # [paths*N, state_dim]

        Z_flat = self.z_net(t_flat, feat_flat)  # [paths*N, m_brownian]
        Z_all = Z_flat.reshape(n_paths, N, self.m_brownian)

        if self.z_to_delta is not None:
            # Apply projection
            delta_flat = self.z_to_delta(Z_flat)
            deltas = delta_flat.reshape(n_paths, N, self.d_traded)
        else:
            deltas = Z_all[:, :, :self.d_traded]

        return deltas
