"""
FNN-5 Hedger: 5-layer feedforward neural network for delta hedging.

Architecture:
- Input: features at time k [batch, feature_dim]
- 5 hidden layers of size 128 with ReLU + LayerNorm + Dropout
- Output: Delta_k in R^{d_traded}
"""

import torch
import torch.nn as nn


class FNN5Hedger(nn.Module):
    """Feedforward neural network hedger with 5 hidden layers.

    At each time step k, takes features X_k and outputs hedge ratio Delta_k.
    The same network is applied at every time step (parameter sharing across time).
    """

    def __init__(
        self,
        input_dim: int,
        d_traded: int,
        hidden_dim: int = 128,
        n_layers: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_traded = d_traded

        layers = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        self.hidden = nn.Sequential(*layers)
        self.output_head = nn.Linear(hidden_dim, d_traded)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute hedge ratio for a single time step.

        Args:
            x: Features [batch, feature_dim]

        Returns:
            delta: Hedge ratios [batch, d_traded]
        """
        h = self.hidden(x)
        delta = self.output_head(h)
        return delta

    def compute_deltas(self, features: torch.Tensor) -> torch.Tensor:
        """Compute hedge ratios for all time steps.

        Args:
            features: [paths, N, feature_dim]

        Returns:
            deltas: [paths, N, d_traded]
        """
        n_paths, N, feat_dim = features.shape
        # Reshape to [paths*N, feat_dim] for batch processing
        x_flat = features.reshape(-1, feat_dim)
        delta_flat = self.forward(x_flat)
        deltas = delta_flat.reshape(n_paths, N, self.d_traded)
        return deltas
