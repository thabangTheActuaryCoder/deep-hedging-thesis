"""
LSTM-5 Hedger: 5-layer stacked LSTM for sequential delta hedging.

Architecture:
- Input: feature sequence [batch, seq_len, feature_dim]
- 5-layer stacked LSTM with hidden_size=128, dropout=0.1 between layers
- Linear output head: hidden -> Delta_k in R^{d_traded}
"""

import torch
import torch.nn as nn


class LSTM5Hedger(nn.Module):
    """Stacked LSTM hedger with 5 layers.

    Processes the feature sequence up to time k and outputs Delta_k.
    Uses the LSTM hidden state at each time step for sequential hedging.
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
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        self.output_head = nn.Linear(hidden_dim, d_traded)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process full sequence and output deltas at each time step.

        Args:
            x: Feature sequence [batch, N, feature_dim]

        Returns:
            deltas: Hedge ratios [batch, N, d_traded]
        """
        # LSTM output: [batch, N, hidden_dim]
        lstm_out, _ = self.lstm(x)
        # Apply output head at each time step
        deltas = self.output_head(lstm_out)  # [batch, N, d_traded]
        return deltas

    def compute_deltas(self, features: torch.Tensor) -> torch.Tensor:
        """Compute hedge ratios for all time steps (same as forward).

        Args:
            features: [paths, N, feature_dim]

        Returns:
            deltas: [paths, N, d_traded]
        """
        return self.forward(features)

    def forward_step(
        self,
        x_k: torch.Tensor,
        hidden: tuple = None,
    ) -> tuple:
        """Process a single time step (for validation/testing).

        Args:
            x_k: Features at time k [batch, feature_dim]
            hidden: LSTM hidden state tuple (h, c)

        Returns:
            delta_k: Hedge ratio [batch, d_traded]
            hidden: Updated LSTM state
        """
        # Add sequence dimension: [batch, 1, feature_dim]
        x_k = x_k.unsqueeze(1)
        lstm_out, hidden = self.lstm(x_k, hidden)
        delta_k = self.output_head(lstm_out.squeeze(1))
        return delta_k, hidden
