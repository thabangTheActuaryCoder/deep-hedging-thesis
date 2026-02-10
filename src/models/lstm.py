"""LSTM baseline hedger (NN1).

Single forward pass over the entire sequence (no Python loop for outputs).
Pre-MLP and head-MLP use the activation schedule.
Gradient checkpointing reduces memory by ~40-60% at ~20% compute cost.
"""
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from src.models.fnn import get_activation


class LSTMHedger(nn.Module):
    """LSTM-based baseline hedger.

    Input: full feature sequence [batch, N, feat_dim]
    Output: hedge ratios [batch, N, d_traded] via single forward pass
    """

    def __init__(self, input_dim, d_traded, num_layers=5, hidden_size=128,
                 act_schedule="relu_all", dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Pre-MLP: project features to hidden_size
        self.pre_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            get_activation(act_schedule, 0),
        )

        # Stacked LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Head MLP: map LSTM output to hedge ratios
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            get_activation(act_schedule, 1),
            nn.Linear(hidden_size, d_traded),
        )

    def _lstm_forward(self, h):
        """LSTM forward — separated for gradient checkpointing."""
        output, _ = self.lstm(h)
        return output

    def forward(self, x):
        """Single forward pass over full sequence.

        Args:
            x: [batch, N, feat_dim]

        Returns:
            delta: [batch, N, d_traded]
        """
        h = self.pre_mlp(x)        # [batch, N, hidden]
        # Gradient checkpointing: recompute LSTM activations during backward
        # instead of storing them all — saves ~40-60% memory
        if self.training and h.requires_grad:
            output = checkpoint(self._lstm_forward, h, use_reentrant=False)
        else:
            output = self._lstm_forward(h)
        delta = self.head(output)   # [batch, N, d_traded]
        return delta

    def forward_with_states(self, x):
        """Forward pass that also returns LSTM hidden states (for diagnostics).

        Returns:
            delta: [batch, N, d_traded]
            (h_n, c_n): final hidden and cell states
        """
        h = self.pre_mlp(x)
        output, states = self.lstm(h)
        delta = self.head(output)
        return delta, states
