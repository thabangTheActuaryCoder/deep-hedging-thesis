"""GRU hedger.

Pre-MLP -> stacked GRU -> head MLP.
Output: d_traded unconstrained hedge positions (direct positions).
Gradient checkpointing for memory efficiency.
"""
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.checkpoint import checkpoint
from src.models.fnn import get_activation


class GRUHedger(nn.Module):
    """GRU-based hedger.

    Input: full feature sequence [batch, N, feat_dim]
    Output: hedge positions [batch, N, d_traded] via single forward pass
    """

    def __init__(self, input_dim, num_layers=2, hidden_size=64,
                 act_schedule="relu_all", dropout=0.1, d_traded=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Pre-MLP: project features to hidden_size
        self.pre_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            get_activation(act_schedule, 0),
        )
        _init_sequential(self.pre_mlp, act_schedule, 0)

        # Stacked GRU
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Head MLP: map GRU output to hedge positions
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            get_activation(act_schedule, 1),
            nn.Linear(hidden_size, d_traded),
        )
        _init_sequential(self.head, act_schedule, 1)

    def _gru_forward(self, h):
        """GRU forward — separated for gradient checkpointing."""
        output, _ = self.gru(h)
        return output

    def forward(self, x):
        """Single forward pass over full sequence.

        Args:
            x: [batch, N, feat_dim]

        Returns:
            h: [batch, N, d_traded] hedge positions
        """
        h = self.pre_mlp(x)
        if self.training and h.requires_grad:
            output = checkpoint(self._gru_forward, h, use_reentrant=False)
        else:
            output = self._gru_forward(h)
        return self.head(output)


def _init_sequential(seq, act_schedule, base_idx):
    """Initialize Linear layers in a Sequential block."""
    for module in seq:
        if isinstance(module, nn.Linear):
            act_name = "tanh" if "tanh" in act_schedule else "relu"
            if act_name == "tanh":
                init.xavier_uniform_(module.weight)
            else:
                init.kaiming_uniform_(module.weight, nonlinearity="relu")
            init.zeros_(module.bias)
