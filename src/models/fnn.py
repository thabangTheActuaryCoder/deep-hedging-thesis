"""FNN baseline hedger (NN1).

Block structure: Linear -> LayerNorm -> Activation -> Dropout
Processes each time step independently (parameter sharing across time).
"""
import torch.nn as nn


def get_activation(schedule, layer_idx):
    """Return activation module based on schedule and layer position."""
    if schedule == "relu_all":
        return nn.ReLU()
    elif schedule == "tanh_all":
        return nn.Tanh()
    elif schedule == "alt_relu_tanh":
        return nn.ReLU() if layer_idx % 2 == 0 else nn.Tanh()
    elif schedule == "alt_tanh_relu":
        return nn.Tanh() if layer_idx % 2 == 0 else nn.ReLU()
    raise ValueError(f"Unknown activation schedule: {schedule}")


class FNNHedger(nn.Module):
    """Feedforward baseline hedger.

    Input: X_k features at a single time step
    Output: Delta0_k hedge ratio in R^d_traded
    """

    def __init__(self, input_dim, d_traded, depth=5, width=128,
                 act_schedule="relu_all", dropout=0.1):
        super().__init__()
        layers = []
        in_dim = input_dim
        for i in range(depth):
            layers.extend([
                nn.Linear(in_dim, width),
                nn.LayerNorm(width),
                get_activation(act_schedule, i),
                nn.Dropout(dropout),
            ])
            in_dim = width
        self.network = nn.Sequential(*layers)
        self.output_head = nn.Linear(width, d_traded)

    def forward(self, x):
        """Forward pass.

        Args:
            x: [batch, feat_dim] single step or [batch, N, feat_dim] sequence

        Returns:
            delta: [batch, d_traded] or [batch, N, d_traded]
        """
        if x.dim() == 3:
            batch, N, feat = x.shape
            h = self.network(x.reshape(-1, feat))
            delta = self.output_head(h)
            return delta.reshape(batch, N, -1)
        h = self.network(x)
        return self.output_head(h)
