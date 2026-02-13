"""FNN hedger with cone (narrowing) architecture.

Cone structure: start_width -> start_width//2 -> ... -> min 4.
Block structure: Linear -> LayerNorm -> CELU -> Dropout
Output: 1 scalar (sigmoid allocation logit), identity output.
Kaiming weight initialization.
"""
import torch.nn as nn
import torch.nn.init as init


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


def cone_layer_widths(start_width, min_width=4):
    """Compute cone layer widths: start_width -> start_width//2 -> ... -> min_width.

    Example: start=64 -> [64, 32, 16, 8, 4]
    """
    widths = []
    w = start_width
    while w >= min_width:
        widths.append(w)
        w = w // 2
    if not widths:
        widths = [min_width]
    return widths


class FNNHedger(nn.Module):
    """Feedforward hedger with cone (narrowing) architecture.

    Input: X_k features at a single time step
    Output: h_k scalar in R (sigmoid allocation logit)
    Hidden layers use CELU activation. Output is identity (no activation).
    """

    def __init__(self, input_dim, start_width=64, dropout=0.1):
        super().__init__()
        widths = cone_layer_widths(start_width)

        layers = []
        in_dim = input_dim
        for i, width in enumerate(widths):
            linear = nn.Linear(in_dim, width)
            init.kaiming_uniform_(linear.weight, nonlinearity="relu")
            init.zeros_(linear.bias)
            layers.extend([
                linear,
                nn.LayerNorm(width),
                nn.CELU(),
                nn.Dropout(dropout),
            ])
            in_dim = width
        self.network = nn.Sequential(*layers)

        self.output_head = nn.Linear(in_dim, 1)
        init.xavier_uniform_(self.output_head.weight)
        init.zeros_(self.output_head.bias)

    def forward(self, x):
        """Forward pass.

        Args:
            x: [batch, feat_dim] single step or [batch, N, feat_dim] sequence

        Returns:
            h: [batch, 1] or [batch, N, 1] allocation logit
        """
        if x.dim() == 3:
            batch, N, feat = x.shape
            h = self.network(x.reshape(-1, feat))
            out = self.output_head(h)
            return out.reshape(batch, N, 1)
        h = self.network(x)
        return self.output_head(h)
