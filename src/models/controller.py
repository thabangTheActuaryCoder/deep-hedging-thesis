"""NN2 Controller: gate + correction for two-stage hedging.

Takes market features and causal P/L features as input.
Outputs a gate scalar g in [0,1] and a correction Delta1 in R^d_traded.
Final hedge: Delta = Delta0 + g * Delta1

Strictly causal: all P/L features at step k use portfolio data up to step k only.
"""
import torch
import torch.nn as nn
from src.models.fnn import get_activation

# Number of P/L features fed to the controller
N_PL_FEATURES = 5  # PL, dPL, DD, intrinsic_gap, rolling_std


class Controller(nn.Module):
    """NN2 gating controller.

    Input: concat([X_k, PL_features_k])  where PL_features has 5 dims.
    Output: gate g_k in [0,1], correction Delta1_k in R^d_traded (clamped).
    """

    def __init__(self, input_dim, d_traded, depth=3, width=64,
                 act_schedule="relu_all", dropout=0.1, delta_clip=5.0):
        super().__init__()
        self.delta_clip = delta_clip
        total_input = input_dim + N_PL_FEATURES

        layers = []
        in_dim = total_input
        for i in range(depth):
            layers.extend([
                nn.Linear(in_dim, width),
                nn.LayerNorm(width),
                get_activation(act_schedule, i),
                nn.Dropout(dropout),
            ])
            in_dim = width
        self.backbone = nn.Sequential(*layers)

        # Gate head: sigmoid -> [0, 1]
        self.gate_head = nn.Sequential(
            nn.Linear(width, 1),
            nn.Sigmoid(),
        )
        # Correction head: raw output clamped via tanh
        self.correction_head = nn.Linear(width, d_traded)

    def forward(self, x_k, pl_features):
        """Forward pass at a single time step.

        Args:
            x_k: [batch, feat_dim] market features at step k
            pl_features: [batch, 5] causal P/L features at step k

        Returns:
            gate: [batch, 1] in [0, 1]
            correction: [batch, d_traded] clamped to [-delta_clip, delta_clip]
        """
        combined = torch.cat([x_k, pl_features], dim=1)
        h = self.backbone(combined)
        gate = self.gate_head(h)
        correction = torch.tanh(self.correction_head(h)) * self.delta_clip
        return gate, correction
