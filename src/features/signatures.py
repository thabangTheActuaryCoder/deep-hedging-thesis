"""
Signature-like features computed without external libraries.

Implements level-1 and level-2 path signature features with no look-ahead.
All features at time k depend only on data up to and including time k.
"""

import torch
from typing import Optional


def compute_signature_features(
    S: torch.Tensor,
    level: int = 2,
) -> torch.Tensor:
    """Compute signature-like features from stock price paths.

    Level 1: Cumulative increments of log-prices up to time k.
    Level 2: Cumulative pairwise products of increments up to time k.

    All features are causal (no look-ahead).

    Args:
        S: Stock prices [paths, N+1, d_traded]
        level: Signature truncation level (1 or 2)

    Returns:
        sig_features: [paths, N, n_sig_features]
            At time index k (0..N-1), features use data up to time k+1
            (i.e., increments from step 0..k are available at decision time k).
    """
    n_paths, N_plus_1, d = S.shape
    N = N_plus_1 - 1

    # Log-prices
    logS = torch.log(S.clamp(min=1e-8))

    # Increments: delta_k = logS_{k+1} - logS_k, for k=0..N-1
    increments = logS[:, 1:, :] - logS[:, :-1, :]  # [paths, N, d]

    features_list = []

    # Level 1: cumulative sum of increments up to time k
    # At decision time k, we use increments 0..k-1 (returns up to current time)
    # For k=0, no increments available yet -> use zeros
    cum_incr = torch.cumsum(increments, dim=1)  # [paths, N, d]
    # Shift: at time k, use sum of increments 0..k-1
    cum_incr_shifted = torch.zeros_like(cum_incr)
    cum_incr_shifted[:, 1:, :] = cum_incr[:, :-1, :]
    features_list.append(cum_incr_shifted)  # [paths, N, d]

    # Include the most recent observed increment as a feature (no look-ahead)
    # At decision time k, the most recent observed increment is logS_k - logS_{k-1}
    # For k=0, no previous increment -> zeros
    prev_incr = torch.zeros_like(increments)
    prev_incr[:, 1:, :] = increments[:, :-1, :]
    features_list.append(prev_incr)  # [paths, N, d]

    if level >= 2:
        # Level 2: cumulative pairwise products of increments
        # For each pair (i, j), compute sum_{s<k} delta_s^i * delta_s^j
        n_pairs = d * d
        pairwise = torch.zeros(n_paths, N, n_pairs, device=S.device)

        for i in range(d):
            for j in range(d):
                # Product of increments at each step
                prod = increments[:, :, i] * increments[:, :, j]  # [paths, N]
                cum_prod = torch.cumsum(prod, dim=1)  # [paths, N]
                # Shift for causality
                cum_prod_shifted = torch.zeros_like(cum_prod)
                cum_prod_shifted[:, 1:] = cum_prod[:, :-1]
                pairwise[:, :, i * d + j] = cum_prod_shifted

        features_list.append(pairwise)

    sig_features = torch.cat(features_list, dim=-1)  # [paths, N, n_features]
    return sig_features


def get_signature_dim(d_traded: int, level: int = 2) -> int:
    """Return the number of signature features for given dimension and level."""
    # Level 1: d (cumulative) + d (current increment) = 2*d
    dim = 2 * d_traded
    if level >= 2:
        dim += d_traded * d_traded  # pairwise products
    return dim
