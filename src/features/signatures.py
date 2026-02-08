"""Signature-like features (pure PyTorch, no external libraries).

Computes cumulative path statistics up to a given truncation level.
Strictly causal: features at time k use only data up to and including time k.
"""
import torch


def compute_signature_features(log_prices, level=2):
    """Compute signature-like features up to given truncation level.

    Level 1: cumulative sums of increments (d features).
    Level 2: cumulative iterated integrals of pairwise products (d*(d+1)/2 features).

    NO LOOK-AHEAD: increment at time k is log_prices_k - log_prices_{k-1}.
    At time k=0, all features are zero.

    Args:
        log_prices: [n_paths, N+1, d] log of discounted prices
        level: truncation level (1 or 2)

    Returns:
        sig_features: [n_paths, N+1, sig_dim]
    """
    n_paths, N_plus_1, d = log_prices.shape

    # Increments: inc_k = log_prices_k - log_prices_{k-1}, inc_0 = 0
    increments = torch.zeros_like(log_prices)
    increments[:, 1:, :] = log_prices[:, 1:, :] - log_prices[:, :-1, :]

    # Level 1: cumulative sum of increments
    cum_inc = torch.cumsum(increments, dim=1)  # [n_paths, N+1, d]
    features = [cum_inc]

    if level >= 2:
        # Level 2: iterated integral approximation
        # S2^{a,b}_k = sum_{j=1}^{k} cum_inc_{j-1}^a * inc_j^b
        # This is causal: at step k it only uses data up to k
        level2_features = []
        for a in range(d):
            for b in range(a, d):
                shifted_cum = torch.zeros_like(cum_inc[:, :, a])
                shifted_cum[:, 1:] = cum_inc[:, :-1, a]  # cum up to j-1
                product = shifted_cum * increments[:, :, b]
                level2 = torch.cumsum(product, dim=1)
                level2_features.append(level2)
        level2_tensor = torch.stack(level2_features, dim=2)
        features.append(level2_tensor)

    return torch.cat(features, dim=2)


def get_signature_dim(d, level=2):
    """Compute output dimension of signature features."""
    dim = d  # level 1
    if level >= 2:
        dim += d * (d + 1) // 2  # level 2 upper triangular pairs
    return dim
