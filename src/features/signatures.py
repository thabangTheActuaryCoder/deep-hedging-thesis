"""Signature-like features (pure PyTorch, no external libraries).

Computes cumulative path statistics up to a given truncation level.
Strictly causal: features at time k use only data up to and including time k.
"""
import torch


def compute_signature_features(log_prices, level=3):
    """Compute signature-like features up to given truncation level.

    Level 1: cumulative sums of increments (d features).
    Level 2: cumulative iterated integrals of pairwise products (d*(d+1)/2 features).
    Level 3: cumulative triple iterated integrals (C(d+2,3) features).

    NO LOOK-AHEAD: increment at time k is log_prices_k - log_prices_{k-1}.
    At time k=0, all features are zero.

    Args:
        log_prices: [n_paths, N+1, d] log of discounted prices
        level: truncation level (1, 2, or 3)

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

    if level >= 3:
        # Level 3: triple iterated integrals
        # S3^{a,b,c}_k = sum_{j=1}^{k} S2^{a,b}_{j-1} * inc_j^c  for a <= b <= c
        # Build index map for level2_features list: (a,b) -> index
        l2_index = {}
        idx = 0
        for a in range(d):
            for b in range(a, d):
                l2_index[(a, b)] = idx
                idx += 1

        level3_features = []
        for a in range(d):
            for b in range(a, d):
                for c in range(b, d):
                    s2_ab = level2_features[l2_index[(a, b)]]
                    shifted_s2 = torch.zeros_like(s2_ab)
                    shifted_s2[:, 1:] = s2_ab[:, :-1]
                    product = shifted_s2 * increments[:, :, c]
                    level3 = torch.cumsum(product, dim=1)
                    level3_features.append(level3)
        level3_tensor = torch.stack(level3_features, dim=2)
        features.append(level3_tensor)

    return torch.cat(features, dim=2)


def get_signature_dim(d, level=3):
    """Compute output dimension of signature features."""
    dim = d  # level 1
    if level >= 2:
        dim += d * (d + 1) // 2  # level 2 upper triangular pairs
    if level >= 3:
        dim += (d + 2) * (d + 1) * d // 6  # level 3 upper triangular triples
    return dim
