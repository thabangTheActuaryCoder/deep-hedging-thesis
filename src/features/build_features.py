"""Feature pipeline: model-specific feature construction.

FNN: base + signatures
GRU: base only
Regression: base only

Base features: [log(S1), log(S2), tau]  (d_traded + 1 dims)

All features are strictly causal (X_k depends only on data up to time k).
Standardization is computed on the training set only.
"""
import torch
from src.features.signatures import compute_signature_features


def build_base_features(S_tilde, time_grid, T):
    """Base features: [log(S1), log(S2), tau].

    Args:
        S_tilde: [n_paths, N+1, d_traded]
        time_grid: [N+1]
        T: terminal time

    Returns:
        base: [n_paths, N+1, d_traded + 1]
    """
    n_paths, N_plus_1, d = S_tilde.shape
    log_S = torch.log(S_tilde.clamp(min=1e-8))
    tau = (T - time_grid).unsqueeze(0).unsqueeze(2).expand(n_paths, N_plus_1, 1)
    return torch.cat([log_S, tau], dim=2)


def build_features_for_model(model_type, S_tilde, time_grid, T,
                              train_idx, val_idx, test_idx,
                              sig_level=2):
    """Build feature tensors appropriate for a specific model type.

    FNN: base + signatures
    GRU: base only
    Regression: base only

    Args:
        model_type: "FNN", "GRU", or "Regression"
        S_tilde: [n_paths, N+1, d_traded]
        time_grid: [N+1]
        T: terminal time
        train_idx, val_idx, test_idx: split indices
        sig_level: signature truncation level (for FNN only)

    Returns:
        features_train, features_val, features_test: standardized feature tensors
        feat_dim: int
    """
    base = build_base_features(S_tilde, time_grid, T)

    if model_type == "FNN":
        log_prices = torch.log(S_tilde.clamp(min=1e-8))
        sig_feats = compute_signature_features(log_prices, level=sig_level)
        all_features = torch.cat([base, sig_feats], dim=2)
    else:
        # GRU and Regression: base only
        all_features = base

    feat_dim = all_features.shape[2]

    # Split
    train_feats = all_features[train_idx]
    val_feats = all_features[val_idx]
    test_feats = all_features[test_idx]

    # Standardize on training set
    flat_train = train_feats.reshape(-1, feat_dim)
    mean = flat_train.mean(dim=0)
    std = flat_train.std(dim=0).clamp(min=1e-8)

    train_feats = (train_feats - mean) / std
    val_feats = (val_feats - mean) / std
    test_feats = (test_feats - mean) / std

    return train_feats, val_feats, test_feats, feat_dim
