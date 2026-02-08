"""Feature pipeline: base + VAE latent + signature features.

All features are strictly causal (X_k depends only on data up to time k).
Standardization is computed on the training set only.
"""
import torch
from src.features.vae import train_vae, encode_paths
from src.features.signatures import compute_signature_features


def build_base_features(S_tilde, time_grid, T):
    """Base features: log(S_tilde^1), log(S_tilde^2), tau_k = T - t_k.

    Returns:
        base: [n_paths, N+1, d_traded + 1]
    """
    n_paths, N_plus_1, d = S_tilde.shape
    log_S = torch.log(S_tilde.clamp(min=1e-8))
    tau = (T - time_grid).unsqueeze(0).unsqueeze(2).expand(n_paths, N_plus_1, 1)
    return torch.cat([log_S, tau], dim=2)


def build_features(S_tilde, time_grid, T, train_idx, val_idx, test_idx,
                   latent_dim=16, sig_level=2, vae_epochs=50, device="cpu"):
    """Build full feature tensor: base + VAE latent + signatures.

    VAE is trained only on training paths to prevent information leakage.
    Standardization (mean/std) is computed on training set only.

    Args:
        S_tilde: [n_paths, N+1, d_traded] discounted prices (all paths)
        time_grid: [N+1]
        T: terminal time
        train_idx, val_idx, test_idx: split indices
        latent_dim: VAE latent dimension
        sig_level: signature truncation level
        vae_epochs: VAE training epochs
        device: torch device

    Returns:
        features_train: [n_train, N+1, feat_dim]
        features_val: [n_val, N+1, feat_dim]
        features_test: [n_test, N+1, feat_dim]
        feat_dim: int
    """
    # Base features
    base = build_base_features(S_tilde, time_grid, T)

    # Signature features
    log_prices = torch.log(S_tilde.clamp(min=1e-8))
    sig_feats = compute_signature_features(log_prices, level=sig_level)

    # VAE features (train on training set only)
    log_S_train = log_prices[train_idx]
    vae_model = train_vae(log_S_train, latent_dim=latent_dim,
                          epochs=vae_epochs, device=device)

    # Encode all paths
    latent_all = encode_paths(vae_model, log_prices, device=device)
    # Repeat latent vector across time steps
    N_plus_1 = S_tilde.shape[1]
    vae_feats = latent_all.unsqueeze(1).expand(-1, N_plus_1, -1)

    # Concatenate all features
    all_features = torch.cat([base, vae_feats, sig_feats], dim=2)
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
