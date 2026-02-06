"""
Build the full feature tensor combining base, VAE latent, and signature features.

All features satisfy the no-look-ahead property:
features at time k depend only on information up to time k.
"""

import torch
from typing import Optional
from src.features.vae import VAE, train_vae, encode_paths
from src.features.signatures import compute_signature_features, get_signature_dim


def build_base_features(
    S: torch.Tensor,
    time: torch.Tensor,
    T: float,
) -> torch.Tensor:
    """Build base features: log-prices and time-to-maturity.

    Args:
        S: [paths, N+1, d_traded]
        time: [N+1]
        T: Terminal time

    Returns:
        base: [paths, N, d_traded + 1]
            log(S) at times 1..N, tau = T - t for times 1..N
            (Features at decision time k use S at time k)
    """
    n_paths, N_plus_1, d = S.shape
    N = N_plus_1 - 1

    # Log-prices at decision times k=0..N-1
    # At decision time k, we observe S_k (current price)
    logS = torch.log(S[:, :N, :].clamp(min=1e-8))  # [paths, N, d]

    # Time to maturity
    tau = (T - time[:N]).unsqueeze(0).unsqueeze(-1).expand(n_paths, N, 1)

    base = torch.cat([logS, tau], dim=-1)  # [paths, N, d+1]
    return base


def build_features(
    S_train: torch.Tensor,
    S_val: torch.Tensor,
    S_test: torch.Tensor,
    time: torch.Tensor,
    T: float,
    latent_dim: int = 16,
    sig_level: int = 2,
    vae_epochs: int = 100,
    vae_lr: float = 1e-3,
    device: str = "cpu",
    seed: int = 0,
) -> dict:
    """Build complete feature tensors for train/val/test.

    Features at decision time k (k=0..N-1):
    - Base: log(S_k^1), log(S_k^2), tau_k
    - VAE: per-path latent vector (repeated across time)
    - Signature: cumulative increments and pairwise products up to k

    Args:
        S_train, S_val, S_test: Stock prices [n, N+1, d_traded]
        time: [N+1]
        T: Terminal time
        latent_dim: VAE latent dimension
        sig_level: Signature truncation level
        vae_epochs: VAE training epochs
        vae_lr: VAE learning rate
        device: torch device
        seed: Random seed

    Returns:
        dict with keys:
            'train', 'val', 'test': feature tensors [n, N, p]
            'vae': trained VAE model
            'feature_dim': total feature dimension p
    """
    N = S_train.shape[1] - 1
    d_traded = S_train.shape[2]

    # 1. Base features
    base_train = build_base_features(S_train, time, T)
    base_val = build_base_features(S_val, time, T)
    base_test = build_base_features(S_test, time, T)

    # 2. Train VAE on training data only
    print("  Training VAE...")
    vae = train_vae(
        S_train, latent_dim=latent_dim, epochs=vae_epochs,
        lr=vae_lr, device=device, seed=seed
    )

    # Encode all splits (no leakage: VAE was trained only on train)
    latent_train = encode_paths(vae, S_train, device)  # [n_train, latent_dim]
    latent_val = encode_paths(vae, S_val, device)
    latent_test = encode_paths(vae, S_test, device)

    # Repeat latent across time steps: [n, latent_dim] -> [n, N, latent_dim]
    latent_train = latent_train.unsqueeze(1).expand(-1, N, -1)
    latent_val = latent_val.unsqueeze(1).expand(-1, N, -1)
    latent_test = latent_test.unsqueeze(1).expand(-1, N, -1)

    # 3. Signature features
    print("  Computing signature features...")
    sig_train = compute_signature_features(S_train, level=sig_level)
    sig_val = compute_signature_features(S_val, level=sig_level)
    sig_test = compute_signature_features(S_test, level=sig_level)

    # Concatenate all features
    feat_train = torch.cat([base_train, latent_train, sig_train], dim=-1)
    feat_val = torch.cat([base_val, latent_val, sig_val], dim=-1)
    feat_test = torch.cat([base_test, latent_test, sig_test], dim=-1)

    feature_dim = feat_train.shape[-1]

    return {
        'train': feat_train,
        'val': feat_val,
        'test': feat_test,
        'vae': vae,
        'feature_dim': feature_dim,
    }


def get_feature_dim(d_traded: int, latent_dim: int, sig_level: int) -> int:
    """Compute total feature dimension."""
    base_dim = d_traded + 1  # log-prices + tau
    sig_dim = get_signature_dim(d_traded, sig_level)
    return base_dim + latent_dim + sig_dim
