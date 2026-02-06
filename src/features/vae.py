"""
Variational Autoencoder for latent feature extraction from price paths.

Trained ONLY on training data; applied to val/test via frozen encoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class VAEEncoder(nn.Module):
    """Encoder: maps a price path to latent distribution parameters."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, latent_dim: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class VAEDecoder(nn.Module):
    """Decoder: maps latent vector back to reconstructed path."""

    def __init__(self, latent_dim: int = 16, hidden_dim: int = 128, output_dim: int = 100):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        return self.fc_out(h)


class VAE(nn.Module):
    """Full VAE for path encoding."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, latent_dim: int = 16):
        super().__init__()
        self.encoder = VAEEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = VAEDecoder(latent_dim, hidden_dim, input_dim)
        self.latent_dim = latent_dim

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode without sampling (use mean as latent)."""
        mu, _ = self.encoder(x)
        return mu


def vae_loss(recon: torch.Tensor, target: torch.Tensor,
             mu: torch.Tensor, logvar: torch.Tensor,
             beta: float = 1.0) -> torch.Tensor:
    """VAE loss = reconstruction MSE + beta * KL divergence."""
    recon_loss = F.mse_loss(recon, target, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss


def train_vae(
    S_train: torch.Tensor,
    latent_dim: int = 16,
    hidden_dim: int = 128,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    beta: float = 1.0,
    device: str = "cpu",
    seed: int = 0,
) -> VAE:
    """Train VAE on flattened training price paths.

    Args:
        S_train: [n_train, N+1, d_traded] training stock prices
        latent_dim: Dimension of latent space
        hidden_dim: Hidden layer dimension
        epochs: Training epochs
        batch_size: Batch size
        lr: Learning rate
        beta: KL weight
        device: torch device
        seed: Random seed

    Returns:
        Trained VAE model (in eval mode)
    """
    torch.manual_seed(seed)

    n_train = S_train.shape[0]
    # Flatten path: [n_train, (N+1)*d_traded]
    X = S_train.reshape(n_train, -1).to(device)
    input_dim = X.shape[1]

    model = VAE(input_dim, hidden_dim, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n_train, device=device)
        total_loss = 0.0
        n_batches = 0

        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            batch = X[idx]

            recon, mu, logvar = model(batch)
            loss = vae_loss(recon, batch, mu, logvar, beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

    model.eval()
    return model


def encode_paths(vae: VAE, S: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    """Encode stock paths to latent vectors using trained VAE.

    Args:
        vae: Trained VAE model
        S: [n_paths, N+1, d_traded]

    Returns:
        latent: [n_paths, latent_dim] per-path latent vector
    """
    n = S.shape[0]
    X = S.reshape(n, -1).to(device)
    with torch.no_grad():
        latent = vae.encode(X)
    return latent
