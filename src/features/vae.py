"""Variational Autoencoder for per-path latent encoding.

Trained only on training set paths to prevent leakage.
Produces a fixed-size latent vector per path that captures path-level information.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, latent_dim=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(recon, x, mu, logvar, beta=1.0):
    """ELBO loss: reconstruction MSE + beta * KL divergence."""
    recon_loss = F.mse_loss(recon, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl


def train_vae(log_prices_train, latent_dim=16, epochs=50,
              batch_size=256, lr=1e-3, device="cpu"):
    """Train VAE on flattened training paths only.

    Args:
        log_prices_train: [n_train, N+1, d] log discounted prices (train set)

    Returns:
        Trained VAE model (in eval mode, frozen)
    """
    n_train, seq_len, d = log_prices_train.shape
    input_dim = seq_len * d
    flat = log_prices_train.reshape(n_train, -1).to(device)

    model = VAE(input_dim, hidden_dim=128, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n_train, device=device)
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            batch = flat[idx]
            recon, mu, logvar = model(batch)
            loss = vae_loss(recon, batch, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def encode_paths(vae_model, log_prices, device="cpu"):
    """Encode paths to per-path latent vectors using frozen encoder.

    Args:
        log_prices: [n_paths, N+1, d]

    Returns:
        latent: [n_paths, latent_dim]
    """
    n_paths, seq_len, d = log_prices.shape
    flat = log_prices.reshape(n_paths, -1).to(device)
    with torch.no_grad():
        mu, _ = vae_model.encode(flat)
    return mu.cpu()
