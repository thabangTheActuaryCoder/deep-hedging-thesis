"""Variational Autoencoder for synthetic path generation.

Trained on training set log-prices. Used to generate synthetic paths
for data augmentation (not feature extraction).
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


def train_path_vae(log_prices_train, latent_dim=16, epochs=50,
                   batch_size=256, lr=1e-3, device="cpu"):
    """Train VAE on flattened training paths.

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


def generate_synthetic_paths(vae_model, n_synthetic, seq_len, d, device="cpu"):
    """Generate synthetic paths by sampling from the VAE prior.

    Args:
        vae_model: trained VAE (frozen)
        n_synthetic: number of synthetic paths to generate
        seq_len: N+1 (sequence length)
        d: number of assets (d_traded)

    Returns:
        synthetic_log_prices: [n_synthetic, seq_len, d]
    """
    with torch.no_grad():
        z = torch.randn(n_synthetic, vae_model.latent_dim, device=device)
        flat = vae_model.decode(z)
    return flat.reshape(n_synthetic, seq_len, d).cpu()


def augment_training_data(S_tilde_train, vae_model, augment_ratio=0.5,
                          device="cpu"):
    """Augment training price paths with VAE-generated synthetic paths.

    Args:
        S_tilde_train: [n_train, N+1, d_traded] real discounted prices
        vae_model: trained VAE
        augment_ratio: fraction of synthetic paths relative to real (e.g. 0.5 = +50%)

    Returns:
        S_tilde_augmented: [n_train + n_synthetic, N+1, d_traded]
    """
    n_train, seq_len, d = S_tilde_train.shape
    n_synthetic = int(n_train * augment_ratio)

    if n_synthetic == 0:
        return S_tilde_train

    # Generate synthetic log-prices and convert to prices
    log_prices_train = torch.log(S_tilde_train.clamp(min=1e-8))
    synthetic_log = generate_synthetic_paths(
        vae_model, n_synthetic, seq_len, d, device=device
    )
    synthetic_prices = torch.exp(synthetic_log)

    return torch.cat([S_tilde_train, synthetic_prices], dim=0)
