"""Experiment configuration."""
from dataclasses import dataclass, field
from typing import List


@dataclass
class ExperimentConfig:
    # Market
    paths: int = 50000
    N: int = 200
    T: float = 1.0
    d_traded: int = 2
    m_brownian: int = 3
    K: float = 1.0
    r: float = 0.0
    vols: List[float] = field(default_factory=lambda: [0.2, 0.2])
    extra_vol: float = 0.05

    # Features
    latent_dim: int = 16
    sig_level: int = 3
    vae_epochs: int = 50
    vae_augment_ratio: float = 0.5

    # Architecture grid (FNN cone)
    start_width_grid: List[int] = field(default_factory=lambda: [16, 32, 64, 128])
    lrs: List[float] = field(default_factory=lambda: [1e-4, 5e-4, 1e-3])
    dropout_grid: List[float] = field(default_factory=lambda: [0.0, 0.05, 0.1, 0.2, 0.3])

    # Training
    epochs: int = 1000
    patience: int = 15
    batch_size: int = 2048
    dropout: float = 0.1

    # Loss (super-hedging)
    lambda_short: float = 10.0
    lambda_over: float = 1.0
    cvar_q: float = 0.95

    # Seeds
    seed_arch: int = 0
    seeds: List[int] = field(default_factory=lambda: [0])

    # Market model
    market_model: str = "gbm"
    market_config: str = ""  # path to market_params JSON (empty = use defaults)

    # Paths
    output_dir: str = "outputs"
    data_dir: str = "data"

    # Quick mode
    quick: bool = False
