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
    sig_level: int = 2
    vae_epochs: int = 50

    # Architecture grid
    depth_grid: List[int] = field(default_factory=lambda: [3, 5, 7])
    width_grid: List[int] = field(default_factory=lambda: [64, 128, 256])
    act_schedules: List[str] = field(
        default_factory=lambda: [
            "relu_all", "tanh_all", "alt_relu_tanh", "alt_tanh_relu"
        ]
    )
    lrs: List[float] = field(default_factory=lambda: [3e-4, 1e-3, 3e-3])

    # Training
    epochs: int = 1000
    patience: int = 15
    batch_size: int = 2048
    dropout: float = 0.1

    # Regularization
    l1: float = 0.0
    l2: float = 1e-4

    # Loss
    objective: str = "cvar_shortfall"
    cvar_q: float = 0.95
    alpha: float = 1.0
    beta: float = 1.0

    # Controller
    use_controller: bool = True
    delta_clip: float = 5.0

    # LSTM
    tbptt: int = 50

    # Seeds
    seed_arch: int = 0
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])

    # DBSDE
    substeps: List[int] = field(default_factory=lambda: [0, 5, 10])

    # Paths
    output_dir: str = "outputs"
    data_dir: str = "data"

    # Quick mode
    quick: bool = False
