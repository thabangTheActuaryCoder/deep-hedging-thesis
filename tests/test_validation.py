"""Validation tests for the deep hedging codebase.

Tests cover:
  - No look-ahead in features
  - Self-financing portfolio constraint
  - Reproducibility (same seed -> identical outputs)
  - Controller causality (NN2 inputs at step k exclude future data)
  - LR grid tested as specified
"""
import sys
import os
import pytest
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.reproducibility import set_seed
from src.sim.simulate_market import (
    simulate_market, compute_european_put_payoff,
    compute_intrinsic_process, split_data,
)
from src.features.build_features import build_base_features
from src.features.signatures import compute_signature_features
from src.models.fnn import FNNHedger
from src.models.lstm import LSTMHedger
from src.models.controller import Controller
from src.models.bsde import DeepBSDE
from src.training.train import forward_portfolio


DEVICE = "cpu"
N_PATHS = 500
N_STEPS = 20
T = 1.0
D_TRADED = 2
M_BROWN = 3


@pytest.fixture
def market_data():
    """Generate small market for testing."""
    set_seed(42)
    S_tilde, dW, time_grid, sigma = simulate_market(
        N_PATHS, N_STEPS, T, D_TRADED, M_BROWN, r=0.0,
        vols=[0.2, 0.2], seed=42, device=DEVICE,
    )
    H_tilde = compute_european_put_payoff(S_tilde, K=1.0, r=0.0, T=T)
    Z = compute_intrinsic_process(S_tilde, K=1.0, r=0.0, time_grid=time_grid)
    return S_tilde, dW, time_grid, sigma, H_tilde, Z


# ─────────────────────────────────────
# No look-ahead tests
# ─────────────────────────────────────

class TestNoLookAhead:
    def test_base_features_no_lookahead(self, market_data):
        """Base features at step k depend only on S_tilde up to k."""
        S_tilde, _, time_grid, _, _, _ = market_data
        base = build_base_features(S_tilde, time_grid, T)

        # Modify future prices — base features at k should not change
        S_mod = S_tilde.clone()
        S_mod[:, 10:, :] = 999.0
        base_mod = build_base_features(S_mod, time_grid, T)

        # Features at k=5 should be identical
        assert torch.allclose(base[:, 5, :], base_mod[:, 5, :]), \
            "Base features at k=5 changed when future prices were modified"

    def test_signature_features_no_lookahead(self, market_data):
        """Signature features at step k are invariant to future price changes."""
        S_tilde, _, _, _, _, _ = market_data
        log_prices = torch.log(S_tilde.clamp(min=1e-8))
        sig = compute_signature_features(log_prices, level=2)

        # Modify future
        log_mod = log_prices.clone()
        log_mod[:, 10:, :] = 0.0
        sig_mod = compute_signature_features(log_mod, level=2)

        # At k=8, signatures should match
        assert torch.allclose(sig[:, 8, :], sig_mod[:, 8, :], atol=1e-6), \
            "Signature features at k=8 changed when future data was modified"

    def test_signature_at_zero(self, market_data):
        """Signature features at k=0 should be zero (no increments yet)."""
        S_tilde, _, _, _, _, _ = market_data
        log_prices = torch.log(S_tilde.clamp(min=1e-8))
        sig = compute_signature_features(log_prices, level=2)
        assert torch.allclose(sig[:, 0, :], torch.zeros_like(sig[:, 0, :])), \
            "Signature features at k=0 are not zero"


# ─────────────────────────────────────
# Self-financing tests
# ─────────────────────────────────────

class TestSelfFinancing:
    def test_portfolio_update(self, market_data):
        """V_{k+1} = V_k + Delta_k^T * (S_{k+1} - S_k)."""
        S_tilde, _, _, _, _, Z = market_data
        N = N_STEPS
        dS = S_tilde[:, 1:, :] - S_tilde[:, :-1, :]

        set_seed(0)
        feat_dim = 8
        features = torch.randn(N_PATHS, N, feat_dim)
        nn1 = FNNHedger(feat_dim, D_TRADED, depth=2, width=16)
        nn1.eval()  # disable dropout for deterministic comparison

        with torch.no_grad():
            Delta = nn1(features)  # [N_PATHS, N, D_TRADED]

        # Manual portfolio simulation
        V = torch.zeros(N_PATHS)
        for k in range(N):
            V_new = V + (Delta[:, k, :] * dS[:, k, :]).sum(dim=1)
            V = V_new

        # Compare with forward_portfolio (no controller)
        with torch.no_grad():
            V_T, info = forward_portfolio(
                nn1, None, features, Z[:, :N], dS,
                use_controller=False, tbptt=0,
            )

        assert torch.allclose(V, V_T, atol=1e-5), \
            "Portfolio terminal value mismatch"

    def test_path_terminal_consistency(self, market_data):
        """V_path[-1] should equal V_T from forward_portfolio."""
        S_tilde, _, _, _, _, Z = market_data
        N = N_STEPS
        dS = S_tilde[:, 1:, :] - S_tilde[:, :-1, :]

        set_seed(0)
        feat_dim = 8
        features = torch.randn(N_PATHS, N, feat_dim)
        nn1 = FNNHedger(feat_dim, D_TRADED, depth=2, width=16)

        with torch.no_grad():
            V_T, info = forward_portfolio(
                nn1, None, features, Z[:, :N], dS,
                use_controller=False, tbptt=0,
            )
        assert torch.allclose(info["V_path"][:, -1], V_T, atol=1e-5)

    def test_no_future_prices_in_delta(self, market_data):
        """FNN delta at step k is independent of future features."""
        set_seed(0)
        feat_dim = 8
        nn1 = FNNHedger(feat_dim, D_TRADED, depth=2, width=16)
        nn1.eval()

        features = torch.randn(10, N_STEPS, feat_dim)
        with torch.no_grad():
            delta_orig = nn1(features[:, 5, :]).clone()

        # Modify future features
        features[:, 6:, :] = 999.0
        with torch.no_grad():
            delta_mod = nn1(features[:, 5, :])

        assert torch.allclose(delta_orig, delta_mod), \
            "FNN delta at k=5 changed when future features were modified"


# ─────────────────────────────────────
# Reproducibility tests
# ─────────────────────────────────────

class TestReproducibility:
    def test_market_simulation(self):
        """Same seed produces identical market data."""
        S1, dW1, _, _ = simulate_market(100, 10, 1.0, 2, 3, 0.0,
                                         [0.2, 0.2], seed=99)
        S2, dW2, _, _ = simulate_market(100, 10, 1.0, 2, 3, 0.0,
                                         [0.2, 0.2], seed=99)
        assert torch.allclose(S1, S2)
        assert torch.allclose(dW1, dW2)

    def test_payoff_reproducibility(self, market_data):
        """Same market -> same payoffs."""
        S, _, _, _, H1, _ = market_data
        H2 = compute_european_put_payoff(S, K=1.0, r=0.0, T=T)
        assert torch.allclose(H1, H2)

    def test_split_reproducibility(self):
        """Same seed -> same split indices."""
        t1, v1, te1 = split_data(1000, seed=42)
        t2, v2, te2 = split_data(1000, seed=42)
        assert np.array_equal(t1, t2)
        assert np.array_equal(v1, v2)
        assert np.array_equal(te1, te2)

    def test_model_forward_reproducibility(self):
        """Same seed -> same model init -> same output."""
        set_seed(7)
        m1 = FNNHedger(10, 2, depth=3, width=32)
        m1.eval()
        x = torch.randn(5, 10)
        with torch.no_grad():
            out1 = m1(x).clone()

        set_seed(7)
        m2 = FNNHedger(10, 2, depth=3, width=32)
        m2.eval()
        with torch.no_grad():
            out2 = m2(x)

        assert torch.allclose(out1, out2, atol=1e-6)


# ─────────────────────────────────────
# Controller causality test
# ─────────────────────────────────────

class TestControllerCausality:
    def test_controller_input_excludes_future(self, market_data):
        """NN2 at step k does not receive dPL_{k+1} or S_{k+1}."""
        S_tilde, _, _, _, _, Z = market_data
        N = N_STEPS
        dS = S_tilde[:, 1:, :] - S_tilde[:, :-1, :]
        feat_dim = 8
        features = torch.randn(N_PATHS, N, feat_dim)

        set_seed(0)
        nn1 = FNNHedger(feat_dim, D_TRADED, depth=2, width=16)
        ctrl = Controller(feat_dim, D_TRADED, depth=2, width=16)

        # Run portfolio up to step 5 with original data
        with torch.no_grad():
            V_T_orig, info_orig = forward_portfolio(
                nn1, ctrl, features, Z[:, :N], dS,
                use_controller=True, tbptt=0,
            )
            gate_5_orig = info_orig["gate"][:, 5].clone()

        # Modify future dS (steps > 5) and rerun
        dS_mod = dS.clone()
        dS_mod[:, 6:, :] = 0.0
        set_seed(0)
        nn1_2 = FNNHedger(feat_dim, D_TRADED, depth=2, width=16)
        ctrl_2 = Controller(feat_dim, D_TRADED, depth=2, width=16)

        with torch.no_grad():
            _, info_mod = forward_portfolio(
                nn1_2, ctrl_2, features, Z[:, :N], dS_mod,
                use_controller=True, tbptt=0,
            )
            gate_5_mod = info_mod["gate"][:, 5]

        assert torch.allclose(gate_5_orig, gate_5_mod, atol=1e-5), \
            "Controller gate at k=5 changed when future dS was modified"


# ─────────────────────────────────────
# LR grid test
# ─────────────────────────────────────

class TestLRGrid:
    def test_lr_values(self):
        """Verify the exact LR grid is {3e-4, 1e-3, 3e-3}."""
        from src.utils.config import ExperimentConfig
        cfg = ExperimentConfig()
        expected = [3e-4, 1e-3, 3e-3]
        assert cfg.lrs == expected, f"LR grid mismatch: {cfg.lrs} != {expected}"


# ─────────────────────────────────────
# LSTM consistency test
# ─────────────────────────────────────

class TestLSTMConsistency:
    def test_lstm_full_vs_slice(self):
        """LSTM output at step k is same whether we feed full seq or slice to k+1."""
        set_seed(0)
        feat_dim = 8
        lstm = LSTMHedger(feat_dim, D_TRADED, num_layers=2, hidden_size=16)
        lstm.eval()  # disable dropout for deterministic comparison
        x = torch.randn(5, 10, feat_dim)

        with torch.no_grad():
            full_out = lstm(x)                # [5, 10, D_TRADED]
            partial_out = lstm(x[:, :6, :])   # [5, 6, D_TRADED]

        # Note: LSTM output at k depends on all inputs 0..k,
        # so slicing to k+1 should give same output at k
        # This is NOT true for standard LSTM because of bidirectional effects,
        # but our LSTM is unidirectional so it should hold.
        # However, the head MLP is applied independently, so this test is valid.
        # Actually for a forward LSTM, output[k] = f(input[0:k+1]), so
        # feeding [0:6] should give same output[0:5] as feeding [0:10].
        assert torch.allclose(full_out[:, :6, :], partial_out, atol=1e-5), \
            "LSTM output at k < 6 differs between full and partial sequence"
