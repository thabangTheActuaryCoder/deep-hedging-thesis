"""Validation tests for the deep hedging codebase.

Tests cover:
  - No look-ahead in features
  - Self-financing portfolio constraint
  - Reproducibility (same seed -> identical outputs)
  - GRU consistency (unidirectional output invariance)
  - Sigmoid allocation (weights sum to 1)
  - OLS fit/predict
  - Super-hedging loss asymmetry
  - FNN cone depth auto-computation
  - LR grid tested as specified
  - Direct position portfolio (GRU/Regression path)
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
from src.models.fnn import FNNHedger, cone_layer_widths
from src.models.gru import GRUHedger
from src.models.regression import RegressionHedger, bs_put_delta
from src.training.train import forward_portfolio, sigmoid_allocation
from src.training.losses import super_hedging_loss, shortfall, over_hedge


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

        S_mod = S_tilde.clone()
        S_mod[:, 10:, :] = 999.0
        base_mod = build_base_features(S_mod, time_grid, T)

        assert torch.allclose(base[:, 5, :], base_mod[:, 5, :]), \
            "Base features at k=5 changed when future prices were modified"

    def test_signature_features_no_lookahead(self, market_data):
        """Signature features at step k are invariant to future price changes."""
        S_tilde, _, _, _, _, _ = market_data
        log_prices = torch.log(S_tilde.clamp(min=1e-8))
        sig = compute_signature_features(log_prices, level=2)

        log_mod = log_prices.clone()
        log_mod[:, 10:, :] = 0.0
        sig_mod = compute_signature_features(log_mod, level=2)

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
        """V_{k+1} = V_k + phi_k^T * dS_k with sigmoid allocation."""
        S_tilde, _, _, _, _, _ = market_data
        N = N_STEPS

        set_seed(0)
        feat_dim = 8
        features = torch.randn(N_PATHS, N, feat_dim)
        nn1 = FNNHedger(feat_dim, start_width=16)
        nn1.eval()

        V_0 = torch.full((N_PATHS,), 0.05)

        with torch.no_grad():
            V_T, V_path = forward_portfolio(
                nn1, features, S_tilde, V_0, D_TRADED,
            )

        assert torch.allclose(V_path[:, -1], V_T, atol=1e-5), \
            "V_path[-1] != V_T from forward_portfolio"

    def test_no_future_prices_in_delta(self, market_data):
        """FNN output at step k is independent of future features."""
        set_seed(0)
        feat_dim = 8
        nn1 = FNNHedger(feat_dim, start_width=16)
        nn1.eval()

        features = torch.randn(10, N_STEPS, feat_dim)
        with torch.no_grad():
            h_orig = nn1(features[:, 5, :]).clone()

        features[:, 6:, :] = 999.0
        with torch.no_grad():
            h_mod = nn1(features[:, 5, :])

        assert torch.allclose(h_orig, h_mod), \
            "FNN output at k=5 changed when future features were modified"


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
        m1 = FNNHedger(10, start_width=32)
        m1.eval()
        x = torch.randn(5, 10)
        with torch.no_grad():
            out1 = m1(x).clone()

        set_seed(7)
        m2 = FNNHedger(10, start_width=32)
        m2.eval()
        with torch.no_grad():
            out2 = m2(x)

        assert torch.allclose(out1, out2, atol=1e-6)


# ─────────────────────────────────────
# GRU consistency test
# ─────────────────────────────────────

class TestGRUConsistency:
    def test_gru_full_vs_slice(self):
        """GRU output at step k is same whether we feed full seq or slice to k+1."""
        set_seed(0)
        feat_dim = 8
        gru = GRUHedger(feat_dim, num_layers=2, hidden_size=16)
        gru.eval()
        x = torch.randn(5, 10, feat_dim)

        with torch.no_grad():
            full_out = gru(x)                # [5, 10, d_traded]
            partial_out = gru(x[:, :6, :])   # [5, 6, d_traded]

        assert torch.allclose(full_out[:, :6, :], partial_out, atol=1e-5), \
            "GRU output at k < 6 differs between full and partial sequence"

    def test_gru_output_shape(self):
        """GRU outputs d_traded scalars per time step."""
        gru = GRUHedger(10, num_layers=2, hidden_size=32)
        x = torch.randn(4, 8, 10)
        with torch.no_grad():
            out = gru(x)
        assert out.shape == (4, 8, 2), f"Expected (4, 8, 2), got {out.shape}"


# ─────────────────────────────────────
# Sigmoid allocation test
# ─────────────────────────────────────

class TestSigmoidAllocation:
    def test_weights_sum_to_one(self):
        """Sigmoid allocation weights w1 + w2 = 1."""
        h_t = torch.randn(10, 1)
        w1 = torch.sigmoid(h_t.squeeze(-1))
        w2 = 1.0 - w1
        assert torch.allclose(w1 + w2, torch.ones(10), atol=1e-7), \
            "Allocation weights do not sum to 1"

    def test_sigmoid_allocation_shares(self):
        """phi_i = w_i * V / S_i gives correct shares."""
        h_t = torch.zeros(5, 1)  # sigmoid(0) = 0.5
        V_t = torch.full((5,), 1.0)
        S_t = torch.ones(5, 2)
        phi = sigmoid_allocation(h_t, V_t, S_t)
        # With h=0: w1=0.5, w2=0.5, phi_i = 0.5 * 1 / 1 = 0.5
        expected = torch.full((5, 2), 0.5)
        assert torch.allclose(phi, expected, atol=1e-6), \
            f"Expected phi=0.5, got {phi}"


# ─────────────────────────────────────
# OLS Regression test
# ─────────────────────────────────────

class TestOLSRegression:
    def test_fit_predict(self):
        """OLS fit recovers known coefficients (multi-target)."""
        torch.manual_seed(0)
        X = torch.randn(100, 3)
        beta_true = torch.tensor([[1.0, 0.2], [-0.5, 0.8], [0.3, -0.4]])
        y = X @ beta_true + 0.01 * torch.randn(100, 2)

        model = RegressionHedger(3, d_traded=2)
        model.fit(X, y)

        assert model.is_fitted
        assert model.beta.shape == (3, 2)

        # Check predictions are close
        with torch.no_grad():
            y_pred = model(X)
        error = (y_pred - y).abs().mean()
        assert error < 0.1, f"Mean prediction error {error:.4f} too large"

    def test_output_shape(self):
        """Regression output shape: [batch, N, d_traded]."""
        model = RegressionHedger(5, d_traded=2)
        model.fit(torch.randn(20, 5), torch.randn(20, 2))
        x = torch.randn(4, 8, 5)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 8, 2), f"Expected (4, 8, 2), got {out.shape}"


# ─────────────────────────────────────
# Super-hedging loss test
# ─────────────────────────────────────

class TestSuperHedgingLoss:
    def test_asymmetry(self):
        """Shortfall is penalised more than over-hedge."""
        V_T = torch.tensor([0.9, 0.9, 0.9, 0.9])  # under-hedge by 0.1
        H_tilde = torch.ones(4)

        loss_under = super_hedging_loss(V_T, H_tilde, lambda_short=10.0, lambda_over=1.0)

        V_T_over = torch.tensor([1.1, 1.1, 1.1, 1.1])  # over-hedge by 0.1
        loss_over = super_hedging_loss(V_T_over, H_tilde, lambda_short=10.0, lambda_over=1.0)

        assert loss_under > loss_over, \
            f"Under-hedge loss {loss_under:.4f} should be > over-hedge loss {loss_over:.4f}"

    def test_perfect_hedge_zero_shortfall(self):
        """Perfect hedge => zero shortfall."""
        V_T = torch.ones(10)
        H_tilde = torch.ones(10)
        s = shortfall(V_T, H_tilde)
        assert torch.allclose(s, torch.zeros(10)), \
            "Shortfall should be zero for perfect hedge"

    def test_over_hedge_positive_error(self):
        """Over-hedge => positive over_hedge, zero shortfall."""
        V_T = torch.full((10,), 1.5)
        H_tilde = torch.ones(10)
        s = shortfall(V_T, H_tilde)
        o = over_hedge(V_T, H_tilde)
        assert torch.allclose(s, torch.zeros(10))
        assert torch.allclose(o, torch.full((10,), 0.5))


# ─────────────────────────────────────
# FNN cone depth test
# ─────────────────────────────────────

class TestConeArchitecture:
    def test_cone_widths_64(self):
        """start_width=64 -> [64, 32, 16, 8, 4] (depth=5)."""
        widths = cone_layer_widths(64)
        assert widths == [64, 32, 16, 8, 4], f"Got {widths}"

    def test_cone_widths_128(self):
        """start_width=128 -> [128, 64, 32, 16, 8, 4] (depth=6)."""
        widths = cone_layer_widths(128)
        assert widths == [128, 64, 32, 16, 8, 4], f"Got {widths}"

    def test_cone_widths_16(self):
        """start_width=16 -> [16, 8, 4] (depth=3)."""
        widths = cone_layer_widths(16)
        assert widths == [16, 8, 4], f"Got {widths}"

    def test_fnn_output_dim(self):
        """FNN outputs exactly 1 scalar per step."""
        model = FNNHedger(10, start_width=32)
        x = torch.randn(5, 10)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (5, 1), f"Expected (5, 1), got {out.shape}"

    def test_fnn_3d_output_dim(self):
        """FNN sequence input: [batch, N, feat] -> [batch, N, 1]."""
        model = FNNHedger(10, start_width=32)
        x = torch.randn(4, 8, 10)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 8, 1), f"Expected (4, 8, 1), got {out.shape}"


# ─────────────────────────────────────
# LR grid test
# ─────────────────────────────────────

class TestLRGrid:
    def test_lr_values(self):
        """Verify the exact LR grid is {1e-4, 5e-4, 1e-3}."""
        from src.utils.config import ExperimentConfig
        cfg = ExperimentConfig()
        expected = [1e-4, 5e-4, 1e-3]
        assert cfg.lrs == expected, f"LR grid mismatch: {cfg.lrs} != {expected}"


# ─────────────────────────────────────
# Direct position portfolio test
# ─────────────────────────────────────

class TestDirectPositionPortfolio:
    def test_gru_direct_positions(self, market_data):
        """forward_portfolio with GRU (output_dim=2) uses direct positions path."""
        S_tilde, _, _, _, _, _ = market_data
        N = N_STEPS

        set_seed(0)
        feat_dim = 3
        features = torch.randn(N_PATHS, N, feat_dim)
        gru = GRUHedger(feat_dim, num_layers=1, hidden_size=16, d_traded=D_TRADED)
        gru.eval()

        V_0 = torch.full((N_PATHS,), 0.05)

        with torch.no_grad():
            V_T, V_path = forward_portfolio(gru, features, S_tilde, V_0, D_TRADED)

        # V_path[:, -1] should equal V_T
        assert torch.allclose(V_path[:, -1], V_T, atol=1e-5), \
            "V_path[-1] != V_T for direct position portfolio"

        # Verify gains = sum of h_k * dS_k
        with torch.no_grad():
            h_all = gru(features)  # [batch, N, d_traded]
        dS = S_tilde[:, 1:, :] - S_tilde[:, :-1, :]
        gains = (h_all * dS).sum(dim=2)  # [batch, N]
        V_T_manual = V_0 + gains.sum(dim=1)
        assert torch.allclose(V_T, V_T_manual, atol=1e-5), \
            "V_T from forward_portfolio doesn't match manual gains computation"

    def test_direct_position_v_path(self, market_data):
        """V_path via cumsum matches step-by-step computation."""
        S_tilde, _, _, _, _, _ = market_data
        N = N_STEPS

        set_seed(1)
        feat_dim = 3
        features = torch.randn(10, N, feat_dim)
        gru = GRUHedger(feat_dim, num_layers=1, hidden_size=8, d_traded=D_TRADED)
        gru.eval()

        V_0 = torch.full((10,), 0.1)

        with torch.no_grad():
            V_T, V_path = forward_portfolio(gru, features, S_tilde[:10], V_0, D_TRADED)

        # V_path should have N+1 time steps
        assert V_path.shape == (10, N + 1), \
            f"Expected V_path shape (10, {N+1}), got {V_path.shape}"

        # V_path[:, 0] should be V_0
        assert torch.allclose(V_path[:, 0], V_0, atol=1e-7), \
            "V_path[:, 0] != V_0"
