"""
Validation tests for the deep hedging codebase.

Tests:
A) No look-ahead: inputs at time k do not use data from k+1..N
B) Self-financing: portfolio update uses only Delta_k and (S_{k+1} - S_k)
C) Reproducibility: same seed -> identical metrics
"""

import sys
import os
import torch
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sim.simulate_market import simulate_market, compute_payoffs, split_data
from src.features.signatures import compute_signature_features
from src.features.build_features import build_base_features
from src.models.fnn import FNN5Hedger
from src.models.lstm import LSTM5Hedger
from src.models.bsde import DeepBSDE
from src.eval.portfolio import simulate_portfolio, simulate_portfolio_path


# ==================== A) No Look-Ahead Tests ====================

class TestNoLookAhead:
    """Verify that features at time k only use data up to time k."""

    def test_base_features_no_lookahead(self):
        """Base features at time k use S_k and tau_k only."""
        torch.manual_seed(0)
        S = torch.randn(10, 11, 2).abs() * 100  # [10, 11, 2]
        time = torch.linspace(0, 1, 11)

        base = build_base_features(S, time, T=1.0)  # [10, 10, 3]

        # At time k, logS should be log(S[:, k, :])
        for k in range(10):
            expected_logS = torch.log(S[:, k, :].clamp(min=1e-8))
            actual_logS = base[:, k, :2]
            assert torch.allclose(actual_logS, expected_logS, atol=1e-6), \
                f"Base feature at k={k} uses wrong price data"

    def test_signature_features_no_lookahead(self):
        """Signature features at time k depend only on data up to k."""
        torch.manual_seed(0)
        S_full = torch.randn(5, 11, 2).abs() * 100

        sig_full = compute_signature_features(S_full, level=2)  # [5, 10, feat]

        # Truncate: keep only first k+2 prices, fill rest with zeros
        for k in range(1, 10):
            S_trunc = S_full.clone()
            S_trunc[:, k + 1:, :] = 999.0  # Corrupt future data

            sig_trunc = compute_signature_features(S_trunc, level=2)

            # Features at time k should be identical
            assert torch.allclose(sig_full[:, k, :], sig_trunc[:, k, :], atol=1e-5), \
                f"Signature features at k={k} leak future data"

    def test_lstm_step_by_step_matches_full(self):
        """LSTM step-by-step output at k matches full-sequence output at k."""
        torch.manual_seed(42)
        feat_dim = 10
        d_traded = 2

        model = LSTM5Hedger(feat_dim, d_traded, hidden_dim=32, n_layers=2, dropout=0.0)
        model.eval()

        features = torch.randn(3, 8, feat_dim)  # [3, 8, 10]

        # Full sequence forward
        with torch.no_grad():
            deltas_full = model.compute_deltas(features)  # [3, 8, 2]

        # Step-by-step forward
        hidden = None
        for k in range(8):
            with torch.no_grad():
                delta_k, hidden = model.forward_step(features[:, k, :], hidden)

            assert torch.allclose(deltas_full[:, k, :], delta_k, atol=1e-5), \
                f"LSTM step-by-step mismatch at k={k}"


# ==================== B) Self-Financing Tests ====================

class TestSelfFinancing:
    """Verify self-financing portfolio dynamics."""

    def test_portfolio_uses_only_delta_and_price_increment(self):
        """V_{k+1} = V_k + sum_j Delta_k^j * (S_{k+1}^j - S_k^j)."""
        torch.manual_seed(0)
        n_paths, N, d = 5, 4, 2
        S = torch.randn(n_paths, N + 1, d).abs() * 100
        deltas = torch.randn(n_paths, N, d)

        V0 = 10.0
        V_path = simulate_portfolio_path(deltas, S, V0=V0)

        # Manually verify
        for k in range(N):
            dS_k = S[:, k + 1, :] - S[:, k, :]
            gain_k = (deltas[:, k, :] * dS_k).sum(dim=-1)
            expected = V_path[:, k] + gain_k
            assert torch.allclose(V_path[:, k + 1], expected, atol=1e-5), \
                f"Self-financing violated at step k={k}"

    def test_portfolio_terminal_matches_path(self):
        """simulate_portfolio terminal value matches simulate_portfolio_path terminal."""
        torch.manual_seed(0)
        n_paths, N, d = 10, 6, 2
        S = torch.randn(n_paths, N + 1, d).abs() * 100
        deltas = torch.randn(n_paths, N, d)

        V_T = simulate_portfolio(deltas, S, V0=5.0)
        V_path = simulate_portfolio_path(deltas, S, V0=5.0)

        assert torch.allclose(V_T, V_path[:, -1], atol=1e-5), \
            "Terminal portfolio value mismatch"

    def test_no_future_prices_in_delta(self):
        """Delta_k should not depend on S_{k+1}..S_N (FNN case)."""
        torch.manual_seed(42)
        feat_dim = 8
        d_traded = 2

        model = FNN5Hedger(feat_dim, d_traded, hidden_dim=32, n_layers=2, dropout=0.0)
        model.eval()

        # Create features and modify future entries
        features = torch.randn(3, 5, feat_dim)
        features2 = features.clone()
        features2[:, 3:, :] = torch.randn(3, 2, feat_dim)  # Change future

        with torch.no_grad():
            d1 = model.compute_deltas(features)
            d2 = model.compute_deltas(features2)

        # Deltas at k=0,1,2 should be identical (FNN is per-step)
        for k in range(3):
            assert torch.allclose(d1[:, k, :], d2[:, k, :], atol=1e-6), \
                f"FNN delta at k={k} depends on future features"


# ==================== C) Reproducibility Tests ====================

class TestReproducibility:
    """Verify that same seed produces identical results."""

    def test_market_simulation_reproducibility(self):
        """Same seed -> identical market data."""
        m1 = simulate_market(100, N=10, T=1.0, d_traded=2, m_brownian=3, seed=123)
        m2 = simulate_market(100, N=10, T=1.0, d_traded=2, m_brownian=3, seed=123)

        assert torch.allclose(m1.S, m2.S), "Stock prices differ with same seed"
        assert torch.allclose(m1.dW, m2.dW), "Brownian increments differ with same seed"

    def test_payoff_reproducibility(self):
        """Same market -> same payoffs."""
        m = simulate_market(50, N=10, T=1.0, d_traded=2, m_brownian=3, seed=42)
        p1_T, p1_path = compute_payoffs(m.S, K=100.0)
        p2_T, p2_path = compute_payoffs(m.S, K=100.0)

        assert torch.allclose(p1_T, p2_T), "Terminal payoffs differ"
        assert torch.allclose(p1_path, p2_path), "Payoff paths differ"

    def test_split_reproducibility(self):
        """Same seed -> same split indices."""
        m = simulate_market(100, N=5, T=1.0, d_traded=2, m_brownian=3, seed=0)

        (S_tr1,), (S_v1,), (S_te1,) = split_data(m.S, seed=99)
        (S_tr2,), (S_v2,), (S_te2,) = split_data(m.S, seed=99)

        assert torch.allclose(S_tr1, S_tr2), "Train split differs with same seed"
        assert torch.allclose(S_v1, S_v2), "Val split differs with same seed"
        assert torch.allclose(S_te1, S_te2), "Test split differs with same seed"

    def test_model_forward_reproducibility(self):
        """Same seed -> identical model initialization -> same output."""
        feat_dim = 8
        d_traded = 2
        x = torch.randn(3, feat_dim)

        torch.manual_seed(77)
        m1 = FNN5Hedger(feat_dim, d_traded, hidden_dim=32, n_layers=2)
        m1.eval()

        torch.manual_seed(77)
        m2 = FNN5Hedger(feat_dim, d_traded, hidden_dim=32, n_layers=2)
        m2.eval()

        with torch.no_grad():
            y1 = m1(x)
            y2 = m2(x)

        assert torch.allclose(y1, y2), "Model outputs differ with same seed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
