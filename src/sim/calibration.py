"""Market calibration: load calibrated GBM parameters.

Default values are calibrated to S&P 500 / CBOE VIX data (Jan 2026).
"""
import json
import os


_DEFAULT_CONFIG = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "market_params_sp500.json"
)


def load_market_params(config_path=None):
    """Load calibrated GBM market parameters.

    Returns:
        dict with keys: r, vols, extra_vol, K, T
    """
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            data = json.load(f)
        return data["gbm"]

    if os.path.exists(_DEFAULT_CONFIG):
        with open(_DEFAULT_CONFIG) as f:
            data = json.load(f)
        return data["gbm"]

    # Fallback defaults (S&P 500 calibrated)
    return {
        "r": 0.043,
        "vols": [0.18, 0.22],
        "extra_vol": 0.06,
        "K": 1.0,
        "T": 1.0,
    }
