# Deep Hedging in Incomplete Markets

Research codebase for MSc Mathematical Statistics thesis comparing deep hedging approaches in incomplete markets with VAE and signature-like features under two market dynamics: **GBM** (constant volatility) and **Heston** (stochastic volatility), both calibrated to S&P 500 data.

## Market Models

| Model | Dynamics | Incompleteness Source |
|-------|----------|----------------------|
| **GBM** | Geometric Brownian Motion, constant volatility | Extra untraded Brownian drivers (`m > d`) |
| **Heston** | Stochastic volatility (Euler-Maruyama, Cholesky correlation, reflection scheme) | Variance process is not directly tradeable |

Both models are calibrated to S&P 500 / CBOE VIX data (see `data/market_params_sp500.json`):
- **GBM**: r=0.043, vols=[0.18, 0.22], extra_vol=0.06
- **Heston**: r=0.043, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04

## Hedging Models

| Model | Description |
|-------|-------------|
| **FNN-5** | 5-layer feedforward network (128 units, ReLU, LayerNorm, Dropout=0.1) |
| **LSTM-5** | 5-layer stacked LSTM (hidden_size=128, inter-layer dropout=0.1) |
| **DBSDE** | Deep BSDE solver with learnable Y0, time-embedded Z-network, and Z-to-Delta projection |

## Quick Start

### Quick test (~10 min on GPU)

```bash
python run_experiment.py --quick --market_model both
```

### GBM only

```bash
python run_experiment.py --quick --market_model gbm
```

### Heston only

```bash
python run_experiment.py --quick --market_model heston
```

### Full run (100k paths, both market models)

```bash
python run_experiment.py \
  --paths 100000 \
  --N 200 \
  --epochs 1000 \
  --patience 15 \
  --batch_size 2048 \
  --n_trials 60 \
  --seeds 0 1 2 3 4 \
  --substeps 0 5 10 \
  --market_model both
```

## Google Colab

Open `colab_run.ipynb` on Google Colab (A100 GPU recommended). The notebook clones the repo, runs tests, and executes the full pipeline. Quick test takes ~10 min; full run takes ~4-8 hours.

## Reproducing Exact Results

Reproducibility is ensured by fixing seeds at all stages:
- Market simulation uses `seed=42` for data generation
- Data splitting uses `seed=42` for consistent train/val/test partitions
- Each model is trained with seeds specified via `--seeds` (default: 0, 1, 2, 3, 4)
- Optuna HP search uses fixed seed per trial
- Results are reported as mean +/- std across seeds

To reproduce: run with the same `--seeds` and `--paths` values.

## Replotting Without Rerunning

After a full run, comparison data is saved to `outputs/comparison/comparison_data.pt`. To adjust figure dimensions, colors, grids, or fonts without rerunning the experiment:

1. Edit the `STYLE` dict at the top of `replot.py`
2. Run:

```bash
python replot.py --data outputs/comparison/comparison_data.pt \
                 --metrics outputs/metrics_summary.json \
                 --out outputs/comparison
```

## Validation Tests

```bash
python -m pytest tests/test_validation.py -v
```

13 tests verify:
- **No look-ahead**: Features at time k use only data up to time k
- **Self-financing**: Portfolio dynamics V_{k+1} = V_k + Delta_k . dS_k
- **Reproducibility**: Same seed produces identical results
- **Heston simulation**: Variance stays non-negative, correct correlation structure
- **Feature dimensions**: GBM (d_X=24) and Heston (d_X=26) feature sizes are correct

## Output Structure

```
outputs/
  gbm/
    plots_val/                    # GBM validation plots
    plots_3d/                     # GBM 3D delta surfaces
    checkpoints/                  # GBM model state dicts
    val_metrics.json              # GBM validation metrics
  heston/
    plots_val/                    # Heston validation plots
    plots_3d/                     # Heston 3D delta surfaces
    plots_heston/                 # Variance paths, implied vol surface
    checkpoints/                  # Heston model state dicts
    val_metrics.json              # Heston validation metrics
  comparison/
    comparison_data.pt            # Saved data for replotting
    gbm_vs_heston_bars.png        # Metric bar chart comparison
    *_hist.png, *_violin.png      # Per-model and cross-model P&L plots
  metrics_summary.json            # Combined metrics (both market models)
```

## Key Concepts

### Total Error
Terminal hedging error: `total_error = V_T - H`, where V_T is the terminal portfolio value and H is the option payoff.

### Worst Error
Worst shortfall across all exercise times: `worst_error = min_k (V_k - Z_k)`, where Z_k is the Bermudan payoff process at time k. Negative values indicate the portfolio value fell below the intrinsic value.

### Elastic Net Regularization
All models are trained with combined L1 + L2 regularization on weight parameters only (biases and LayerNorm parameters excluded):
```
TotalLoss = DataLoss + l1_lambda * sum(|W|) + l2_lambda * sum(W^2)
```

### Features
Base features differ by market model:
- **GBM**: log(S_tilde), tau (time-to-maturity) — d_base = 2 * d_traded + 1
- **Heston**: log(S_tilde), log(v), tau — d_base = 3 * d_traded + 1

All models additionally use VAE latent features and level-2 signature features.

## Project Structure

```
src/
  sim/simulate_market.py        # GBM Euler-Maruyama with correlated BM
  sim/simulate_heston.py        # Heston stochastic vol (Euler-Maruyama, reflection)
  sim/calibration.py            # Load calibrated market parameters
  features/vae.py               # VAE for latent path features
  features/signatures.py        # Signature-like features (no external deps)
  features/build_features.py    # Combined feature pipeline (GBM / Heston)
  models/fnn.py                 # FNN-5 hedger
  models/lstm.py                # LSTM-5 hedger
  models/bsde.py                # Deep BSDE solver/hedger
  training/train.py             # Training loop + Optuna HP search
  training/early_stopping.py    # Early stopping
  eval/portfolio.py             # Self-financing portfolio simulation
  eval/metrics.py               # MAE, MSE, R2, error statistics
  eval/plots.py                 # Validation and diagnostic plots
  eval/plots_heston.py          # GBM vs Heston comparison plots
data/
  market_params_sp500.json      # Calibrated S&P 500 parameters
tests/
  test_validation.py            # Validation test suite (13 tests)
run_experiment.py               # Main entrypoint
replot.py                       # Standalone figure regeneration
colab_run.ipynb                 # Google Colab notebook
algorithms.md                   # Pseudocode for all algorithms
```

## Dependencies

- Python >= 3.8
- PyTorch >= 1.12
- NumPy
- Matplotlib
- Optuna (hyperparameter optimization)
- pytest (for tests)
