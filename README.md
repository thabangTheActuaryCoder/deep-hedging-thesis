# Deep Hedging in Incomplete Markets

Research codebase for MSc Mathematical Statistics thesis comparing deep hedging approaches in incomplete markets with VAE and signature-like features.

## Models

| Model | Description |
|-------|-------------|
| **FNN-5** | 5-layer feedforward network (128 units, ReLU, LayerNorm, Dropout=0.1) |
| **LSTM-5** | 5-layer stacked LSTM (hidden_size=128, inter-layer dropout=0.1) |
| **DBSDE** | Deep BSDE solver with learnable Y0, time-embedded Z-network, and Z-to-Delta projection |

## Quick Start

### Trial run (fast, ~5k paths)

```bash
python run_experiment.py --quick
```

### Full run (20k paths, 3 seeds, HP sweep)

```bash
python run_experiment.py --paths 20000 --N 50 --T 1.0 --d_traded 2 --m_brownian 3 \
  --latent_dim 16 --sig_level 2 \
  --hidden 128 --lstm_layers 5 \
  --l1 1e-6 --l2 1e-4 \
  --lrs 3e-4 1e-3 3e-3 \
  --substeps 0 5 10 20 30 \
  --seeds 0 1 2
```

### Large-scale run (100k paths)

```bash
python run_experiment.py --paths 100000 --epochs 300 --seeds 0 1 2
```

## Reproducing Exact Results

Reproducibility is ensured by fixing seeds at all stages:
- Market simulation uses `seed=42` for data generation
- Data splitting uses `seed=42` for consistent train/val/test partitions
- Each model is trained with seeds specified via `--seeds` (default: 0, 1, 2)
- Results are reported as mean +/- std across seeds

To reproduce: run with the same `--seeds` and `--paths` values.

## Validation Tests

```bash
python -m pytest tests/test_validation.py -v
```

Tests verify:
- **No look-ahead**: Features at time k use only data up to time k
- **Self-financing**: Portfolio dynamics V_{k+1} = V_k + Delta_k . dS_k
- **Reproducibility**: Same seed produces identical results

## Output Structure

```
outputs/
  metrics_summary.csv     # Mean +/- std of all metrics across seeds
  metrics_summary.json    # Full results including hyperparameters
  data/                   # Saved features and payoffs
    features_{train,val,test}.pt
    payoff_T_{train,val,test}.pt
    payoff_path_{train,val,test}.pt
  plots/
    {Model}_seed{s}_loss_curves.png     # Train vs validation loss
    {Model}_seed{s}_error_hist.png      # Total vs worst error histograms
    {Model}_seed{s}_scatter.png         # V_T vs H scatter
    substeps_convergence.png            # MSE/worst error vs substeps
    {Model}_function_shape.png          # Hedge ratio shape over time
    summary_table.png                   # Final comparison table
  checkpoints/
    {Model}_seed{s}.pt                  # Model state dicts
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

### Incomplete Market
The market has `m_brownian` > `d_traded` Brownian drivers, creating unhedgeable risk. Each traded asset loads on its own Brownian factor (vol=0.2) plus small loadings (vol=0.05) on extra factors.

## Project Structure

```
src/
  sim/simulate_market.py      # Euler-Maruyama with correlated BM
  features/vae.py             # VAE for latent path features
  features/signatures.py      # Signature-like features (no external deps)
  features/build_features.py  # Combined feature pipeline
  models/fnn.py               # FNN-5 hedger
  models/lstm.py              # LSTM-5 hedger
  models/bsde.py              # Deep BSDE solver/hedger
  training/train.py           # Training loop + HP sweep
  training/early_stopping.py  # Early stopping
  eval/portfolio.py           # Self-financing portfolio simulation
  eval/metrics.py             # MAE, MSE, R2, error statistics
  eval/plots.py               # All plotting functions
tests/
  test_validation.py          # Validation test suite
run_experiment.py             # Main entrypoint
```

## Dependencies

- Python >= 3.8
- PyTorch >= 1.12
- NumPy
- Matplotlib
- pytest (for tests)
