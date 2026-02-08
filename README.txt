================================================================================
  DEEP HEDGING IN INCOMPLETE MARKETS
  Neural-Network Hedging with Two-Stage Controller
================================================================================

1. OVERVIEW
-----------
This codebase implements neural-network-based hedging strategies for European
put options in an incomplete diffusion market. Three model classes are compared:

  - FNN:   Feedforward neural network baseline hedger + NN2 controller
  - LSTM:  Long Short-Term Memory baseline hedger + NN2 controller
  - DBSDE: Deep Backward Stochastic Differential Equation solver/hedger

The market is incomplete: m_brownian=3 Brownian drivers drive d_traded=2 assets,
creating unhedgeable risk. N=200 time steps discretize T=1.0 years.


2. TWO-STAGE HEDGER (NN1 + NN2 CONTROLLER)
-------------------------------------------
For FNN and LSTM experiments, a two-stage architecture is used:

  NN1 (baseline hedger):
    - Inputs: market features X_k at time step k
    - Outputs: baseline hedge Delta0_k in R^d_traded

  NN2 (controller):
    - Inputs: market features X_k + causal P/L features at step k
    - Outputs: gate g_k in [0,1] (sigmoid) and correction Delta1_k
    - Correction is clamped: Delta1_k = tanh(raw) * delta_clip

  Final hedge: Delta_k = Delta0_k + g_k * Delta1_k

  P/L features (strictly causal -- only use data up to step k):
    - PL_k:          cumulative P/L (V_k - V_0)
    - dPL_k:         daily P/L (V_k - V_{k-1})
    - DD_k:          drawdown (V_k - running_max_V, always <= 0)
    - intrinsic_gap: V_k - Z_k (portfolio vs intrinsic value)
    - rolling_std:   rolling std of dPL over last 10 steps

The controller allows the model to adapt its hedging strategy based on how
the portfolio has performed so far, adding risk management on top of NN1.


3. DISCOUNTED SELF-FINANCING FORMULATION
-----------------------------------------
All prices use discounted values: S_tilde_k = exp(-r*t_k) * S_k.
Portfolio dynamics (no cash injections):

  V_{k+1} = V_k + Delta_k^T * (S_tilde_{k+1} - S_tilde_k)

Payoff: H_tilde = exp(-r*T) * max(K - S_T^1, 0)
Terminal error: e_T = V_T - H_tilde
Shortfall: s = max(H_tilde - V_T, 0) = max(-e_T, 0)


4. FEATURES (STRICTLY CAUSAL)
------------------------------
Features at time k depend only on data up to and including time k:

  Base:        log(S_tilde_k^1), log(S_tilde_k^2), tau_k = T - t_k
  VAE:         per-path latent vector (trained on train set only, frozen)
  Signatures:  level-2 cumulative path statistics (no external libraries)

Standardization: mean/std computed on training set only, applied to all sets.


5. TRAINING OBJECTIVE
---------------------
Loss = MSE(e_T) + alpha * mean(shortfall) + beta * CVaR_q(shortfall)

Defaults: alpha=1.0, beta=1.0, q=0.95.
CVaR_q(shortfall) = average of the worst (1-q)% shortfalls in the batch.

Elastic net regularization on weight tensors only (excludes biases, LayerNorm):
  Reg = l1 * sum(|W|) + l2 * sum(W^2)

DBSDE uses MSE(Y_N - H_tilde) as training loss, with selection still based
on validation CVaR95(shortfall).


6. WHY NEGATIVE ERRORS AND CVaR
---------------------------------
In incomplete markets, perfect replication is impossible. Terminal errors e_T
can be positive (over-hedge, V_T > H_tilde) or negative (under-hedge).

Negative errors are expected. The focus is on controlling the tail of
under-hedging (shortfall). CVaR provides a coherent risk measure that
penalizes the worst-case shortfalls, encouraging the model to avoid
catastrophic under-hedging even if average performance is slightly worse.


7. ARCHITECTURE + LR SELECTION PROTOCOL
-----------------------------------------
Two-stage bias control:

STAGE 1 (architecture + LR selection):
  - Fix seed = seed_arch (default 0)
  - Fix simulation (one dataset), fixed train/val/test split
  - For each model class, search over:
      depth x width x activation_schedule x learning_rate
  - Select by: (1) validation CVaR95(shortfall), (2) validation MSE, (3) smaller LR
  - Rationale: smaller LRs often converge more stably for LSTM/DBSDE and CVaR

STAGE 2 (seed robustness):
  - Use best (architecture + LR) per model class from Stage 1
  - Run seeds in {0, 1, 2, 3, 4}
  - Report mean +/- std + 95% CI across seeds
  - Representative seed = argmin validation CVaR95(shortfall)
  - Do NOT retune LR per seed

Why prefer smaller LR when metrics are close:
  - LSTM hidden state dynamics are sensitive to large gradient updates
  - CVaR loss landscapes can be noisy; smaller LR smooths convergence
  - DBSDE Y0 parameter benefits from gradual optimization


8. LEARNING RATES
-----------------
Exactly three learning rates: {3e-4, 1e-3, 3e-3}.

Model-specific notes:
  - LSTM: if training instability (NaN/spikes), automatically drop to next smaller
  - DBSDE: {3e-4, 1e-3} preferred; 3e-3 only if stable


9. DBSDE SOLVER
---------------
Separate module (no NN2 controller chaining):
  - Learns Y0 (scalar, initial portfolio value ~ option price)
  - z_net: maps (features, time_embedding) -> Z_k in R^m_brownian
  - Propagation: Y_{k+1} = Y_k + <Z_k, dW_k>  (driver f=0)
  - Loss: MSE(Y_N - H_tilde)
  - Z-to-Delta projection: Delta = pinv(sigma^T) @ Z / S_tilde

Substeps study: test substeps in {0, 5, 10} to verify N=200 is sufficient.


10. RUNNING THE EXPERIMENT
--------------------------
Quick test (fast, small):
  python run_experiment.py --quick

Full experiment:
  python run_experiment.py \
    --paths 20000 --N 200 --T 1.0 --d_traded 2 --m_brownian 3 \
    --K 1.0 --r 0.0 \
    --latent_dim 16 --sig_level 2 \
    --depth_grid 3 5 --width_grid 64 128 \
    --act_schedules relu_all tanh_all alt_relu_tanh alt_tanh_relu \
    --lrs 3e-4 1e-3 3e-3 \
    --seed_arch 0 \
    --seeds 0 1 2 3 4 \
    --use_controller 1 \
    --tbptt 50 \
    --l1 0 --l2 1e-4 \
    --objective cvar_shortfall \
    --cvar_q 0.95 \
    --substeps 0 5 10

Full production run (100k paths):
  python run_experiment.py --paths 100000 --epochs 1000 --patience 10


11. OUTPUT STRUCTURE
--------------------
  outputs/
    metrics_summary.csv          Aggregated metrics table
    metrics_summary.json         Full results + configs + grid logs
    data/
      features_{train,val,test}.pt    Standardized feature tensors
      payoff_T_{train,val,test}.pt    Terminal payoffs
      payoff_path_{train,val,test}.pt Intrinsic value process
    checkpoints/
      {FNN,LSTM,DBSDE}_seed{N}.pt    Best model checkpoints per seed
    run_configs/
      split_indices.json              Train/val/test indices + hash
      stage1_grid.json                Full grid search results
    plots/
      {model}_seed{N}_loss_curves.png
      {model}_seed{N}_error_hist.png
      {model}_seed{N}_shortfall_hist.png
      {model}_seed{N}_overlay_hist.png
      {model}_seed{N}_cvar_curve.png
      {model}_seed{N}_scatter.png
      {model}_seed{N}_daily_pnl.png
      {model}_seed{N}_drawdown.png
      substeps_convergence.png
      summary_table.png
    plots_3d/
      {model}_delta_surface_k{0,100,199}.html   Interactive Plotly surfaces


12. TESTS
---------
Run validation tests:
  python -m pytest tests/test_validation.py -v

Tests verify:
  - No look-ahead in base and signature features
  - Self-financing portfolio constraint
  - Controller causality (NN2 inputs exclude future data)
  - Reproducibility across identical seeds
  - LR grid matches specification
  - LSTM output consistency


13. DEPENDENCIES
----------------
  torch >= 2.0
  numpy
  matplotlib
  plotly (for 3D interactive plots)
  pytest (for tests)


14. REFERENCE
-------------
Inspired by: Guo, Langrene, Wu (2023) - "Simultaneous upper and lower
bounds of American-style option prices with hedging via neural networks"
================================================================================
