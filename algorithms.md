# Algorithms for Deep Hedging in Incomplete Markets

All pseudocode below is derived directly from the implementation.
Notation follows the thesis: $\tilde{S}$ = discounted price, $\tilde{H}$ = discounted payoff,
$e_T$ = terminal error, $s$ = shortfall, $Z$ = intrinsic process.

---

## Algorithm 1: GBM Market Simulation (Log-Euler Scheme, Calibrated)

**Input:** $n$ paths, $N$ time steps, maturity $T$, $d$ traded assets, $m$ Brownian drivers ($m > d$),
calibrated volatilities $\{\sigma_i\}_{i=1}^d$, extra loading $\sigma_{\text{extra}}$, risk-free rate $r$, strike $K$.

**Calibrated defaults** (S\&P 500, Jan 2026): $r = 0.043$, $\sigma = [0.18, 0.22]$, $\sigma_{\text{extra}} = 0.06$, $K = 1.0$, $T = 1.0$.

1. Build diffusion matrix $\sigma \in \mathbb{R}^{d \times m}$:
   - $\sigma_{i,i} = \text{vol}_i$ for $i = 1, \dots, d$
   - $\sigma_{i,j} = \sigma_{\text{extra}}$ for $i = 1, \dots, d$ and $j = d+1, \dots, m$
2. Set $\Delta t = T / N$, $\tilde{S}_0^i = 1$ for all assets $i$
3. **for** $k = 0, 1, \dots, N-1$ **do**
   - Sample $\Delta W_k \sim \mathcal{N}(0, \Delta t \cdot I_m)$, shape $[n, m]$
   - **for** each asset $i = 1, \dots, d$ **do**
     - $\log(\tilde{S}_{k+1}^i / \tilde{S}_k^i) = -\tfrac{1}{2}\|\sigma_i\|^2 \Delta t + \sigma_i \cdot \Delta W_k$
4. Compute discounted payoff: $\tilde{H} = e^{-rT} \max(K - S_T^1, 0)$
5. Compute intrinsic process: $Z_k = e^{-r t_k} \max(K - S_k^1, 0)$ for $k = 0, \dots, N$
6. Split data: 60% train, 20% validation, 20% test (deterministic permutation)

**Output:** $\tilde{S} \in \mathbb{R}^{n \times (N+1) \times d}$, $\Delta W \in \mathbb{R}^{n \times N \times m}$, $\tilde{H} \in \mathbb{R}^n$, $Z \in \mathbb{R}^{n \times (N+1)}$

---

## Algorithm 2: Heston Stochastic Volatility Simulation (Euler-Maruyama)

**Input:** $n$ paths, $N$ time steps, maturity $T$, $d$ traded assets, Heston parameters
$\{\kappa_i, \theta_i, \xi_i, \rho_i, v_0^i\}_{i=1}^d$, extra loading $\sigma_{\text{extra}}$, risk-free rate $r$, strike $K$.

**Calibrated defaults** (S\&P 500, Jan 2026): $r = 0.043$, $\kappa = [2.0, 2.0]$, $\theta = [0.04, 0.05]$,
$\xi = [0.3, 0.35]$, $\rho = [-0.7, -0.65]$, $v_0 = [0.04, 0.05]$.

**Dynamics** for each asset $i$:
$$dS_i = r \, S_i \, dt + \sqrt{v_i} \, S_i \, dW_i^S$$
$$dv_i = \kappa_i (\theta_i - v_i) \, dt + \xi_i \sqrt{v_i} \, dW_i^v$$
$$\text{corr}(dW_i^S, dW_i^v) = \rho_i$$

**Brownian structure:** $m = 2d + \mathbb{1}[\sigma_{\text{extra}} > 0]$ drivers.
Columns: $W_1^S, W_1^v, W_2^S, W_2^v, \dots, W_{\text{extra}}$.

1. Set $\Delta t = T / N$, $\tilde{S}_0^i = 1$, $v_0^i$ from parameters
2. Generate independent standard normals $Z \in \mathbb{R}^{n \times N \times m}$
3. Apply Cholesky correlation for each asset $i$:
   - $\Delta W_i^S = Z_{2i} \cdot \sqrt{\Delta t}$
   - $\Delta W_i^v = \bigl(\rho_i \, Z_{2i} + \sqrt{1 - \rho_i^2} \, Z_{2i+1}\bigr) \cdot \sqrt{\Delta t}$
4. **for** $k = 0, 1, \dots, N-1$ **do**
   - **for** each asset $i = 1, \dots, d$ **do**
     - $\sqrt{v} \leftarrow \sqrt{\max(v_k^i, 0)}$
     - Variance: $v_{k+1}^i = |v_k^i + \kappa_i (\theta_i - v_k^i) \Delta t + \xi_i \sqrt{v} \, \Delta W_{k,i}^v|$ (reflection)
     - Price: $\log(\tilde{S}_{k+1}^i / \tilde{S}_k^i) = -\tfrac{1}{2} v_k^i \, \Delta t + \sqrt{v} \, \Delta W_{k,i}^S + \sigma_{\text{extra}} \, \Delta W_{k}^{\text{extra}}$
5. Build effective diffusion matrix $\bar{\sigma} \in \mathbb{R}^{d \times m}$:
   - $\bar{\sigma}_{i, 2i} = \sqrt{\theta_i}$ (long-run vol approximation)
   - $\bar{\sigma}_{i, -1} = \sigma_{\text{extra}}$ (if extra driver exists)
6. Compute payoff, intrinsic process, and split data (same as Algorithm 1, steps 4--6)

**Incompleteness:** Variance $v_i$ is not directly tradeable; the market has $m > d$ risk sources.

**Output:** $\tilde{S}$, $\Delta W$, time grid, $\bar{\sigma}$, $V \in \mathbb{R}^{n \times (N+1) \times d}$ (variance paths)

---

## Algorithm 3: Feature Construction

**Input:** Discounted prices $\tilde{S}$, time grid $\{t_k\}$, maturity $T$, train/val/test indices,
VAE latent dimension $d_z$, signature level $L = 2$, optional variance paths $V$ (Heston).

**Part A: Base features**
- GBM: $X_k^{\text{base}} = [\log \tilde{S}_k^1, \dots, \log \tilde{S}_k^d, \; T - t_k]$ ($d + 1$ dims)
- Heston: $X_k^{\text{base}} = [\log \tilde{S}_k^1, \dots, \log \tilde{S}_k^d, \; T - t_k, \; \log v_k^1, \dots, \log v_k^d]$ ($2d + 1$ dims)

**Part B: Signature-like features** (causal, no look-ahead)
1. Increments: $\Delta_k^a = \log \tilde{S}_k^a - \log \tilde{S}_{k-1}^a$ for $k \geq 1$, $\Delta_0 = 0$
2. Level 1: $C_k^a = \sum_{j=1}^{k} \Delta_j^a$ (cumulative sum, $d$ features)
3. Level 2: $S^{(2),a,b}_k = \sum_{j=1}^{k} C_{j-1}^a \cdot \Delta_j^b$ for $a \leq b$ ($d(d+1)/2$ features)

**Part C: VAE latent features** ($d_z$ dimensions)
1. Train VAE on **training paths only**: encoder maps flattened $\log \tilde{S}$ to $(\mu, \log \sigma^2) \in \mathbb{R}^{d_z}$
2. Sample $z = \mu + \sigma \odot \epsilon$, $\epsilon \sim \mathcal{N}(0, I)$
3. Encode all paths; repeat latent vector $z$ across all $N+1$ time steps

**Combine and standardise:**
1. Concatenate: $X_k = [X_k^{\text{base}} \,\|\, z \,\|\, C_k \,\|\, S^{(2)}_k]$, dimension $d_X$
   - GBM: $d_X = (d+1) + d_z + d + d(d+1)/2 = 24$
   - Heston: $d_X = (2d+1) + d_z + d + d(d+1)/2 = 26$
2. Compute mean $\hat{\mu}$, std $\hat{\sigma}$ on **training set only**
3. Standardise all splits: $X_k \leftarrow (X_k - \hat{\mu}) / \hat{\sigma}$

**Output:** $X^{\text{train}}, X^{\text{val}}, X^{\text{test}} \in \mathbb{R}^{\cdot \times (N+1) \times d_X}$

---

## Algorithm 4: Two-Stage Hedging (NN1 + NN2 Controller)

**Input:** Features $X \in \mathbb{R}^{n \times N \times d_X}$, intrinsic process $Z$,
price increments $\Delta S_k = \tilde{S}_{k+1} - \tilde{S}_k$, controller flag.

1. Compute baseline hedge ratios: $\Delta_k^0 = \text{NN1}(X_k)$ for all $k$ (batched)
2. Set $V_0 = 0$, $V_{\max} = 0$

3. **for** $k = 0, 1, \dots, N-1$ **do**

   **If using controller (NN2):**
   - Compute causal P/L features:
     - $\text{PL}_k = V_k$ (cumulative P/L)
     - $\text{dPL}_k = V_k - V_{k-1}$ (step P/L; 0 if $k = 0$)
     - $V_{\max} \leftarrow \max(V_{\max}, V_k)$
     - $\text{DD}_k = V_k - V_{\max}$ (drawdown, $\leq 0$)
     - $\text{Gap}_k = V_k - Z_k$ (intrinsic gap)
     - $\text{RolStd}_k = \text{std}(\text{dPL}_{k-9:k})$ (rolling std, 10-step window)
   - Concatenate: $F_k = [X_k \,\|\, \text{PL}_k, \text{dPL}_k, \text{DD}_k, \text{Gap}_k, \text{RolStd}_k]$
   - $(g_k, \Delta_k^1) = \text{NN2}(F_k)$ where $g_k = \text{sigmoid}(\cdot) \in [0,1]$, $\Delta_k^1 = \delta_{\text{clip}} \cdot \tanh(\cdot)$
   - Final hedge: $\Delta_k = \Delta_k^0 + g_k \cdot \Delta_k^1$

   **If no controller:**
   - $\Delta_k = \Delta_k^0$

   **Self-financing update:**
   - $V_{k+1} = V_k + \Delta_k^\top \Delta S_k$

**Output:** $V_T \in \mathbb{R}^n$ (terminal portfolio values), $\{V_k\}_{k=0}^N$ (portfolio path)

---

## Algorithm 5: Deep BSDE Forward Propagation

**Input:** Features $X \in \mathbb{R}^{n \times (N+1) \times d_X}$, Brownian increments $\Delta W \in \mathbb{R}^{n \times N \times m}$,
time grid $\{t_k\}$, diffusion matrix $\sigma$, number of sub-steps $M$, optional effective sigma $\bar{\sigma}$.

1. Initialise: $Y_0 = \hat{Y}_0$ (learnable scalar parameter)
2. Pre-compute pseudo-inverse for Z-to-Delta projection:
   - If $\bar{\sigma}$ provided (Heston): $(\bar{\sigma}^\top)^+ = \text{pinv}(\bar{\sigma}^\top) \in \mathbb{R}^{d \times m}$
   - Else (GBM): $(\sigma^\top)^+ = \text{pinv}(\sigma^\top) \in \mathbb{R}^{d \times m}$

3. **for** $k = 0, 1, \dots, N-1$ **do**

   **If** $M = 0$ (standard):
   - $\tau_k = \text{SinEmbed}(t_k)$ (sinusoidal embedding, 32 dims)
   - $Z_k = f_\theta([X_k \,\|\, \tau_k])$ via Z-network, $Z_k \in \mathbb{R}^m$
   - $Y_{k+1} = Y_k + Z_k \cdot \Delta W_k$

   **If** $M > 0$ (sub-stepping):
   - $\delta t = (t_{k+1} - t_k) / (M + 1)$
   - Split $\Delta W_k$ into $M+1$ sub-increments $\{\delta W_s\}$ summing to $\Delta W_k$
   - **for** $s = 0, \dots, M$ **do**
     - $t_s = t_k + s \cdot \delta t$
     - $Z_s = f_\theta([X_k \,\|\, \text{SinEmbed}(t_s)])$
     - $Y \leftarrow Y + Z_s \cdot \delta W_s$

4. **Z-to-Delta projection** (recover tradeable hedge ratios):
   - $\Delta_k = (\bar{\sigma}^\top)^+ Z_k \;/\; \tilde{S}_k$ (element-wise division by asset prices)

**Output:** $Y_N \in \mathbb{R}^n$ (terminal BSDE value), $\{Y_k\}$, $\{Z_k\}$, $\{\Delta_k\}$

---

## Algorithm 6: Hedger Training (FNN/LSTM + Controller)

**Input:** Training data $(X^{\text{tr}}, Z^{\text{tr}}, \Delta S^{\text{tr}}, \tilde{H}^{\text{tr}})$,
validation data $(X^{\text{val}}, \dots)$, hyperparameters $(\eta, B, \alpha, \beta, q, \lambda_1, \lambda_2, P)$.

1. Initialise NN1 (and NN2 if using controller)
2. $\theta \leftarrow$ all trainable parameters of NN1 (and NN2)
3. Optimiser $\leftarrow \text{Adam}(\theta, \text{lr} = \eta)$
4. Best validation metric $\leftarrow \infty$, patience counter $\leftarrow 0$

5. **for** epoch $= 1, 2, \dots, E_{\max}$ **do**
   - Randomly permute training indices
   - **for** each mini-batch $\mathcal{B}$ of size $B$ **do**
     - Forward pass: $V_T = \text{Algorithm 4}(X_\mathcal{B}, Z_\mathcal{B}, \Delta S_\mathcal{B})$
     - Terminal error: $e_T = V_T - \tilde{H}_\mathcal{B}$
     - Shortfall: $s = \max(\tilde{H}_\mathcal{B} - V_T, 0)$
     - Data loss: $\mathcal{L}_{\text{data}} = \text{MSE}(e_T) + \alpha \cdot \bar{s} + \beta \cdot \text{CVaR}_q(s)$
     - Regularisation: $\mathcal{L}_{\text{reg}} = \lambda_1 \sum_W |W| + \lambda_2 \sum_W W^2$ (weights only, excludes biases/LayerNorm/$Y_0$)
     - Total loss: $\mathcal{L} = \mathcal{L}_{\text{data}} + \mathcal{L}_{\text{reg}}$
     - $\theta \leftarrow \theta - \eta \cdot \text{Adam}(\nabla_\theta \mathcal{L})$ with $\|\nabla\|$ clipped to 10

   - **Validation:** $\text{CVaR}_q^{\text{val}} = \text{CVaR}_q\bigl(\max(\tilde{H}^{\text{val}} - V_T^{\text{val}}, 0)\bigr)$
   - **Early stopping:** if $\text{CVaR}_q^{\text{val}}$ improves, save $\theta^*$ and reset counter; else increment counter
   - **if** counter $\geq P$ **then** break

6. Restore $\theta \leftarrow \theta^*$ (best weights)

**Output:** Trained NN1 (and NN2), training/validation loss curves

---

## Algorithm 7: Deep BSDE Training

**Input:** Training data $(X^{\text{tr}}, \Delta W^{\text{tr}}, \tilde{H}^{\text{tr}}, \{t_k\})$,
validation data, hyperparameters $(\eta, B, \lambda_1, \lambda_2, q, M, P)$.

1. Initialise DeepBSDE model (learnable $\hat{Y}_0$, Z-network $f_\theta$)
2. Optimiser $\leftarrow \text{Adam}(\theta, \text{lr} = \eta)$

3. **for** epoch $= 1, 2, \dots, E_{\max}$ **do**
   - **for** each mini-batch $\mathcal{B}$ of size $B$ **do**
     - Forward: $Y_N = \text{Algorithm 5}(X_\mathcal{B}, \Delta W_\mathcal{B}, \{t_k\}, M)$
     - Loss: $\mathcal{L} = \text{MSE}(Y_N - \tilde{H}_\mathcal{B}) + \lambda_1 \sum_W |W| + \lambda_2 \sum_W W^2$
     - $\theta \leftarrow \theta - \eta \cdot \text{Adam}(\nabla_\theta \mathcal{L})$ with $\|\nabla\|$ clipped to 10

   - **Validation:** $\text{CVaR}_q^{\text{val}} = \text{CVaR}_q\bigl(\max(\tilde{H}^{\text{val}} - Y_N^{\text{val}}, 0)\bigr)$
   - **Early stopping** with patience $P$ (same as Algorithm 6)

4. Restore best weights

**Output:** Trained DeepBSDE, learned $\hat{Y}_0$

---

## Algorithm 8: Two-Stage Bias Control Protocol

**Input:** Model classes $\mathcal{M} = \{\text{FNN}, \text{LSTM}, \text{DBSDE}\}$,
hyperparameter grid $\mathcal{G}$, architecture seed $s_0$, robustness seeds $\{s_1, \dots, s_R\}$,
market models $\mathcal{K} = \{\text{GBM}, \text{Heston}\}$.

The full protocol runs independently for each market model $\mathcal{K}_j$ using
the same data split (shared indices across market models).

**Stage 1: Optuna HP Search (TPE Bayesian optimisation)**

1. **for** each model class $m \in \mathcal{M}$ **do**
   - Set seed $s_0$
   - **for** trial $= 1, \dots, T_{\max}$ (TPE-sampled) **do**
     - Sample: depth $\in \{3, 5, 7\}$, width $\in \{64, 128, 256\}$, activation $\in \{\text{relu}, \text{tanh}, \text{alt}_1, \text{alt}_2\}$, lr $\in \{3\!\times\!10^{-4}, 10^{-3}, 3\!\times\!10^{-3}\}$
     - Train model $m$ with sampled config (Algorithm 6 or 7)
     - Record $\text{CVaR}_{0.95}^{\text{val}}$ and $\text{MSE}^{\text{val}}$
   - Select best: $\arg\min$ CVaR$_{0.95}^{\text{val}}$, tie-break by MSE, then prefer smaller lr
   - Store $c_m^* = (\text{depth}^*, \text{width}^*, \text{act}^*, \text{lr}^*)$

**Stage 2: Seed Robustness (Validation)**

2. **for** each model class $m \in \mathcal{M}$ **do**
   - **for** each seed $s \in \{s_1, \dots, s_R\}$ **do**
     - Train $m$ with config $c_m^*$ and seed $s$ (Algorithm 6 or 7)
     - Evaluate on validation set: $\text{metrics}_s$
     - Generate per-model diagnostic plots (8 per seed)
   - Aggregate: $\bar{\mu} \pm \sigma$ across seeds for each metric

**Validation Analysis:**

3. Best model: $m^* = \arg\min_{m \in \mathcal{M}} \bar{\text{CVaR}}_{0.95}^{\text{val}}(m)$
4. Generate comparison plots: grouped bars, error overlays, CVaR curves, highlighted summary

**Cross-model comparison** (when both market models run):

5. Compare P\&L distributions per model across GBM vs Heston (histogram + violin)
6. Compare all models under each regime (all-model overlay per market model)
7. Grouped metric bar charts: GBM vs Heston for each metric

**Output:** Best configs $\{c_m^*\}$, aggregated validation metrics per market model, best model $m^*$, all diagnostic and comparison plots

---

## Algorithm 9: FNN Hedger (Method I: Feedforward Neural Network)

**Input:** Features $X_k \in \mathbb{R}^{d_X}$ at time step $k$, network depth $D$, width $W$,
activation schedule $\phi$, dropout rate $p$.

**Architecture** (parameter-shared across all time steps):

1. $h^{(0)} = X_k$
2. **for** $\ell = 1, \dots, D$ **do**
   - $h^{(\ell)} = \text{Dropout}_p\bigl(\phi_\ell\bigl(\text{LayerNorm}(W^{(\ell)} h^{(\ell-1)} + b^{(\ell)})\bigr)\bigr)$
3. $\Delta_k^0 = W^{\text{out}} h^{(D)} + b^{\text{out}} \in \mathbb{R}^d$

**Batched forward pass** (all time steps at once):
1. Reshape: $[n, N, d_X] \to [n \cdot N, \; d_X]$
2. Apply network: $\Delta^0 \in \mathbb{R}^{n \cdot N \times d}$
3. Reshape: $[n \cdot N, \; d] \to [n, N, d]$

**Note:** Each block is Linear $\to$ LayerNorm $\to$ Activation $\to$ Dropout.
Activation $\phi_\ell$ depends on the schedule: all-ReLU, all-Tanh, or alternating.

**Output:** $\Delta^0 \in \mathbb{R}^{n \times N \times d}$ (baseline hedge ratios for all paths and time steps)

---

## Algorithm 10: LSTM Hedger (Method II: Long Short-Term Memory Network)

**Input:** Feature sequence $X = \{X_k\}_{k=0}^{N-1} \in \mathbb{R}^{n \times N \times d_X}$,
LSTM layers $L$, hidden size $H$, activation schedule $\phi$, dropout rate $p$.

**Architecture:**

1. **Pre-MLP** (project features to hidden dimension):
   - $\tilde{X}_k = \phi_0\bigl(\text{LayerNorm}(W^{\text{pre}} X_k + b^{\text{pre}})\bigr) \in \mathbb{R}^H$ for all $k$

2. **Stacked LSTM** (single forward pass over full sequence):
   - Initialise: $h_0^{(\ell)} = 0$, $c_0^{(\ell)} = 0$ for layers $\ell = 1, \dots, L$
   - **for** $k = 0, 1, \dots, N-1$ **do** (computed in parallel by PyTorch):
     - Input to layer 1: $\tilde{X}_k$
     - **for** $\ell = 1, \dots, L$ **do**
       - $(h_k^{(\ell)}, c_k^{(\ell)}) = \text{LSTMCell}^{(\ell)}(h_{k-1}^{(\ell)}, c_{k-1}^{(\ell)}, h_k^{(\ell-1)})$
       - Inter-layer dropout ($p$) between layers if $L > 1$
   - Output: $O_k = h_k^{(L)} \in \mathbb{R}^H$ for all $k$

3. **Head MLP** (map LSTM output to hedge ratios):
   - $\Delta_k^0 = W^{\text{out}}_2 \cdot \phi_1\bigl(\text{LayerNorm}(W^{\text{out}}_1 O_k + b_1^{\text{out}})\bigr) + b_2^{\text{out}} \in \mathbb{R}^d$

**Output:** $\Delta^0 \in \mathbb{R}^{n \times N \times d}$ (baseline hedge ratios)

---

## Algorithm 11: Deep BSDE Solver (Method III)

**Input:** Features $X \in \mathbb{R}^{n \times (N+1) \times d_X}$, Brownian increments $\Delta W \in \mathbb{R}^{n \times N \times m}$,
time grid $\{t_k\}_{k=0}^N$, diffusion matrix $\sigma \in \mathbb{R}^{d \times m}$,
network depth $D$, width $W$, sub-steps $M$, optional effective sigma $\bar{\sigma} \in \mathbb{R}^{d \times m}$.

**Architecture:**

**Learnable initial value:** $\hat{Y}_0 \in \mathbb{R}$ (scalar parameter, initialised to 0.05)

**Sinusoidal time embedding** ($d_\tau = 32$):
- Frequencies: $\omega_j = e^{-4j/(d_\tau/2 - 1)}$ for $j = 0, \dots, d_\tau/2 - 1$
- $\tau(t) = [\sin(\omega_0 t), \dots, \sin(\omega_{d_\tau/2-1} t), \; \cos(\omega_0 t), \dots, \cos(\omega_{d_\tau/2-1} t)]$

**Z-network** $f_\theta: \mathbb{R}^{d_X + d_\tau} \to \mathbb{R}^m$:
1. Input: $[X_k \,\|\, \tau(t_k)]$
2. $D$ blocks of Linear $\to$ LayerNorm $\to$ Activation $\to$ Dropout
3. Linear head $\to Z_k \in \mathbb{R}^m$

**Forward propagation** (BSDE dynamics with $f \equiv 0$):
1. $Y_0 = \hat{Y}_0$ (broadcast to batch)
2. **for** $k = 0, 1, \dots, N-1$ **do**
   - $Z_k = f_\theta([X_k \,\|\, \tau(t_k)])$
   - $Y_{k+1} = Y_k + \sum_{j=1}^m Z_k^j \cdot \Delta W_k^j$
3. Terminal loss: $\mathcal{L} = \frac{1}{n}\sum_{i=1}^n (Y_N^{(i)} - \tilde{H}^{(i)})^2$

**Z-to-Delta projection** (recover tradeable hedge from BSDE control):
- Relationship: $Z_k = \sigma^\top \, \text{diag}(\tilde{S}_k) \, \Delta_k$
- GBM: $\Delta_k = (\sigma^\top)^+ Z_k \;/\; \tilde{S}_k$
- Heston: $\Delta_k = (\bar{\sigma}^\top)^+ Z_k \;/\; \tilde{S}_k$ where $\bar{\sigma}_{i,2i} = \sqrt{\theta_i}$ (long-run vol)

**Output:** $Y_N$ (terminal value), $\hat{Y}_0$ (learned option price), $\{Z_k\}$, $\{\Delta_k\}$ (hedge ratios)

---

## Algorithm 12: Loss Functions

**Terminal error:** $e_T = V_T - \tilde{H}$ (positive = over-hedge, negative = under-hedge)

**Shortfall:** $s = \max(\tilde{H} - V_T, \; 0) \geq 0$ (under-hedging cost)

**CVaR computation:**
Given loss vector $\ell \in \mathbb{R}^n$ and quantile $q \in (0,1)$:
1. $k = \max(1, \lfloor (1 - q) \cdot n \rfloor)$
2. Sort $\ell$ in descending order: $\ell_{(1)} \geq \ell_{(2)} \geq \dots \geq \ell_{(n)}$
3. $\text{CVaR}_q(\ell) = \frac{1}{k} \sum_{i=1}^{k} \ell_{(i)}$

**Hedging loss** (FNN/LSTM): $\mathcal{L} = \text{MSE}(e_T) + \alpha \cdot \mathbb{E}[s] + \beta \cdot \text{CVaR}_q(s)$

**BSDE loss:** $\mathcal{L} = \text{MSE}(Y_N - \tilde{H})$

**Elastic net** (weights only, excludes biases, LayerNorm, $Y_0$): $\mathcal{R} = \lambda_1 \sum_W |W| + \lambda_2 \sum_W W^2$
