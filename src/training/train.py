"""Training loop for FNN/GRU hedgers and OLS regression.

FNN: 1 output -> sigmoid allocation.
GRU/Regression: d_traded outputs -> direct positions.
Super-hedging loss. NAdam optimizer. No regularization.
V_0 initialized to Black-Scholes price.
"""
import torch
from torch.amp import autocast, GradScaler
from src.training.losses import super_hedging_loss, shortfall, cvar
from src.training.early_stopping import EarlyStopping


def sigmoid_allocation(h_t, V_t, S_tilde_t):
    """Convert model output h_t to portfolio positions via sigmoid allocation.

    w1 = sigmoid(h_t), w2 = 1 - sigmoid(h_t)
    phi_i = w_i * V_t / S_tilde_i

    Args:
        h_t: [batch, 1] model output (logit)
        V_t: [batch] current portfolio value
        S_tilde_t: [batch, d_traded] current discounted prices

    Returns:
        phi: [batch, d_traded] position in shares
    """
    w1 = torch.sigmoid(h_t.squeeze(-1))  # [batch]
    d_traded = S_tilde_t.shape[1]
    if d_traded == 2:
        w = torch.stack([w1, 1.0 - w1], dim=1)  # [batch, 2]
    else:
        # For d_traded=1, w1 is the full allocation
        w = w1.unsqueeze(1)  # [batch, 1]
    phi = w * V_t.unsqueeze(1) / S_tilde_t.clamp(min=1e-8)
    return phi


def direct_position_gains(h_all, dS):
    """Compute gains from direct hedge positions.

    gains_k = h_k * dS_k, summed over assets.

    Args:
        h_all: [batch, N, d_traded] hedge positions
        dS: [batch, N, d_traded] price increments

    Returns:
        gains: [batch, N] per-step gains (summed over assets)
    """
    return (h_all * dS).sum(dim=2)


def forward_portfolio(model, features, S_tilde, V_0, d_traded):
    """Simulate self-financing portfolio.

    Detects output_dim from model output:
      - output_dim == 1 (FNN): sigmoid allocation loop
      - output_dim > 1 (GRU/Regression): vectorized direct-position path

    Args:
        model: hedger model (FNN, GRU, or Regression)
        features: [batch, N, feat_dim] features at k=0..N-1
        S_tilde: [batch, N+1, d_traded] discounted prices
        V_0: [batch] initial portfolio value (BS price)
        d_traded: int

    Returns:
        V_T: [batch] terminal portfolio value
        V_path: [batch, N+1] full portfolio path
    """
    batch, N, _ = features.shape

    # Get all model outputs at once
    h_all = model(features)  # [batch, N, output_dim]

    dS = S_tilde[:, 1:, :] - S_tilde[:, :-1, :]  # [batch, N, d_traded]

    output_dim = h_all.shape[2]

    if output_dim == 1:
        # FNN path: sigmoid allocation (sequential)
        V = V_0.clone()
        V_path = [V]

        for k in range(N):
            phi_k = sigmoid_allocation(h_all[:, k:k+1, :].squeeze(1),
                                       V, S_tilde[:, k, :])
            V = V + (phi_k * dS[:, k, :]).sum(dim=1)
            V_path.append(V)

        V_path = torch.stack(V_path, dim=1)
        return V, V_path
    else:
        # GRU/Regression path: direct positions (vectorized)
        gains = direct_position_gains(h_all, dS)  # [batch, N]
        cum_gains = gains.cumsum(dim=1)  # [batch, N]
        V_T = V_0 + cum_gains[:, -1]
        V_path = torch.cat([
            V_0.unsqueeze(1),
            V_0.unsqueeze(1) + cum_gains,
        ], dim=1)  # [batch, N+1]
        return V_T, V_path


def train_hedger(model, train_data, val_data, config, device="cpu"):
    """Train FNN/GRU hedger with NAdam and super-hedging loss.

    Args:
        model: hedger model (on device)
        train_data: dict with features, S_tilde, H_tilde, V_0
        val_data: dict with same keys
        config: dict with training params

    Returns:
        train_losses: list of per-epoch training losses
        val_losses: list of per-epoch validation CVaR95(shortfall)
    """
    lr = config["lr"]
    epochs = config["epochs"]
    patience = config["patience"]
    batch_size = config["batch_size"]
    lambda_short = config["lambda_short"]
    lambda_over = config["lambda_over"]
    cvar_q = config["cvar_q"]
    d_traded = config["d_traded"]

    optimizer = torch.optim.NAdam(model.parameters(), lr=lr)

    use_amp = device != "cpu" and torch.cuda.is_available()
    scaler = GradScaler("cuda") if use_amp else None

    stopper = EarlyStopping(patience=patience, mode="min")
    train_losses = []
    val_losses = []

    n_train = train_data["features"].shape[0]

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            feat = train_data["features"][idx]
            S_t = train_data["S_tilde"][idx]
            H = train_data["H_tilde"][idx]
            v0 = train_data["V_0"][idx]

            optimizer.zero_grad()

            with autocast("cuda", enabled=use_amp):
                V_T, _ = forward_portfolio(model, feat, S_t, v0, d_traded)
                loss = super_hedging_loss(V_T, H,
                                          lambda_short=lambda_short,
                                          lambda_over=lambda_over,
                                          cvar_q=cvar_q)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        train_losses.append(epoch_loss / max(n_batches, 1))

        # Validation
        val_cvar = evaluate_hedger(model, val_data, cvar_q, d_traded, device)
        val_losses.append(val_cvar)

        if stopper.step(val_cvar, model):
            break

    stopper.load_best(model)
    return train_losses, val_losses


def evaluate_hedger(model, data, cvar_q, d_traded, device):
    """Evaluate hedger on a dataset and return CVaR of shortfall."""
    model.eval()
    with torch.no_grad():
        V_T, _ = forward_portfolio(
            model, data["features"], data["S_tilde"],
            data["V_0"], d_traded,
        )
        s = shortfall(V_T, data["H_tilde"])
        val_cvar = cvar(s, q=cvar_q).item()
    return val_cvar


def train_regression(model, train_data, config, device="cpu"):
    """Fit OLS regression model (closed-form, no iterative training).

    Args:
        model: RegressionHedger
        train_data: dict with features, target (BS delta proxy)

    Returns:
        train_losses: empty list (no iterative training)
        val_losses: empty list
    """
    X = train_data["features"]  # [n_train * N, feat_dim]
    y = train_data["target"]    # [n_train * N, d_traded]
    model.fit(X.to(device), y.to(device))
    return [], []
