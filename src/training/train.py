"""Training loop for FNN/LSTM + Controller and Deep BSDE.

Supports:
- Two-stage hedger (NN1 + NN2 controller) for FNN and LSTM
- Standalone BSDE training
- TBPTT for LSTM portfolio propagation
- Elastic net regularization
- Early stopping on validation CVaR95(shortfall)
"""
import math
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from src.training.losses import (
    hedging_loss, bsde_loss, elastic_net_penalty,
    shortfall, cvar, terminal_error,
)
from src.training.early_stopping import EarlyStopping


def forward_portfolio(nn1, controller, features, Z_intrinsic, dS,
                      use_controller=True, tbptt=0):
    """Simulate self-financing portfolio with NN1 + optional NN2 controller.

    Discounted self-financing: V_{k+1} = V_k + Delta_k^T * dS_k
    where dS_k = S_tilde_{k+1} - S_tilde_k.

    Args:
        nn1: baseline hedger (FNN or LSTM)
        controller: NN2 controller (or None)
        features: [batch, N, feat_dim] features at k=0..N-1
        Z_intrinsic: [batch, N] intrinsic values at k=0..N-1
        dS: [batch, N, d_traded] price increments
        use_controller: whether to apply NN2 gating
        tbptt: truncated BPTT length (0 = full backprop through portfolio)

    Returns:
        V_T: [batch] terminal portfolio value
        info: dict with Delta, V_path, gate for diagnostics
    """
    batch, N, _ = features.shape
    device = features.device

    # NN1 baseline deltas (batched over full sequence)
    Delta0 = nn1(features)  # [batch, N, d_traded]

    if not use_controller or controller is None:
        # Simple portfolio: no controller
        V = torch.zeros(batch, device=device)
        V_path = [V]
        for k in range(N):
            if tbptt > 0 and k > 0 and k % tbptt == 0:
                V = V.detach()
            V = V + (Delta0[:, k, :] * dS[:, k, :]).sum(dim=1)
            V_path.append(V)
        return V, {"Delta": Delta0, "V_path": torch.stack(V_path, dim=1)}

    # Two-stage hedger with controller
    V = torch.zeros(batch, device=device)
    V_path = [V]
    prev_V = torch.zeros_like(V)
    running_max_V = torch.zeros_like(V)
    dPL_buffer = []
    gate_all = []

    for k in range(N):
        if tbptt > 0 and k > 0 and k % tbptt == 0:
            V = V.detach()
            prev_V = prev_V.detach()
            running_max_V = running_max_V.detach()

        # Causal P/L features at step k
        PL_k = V                                         # cumulative P/L
        dPL_k = (V - prev_V) if k > 0 else torch.zeros_like(V)
        running_max_V = torch.max(running_max_V, V)
        DD_k = V - running_max_V                         # drawdown <= 0
        intrinsic_gap_k = V - Z_intrinsic[:, k]

        dPL_buffer.append(dPL_k.detach())
        if len(dPL_buffer) >= 2:
            recent = torch.stack(dPL_buffer[-10:], dim=1)
            rolling_std = recent.std(dim=1)
        else:
            rolling_std = torch.zeros_like(V)

        pl_feats = torch.stack(
            [PL_k, dPL_k, DD_k, intrinsic_gap_k, rolling_std], dim=1
        )  # [batch, 5]

        gate, correction = controller(features[:, k, :], pl_feats)
        gate_all.append(gate.squeeze(1))

        Delta_k = Delta0[:, k, :] + gate * correction
        prev_V = V
        V = V + (Delta_k * dS[:, k, :]).sum(dim=1)
        V_path.append(V)

    return V, {
        "Delta0": Delta0,
        "gate": torch.stack(gate_all, dim=1),
        "V_path": torch.stack(V_path, dim=1),
    }


def train_hedger(nn1, controller, train_data, val_data, config, device="cpu"):
    """Train FNN/LSTM hedger with optional controller.

    Args:
        nn1: baseline hedger model (on device)
        controller: NN2 controller model (on device) or None
        train_data: dict with features, Z_intrinsic, dS, H_tilde
        val_data: dict with same keys
        config: ExperimentConfig or dict with training params

    Returns:
        train_losses: list of per-epoch training losses
        val_losses: list of per-epoch validation CVaR95(shortfall)
    """
    lr = config["lr"]
    epochs = config["epochs"]
    patience = config["patience"]
    batch_size = config["batch_size"]
    l1 = config["l1"]
    l2 = config["l2"]
    alpha = config["alpha"]
    beta = config["beta"]
    cvar_q = config["cvar_q"]
    use_ctrl = config["use_controller"] and controller is not None
    tbptt = config.get("tbptt", 0)

    # Collect all trainable parameters
    params = list(nn1.parameters())
    if use_ctrl:
        params += list(controller.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    # Mixed precision: use AMP on CUDA to halve LSTM memory
    use_amp = device != "cpu" and torch.cuda.is_available()
    scaler = GradScaler("cuda") if use_amp else None

    stopper = EarlyStopping(patience=patience, mode="min")
    train_losses = []
    val_losses = []

    n_train = train_data["features"].shape[0]

    for epoch in range(epochs):
        nn1.train()
        if use_ctrl:
            controller.train()

        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            feat = train_data["features"][idx]
            Z_int = train_data["Z_intrinsic"][idx]
            ds = train_data["dS"][idx]
            H = train_data["H_tilde"][idx]

            optimizer.zero_grad()

            with autocast("cuda", enabled=use_amp):
                V_T, _ = forward_portfolio(
                    nn1, controller, feat, Z_int, ds,
                    use_controller=use_ctrl, tbptt=tbptt,
                )
                loss = hedging_loss(V_T, H, alpha=alpha, beta=beta, cvar_q=cvar_q)

                # Elastic net on all models
                reg = elastic_net_penalty(nn1, l1=l1, l2=l2)
                if use_ctrl:
                    reg = reg + elastic_net_penalty(controller, l1=l1, l2=l2)
                loss = loss + reg

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
                optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        train_losses.append(epoch_loss / max(n_batches, 1))

        # Validation
        val_cvar = evaluate_hedger(
            nn1, controller, val_data, use_ctrl, tbptt, cvar_q, device
        )
        val_losses.append(val_cvar)

        if stopper.step(val_cvar, nn1):
            break

    stopper.load_best(nn1)
    return train_losses, val_losses


def evaluate_hedger(nn1, controller, data, use_controller, tbptt, cvar_q, device):
    """Evaluate hedger on a dataset and return CVaR of shortfall."""
    nn1.eval()
    if controller is not None:
        controller.eval()

    with torch.no_grad():
        V_T, _ = forward_portfolio(
            nn1, controller,
            data["features"], data["Z_intrinsic"],
            data["dS"], use_controller=use_controller, tbptt=0,
        )
        s = shortfall(V_T, data["H_tilde"])
        val_cvar = cvar(s, q=cvar_q).item()

    return val_cvar


def train_bsde(model, train_data, val_data, config, device="cpu"):
    """Train Deep BSDE model.

    Loss: MSE(Y_N - H_tilde) + elastic net regularization.

    Args:
        model: DeepBSDE model (on device)
        train_data: dict with features, dW, H_tilde, time_grid
        val_data: dict with same keys
        config: dict with training params

    Returns:
        train_losses: list of per-epoch losses
        val_losses: list of per-epoch validation CVaR95(shortfall)
    """
    lr = config["lr"]
    epochs = config["epochs"]
    patience = config["patience"]
    batch_size = config["batch_size"]
    l1 = config["l1"]
    l2 = config["l2"]
    cvar_q = config["cvar_q"]
    substeps = config.get("substeps", 0)
    time_grid = train_data["time_grid"]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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
            dw = train_data["dW"][idx]
            H = train_data["H_tilde"][idx]

            Y_T, _, _ = model(feat, dw, time_grid, substeps=substeps)
            loss = bsde_loss(Y_T, H) + elastic_net_penalty(model, l1=l1, l2=l2)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        train_losses.append(epoch_loss / max(n_batches, 1))

        # Validation: compute CVaR95(shortfall) using BSDE terminal value as V_T
        val_cvar = evaluate_bsde(model, val_data, time_grid, cvar_q, substeps, device)
        val_losses.append(val_cvar)

        if stopper.step(val_cvar, model):
            break

    stopper.load_best(model)
    return train_losses, val_losses


def evaluate_bsde(model, data, time_grid, cvar_q, substeps, device):
    """Evaluate BSDE model, return CVaR of shortfall."""
    model.eval()
    with torch.no_grad():
        Y_T, _, _ = model(data["features"], data["dW"], time_grid, substeps=substeps)
        s = shortfall(Y_T, data["H_tilde"])
        return cvar(s, q=cvar_q).item()
