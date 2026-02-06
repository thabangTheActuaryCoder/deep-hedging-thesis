"""
Training loop for all models (FNN-5, LSTM-5, DBSDE) with:
- Elastic Net regularization (L1 + L2) on weights only
- Gradient clipping
- Early stopping
- Optuna Bayesian HP search (TPE) over lr, l1, l2, dropout, hidden_dim
"""

import torch
import torch.nn as nn
import numpy as np
import copy
import optuna
from typing import Dict, List, Optional, Tuple
from src.training.early_stopping import EarlyStopping
from src.eval.portfolio import simulate_portfolio

# Silence Optuna's per-trial INFO logs (we print our own)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def elastic_net_penalty(model: nn.Module, l1_lambda: float, l2_lambda: float) -> torch.Tensor:
    """Compute elastic net penalty on model WEIGHTS only (exclude biases, LayerNorm params)."""
    l1_term = torch.tensor(0.0, device=next(model.parameters()).device)
    l2_term = torch.tensor(0.0, device=next(model.parameters()).device)

    for name, param in model.named_parameters():
        if "bias" in name or "LayerNorm" in name or "layer_norm" in name or "norm" in name.split(".")[-1]:
            continue
        if "Y0" in name:
            continue
        l1_term = l1_term + param.abs().sum()
        l2_term = l2_term + (param ** 2).sum()

    return l1_lambda * l1_term + l2_lambda * l2_term


def train_hedger(
    model: nn.Module,
    model_type: str,
    features_train: torch.Tensor,
    features_val: torch.Tensor,
    S_train: torch.Tensor,
    S_val: torch.Tensor,
    payoff_T_train: torch.Tensor,
    payoff_T_val: torch.Tensor,
    dW_train: Optional[torch.Tensor] = None,
    dW_val: Optional[torch.Tensor] = None,
    time: Optional[torch.Tensor] = None,
    lr: float = 1e-3,
    l1_lambda: float = 1e-6,
    l2_lambda: float = 1e-4,
    epochs: int = 1000,
    batch_size: int = 512,
    patience: int = 30,
    grad_clip: float = 1.0,
    V0: float = 0.0,
    device: str = "cpu",
    optuna_trial: Optional[optuna.Trial] = None,
) -> Dict:
    """Train a hedging model.

    For FNN-5 and LSTM-5:
        Loss = MSE(V_T - H) via self-financing portfolio
    For DBSDE:
        Loss = MSE(Y_N - H) via BSDE propagation

    If optuna_trial is provided, reports intermediate values and supports
    pruning of unpromising trials.

    Returns:
        dict with 'train_losses', 'val_losses', 'best_epoch', 'best_val_loss'
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    early_stop = EarlyStopping(patience=patience)

    n_train = features_train.shape[0]

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]

            feat_batch = features_train[idx].to(device)
            S_batch = S_train[idx].to(device)
            H_batch = payoff_T_train[idx].to(device)

            if model_type == "bsde":
                dW_batch = dW_train[idx].to(device)
                Y_T, _, _ = model(feat_batch, dW_batch, time.to(device))
                data_loss = torch.mean((Y_T - H_batch) ** 2)
            else:
                deltas = model.compute_deltas(feat_batch)
                V_T = simulate_portfolio(deltas, S_batch, V0=V0)
                data_loss = torch.mean((V_T - H_batch) ** 2)

            reg_loss = elastic_net_penalty(model, l1_lambda, l2_lambda)
            loss = data_loss + reg_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            epoch_loss += data_loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            feat_v = features_val.to(device)
            S_v = S_val.to(device)
            H_v = payoff_T_val.to(device)

            if model_type == "bsde":
                dW_v = dW_val.to(device)
                Y_T_val, _, _ = model(feat_v, dW_v, time.to(device))
                val_loss = torch.mean((Y_T_val - H_v) ** 2).item()
            else:
                deltas_val = model.compute_deltas(feat_v)
                V_T_val = simulate_portfolio(deltas_val, S_v, V0=V0)
                val_loss = torch.mean((V_T_val - H_v) ** 2).item()

        val_losses.append(val_loss)

        if epoch % 50 == 0:
            print(f"    Epoch {epoch:4d} | Train MSE: {avg_train_loss:.6f} | Val MSE: {val_loss:.6f}")

        # Optuna pruning: kill bad trials early
        if optuna_trial is not None:
            optuna_trial.report(val_loss, epoch)
            if optuna_trial.should_prune():
                print(f"    Pruned at epoch {epoch} (val={val_loss:.4f})")
                raise optuna.TrialPruned()

        if early_stop.step(val_loss, model, epoch):
            print(f"    Early stopping at epoch {epoch}, best epoch {early_stop.best_epoch}")
            break

    early_stop.load_best(model)

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_epoch': early_stop.best_epoch,
        'best_val_loss': early_stop.best_score,
    }


# =========================================================================
# Optuna Bayesian HP search
# =========================================================================

def optuna_search(
    model_factory_fn,
    model_type: str,
    feature_dim: int,
    features_train: torch.Tensor,
    features_val: torch.Tensor,
    S_train: torch.Tensor,
    S_val: torch.Tensor,
    payoff_T_train: torch.Tensor,
    payoff_T_val: torch.Tensor,
    dW_train: Optional[torch.Tensor] = None,
    dW_val: Optional[torch.Tensor] = None,
    time: Optional[torch.Tensor] = None,
    n_trials: int = 12,
    epochs: int = 1000,
    batch_size: int = 512,
    patience: int = 30,
    V0: float = 0.0,
    device: str = "cpu",
    seed: int = 0,
) -> Tuple[nn.Module, Dict, Dict, List[Dict]]:
    """Bayesian hyperparameter search using Optuna TPE.

    Searches over:
        lr        : log-uniform [1e-4, 5e-3]
        l1        : log-uniform [1e-8, 1e-4] or 0 (20% chance of 0)
        l2        : log-uniform [1e-6, 1e-3] or 0 (15% chance of 0)
        dropout   : uniform [0.0, 0.35]
        hidden    : categorical {64, 96, 128, 192, 256}

    Pruning via MedianPruner: kills trials whose val loss at epoch E
    is worse than the median of completed trials at epoch E.

    Args:
        model_factory_fn: Callable(hidden, dropout) -> nn.Module
        n_trials: Number of Optuna trials (12 ≈ 25+ random)

    Returns:
        best_model, best_result, best_hparams, all_trials
    """
    best_model_state = [None]
    best_model_obj = [None]
    best_result_holder = [None]
    all_trials = []

    def objective(trial: optuna.Trial) -> float:
        # --- Sample hyperparameters ---
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)

        use_l1 = trial.suggest_categorical("use_l1", [True, False])
        l1 = trial.suggest_float("l1", 1e-8, 1e-4, log=True) if use_l1 else 0.0

        use_l2 = trial.suggest_categorical("use_l2", [True, False])
        l2 = trial.suggest_float("l2", 1e-6, 1e-3, log=True) if use_l2 else 0.0

        dropout = trial.suggest_float("dropout", 0.0, 0.35)
        hidden = trial.suggest_categorical("hidden", [64, 96, 128, 192, 256])

        hp = {'lr': lr, 'l1': l1, 'l2': l2, 'dropout': round(dropout, 3), 'hidden': hidden}
        trial_num = trial.number + 1
        print(f"\n  Trial {trial_num}/{n_trials} | "
              f"lr={lr:.2e}  l1={l1:.2e}  l2={l2:.2e}  "
              f"drop={dropout:.3f}  hidden={hidden}")

        torch.manual_seed(seed)
        model = model_factory_fn(hidden=hidden, dropout=dropout)

        result = train_hedger(
            model=model,
            model_type=model_type,
            features_train=features_train,
            features_val=features_val,
            S_train=S_train, S_val=S_val,
            payoff_T_train=payoff_T_train,
            payoff_T_val=payoff_T_val,
            dW_train=dW_train, dW_val=dW_val,
            time=time,
            lr=lr, l1_lambda=l1, l2_lambda=l2,
            epochs=epochs, batch_size=batch_size,
            patience=patience, V0=V0, device=device,
            optuna_trial=trial,
        )

        val_mse = result['best_val_loss']
        trial_record = {**hp, 'val_loss': val_mse, 'best_epoch': result['best_epoch']}
        all_trials.append(trial_record)

        # Keep track of the actual best model object
        if best_model_state[0] is None or val_mse < best_result_holder[0]['best_val_loss']:
            best_model_state[0] = copy.deepcopy(model.state_dict())
            best_model_obj[0] = model
            best_result_holder[0] = result
            print(f"    -> New best! Val MSE: {val_mse:.6f}  (epoch {result['best_epoch']})")
        else:
            print(f"    Val MSE: {val_mse:.6f}  (best so far: {best_result_holder[0]['best_val_loss']:.6f})")

        return val_mse

    # Create study with TPE sampler + median pruner
    sampler = optuna.samplers.TPESampler(seed=seed + 7919)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=3,    # Don't prune first 3 trials (need baseline)
        n_warmup_steps=20,     # Don't prune before epoch 20
    )
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )
    study.optimize(objective, n_trials=n_trials)

    # Extract best
    best_trial = study.best_trial
    best_hparams = {
        'lr': best_trial.params['lr'],
        'l1': best_trial.params.get('l1', 0.0) if best_trial.params.get('use_l1', False) else 0.0,
        'l2': best_trial.params.get('l2', 0.0) if best_trial.params.get('use_l2', False) else 0.0,
        'dropout': round(best_trial.params['dropout'], 3),
        'hidden': best_trial.params['hidden'],
    }

    # Rebuild best model with saved state
    best_model = model_factory_fn(hidden=best_hparams['hidden'],
                                  dropout=best_hparams['dropout'])
    best_model.load_state_dict(best_model_state[0])
    best_model.eval()

    n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"\n  Search done: {n_complete} completed, {n_pruned} pruned")
    print(f"  Best config: {best_hparams}  |  Val MSE: {best_trial.value:.6f}")

    return best_model, best_result_holder[0], best_hparams, all_trials
