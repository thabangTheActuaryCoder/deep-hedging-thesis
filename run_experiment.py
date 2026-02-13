#!/usr/bin/env python3
"""Main entrypoint for deep hedging experiments.

Two-stage bias control protocol:
  Stage 1: Architecture + LR selection via Optuna at fixed seed (seed_arch)
  Stage 2: Seed robustness on best configs with multiple seeds

Models: FNN (cone) + GRU (neural) + OLS Regression (closed-form)
All models output 1 scalar -> sigmoid allocation -> super-hedging loss.
VAE used for path augmentation (not feature extraction).
"""
import argparse
import os
import sys
import time
import math
import logging

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import numpy as np
import optuna

from src.utils.config import ExperimentConfig
from src.utils.reproducibility import set_seed, get_device, clear_gpu_cache
from src.utils.io import save_json, load_json, save_checkpoint, ensure_dir, compute_split_hash
from src.sim.simulate_market import (
    simulate_market, compute_european_put_payoff,
    compute_intrinsic_process, split_data,
)
from src.sim.calibration import load_market_params
from src.features.build_features import build_features_for_model, build_base_features
from src.features.vae import train_path_vae, augment_training_data
from src.models.fnn import FNNHedger, cone_layer_widths
from src.models.gru import GRUHedger
from src.models.regression import RegressionHedger, bs_put_delta
from src.training.train import (
    train_hedger, train_regression, forward_portfolio,
    evaluate_hedger,
)
from src.training.losses import shortfall, cvar, terminal_error
from src.eval.metrics import compute_metrics, aggregate_seed_metrics, select_representative_seed
from src.eval.portfolio import simulate_portfolio_path
from src.eval.plots import (
    generate_all_plots, plot_summary_table,
    plot_model_comparison_bars, plot_model_comparison_errors,
    plot_model_comparison_cvar, plot_validation_summary,
    plot_optuna_validation_loss,
)
from src.eval.plots_3d import generate_3d_plots


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Deep Hedging Experiment")

    # Market
    p.add_argument("--paths", type=int, default=100000)
    p.add_argument("--N", type=int, default=200)
    p.add_argument("--T", type=float, default=1.0)
    p.add_argument("--d_traded", type=int, default=2)
    p.add_argument("--m_brownian", type=int, default=3)
    p.add_argument("--K", type=float, default=1.0)
    p.add_argument("--r", type=float, default=0.0)
    p.add_argument("--market_config", type=str, default="")

    # Features
    p.add_argument("--latent_dim", type=int, default=16)
    p.add_argument("--sig_level", type=int, default=2)
    p.add_argument("--vae_augment_ratio", type=float, default=0.5)

    # Architecture grid (FNN cone)
    p.add_argument("--start_width_grid", type=int, nargs="+", default=[16, 32, 64, 128])
    p.add_argument("--act_schedules", type=str, nargs="+",
                   default=["relu_all", "tanh_all", "alt_relu_tanh", "alt_tanh_relu"])
    p.add_argument("--lrs", type=float, nargs="+", default=[3e-4, 1e-3, 3e-3])

    # GRU grid
    p.add_argument("--gru_num_layers_grid", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--gru_hidden_size_grid", type=int, nargs="+", default=[32, 64, 128])

    # Training
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=2048)

    # Loss (super-hedging)
    p.add_argument("--lambda_short", type=float, default=10.0)
    p.add_argument("--lambda_over", type=float, default=1.0)
    p.add_argument("--cvar_q", type=float, default=0.95)

    # Seeds
    p.add_argument("--seed_arch", type=int, default=0)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])

    # Optuna HP search
    p.add_argument("--n_trials", type=int, default=60)

    # Quick mode
    p.add_argument("--quick", action="store_true")

    args = p.parse_args()

    if args.quick:
        args.paths = 5000
        args.epochs = 50
        args.patience = 5
        args.seeds = [0]
        args.n_trials = 3
        args.start_width_grid = [16, 32]
        args.gru_num_layers_grid = [1, 2]
        args.gru_hidden_size_grid = [32, 64]
        args.batch_size = 1024

    return args


# ──────────────────────────────────────────────
# Model factories
# ──────────────────────────────────────────────

def make_fnn(feat_dim, start_width, act_schedule, dropout, device):
    return FNNHedger(feat_dim, start_width=start_width,
                     act_schedule=act_schedule, dropout=dropout).to(device)


def make_gru(feat_dim, num_layers, hidden_size, act_schedule, dropout, device):
    return GRUHedger(feat_dim, num_layers=num_layers, hidden_size=hidden_size,
                     act_schedule=act_schedule, dropout=dropout).to(device)


def make_regression(feat_dim, device):
    return RegressionHedger(feat_dim).to(device)


# ──────────────────────────────────────────────
# Black-Scholes V_0 computation
# ──────────────────────────────────────────────

def compute_bs_price(S0, K, r, T, sigma):
    """Black-Scholes European put price."""
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    from scipy.stats import norm
    return K * math.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)


def get_V0(S_tilde, K, r, T, sigma_avg, n_paths):
    """Compute initial portfolio value V_0 = BS put price for each path.

    Uses the first asset's initial price for the BS formula.
    Returns [n_paths] tensor.
    """
    bs_price = compute_bs_price(1.0, K, r, T, sigma_avg)
    return torch.full((n_paths,), bs_price)


# ──────────────────────────────────────────────
# Data preparation helpers
# ──────────────────────────────────────────────

def prepare_hedger_data(features, S_tilde_split, H_split, V_0, device):
    """Prepare data dict for hedger training/eval."""
    N = features.shape[1] - 1
    return {
        "features": features[:, :N, :].to(device),
        "S_tilde": S_tilde_split.to(device),
        "H_tilde": H_split.to(device),
        "V_0": V_0.to(device),
    }


def prepare_regression_data(features, S_tilde, K, r, T, time_grid, sigma_avg, device):
    """Prepare OLS regression training data.

    Target: BS put delta as proxy for h_t.
    Features: flattened across time steps.
    """
    n_paths, N_plus_1, feat_dim = features.shape
    N = N_plus_1 - 1

    # Flatten features: [n_paths * N, feat_dim]
    X = features[:, :N, :].reshape(-1, feat_dim)

    # Compute BS delta target for each path x step
    S = S_tilde[:, :N, 0]  # [n_paths, N] first asset
    tau = (T - time_grid[:N]).unsqueeze(0).expand(n_paths, -1)
    y = bs_put_delta(S.reshape(-1), K, r, T, sigma_avg, tau.reshape(-1))

    return {"features": X, "target": y}


# ──────────────────────────────────────────────
# Stage 1: Optuna HP search
# ──────────────────────────────────────────────

def run_optuna_search(model_class, feat_dim, d_traded,
                      train_data, val_data, args, device,
                      output_dir=None):
    """Run Optuna TPE search for FNN or GRU.

    FNN search space: start_width, act_schedule, lr (depth auto from cone)
    GRU search space: num_layers, hidden_size, act_schedule, lr

    Returns:
        best_config: dict with best params
        trial_log: list of all tried configs with val metrics
    """
    trial_log = []
    _seen_configs = {}

    def _data_to(data, target_device):
        return {k: v.to(target_device) if isinstance(v, torch.Tensor) else v
                for k, v in data.items()}

    def _run_trial_inner(trial_params, batch_size):
        set_seed(args.seed_arch)

        train_cfg = {
            "lr": trial_params["lr"],
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": batch_size,
            "lambda_short": args.lambda_short,
            "lambda_over": args.lambda_over,
            "cvar_q": args.cvar_q,
            "d_traded": d_traded,
        }

        t_data = _data_to(train_data, device)
        v_data = _data_to(val_data, device)

        model = None
        try:
            if model_class == "FNN":
                model = make_fnn(feat_dim, trial_params["start_width"],
                                 trial_params["act_schedule"], 0.1, device)
            else:  # GRU
                model = make_gru(feat_dim, trial_params["num_layers"],
                                 trial_params["hidden_size"],
                                 trial_params["act_schedule"], 0.1, device)

            t_losses, v_losses = train_hedger(
                model, t_data, v_data, train_cfg, device
            )
            val_cvar = v_losses[-1] if v_losses else float("inf")

            with torch.no_grad():
                model.eval()
                V_T, _ = forward_portfolio(
                    model, v_data["features"], v_data["S_tilde"],
                    v_data["V_0"], d_traded,
                )
                e = terminal_error(V_T, v_data["H_tilde"])
                val_mse = (e ** 2).mean().item()

            return val_cvar, val_mse

        finally:
            del model, t_data, v_data
            clear_gpu_cache()

    def objective(trial):
        if model_class == "FNN":
            start_width = trial.suggest_categorical("start_width", args.start_width_grid)
            act = trial.suggest_categorical("act_schedule", args.act_schedules)
            lr = trial.suggest_categorical("lr", args.lrs)
            config_key = (start_width, act, lr)
            trial_params = {"start_width": start_width, "act_schedule": act, "lr": lr}
        else:  # GRU
            num_layers = trial.suggest_categorical("num_layers", args.gru_num_layers_grid)
            hidden_size = trial.suggest_categorical("hidden_size", args.gru_hidden_size_grid)
            act = trial.suggest_categorical("act_schedule", args.act_schedules)
            lr = trial.suggest_categorical("lr", args.lrs)
            config_key = (num_layers, hidden_size, act, lr)
            trial_params = {"num_layers": num_layers, "hidden_size": hidden_size,
                            "act_schedule": act, "lr": lr}

        if config_key in _seen_configs:
            val_cvar, val_mse = _seen_configs[config_key]
            trial.set_user_attr("val_MSE", val_mse)
            entry = dict(trial_params)
            entry["val_CVaR95"] = val_cvar
            entry["val_MSE"] = val_mse
            trial_log.append(entry)
            print(f"    Trial {trial.number}: {trial_params} "
                  f"-> CVaR95={val_cvar:.6f}  MSE={val_mse:.6f}  [cached]")
            return val_cvar

        batch_size = args.batch_size
        min_batch = max(64, args.batch_size // 16)
        val_cvar, val_mse = float("inf"), float("inf")
        while batch_size >= min_batch:
            try:
                val_cvar, val_mse = _run_trial_inner(trial_params, batch_size)
                break
            except RuntimeError as ex:
                if "out of memory" in str(ex).lower():
                    clear_gpu_cache()
                    batch_size = batch_size // 2
                    if batch_size >= min_batch:
                        print(f"    OOM -> retrying with batch_size={batch_size}")
                    else:
                        print(f"    FAILED: OOM even at batch_size={batch_size * 2}")
                else:
                    print(f"    FAILED: {ex}")
                    clear_gpu_cache()
                    break

        _seen_configs[config_key] = (val_cvar, val_mse)
        trial.set_user_attr("val_MSE", val_mse)

        entry = dict(trial_params)
        entry["val_CVaR95"] = val_cvar
        entry["val_MSE"] = val_mse
        trial_log.append(entry)

        print(f"    Trial {trial.number}: {trial_params} "
              f"-> CVaR95={val_cvar:.6f}  MSE={val_mse:.6f}")

        return val_cvar

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if model_class == "FNN":
        total_configs = (len(args.start_width_grid) * len(args.act_schedules) * len(args.lrs))
    else:
        total_configs = (len(args.gru_num_layers_grid) * len(args.gru_hidden_size_grid)
                         * len(args.act_schedules) * len(args.lrs))

    n_trials = min(args.n_trials, total_configs)

    print(f"\n  Optuna search for {model_class}: up to {n_trials} trials "
          f"(search space = {total_configs} configs)")

    sampler = optuna.samplers.TPESampler(seed=args.seed_arch)

    storage = None
    if output_dir is not None:
        db_path = os.path.join(output_dir, "run_configs",
                               f"optuna_{model_class}.db")
        storage = f"sqlite:///{db_path}"

    study = optuna.create_study(direction="minimize", sampler=sampler,
                                study_name=f"{model_class}_stage1",
                                storage=storage, load_if_exists=True)

    done = len([t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, n_trials - done)
    if done > 0:
        print(f"  Resuming: {done}/{n_trials} trials already complete")
        for t in study.trials:
            if t.state == optuna.trial.TrialState.COMPLETE:
                if model_class == "FNN":
                    key = (t.params["start_width"], t.params["act_schedule"], t.params["lr"])
                else:
                    key = (t.params["num_layers"], t.params["hidden_size"],
                           t.params["act_schedule"], t.params["lr"])
                mse = t.user_attrs.get("val_MSE", float("inf"))
                _seen_configs[key] = (t.value, mse)
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, show_progress_bar=False)

    # Rebuild trial_log from all completed study trials
    trial_log = []
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            entry = dict(t.params)
            entry["val_CVaR95"] = t.value
            entry["val_MSE"] = t.user_attrs.get("val_MSE", float("inf"))
            trial_log.append(entry)

    trial_log.sort(key=lambda x: (x["val_CVaR95"], x["val_MSE"], x["lr"]))
    best = trial_log[0]

    print(f"\n  Best {model_class}: {best}")

    return best, trial_log


# ──────────────────────────────────────────────
# Stage 2: Seed robustness
# ──────────────────────────────────────────────

def run_seed_robustness(model_class, best_config, feat_dim, d_traded,
                        train_data, val_data, args, device,
                        S_tilde_val, time_grid=None):
    """Train best config across multiple seeds and evaluate on validation set."""
    seed_results = []
    last_VT = None
    last_H = None

    print(f"\n  Seed robustness for {model_class}: seeds={args.seeds}")

    for seed in args.seeds:
        clear_gpu_cache()
        print(f"\n    Seed {seed}:", end=" ", flush=True)

        ckpt_path = os.path.join(args.output_dir, "checkpoints",
                                 f"{model_class}_seed{seed}.pt")
        seed_metric_path = os.path.join(args.output_dir, "run_configs",
                                         f"{model_class}_seed{seed}_metrics.json")

        if os.path.exists(ckpt_path) and os.path.exists(seed_metric_path):
            print("[cached] loading checkpoint")
            val_metrics = load_json(seed_metric_path)
            seed_results.append({
                "seed": seed,
                "val_metrics": val_metrics,
                "train_losses": [],
                "val_losses": [],
            })
            continue

        set_seed(seed)

        train_cfg = {
            "lr": best_config["lr"],
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "lambda_short": args.lambda_short,
            "lambda_over": args.lambda_over,
            "cvar_q": args.cvar_q,
            "d_traded": d_traded,
        }

        if model_class == "FNN":
            model = make_fnn(feat_dim, best_config["start_width"],
                             best_config["act_schedule"], 0.1, device)
        elif model_class == "GRU":
            model = make_gru(feat_dim, best_config["num_layers"],
                             best_config["hidden_size"],
                             best_config["act_schedule"], 0.1, device)
        else:
            raise ValueError(f"Unexpected model_class for seed robustness: {model_class}")

        t_losses, v_losses = train_hedger(
            model, train_data, val_data, train_cfg, device
        )

        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            V_T, V_path = forward_portfolio(
                model, val_data["features"], val_data["S_tilde"],
                val_data["V_0"], d_traded,
            )
        val_metrics = compute_metrics(V_T, val_data["H_tilde"], V_path=V_path)

        save_checkpoint(model.state_dict(), ckpt_path)
        save_json(val_metrics, seed_metric_path)

        generate_all_plots(
            model_class, seed, V_T.cpu(), val_data["H_tilde"].cpu(),
            V_path.cpu(), t_losses, v_losses,
            output_dir=os.path.join(args.output_dir, "plots_val"),
        )

        last_VT = V_T.cpu()
        last_H = val_data["H_tilde"].cpu()

        # 3D plots for first seed
        if seed == args.seeds[0]:
            generate_3d_plots(
                model_class, model, feat_dim, args.N, args.T,
                output_dir=os.path.join(args.output_dir, "plots_3d"),
            )

        print(f"CVaR95(val)={val_metrics['CVaR95_shortfall']:.6f}")

        seed_results.append({
            "seed": seed,
            "val_metrics": val_metrics,
            "train_losses": t_losses,
            "val_losses": v_losses,
        })

        del model
        clear_gpu_cache()

    val_metrics_list = [r["val_metrics"] for r in seed_results]
    agg = aggregate_seed_metrics(val_metrics_list)
    rep_seed = select_representative_seed(seed_results)
    agg["representative_seed"] = rep_seed

    comparison_data = {
        "V_T": last_VT,
        "H_tilde": last_H,
    }

    return seed_results, agg, comparison_data


def run_regression_eval(feat_dim, d_traded, reg_train_data,
                        hedger_val_data, args, device,
                        S_tilde_val, time_grid):
    """Fit and evaluate OLS regression (no HP search, no seed robustness)."""
    print(f"\n  Regression: fitting OLS (closed-form)")

    ckpt_path = os.path.join(args.output_dir, "checkpoints", "Regression_seed0.pt")
    metric_path = os.path.join(args.output_dir, "run_configs",
                               "Regression_seed0_metrics.json")

    model = make_regression(feat_dim, device)
    train_regression(model, reg_train_data, {}, device)

    model.eval()
    with torch.no_grad():
        V_T, V_path = forward_portfolio(
            model, hedger_val_data["features"], hedger_val_data["S_tilde"],
            hedger_val_data["V_0"], d_traded,
        )
    val_metrics = compute_metrics(V_T, hedger_val_data["H_tilde"], V_path=V_path)

    save_checkpoint(model.state_dict(), ckpt_path)
    save_json(val_metrics, metric_path)

    generate_all_plots(
        "Regression", 0, V_T.cpu(), hedger_val_data["H_tilde"].cpu(),
        V_path.cpu(), [], [],
        output_dir=os.path.join(args.output_dir, "plots_val"),
    )

    # 3D plots
    generate_3d_plots(
        "Regression", model, feat_dim, args.N, args.T,
        output_dir=os.path.join(args.output_dir, "plots_3d"),
    )

    print(f"  Regression: CVaR95={val_metrics['CVaR95_shortfall']:.6f}  "
          f"P(V_T>=H)={val_metrics['P_positive_error']:.1%}")

    # Wrap in aggregated format (single seed)
    agg = aggregate_seed_metrics([val_metrics])
    agg["representative_seed"] = 0

    comparison_data = {
        "V_T": V_T.cpu(),
        "H_tilde": hedger_val_data["H_tilde"].cpu(),
    }

    return agg, comparison_data


# ──────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────

def run_pipeline(args, device, S_tilde, dW, time_grid, sigma,
                 H_tilde, Z_intrinsic, train_idx, val_idx, test_idx,
                 split_hash, sigma_avg_scalar):
    """Run the full training pipeline (GBM market)."""
    output_dir = os.path.join(args.output_dir, "gbm")
    for sub in ["plots", "plots_val", "plots_3d", "checkpoints", "run_configs",
                "data"]:
        ensure_dir(os.path.join(output_dir, sub))

    d_traded = args.d_traded

    print(f"\n{'='*60}")
    print(f"  Pipeline: GBM market")
    print(f"{'='*60}")

    save_json({"train": train_idx.tolist(), "val": val_idx.tolist(),
               "test": test_idx.tolist(), "hash": split_hash},
              os.path.join(output_dir, "run_configs", "split_indices.json"))

    # ── VAE path augmentation ──
    print(f"\n=== VAE Path Augmentation ===")
    S_tilde_train = S_tilde[train_idx]
    log_prices_train = torch.log(S_tilde_train.clamp(min=1e-8))
    vae_model = train_path_vae(
        log_prices_train, latent_dim=args.latent_dim,
        epochs=50 if not args.quick else 10, device=device,
    )
    S_tilde_aug = augment_training_data(
        S_tilde_train, vae_model,
        augment_ratio=args.vae_augment_ratio, device=device,
    )
    n_aug = S_tilde_aug.shape[0]
    n_real = S_tilde_train.shape[0]
    print(f"  Real paths: {n_real}  Augmented total: {n_aug}")

    # Create augmented indices for augmented data
    aug_train_idx = np.arange(n_aug)
    # Val/test use original indices mapped to full S_tilde
    S_tilde_val = S_tilde[val_idx]
    S_tilde_test = S_tilde[test_idx]
    H_train = H_tilde[train_idx]
    H_val = H_tilde[val_idx]
    H_test = H_tilde[test_idx]

    # Augment H_tilde for synthetic paths
    if n_aug > n_real:
        n_syn = n_aug - n_real
        H_syn = compute_european_put_payoff(S_tilde_aug[n_real:], args.K, args.r, args.T)
        H_train_aug = torch.cat([H_train, H_syn], dim=0)
    else:
        H_train_aug = H_train

    # Compute V_0 (BS price)
    V_0_val = get_V0(S_tilde_val, args.K, args.r, args.T, sigma_avg_scalar, len(val_idx))
    V_0_train = get_V0(S_tilde_aug, args.K, args.r, args.T, sigma_avg_scalar, n_aug)

    all_models = ["FNN", "GRU", "Regression"]

    # ── Build features per model type ──
    print(f"\n=== Feature Construction ===")

    model_features = {}
    for model_type in all_models:
        # FNN uses augmented S_tilde for training features
        if model_type == "FNN":
            # Build features on augmented training data
            f_train, _, _, feat_dim = build_features_for_model(
                model_type, S_tilde_aug, time_grid, args.T,
                np.arange(n_aug), np.array([0]), np.array([0]),
                sig_level=args.sig_level,
            )
            _, f_val, f_test, _ = build_features_for_model(
                model_type, S_tilde, time_grid, args.T,
                train_idx, val_idx, test_idx,
                sig_level=args.sig_level,
            )
            # Re-standardize val/test using augmented training stats
            flat_train = f_train.reshape(-1, feat_dim)
            mean = flat_train.mean(dim=0)
            std = flat_train.std(dim=0).clamp(min=1e-8)
            # f_train is already standardized from build_features_for_model
            # but val/test need re-standardization with augmented stats
            # Actually build_features_for_model already standardizes internally,
            # so for simplicity use separate calls
        else:
            f_train, _, _, feat_dim = build_features_for_model(
                model_type, S_tilde_aug, time_grid, args.T,
                np.arange(n_aug), np.array([0]), np.array([0]),
            )
            _, f_val, f_test, _ = build_features_for_model(
                model_type, S_tilde, time_grid, args.T,
                train_idx, val_idx, test_idx,
            )

        model_features[model_type] = {
            "train": f_train,
            "val": f_val,
            "test": f_test,
            "feat_dim": feat_dim,
        }
        print(f"  {model_type}: feat_dim={feat_dim}")

    # Temporarily override output_dir for per-market paths
    orig_output_dir = args.output_dir
    args.output_dir = output_dir

    # ── Stage 1: Optuna HP search for FNN and GRU ──
    print(f"\n=== Stage 1: Optuna HP Search ===")

    stage1_path = os.path.join(output_dir, "run_configs", "stage1_optuna.json")
    best_configs = {}
    trial_logs = {}
    if os.path.exists(stage1_path):
        stage1_saved = load_json(stage1_path)
        best_configs = stage1_saved.get("best_configs", {})
        trial_logs = stage1_saved.get("trial_logs", {})

    for model_class in ["FNN", "GRU"]:
        if model_class in best_configs:
            bc = best_configs[model_class]
            print(f"\n--- {model_class} [cached] ---")
            print(f"  Best: {bc}")
            continue

        print(f"\n--- {model_class} ---")
        mf = model_features[model_class]
        feat_dim = mf["feat_dim"]

        hedger_train = prepare_hedger_data(
            mf["train"], S_tilde_aug, H_train_aug, V_0_train, device
        )
        hedger_val = prepare_hedger_data(
            mf["val"], S_tilde_val, H_val, V_0_val, device
        )

        # Per-trial isolation: pass CPU data
        _train = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                  for k, v in hedger_train.items()}
        _val = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                for k, v in hedger_val.items()}
        clear_gpu_cache()

        best, tlog = run_optuna_search(
            model_class, feat_dim, d_traded,
            _train, _val, args, device,
            output_dir=output_dir,
        )
        del _train, _val
        clear_gpu_cache()

        best_configs[model_class] = best
        trial_logs[model_class] = tlog

        save_json({"best_configs": best_configs, "trial_logs": trial_logs},
                  stage1_path)
        print(f"  [saved Stage 1 progress: {list(best_configs.keys())}]")

    # Generate Optuna plots
    plots_val_dir = os.path.join(output_dir, "plots_val")
    for mc in ["FNN", "GRU"]:
        tlog = trial_logs.get(mc, [])
        if tlog:
            plot_optuna_validation_loss(tlog, mc, output_dir=plots_val_dir)

    # ── Stage 2: Seed robustness for FNN and GRU ──
    print(f"\n=== Stage 2: Seed Robustness ===")

    all_agg = {}
    all_comparison = {}

    for model_class in ["FNN", "GRU"]:
        print(f"\n--- {model_class} ---")
        mf = model_features[model_class]
        feat_dim = mf["feat_dim"]

        hedger_train = prepare_hedger_data(
            mf["train"], S_tilde_aug, H_train_aug, V_0_train, device
        )
        hedger_val = prepare_hedger_data(
            mf["val"], S_tilde_val, H_val, V_0_val, device
        )

        seed_res, agg, comp_data = run_seed_robustness(
            model_class, best_configs[model_class],
            feat_dim, d_traded,
            hedger_train, hedger_val,
            args, device,
            S_tilde_val=S_tilde_val.to(device),
        )
        all_agg[model_class] = agg
        all_comparison[model_class] = comp_data

        clear_gpu_cache()

    # ── Regression (no HP search, no seed robustness) ──
    print(f"\n=== Regression (OLS) ===")
    mf = model_features["Regression"]
    feat_dim = mf["feat_dim"]

    hedger_val_reg = prepare_hedger_data(
        mf["val"], S_tilde_val, H_val, V_0_val, device
    )

    reg_train_data = prepare_regression_data(
        mf["train"], S_tilde_aug, args.K, args.r, args.T,
        time_grid, sigma_avg_scalar, device,
    )

    reg_agg, reg_comp = run_regression_eval(
        feat_dim, d_traded, reg_train_data,
        hedger_val_reg, args, device,
        S_tilde_val, time_grid,
    )
    all_agg["Regression"] = reg_agg
    all_comparison["Regression"] = reg_comp

    # Restore original output_dir
    args.output_dir = orig_output_dir

    # ── Validation Analysis ──
    print(f"\n=== Validation Analysis ===")

    best_model = min(
        all_agg.keys(),
        key=lambda m: (all_agg[m]["CVaR95_shortfall"]["mean"]
                        if isinstance(all_agg[m].get("CVaR95_shortfall"), dict)
                        else float("inf")),
    )
    print(f"  Best model: {best_model}")

    plot_summary_table(all_agg, output_dir=plots_val_dir,
                       title="Model Comparison – GBM (Validation Set)")
    plot_validation_summary(all_agg, best_model, output_dir=plots_val_dir)
    plot_model_comparison_bars(all_agg, output_dir=plots_val_dir)

    if all_comparison:
        model_errors_dict = {}
        model_VT_dict = {}
        model_H_dict = {}
        for mc, comp in all_comparison.items():
            V_T_c = comp["V_T"]
            H_c = comp["H_tilde"]
            e = terminal_error(V_T_c, H_c).numpy()
            model_errors_dict[mc] = e
            model_VT_dict[mc] = V_T_c
            model_H_dict[mc] = H_c

        plot_model_comparison_errors(model_errors_dict, output_dir=plots_val_dir)
        plot_model_comparison_cvar(model_VT_dict, model_H_dict,
                                   output_dir=plots_val_dir)

    # Save results
    summary_path = os.path.join(output_dir, "metrics_summary.json")
    summary = {
        "best_configs": best_configs,
        "aggregated_val_metrics": all_agg,
        "best_model": best_model,
        "market_model": "gbm",
        "config": {
            "paths": args.paths, "N": args.N, "T": args.T,
            "d_traded": args.d_traded, "K": args.K, "r": args.r,
            "seed_arch": args.seed_arch, "seeds": args.seeds,
            "split_hash": split_hash,
        },
    }
    save_json(summary, summary_path)
    _write_csv_summary(all_agg, os.path.join(output_dir, "val_metrics_summary.csv"))

    print(f"\n  Done. Results saved to {output_dir}/")
    return all_agg, all_comparison


def main():
    args = parse_args()
    device = get_device()
    print(f"Device: {device}")
    print(f"Quick mode: {args.quick}")

    args.output_dir = "outputs"
    ensure_dir(args.output_dir)

    config_path = args.market_config or None
    gbm_params = load_market_params(config_path)

    args.r = gbm_params.get("r", args.r)
    args.K = gbm_params.get("K", args.K)
    args.T = gbm_params.get("T", args.T)

    set_seed(args.seed_arch)
    train_idx, val_idx, test_idx = split_data(args.paths, seed=args.seed_arch)
    split_hash = compute_split_hash(train_idx, val_idx, test_idx)
    print(f"Paths={args.paths}  Split hash={split_hash}")
    print(f"Train={len(train_idx)}  Val={len(val_idx)}  Test={len(test_idx)}")

    # ── GBM pipeline ──
    print("\n" + "=" * 60)
    print("  MARKET MODEL: GBM (calibrated)")
    print("=" * 60)

    vols = gbm_params["vols"]
    extra_vol = gbm_params.get("extra_vol", 0.06)
    sigma_avg_scalar = vols[0]  # use first asset's vol for BS price

    S_tilde, dW, time_grid, sigma = simulate_market(
        args.paths, args.N, args.T, args.d_traded, args.m_brownian,
        args.r, vols, extra_vol=extra_vol,
        seed=args.seed_arch, device="cpu",
    )
    H_tilde = compute_european_put_payoff(S_tilde, args.K, args.r, args.T)
    Z_intrinsic = compute_intrinsic_process(S_tilde, args.K, args.r, time_grid)

    print(f"  GBM: r={args.r}, vols={vols}, extra_vol={extra_vol}")

    run_pipeline(
        args, device, S_tilde, dW, time_grid, sigma,
        H_tilde, Z_intrinsic, train_idx, val_idx, test_idx, split_hash,
        sigma_avg_scalar,
    )

    print("\n=== All Done ===")
    print(f"Results saved to {args.output_dir}/")


def _write_csv_summary(all_agg, path):
    """Write aggregated metrics to CSV."""
    import csv
    models = list(all_agg.keys())
    if not models:
        return
    first = all_agg[models[0]]
    metric_keys = [k for k in first.keys() if k != "representative_seed"]

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["model"] + [f"{k}_mean" for k in metric_keys] + \
                 [f"{k}_std" for k in metric_keys]
        writer.writerow(header)
        for model in models:
            row = [model]
            for k in metric_keys:
                row.append(f"{all_agg[model][k]['mean']:.6f}")
            for k in metric_keys:
                row.append(f"{all_agg[model][k]['std']:.6f}")
            writer.writerow(row)


if __name__ == "__main__":
    main()
