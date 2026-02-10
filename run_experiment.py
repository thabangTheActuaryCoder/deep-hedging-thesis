#!/usr/bin/env python3
"""Main entrypoint for deep hedging experiments.

Two-stage bias control protocol:
  Stage 1: Architecture + LR selection at fixed seed (seed_arch)
  Stage 2: Seed robustness on best configs with multiple seeds

Models: FNN + Controller, LSTM + Controller, Deep BSDE (standalone)
"""
import argparse
import os
import sys
import time
import copy
import logging

# Reduce CUDA memory fragmentation (recommended by PyTorch for large models)
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
from src.sim.simulate_heston import simulate_heston_market
from src.sim.calibration import load_market_params, load_heston_params
from src.features.build_features import build_features
from src.models.fnn import FNNHedger
from src.models.lstm import LSTMHedger
from src.models.controller import Controller, N_PL_FEATURES
from src.models.bsde import DeepBSDE
from src.training.train import (
    train_hedger, train_bsde, forward_portfolio,
    evaluate_hedger, evaluate_bsde,
)
from src.training.losses import shortfall, cvar, terminal_error
from src.eval.metrics import compute_metrics, aggregate_seed_metrics, select_representative_seed
from src.eval.portfolio import simulate_portfolio_path, simulate_bsde_portfolio
from src.eval.plots import (
    generate_all_plots, plot_substeps_convergence, plot_summary_table,
    plot_model_comparison_bars, plot_model_comparison_errors,
    plot_model_comparison_cvar, plot_validation_summary,
    plot_optuna_validation_loss,
)
from src.eval.plots_3d import generate_3d_plots, generate_bsde_3d_plots
from src.eval.plots_heston import (
    plot_variance_paths, plot_implied_vol_surface,
    plot_vol_distribution, plot_leverage_effect,
    plot_vol_of_vol, plot_heston_summary,
    plot_price_vs_gbm,
    plot_model_comparison_gbm_vs_heston,
    plot_pnl_gbm_vs_heston,
    plot_pnl_violin_gbm_vs_heston,
    plot_pnl_per_model_hist,
    plot_pnl_per_model_violin,
    plot_pnl_all_models_by_regime_hist,
    plot_pnl_all_models_by_regime_violin,
)


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
    p.add_argument("--market_model", type=str, default="both",
                   choices=["gbm", "heston", "both"],
                   help="Market model to use: gbm, heston, or both")
    p.add_argument("--market_config", type=str, default="",
                   help="Path to market params JSON (default: data/market_params_sp500.json)")

    # Features
    p.add_argument("--latent_dim", type=int, default=16)
    p.add_argument("--sig_level", type=int, default=2)

    # Architecture grid
    p.add_argument("--depth_grid", type=int, nargs="+", default=[3, 5, 7])
    p.add_argument("--width_grid", type=int, nargs="+", default=[64, 128, 256])
    p.add_argument("--act_schedules", type=str, nargs="+",
                   default=["relu_all", "tanh_all", "alt_relu_tanh", "alt_tanh_relu"])
    p.add_argument("--lrs", type=float, nargs="+", default=[3e-4, 1e-3, 3e-3])

    # Training
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=2048)

    # Regularization
    p.add_argument("--l1", type=float, default=0.0)
    p.add_argument("--l2", type=float, default=1e-4)

    # Loss
    p.add_argument("--objective", type=str, default="cvar_shortfall")
    p.add_argument("--cvar_q", type=float, default=0.95)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=1.0)

    # Controller
    p.add_argument("--use_controller", type=int, default=1)
    p.add_argument("--delta_clip", type=float, default=5.0)

    # LSTM
    p.add_argument("--tbptt", type=int, default=50)

    # Seeds
    p.add_argument("--seed_arch", type=int, default=0)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])

    # DBSDE
    p.add_argument("--substeps", type=int, nargs="+", default=[0, 5, 10])

    # Optuna HP search
    p.add_argument("--n_trials", type=int, default=60,
                   help="Number of Optuna trials per model class in Stage 1")

    # Quick mode
    p.add_argument("--quick", action="store_true")

    args = p.parse_args()

    if args.quick:
        args.paths = 5000
        args.epochs = 50
        args.patience = 5
        args.seeds = [0]
        args.n_trials = 3
        args.substeps = [0]
        # Use smaller architectures that fit on local GPU (MPS/laptop)
        args.depth_grid = [3, 5]
        args.width_grid = [64, 128]
        args.batch_size = 1024

    return args


# ──────────────────────────────────────────────
# Model factories
# ──────────────────────────────────────────────

def make_fnn(feat_dim, d_traded, depth, width, act_schedule, dropout, device):
    return FNNHedger(feat_dim, d_traded, depth=depth, width=width,
                     act_schedule=act_schedule, dropout=dropout).to(device)


def make_lstm(feat_dim, d_traded, depth, width, act_schedule, dropout, device):
    return LSTMHedger(feat_dim, d_traded, num_layers=depth, hidden_size=width,
                      act_schedule=act_schedule, dropout=dropout).to(device)


def make_controller(feat_dim, d_traded, depth, width, act_schedule,
                    dropout, delta_clip, device):
    return Controller(feat_dim, d_traded, depth=depth, width=width,
                      act_schedule=act_schedule, dropout=dropout,
                      delta_clip=delta_clip).to(device)


def make_bsde(feat_dim, d_traded, m_brownian, sigma, depth, width,
              act_schedule, dropout, device, sigma_avg=None):
    return DeepBSDE(feat_dim, d_traded, m_brownian, sigma,
                    depth=depth, width=width, act_schedule=act_schedule,
                    dropout=dropout, sigma_avg=sigma_avg).to(device)


# ──────────────────────────────────────────────
# Data preparation helpers
# ──────────────────────────────────────────────

def prepare_hedger_data(features, S_tilde_split, Z_split, H_split, device):
    """Prepare data dict for hedger training/eval."""
    N = features.shape[1] - 1
    dS = S_tilde_split[:, 1:, :] - S_tilde_split[:, :-1, :]
    return {
        "features": features[:, :N, :].to(device),
        "Z_intrinsic": Z_split[:, :N].to(device),
        "dS": dS.to(device),
        "H_tilde": H_split.to(device),
    }


def prepare_bsde_data(features, dW_split, H_split, time_grid, device):
    """Prepare data dict for BSDE training/eval."""
    return {
        "features": features.to(device),
        "dW": dW_split.to(device),
        "H_tilde": H_split.to(device),
        "time_grid": time_grid.to(device),
    }


# ──────────────────────────────────────────────
# Stage 1: Optuna HP search (TPE Bayesian)
# ──────────────────────────────────────────────

def run_optuna_search(model_class, feat_dim, d_traded, m_brownian, sigma,
                      train_data, val_data, args, device, time_grid=None,
                      output_dir=None):
    """Run Optuna TPE search over architecture + LR for one model class.

    Search space (categorical, matching spec):
        depth:       {3, 5}
        width:       {64, 128}
        act_schedule: {relu_all, tanh_all, alt_relu_tanh, alt_tanh_relu}
        lr:          {3e-4, 1e-3, 3e-3}

    Objective: minimize validation CVaR95(shortfall).
    Tie-break: validation MSE, then prefer smaller LR.

    Returns:
        best_config: dict with best (depth, width, act_schedule, lr)
        trial_log: list of all tried configs with val metrics
    """
    trial_log = []

    # Keep data on CPU and move to device per-trial; wipe after each trial
    # so every trial starts with a clean memory slate (works on laptop + GPU)
    _needs_isolate = True

    def _data_to(data, target_device):
        """Move all tensors in a data dict to target_device."""
        return {k: v.to(target_device) if isinstance(v, torch.Tensor) else v
                for k, v in data.items()}

    def _run_trial_inner(depth, width, act, lr, batch_size):
        """Run a single trial; returns (val_cvar, val_mse)."""
        set_seed(args.seed_arch)

        train_cfg = {
            "lr": lr, "epochs": args.epochs, "patience": args.patience,
            "batch_size": batch_size, "l1": args.l1, "l2": args.l2,
            "alpha": args.alpha, "beta": args.beta, "cvar_q": args.cvar_q,
            "use_controller": bool(args.use_controller),
            "tbptt": args.tbptt if model_class == "LSTM" else 0,
        }

        # Move data to GPU only for this trial (LSTM isolation)
        if _needs_isolate:
            t_data = _data_to(train_data, device)
            v_data = _data_to(val_data, device)
        else:
            t_data = train_data
            v_data = val_data

        nn1 = ctrl = model = None
        try:
            if model_class in ("FNN", "LSTM"):
                factory = make_fnn if model_class == "FNN" else make_lstm
                nn1 = factory(feat_dim, d_traded, depth, width, act, 0.1, device)
                ctrl = None
                if bool(args.use_controller):
                    ctrl = make_controller(
                        feat_dim, d_traded, depth, width, act,
                        0.1, args.delta_clip, device,
                    )
                t_losses, v_losses = train_hedger(
                    nn1, ctrl, t_data, v_data, train_cfg, device
                )
                val_cvar = v_losses[-1] if v_losses else float("inf")
                with torch.no_grad():
                    nn1.eval()
                    if ctrl is not None:
                        ctrl.eval()
                    V_T, _ = forward_portfolio(
                        nn1, ctrl, v_data["features"], v_data["Z_intrinsic"],
                        v_data["dS"], use_controller=bool(args.use_controller),
                        tbptt=0,
                    )
                    e = terminal_error(V_T, v_data["H_tilde"])
                    val_mse = (e ** 2).mean().item()

            else:  # DBSDE
                model = make_bsde(feat_dim, d_traded, m_brownian, sigma,
                                  depth, width, act, 0.1, device)
                train_cfg["substeps"] = 0
                t_losses, v_losses = train_bsde(
                    model, t_data, v_data, train_cfg, device
                )
                val_cvar = v_losses[-1] if v_losses else float("inf")
                with torch.no_grad():
                    model.eval()
                    Y_T, _, _ = model(
                        v_data["features"], v_data["dW"],
                        v_data["time_grid"], substeps=0,
                    )
                    e = terminal_error(Y_T, v_data["H_tilde"])
                    val_mse = (e ** 2).mean().item()

            return val_cvar, val_mse

        finally:
            del nn1, ctrl, model
            # For LSTM: delete GPU copies of data, wipe everything
            if _needs_isolate:
                del t_data, v_data
            clear_gpu_cache()

    # Cache: skip duplicate hyperparameter combos
    _seen_configs = {}   # (depth, width, act, lr) -> (val_cvar, val_mse)

    def objective(trial):
        depth = trial.suggest_categorical("depth", args.depth_grid)
        width = trial.suggest_categorical("width", args.width_grid)
        act = trial.suggest_categorical("act_schedule", args.act_schedules)
        lr = trial.suggest_categorical("lr", args.lrs)

        # Skip if this exact config already ran
        config_key = (depth, width, act, lr)
        if config_key in _seen_configs:
            val_cvar, val_mse = _seen_configs[config_key]
            trial.set_user_attr("val_MSE", val_mse)
            trial_log.append({
                "depth": depth, "width": width, "act_schedule": act, "lr": lr,
                "val_CVaR95": val_cvar, "val_MSE": val_mse,
            })
            print(f"    Trial {trial.number}: depth={depth} width={width} "
                  f"act={act} lr={lr} -> CVaR95={val_cvar:.6f}  MSE={val_mse:.6f}  [cached]")
            return val_cvar

        # Try full batch; on OOM keep halving batch (up to 3 retries)
        batch_size = args.batch_size
        min_batch = max(64, args.batch_size // 16)
        val_cvar, val_mse = float("inf"), float("inf")
        while batch_size >= min_batch:
            try:
                val_cvar, val_mse = _run_trial_inner(
                    depth, width, act, lr, batch_size,
                )
                break  # success
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

        trial_log.append({
            "depth": depth, "width": width, "act_schedule": act, "lr": lr,
            "val_CVaR95": val_cvar, "val_MSE": val_mse,
        })

        print(f"    Trial {trial.number}: depth={depth} width={width} "
              f"act={act} lr={lr} -> CVaR95={val_cvar:.6f}  MSE={val_mse:.6f}")

        return val_cvar

    # Suppress Optuna internal logs, keep our prints
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    n_trials = args.n_trials
    total_configs = (len(args.depth_grid) * len(args.width_grid)
                     * len(args.act_schedules) * len(args.lrs))
    # Cap trials at total possible configs (no duplicates beyond that)
    n_trials = min(n_trials, total_configs)

    print(f"\n  Optuna search for {model_class}: up to {n_trials} trials "
          f"(search space = {total_configs} configs)")

    sampler = optuna.samplers.TPESampler(seed=args.seed_arch)

    # Use SQLite storage for checkpoint/resume (survives Colab disconnects)
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
        # Seed the dedup cache from previously completed trials
        for t in study.trials:
            if t.state == optuna.trial.TrialState.COMPLETE:
                key = (t.params["depth"], t.params["width"],
                       t.params["act_schedule"], t.params["lr"])
                mse = t.user_attrs.get("val_MSE", float("inf"))
                _seen_configs[key] = (t.value, mse)
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, show_progress_bar=False)

    # Rebuild trial_log from all completed study trials (covers resumed + new)
    trial_log = []
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            trial_log.append({
                "depth": t.params["depth"], "width": t.params["width"],
                "act_schedule": t.params["act_schedule"], "lr": t.params["lr"],
                "val_CVaR95": t.value,
                "val_MSE": t.user_attrs.get("val_MSE", float("inf")),
            })

    # Select best: primary = CVaR95, tie-break = MSE, prefer smaller LR
    trial_log.sort(key=lambda x: (x["val_CVaR95"], x["val_MSE"], x["lr"]))
    best = trial_log[0]

    print(f"\n  Best {model_class}: depth={best['depth']} width={best['width']} "
          f"act={best['act_schedule']} lr={best['lr']} "
          f"CVaR95={best['val_CVaR95']:.6f}  MSE={best['val_MSE']:.6f}")

    return best, trial_log


# ──────────────────────────────────────────────
# Stage 2: Seed robustness
# ──────────────────────────────────────────────

def run_seed_robustness(model_class, best_config, feat_dim, d_traded, m_brownian,
                        sigma, train_data, val_data, args, device,
                        S_tilde_val, Z_val, time_grid=None, dW_val=None):
    """Train best config across multiple seeds and evaluate on validation set.

    Returns:
        seed_results: list of per-seed results
        agg_metrics: aggregated validation metrics
        comparison_data: dict with V_T, H_tilde, errors per representative seed
    """
    depth = best_config["depth"]
    width = best_config["width"]
    act = best_config["act_schedule"]
    lr = best_config["lr"]
    seed_results = []
    # Track V_T and H_tilde per seed for comparison plots (keep last seed)
    last_VT = None
    last_H = None

    print(f"\n  Seed robustness for {model_class}: seeds={args.seeds}")

    for seed in args.seeds:
        clear_gpu_cache()
        print(f"\n    Seed {seed}:", end=" ", flush=True)

        # ── Per-seed resume: skip if checkpoint + metrics already exist ──
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
            "lr": lr, "epochs": args.epochs, "patience": args.patience,
            "batch_size": args.batch_size, "l1": args.l1, "l2": args.l2,
            "alpha": args.alpha, "beta": args.beta, "cvar_q": args.cvar_q,
            "use_controller": bool(args.use_controller),
            "tbptt": args.tbptt if model_class == "LSTM" else 0,
        }

        if model_class in ("FNN", "LSTM"):
            factory = make_fnn if model_class == "FNN" else make_lstm
            nn1 = factory(feat_dim, d_traded, depth, width, act, 0.1, device)
            ctrl = None
            if bool(args.use_controller):
                ctrl = make_controller(feat_dim, d_traded, depth, width, act,
                                       0.1, args.delta_clip, device)

            t_losses, v_losses = train_hedger(
                nn1, ctrl, train_data, val_data, train_cfg, device
            )

            # Evaluate on validation set (full metrics with paths)
            val_metrics = _eval_hedger_full(
                nn1, ctrl, val_data, Z_val,
                bool(args.use_controller), args.cvar_q
            )

            # Save checkpoint
            state = {"nn1": nn1.state_dict()}
            if ctrl is not None:
                state["controller"] = ctrl.state_dict()
            save_checkpoint(state, ckpt_path)

            # Save per-seed metrics for resume
            save_json(val_metrics, seed_metric_path)

            # Plots on validation data
            with torch.no_grad():
                V_T, info = forward_portfolio(
                    nn1, ctrl, val_data["features"],
                    val_data["Z_intrinsic"], val_data["dS"],
                    use_controller=bool(args.use_controller), tbptt=0,
                )
                V_path = info["V_path"]

            generate_all_plots(
                model_class, seed, V_T.cpu(), val_data["H_tilde"].cpu(),
                V_path.cpu(), Z_val.cpu(),
                t_losses, v_losses,
                output_dir=os.path.join(args.output_dir, "plots_val"),
            )

            last_VT = V_T.cpu()
            last_H = val_data["H_tilde"].cpu()

            # 3D plots for representative seed
            if seed == args.seeds[0]:
                generate_3d_plots(
                    model_class, nn1, ctrl, feat_dim, d_traded,
                    args.N, args.T,
                    output_dir=os.path.join(args.output_dir, "plots_3d"),
                )

        else:  # DBSDE
            model = make_bsde(feat_dim, d_traded, m_brownian, sigma,
                              depth, width, act, 0.1, device)
            train_cfg["substeps"] = 0
            t_losses, v_losses = train_bsde(
                model, train_data, val_data, train_cfg, device
            )

            # Evaluate on validation set (full metrics with paths)
            val_metrics = _eval_bsde_full(
                model, val_data, S_tilde_val, Z_val, time_grid,
                args.cvar_q, device
            )

            save_checkpoint(model.state_dict(), ckpt_path)

            # Save per-seed metrics for resume
            save_json(val_metrics, seed_metric_path)

            # Plots on validation data
            with torch.no_grad():
                V_path_h, Y_path, Delta_all = simulate_bsde_portfolio(
                    model, val_data["features"].to(device),
                    S_tilde_val.to(device), dW_val.to(device),
                    time_grid.to(device),
                )
                V_T = V_path_h[:, -1]

            generate_all_plots(
                model_class, seed, V_T.cpu(), val_data["H_tilde"].cpu(),
                V_path_h.cpu(), Z_val.cpu(),
                t_losses, v_losses,
                output_dir=os.path.join(args.output_dir, "plots_val"),
            )

            last_VT = V_T.cpu()
            last_H = val_data["H_tilde"].cpu()

            if seed == args.seeds[0]:
                generate_bsde_3d_plots(
                    model_class, model, feat_dim, d_traded,
                    args.N, args.T,
                    output_dir=os.path.join(args.output_dir, "plots_3d"),
                )

        print(f"CVaR95(val)={val_metrics['CVaR95_shortfall']:.6f}")

        seed_results.append({
            "seed": seed,
            "val_metrics": val_metrics,
            "train_losses": t_losses,
            "val_losses": v_losses,
        })

    # Aggregate on validation metrics
    val_metrics_list = [r["val_metrics"] for r in seed_results]
    agg = aggregate_seed_metrics(val_metrics_list)
    rep_seed = select_representative_seed(seed_results)
    agg["representative_seed"] = rep_seed

    # Comparison data: V_T and H from representative (last) seed
    comparison_data = {
        "V_T": last_VT,
        "H_tilde": last_H,
    }

    return seed_results, agg, comparison_data


# ──────────────────────────────────────────────
# Evaluation helpers
# ──────────────────────────────────────────────

def _eval_hedger_full(nn1, ctrl, data, Z_split, use_ctrl, cvar_q):
    """Full eval with path-level metrics."""
    nn1.eval()
    if ctrl is not None:
        ctrl.eval()
    with torch.no_grad():
        V_T, info = forward_portfolio(
            nn1, ctrl, data["features"], data["Z_intrinsic"],
            data["dS"], use_controller=use_ctrl, tbptt=0,
        )
        V_path = info["V_path"]
    return compute_metrics(V_T, data["H_tilde"], V_path=V_path,
                           Z_intrinsic=Z_split[:, :V_path.shape[1]].to(V_path.device),
                           cvar_q=cvar_q)


def _eval_bsde_full(model, data, S_tilde_split, Z_split, time_grid, cvar_q, device):
    """Full BSDE eval with portfolio path."""
    model.eval()
    with torch.no_grad():
        V_path, _, _ = simulate_bsde_portfolio(
            model, data["features"].to(device), S_tilde_split.to(device),
            data["dW"].to(device), time_grid.to(device),
        )
        V_T = V_path[:, -1]
    return compute_metrics(V_T, data["H_tilde"].to(device),
                           V_path=V_path,
                           Z_intrinsic=Z_split[:, :V_path.shape[1]].to(device),
                           cvar_q=cvar_q)


# ──────────────────────────────────────────────
# Substeps study (DBSDE)
# ──────────────────────────────────────────────

def run_substeps_study(best_bsde_config, feat_dim, d_traded, m_brownian, sigma,
                       train_data, val_data, substeps_list, args, device, time_grid):
    """Train DBSDE at various substep settings, report convergence."""
    results = []
    depth = best_bsde_config["depth"]
    width = best_bsde_config["width"]
    act = best_bsde_config["act_schedule"]
    lr = best_bsde_config["lr"]

    print(f"\n  Substeps study: {substeps_list}")

    for subs in substeps_list:
        clear_gpu_cache()
        print(f"    substeps={subs}", end=" ... ", flush=True)
        set_seed(args.seed_arch)

        model = make_bsde(feat_dim, d_traded, m_brownian, sigma,
                          depth, width, act, 0.1, device)

        train_cfg = {
            "lr": lr, "epochs": args.epochs, "patience": args.patience,
            "batch_size": args.batch_size, "l1": args.l1, "l2": args.l2,
            "cvar_q": args.cvar_q, "substeps": subs,
        }
        _, v_losses = train_bsde(model, train_data, val_data, train_cfg, device)

        # Final val metrics
        model.eval()
        with torch.no_grad():
            Y_T, _, _ = model(
                val_data["features"], val_data["dW"],
                time_grid.to(device), substeps=subs,
            )
            e = terminal_error(Y_T, val_data["H_tilde"])
            s = shortfall(Y_T, val_data["H_tilde"])

        val_mse = (e ** 2).mean().item()
        val_cvar = cvar(s, q=args.cvar_q).item()
        print(f"MSE={val_mse:.6f}  CVaR95={val_cvar:.6f}")

        results.append({
            "substeps": subs,
            "val_MSE": val_mse,
            "val_CVaR95": val_cvar,
        })

    return results


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def run_pipeline(market_label, args, device, S_tilde, dW, time_grid, sigma,
                 H_tilde, Z_intrinsic, train_idx, val_idx, test_idx,
                 split_hash, V_paths=None, heston_params=None):
    """Run the full training pipeline (Steps 2-7) for one market model.

    Args:
        market_label: "gbm" or "heston"
        args: parsed CLI args
        device: torch device
        S_tilde, dW, time_grid, sigma: simulated market data
            For Heston, sigma should be the effective sigma_avg matrix.
        H_tilde: discounted payoffs
        Z_intrinsic: intrinsic value process
        train_idx, val_idx, test_idx: split indices
        split_hash: reproducibility hash
        V_paths: [n_paths, N+1, d_traded] variance paths (Heston only)
        heston_params: dict with kappa, theta, xi, rho, v0 (Heston only)

    Returns:
        all_agg: {model_name: aggregated val metrics}
    """
    output_dir = os.path.join(args.output_dir, market_label)
    for sub in ["plots", "plots_val", "plots_3d", "checkpoints", "run_configs",
                "data"]:
        ensure_dir(os.path.join(output_dir, sub))
    if market_label == "heston":
        ensure_dir(os.path.join(output_dir, "plots_heston"))

    m_brownian = dW.shape[2]

    print(f"\n{'='*60}")
    print(f"  Pipeline: {market_label.upper()} market  (m_brownian={m_brownian})")
    print(f"{'='*60}")

    # Save split indices
    save_json({"train": train_idx.tolist(), "val": val_idx.tolist(),
               "test": test_idx.tolist(), "hash": split_hash},
              os.path.join(output_dir, "run_configs", "split_indices.json"))

    # ── Step 2: Build features ──────────────────
    print(f"\n=== [{market_label}] Step 2: Feature Construction ===")

    data_dir = os.path.join(output_dir, "data")
    feat_paths = [os.path.join(data_dir, f"features_{s}.pt")
                  for s in ("train", "val", "test")]

    if all(os.path.exists(p) for p in feat_paths):
        print("  Loading cached features...")
        features_train = torch.load(feat_paths[0], weights_only=True)
        features_val = torch.load(feat_paths[1], weights_only=True)
        features_test = torch.load(feat_paths[2], weights_only=True)
        feat_dim = features_train.shape[2]
    else:
        features_train, features_val, features_test, feat_dim = build_features(
            S_tilde, time_grid, args.T, train_idx, val_idx, test_idx,
            latent_dim=args.latent_dim, sig_level=args.sig_level,
            vae_epochs=50 if not args.quick else 10, device=device,
            V_paths=V_paths,
        )
    print(f"  Feature dim: {feat_dim}")

    # Split other data
    S_tilde_train = S_tilde[train_idx]
    S_tilde_val = S_tilde[val_idx]
    S_tilde_test = S_tilde[test_idx]
    Z_train = Z_intrinsic[train_idx]
    Z_val = Z_intrinsic[val_idx]
    Z_test = Z_intrinsic[test_idx]
    H_train = H_tilde[train_idx]
    H_val = H_tilde[val_idx]
    H_test = H_tilde[test_idx]
    dW_train = dW[train_idx]
    dW_val = dW[val_idx]
    dW_test = dW[test_idx]

    # Prepare data dicts
    hedger_train = prepare_hedger_data(features_train, S_tilde_train, Z_train, H_train, device)
    hedger_val = prepare_hedger_data(features_val, S_tilde_val, Z_val, H_val, device)

    bsde_train = prepare_bsde_data(features_train, dW_train, H_train, time_grid, device)
    bsde_val = prepare_bsde_data(features_val, dW_val, H_val, time_grid, device)

    # Save feature tensors
    data_dir = os.path.join(output_dir, "data")
    for name, tensor in [("features_train", features_train),
                         ("features_val", features_val),
                         ("features_test", features_test),
                         ("payoff_T_train", H_train),
                         ("payoff_T_val", H_val),
                         ("payoff_T_test", H_test),
                         ("payoff_path_train", Z_train),
                         ("payoff_path_val", Z_val),
                         ("payoff_path_test", Z_test)]:
        torch.save(tensor, os.path.join(data_dir, f"{name}.pt"))

    # ── Step 3: Stage 1 – Optuna HP search ─────
    print(f"\n=== [{market_label}] Step 3: Stage 1 – Optuna HP Search (TPE) ===")

    all_models = ["FNN", "LSTM", "DBSDE"]
    stage1_path = os.path.join(output_dir, "run_configs", "stage1_optuna.json")

    best_configs = {}
    trial_logs = {}
    if os.path.exists(stage1_path):
        stage1_saved = load_json(stage1_path)
        best_configs = stage1_saved.get("best_configs", {})
        trial_logs = stage1_saved.get("trial_logs", {})

    stage1_done = all(mc in best_configs for mc in all_models)
    if stage1_done:
        print("  Found completed Stage 1 results, skipping HP search.")
        for mc in all_models:
            bc = best_configs[mc]
            print(f"    {mc}: depth={bc['depth']} width={bc['width']} "
                  f"act={bc['act_schedule']} lr={bc['lr']} "
                  f"CVaR95={bc['val_CVaR95']:.6f}")
    else:
        for model_class in all_models:
            if model_class in best_configs:
                bc = best_configs[model_class]
                print(f"\n--- {model_class} [cached] ---")
                print(f"  Best {model_class}: depth={bc['depth']} "
                      f"width={bc['width']} act={bc['act_schedule']} "
                      f"lr={bc['lr']} CVaR95={bc['val_CVaR95']:.6f}")
                continue

            print(f"\n--- {model_class} ---")
            if model_class in ("FNN", "LSTM"):
                # Per-trial isolation: pass CPU data so each trial starts clean
                _train = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                          for k, v in hedger_train.items()}
                _val = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                        for k, v in hedger_val.items()}
                clear_gpu_cache()
                best, tlog = run_optuna_search(
                    model_class, feat_dim, args.d_traded, m_brownian, sigma,
                    _train, _val, args, device,
                    output_dir=output_dir,
                )
                del _train, _val
                clear_gpu_cache()
            else:
                # BSDE also gets per-trial isolation
                _train = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                          for k, v in bsde_train.items()}
                _val = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                        for k, v in bsde_val.items()}
                clear_gpu_cache()
                best, tlog = run_optuna_search(
                    model_class, feat_dim, args.d_traded, m_brownian, sigma,
                    _train, _val, args, device, time_grid=time_grid,
                    output_dir=output_dir,
                )
                del _train, _val
                clear_gpu_cache()
            best_configs[model_class] = best
            trial_logs[model_class] = tlog

            save_json({"best_configs": best_configs, "trial_logs": trial_logs},
                      stage1_path)
            print(f"  [saved Stage 1 progress: {list(best_configs.keys())}]")

            # Full memory wipe between models so next model starts fresh
            clear_gpu_cache()
            print(f"  [GPU memory cleared after {model_class}]")

    # ── Consolidated trials database (all models in one table) ──
    import sqlite3
    consolidated_db = os.path.join(output_dir, "run_configs", "all_trials.db")
    with sqlite3.connect(consolidated_db) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trials (
                model       TEXT,
                trial_num   INTEGER,
                depth       INTEGER,
                width       INTEGER,
                act_schedule TEXT,
                lr          REAL,
                val_CVaR95  REAL,
                val_MSE     REAL,
                is_best     INTEGER DEFAULT 0,
                PRIMARY KEY (model, trial_num)
            )
        """)
        for mc in all_models:
            tlog = trial_logs.get(mc, [])
            bc = best_configs.get(mc, {})
            best_key = (bc.get("depth"), bc.get("width"),
                        bc.get("act_schedule"), bc.get("lr"))
            for i, t in enumerate(tlog):
                t_key = (t["depth"], t["width"], t["act_schedule"], t["lr"])
                conn.execute("""
                    INSERT OR REPLACE INTO trials
                    (model, trial_num, depth, width, act_schedule, lr,
                     val_CVaR95, val_MSE, is_best)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (mc, i, t["depth"], t["width"], t["act_schedule"],
                      t["lr"], t["val_CVaR95"], t["val_MSE"],
                      1 if t_key == best_key else 0))
        conn.commit()
    print(f"  Consolidated trials saved to {consolidated_db}")

    # Also write a CSV for easy viewing
    import csv
    csv_path = os.path.join(output_dir, "run_configs", "all_trials.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "trial", "depth", "width", "act_schedule",
                     "lr", "val_CVaR95", "val_MSE", "is_best"])
        for mc in all_models:
            tlog = trial_logs.get(mc, [])
            bc = best_configs.get(mc, {})
            best_key = (bc.get("depth"), bc.get("width"),
                        bc.get("act_schedule"), bc.get("lr"))
            for i, t in enumerate(tlog):
                t_key = (t["depth"], t["width"], t["act_schedule"], t["lr"])
                w.writerow([mc, i, t["depth"], t["width"], t["act_schedule"],
                            t["lr"], f"{t['val_CVaR95']:.6f}",
                            f"{t['val_MSE']:.6f}",
                            1 if t_key == best_key else 0])
    print(f"  Consolidated CSV saved to {csv_path}")

    # Generate Optuna validation loss plots per model
    plots_val_dir = os.path.join(output_dir, "plots_val")
    for model_class in all_models:
        tlog = trial_logs.get(model_class, [])
        if tlog:
            plot_optuna_validation_loss(tlog, model_class, output_dir=plots_val_dir)
    if trial_logs:
        print("  Generated Optuna validation loss plots")

    # ── Step 4: Stage 2 – Seed robustness (val only) ──
    print(f"\n=== [{market_label}] Step 4: Stage 2 – Seed Robustness (Validation) ===")

    summary_path = os.path.join(output_dir, "metrics_summary.json")
    val_metrics_path = os.path.join(output_dir, "val_metrics.json")

    # Temporarily override output_dir in args for seed robustness
    orig_output_dir = args.output_dir
    args.output_dir = output_dir

    all_results = {}
    all_agg = {}
    all_comparison = {}
    if os.path.exists(summary_path):
        summary_saved = load_json(summary_path)
        all_agg = summary_saved.get("aggregated_val_metrics", {})

    stage2_done = all(mc in all_agg for mc in all_models)
    if stage2_done:
        print("  Found completed Stage 2 results, skipping seed robustness.")
        for mc in all_models:
            agg = all_agg[mc]
            rep = agg.get("representative_seed", "?")
            cvar_key = [k for k in agg if "CVaR" in k]
            if cvar_key:
                mean = agg[cvar_key[0]].get("mean", "?")
                std = agg[cvar_key[0]].get("std", "?")
                print(f"    {mc}: {cvar_key[0]} = {mean} +/- {std}  (rep seed={rep})")
            else:
                print(f"    {mc}: loaded (rep seed={rep})")
    else:
        for model_class in all_models:
            if model_class in all_agg:
                print(f"\n--- {model_class} [cached] ---")
                continue

            print(f"\n--- {model_class} ---")
            if model_class in ("FNN", "LSTM"):
                seed_res, agg, comp_data = run_seed_robustness(
                    model_class, best_configs[model_class],
                    feat_dim, args.d_traded, m_brownian, sigma,
                    hedger_train, hedger_val,
                    args, device,
                    S_tilde_val=S_tilde_val.to(device),
                    Z_val=Z_val.to(device),
                )
            else:
                seed_res, agg, comp_data = run_seed_robustness(
                    model_class, best_configs[model_class],
                    feat_dim, args.d_traded, m_brownian, sigma,
                    bsde_train, bsde_val,
                    args, device,
                    S_tilde_val=S_tilde_val.to(device),
                    Z_val=Z_val.to(device),
                    time_grid=time_grid,
                    dW_val=dW_val,
                )
            all_results[model_class] = seed_res
            all_agg[model_class] = agg
            all_comparison[model_class] = comp_data

            save_json(
                {"best_configs": best_configs,
                 "aggregated_val_metrics": {m: all_agg[m] for m in all_agg},
                 "config": {
                     "paths": args.paths, "N": args.N, "T": args.T,
                     "d_traded": args.d_traded, "m_brownian": m_brownian,
                     "K": args.K, "r": args.r, "seed_arch": args.seed_arch,
                     "seeds": args.seeds, "split_hash": split_hash,
                     "market_model": market_label,
                 }},
                summary_path,
            )
            _write_csv_summary(all_agg, os.path.join(output_dir, "val_metrics_summary.csv"))
            print(f"  [saved Stage 2 progress: {list(all_agg.keys())}]")

            # Full memory wipe between models so next model starts fresh
            clear_gpu_cache()
            print(f"  [GPU memory cleared after {model_class}]")

    # Restore original output_dir
    args.output_dir = orig_output_dir

    # ── Step 5: Validation Analysis ────────────
    print(f"\n=== [{market_label}] Step 5: Validation Analysis ===")

    plots_val_dir = os.path.join(output_dir, "plots_val")

    best_model = min(
        all_agg.keys(),
        key=lambda m: (all_agg[m]["CVaR95_shortfall"]["mean"]
                        if isinstance(all_agg[m].get("CVaR95_shortfall"), dict)
                        else float("inf")),
    )
    print(f"  Best model: {best_model}")

    plot_summary_table(all_agg, output_dir=plots_val_dir,
                       title=f"Model Comparison – {market_label.upper()} (Validation Set)")
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
        print("  Generated comparison plots: bars, errors overlay, CVaR overlay")
    else:
        print("  Comparison plots skipped (loaded from cache, no tensor data)")

    # Heston-specific stochastic volatility plots
    if market_label == "heston" and V_paths is not None:
        heston_plot_dir = os.path.join(output_dir, "plots_heston")
        plot_variance_paths(V_paths, time_grid, heston_plot_dir)
        plot_implied_vol_surface(S_tilde, V_paths, args.K, args.T,
                                 time_grid, heston_plot_dir)
        plot_vol_of_vol(V_paths, time_grid, heston_plot_dir)
        plot_leverage_effect(S_tilde, V_paths, time_grid, heston_plot_dir)
        if heston_params is not None:
            plot_vol_distribution(V_paths, time_grid, heston_params, heston_plot_dir)
            plot_heston_summary(V_paths, S_tilde, time_grid, heston_params,
                                heston_plot_dir)
        print("  Generated Heston stochastic volatility plots:"
              " variance paths, vol smile, vol distribution,"
              " leverage effect, vol-of-vol, summary")

    save_json(
        {"aggregated_val_metrics": all_agg, "best_model": best_model},
        val_metrics_path,
    )
    _write_csv_summary(all_agg, os.path.join(output_dir, "val_metrics_summary.csv"))
    print(f"  Saved val_metrics.json and val_metrics_summary.csv")

    # ── Step 6: DBSDE substeps study ────────────
    print(f"\n=== [{market_label}] Step 6: DBSDE Substeps Study ===")

    sub_results = []
    if os.path.exists(summary_path):
        summary_saved = load_json(summary_path)
        sub_results = summary_saved.get("substeps_study", [])

    # Temporarily set output_dir for substeps
    args.output_dir = output_dir

    if sub_results:
        print("  Found completed substeps results, skipping.")
        for sr in sub_results:
            print(f"    substeps={sr['substeps']}: "
                  f"MSE={sr['val_MSE']:.6f}  CVaR95={sr['val_CVaR95']:.6f}")
    elif len(args.substeps) > 1:
        sub_results = run_substeps_study(
            best_configs["DBSDE"], feat_dim, args.d_traded, m_brownian,
            sigma, bsde_train, bsde_val, args.substeps, args, device, time_grid,
        )
        plot_substeps_convergence(
            sub_results, output_dir=os.path.join(output_dir, "plots"),
        )
    else:
        sub_results = []
        print("  Skipped (only 1 substep value).")

    args.output_dir = orig_output_dir

    # ── Step 7: Save final results ──
    print(f"\n=== [{market_label}] Step 7: Saving Final Results ===")

    summary = {
        "best_configs": best_configs,
        "aggregated_val_metrics": {},
        "best_model": best_model,
        "substeps_study": sub_results,
        "market_model": market_label,
        "config": {
            "paths": args.paths, "N": args.N, "T": args.T,
            "d_traded": args.d_traded, "m_brownian": m_brownian,
            "K": args.K, "r": args.r, "seed_arch": args.seed_arch,
            "seeds": args.seeds, "split_hash": split_hash,
        },
    }
    for mc in all_models:
        summary["aggregated_val_metrics"][mc] = all_agg[mc]

    save_json(summary, summary_path)
    _write_csv_summary(all_agg, os.path.join(output_dir, "val_metrics_summary.csv"))

    print(f"\n  [{market_label}] Done. Results saved to {output_dir}/")
    return all_agg, all_comparison


def main():
    args = parse_args()
    device = get_device()
    print(f"Device: {device}")
    print(f"Quick mode: {args.quick}")
    print(f"Market model: {args.market_model}")

    args.output_dir = "outputs"
    ensure_dir(args.output_dir)

    # Load calibrated parameters
    config_path = args.market_config or None
    gbm_params = load_market_params(config_path)
    heston_params = load_heston_params(config_path)

    # Override r/K/T from calibrated params (CLI defaults may differ)
    r_gbm = gbm_params.get("r", args.r)
    r_heston = heston_params.get("r", args.r)
    K = gbm_params.get("K", args.K)
    T = gbm_params.get("T", args.T)
    args.K = K
    args.T = T

    # Common data split (same for both market models — 100k paths)
    set_seed(args.seed_arch)
    train_idx, val_idx, test_idx = split_data(args.paths, seed=args.seed_arch)
    split_hash = compute_split_hash(train_idx, val_idx, test_idx)
    print(f"Paths={args.paths}  Split hash={split_hash}")
    print(f"Train={len(train_idx)}  Val={len(val_idx)}  Test={len(test_idx)}")

    gbm_agg = None
    gbm_comp = {}
    heston_agg = None
    heston_comp = {}
    S_tilde = None
    S_tilde_h = None
    time_grid = None

    # ── GBM pipeline ──
    if args.market_model in ("gbm", "both"):
        print("\n" + "=" * 60)
        print("  MARKET MODEL: GBM (calibrated)")
        print("=" * 60)
        args.r = r_gbm

        gbm_done_path = os.path.join(args.output_dir, "gbm", "metrics_summary.json")
        gbm_complete = False
        if os.path.exists(gbm_done_path):
            gbm_saved = load_json(gbm_done_path)
            saved_agg = gbm_saved.get("aggregated_val_metrics", {})
            # Only skip if all 3 model classes have results
            if all(mc in saved_agg for mc in ("FNN", "LSTM", "DBSDE")):
                gbm_complete = True

        if gbm_complete:
            print("  GBM pipeline already complete, loading results...")
            gbm_agg = saved_agg
        else:
            vols = gbm_params["vols"]
            extra_vol = gbm_params.get("extra_vol", 0.06)

            S_tilde, dW, time_grid, sigma = simulate_market(
                args.paths, args.N, args.T, args.d_traded, args.m_brownian,
                args.r, vols, extra_vol=extra_vol,
                seed=args.seed_arch, device="cpu",
            )
            H_tilde = compute_european_put_payoff(S_tilde, K, args.r, args.T)
            Z_intrinsic = compute_intrinsic_process(S_tilde, K, args.r, time_grid)

            print(f"  GBM: r={args.r}, vols={vols}, extra_vol={extra_vol}")

            gbm_agg, gbm_comp = run_pipeline(
                "gbm", args, device, S_tilde, dW, time_grid, sigma,
                H_tilde, Z_intrinsic, train_idx, val_idx, test_idx, split_hash,
            )

    # ── Heston pipeline ──
    if args.market_model in ("heston", "both"):
        print("\n" + "=" * 60)
        print("  MARKET MODEL: Heston (calibrated)")
        print("=" * 60)
        args.r = r_heston

        heston_done_path = os.path.join(args.output_dir, "heston", "metrics_summary.json")
        heston_complete = False
        if os.path.exists(heston_done_path):
            heston_saved = load_json(heston_done_path)
            saved_agg = heston_saved.get("aggregated_val_metrics", {})
            if all(mc in saved_agg for mc in ("FNN", "LSTM", "DBSDE")):
                heston_complete = True

        if heston_complete:
            print("  Heston pipeline already complete, loading results...")
            heston_agg = saved_agg
        else:
            extra_vol_h = gbm_params.get("extra_vol", 0.06)

            S_tilde_h, dW_h, time_grid_h, sigma_avg_h, V_paths_h = \
                simulate_heston_market(
                    args.paths, args.N, args.T, args.d_traded, heston_params,
                    r=args.r, extra_vol=extra_vol_h,
                    seed=args.seed_arch, device="cpu",
                )
            H_tilde_h = compute_european_put_payoff(S_tilde_h, K, args.r, args.T)
            Z_intrinsic_h = compute_intrinsic_process(S_tilde_h, K, args.r, time_grid_h)

            m_brown_h = dW_h.shape[2]
            # For Heston, sigma for the hedger is the sigma_avg (constant approx)
            # and m_brownian is updated to match
            args.m_brownian = m_brown_h

            print(f"  Heston: r={args.r}, kappa={heston_params['kappa']}, "
                  f"theta={heston_params['theta']}, xi={heston_params['xi']}, "
                  f"rho={heston_params['rho']}")

            heston_agg, heston_comp = run_pipeline(
                "heston", args, device, S_tilde_h, dW_h, time_grid_h, sigma_avg_h,
                H_tilde_h, Z_intrinsic_h, train_idx, val_idx, test_idx, split_hash,
                V_paths=V_paths_h, heston_params=heston_params,
            )

    # ── Step 8: Cross-model comparison ──
    if args.market_model == "both" and gbm_agg is not None and heston_agg is not None:
        print("\n" + "=" * 60)
        print("  Step 8: GBM vs Heston Comparison")
        print("=" * 60)

        comparison_dir = os.path.join(args.output_dir, "comparison")
        comparison_data_path = os.path.join(comparison_dir, "comparison_data.pt")

        # If both pipelines were loaded from cache, comparison plots already exist
        if not gbm_comp and not heston_comp and os.path.exists(comparison_data_path):
            print("  Both pipelines loaded from cache, comparison plots already exist.")
            # Still save combined summary (cheap)
            combined = {"gbm": gbm_agg, "heston": heston_agg}
            save_json(combined, os.path.join(args.output_dir, "metrics_summary.json"))
        else:
            ensure_dir(comparison_dir)
            plot_model_comparison_gbm_vs_heston(gbm_agg, heston_agg, comparison_dir)

            # Price path comparison (Heston vs GBM) — only if tensors available
            if S_tilde_h is not None and S_tilde is not None:
                plot_price_vs_gbm(S_tilde_h, S_tilde, time_grid, comparison_dir)

            # P&L comparison: GBM vs Heston per model
            if gbm_comp and heston_comp:
                plot_pnl_gbm_vs_heston(gbm_comp, heston_comp, comparison_dir)
                plot_pnl_violin_gbm_vs_heston(gbm_comp, heston_comp, comparison_dir)

                # Per-model individual figures (histogram + violin)
                plot_pnl_per_model_hist(gbm_comp, heston_comp, comparison_dir)
                plot_pnl_per_model_violin(gbm_comp, heston_comp, comparison_dir)

                # Cross-model same-regime figures (histogram + violin)
                plot_pnl_all_models_by_regime_hist(gbm_comp, heston_comp, comparison_dir)
                plot_pnl_all_models_by_regime_violin(gbm_comp, heston_comp, comparison_dir)

            # Save comparison data for standalone replotting
            comp_data = {}
            if gbm_comp and heston_comp:
                comp_data["gbm_comp"] = {
                    m: {k: v.cpu() if torch.is_tensor(v) else torch.tensor(v)
                        for k, v in d.items()}
                    for m, d in gbm_comp.items()
                }
                comp_data["heston_comp"] = {
                    m: {k: v.cpu() if torch.is_tensor(v) else torch.tensor(v)
                        for k, v in d.items()}
                    for m, d in heston_comp.items()
                }
            if S_tilde is not None and S_tilde_h is not None:
                # Subsample price paths (only need ~50 for visual)
                n_save = min(50, S_tilde.shape[0])
                comp_data["S_tilde_gbm_sample"] = S_tilde[:n_save].cpu()
                comp_data["S_tilde_heston_sample"] = S_tilde_h[:n_save].cpu()
                comp_data["time_grid"] = time_grid.cpu()
            if comp_data:
                torch.save(comp_data, os.path.join(comparison_dir, "comparison_data.pt"))

            # Save combined summary
            combined = {"gbm": gbm_agg, "heston": heston_agg}
            save_json(combined, os.path.join(args.output_dir, "metrics_summary.json"))
            print(f"  Saved GBM vs Heston comparison to {comparison_dir}/")
            if comp_data:
                print(f"  Saved replot data to {comparison_dir}/comparison_data.pt")

    print("\n=== All Done ===")
    print(f"Results saved to {args.output_dir}/")


def _write_csv_summary(all_agg, path):
    """Write aggregated metrics to CSV."""
    import csv
    models = list(all_agg.keys())
    if not models:
        return
    # Get metric keys from first model
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
