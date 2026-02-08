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
)
from src.eval.plots_3d import generate_3d_plots, generate_bsde_3d_plots


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Deep Hedging Experiment")

    # Market
    p.add_argument("--paths", type=int, default=50000)
    p.add_argument("--N", type=int, default=200)
    p.add_argument("--T", type=float, default=1.0)
    p.add_argument("--d_traded", type=int, default=2)
    p.add_argument("--m_brownian", type=int, default=3)
    p.add_argument("--K", type=float, default=1.0)
    p.add_argument("--r", type=float, default=0.0)

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
              act_schedule, dropout, device):
    return DeepBSDE(feat_dim, d_traded, m_brownian, sigma,
                    depth=depth, width=width, act_schedule=act_schedule,
                    dropout=dropout).to(device)


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
                      train_data, val_data, args, device, time_grid=None):
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
                    nn1, ctrl, train_data, val_data, train_cfg, device
                )
                val_cvar = v_losses[-1] if v_losses else float("inf")
                with torch.no_grad():
                    nn1.eval()
                    if ctrl is not None:
                        ctrl.eval()
                    V_T, _ = forward_portfolio(
                        nn1, ctrl, val_data["features"], val_data["Z_intrinsic"],
                        val_data["dS"], use_controller=bool(args.use_controller),
                        tbptt=0,
                    )
                    e = terminal_error(V_T, val_data["H_tilde"])
                    val_mse = (e ** 2).mean().item()

            else:  # DBSDE
                model = make_bsde(feat_dim, d_traded, m_brownian, sigma,
                                  depth, width, act, 0.1, device)
                train_cfg["substeps"] = 0
                t_losses, v_losses = train_bsde(
                    model, train_data, val_data, train_cfg, device
                )
                val_cvar = v_losses[-1] if v_losses else float("inf")
                with torch.no_grad():
                    model.eval()
                    Y_T, _, _ = model(
                        val_data["features"], val_data["dW"],
                        val_data["time_grid"], substeps=0,
                    )
                    e = terminal_error(Y_T, val_data["H_tilde"])
                    val_mse = (e ** 2).mean().item()

            return val_cvar, val_mse

        finally:
            del nn1, ctrl, model
            clear_gpu_cache()

    def objective(trial):
        depth = trial.suggest_categorical("depth", args.depth_grid)
        width = trial.suggest_categorical("width", args.width_grid)
        act = trial.suggest_categorical("act_schedule", args.act_schedules)
        lr = trial.suggest_categorical("lr", args.lrs)

        # Try full batch; on OOM retry once with half batch
        batch_size = args.batch_size
        try:
            val_cvar, val_mse = _run_trial_inner(depth, width, act, lr, batch_size)
        except RuntimeError as ex:
            if "out of memory" in str(ex).lower():
                clear_gpu_cache()
                batch_size = batch_size // 2
                print(f"    OOM -> retrying with batch_size={batch_size}")
                try:
                    val_cvar, val_mse = _run_trial_inner(
                        depth, width, act, lr, batch_size,
                    )
                except RuntimeError as ex2:
                    print(f"    FAILED (retry): {ex2}")
                    clear_gpu_cache()
                    val_cvar, val_mse = float("inf"), float("inf")
            else:
                print(f"    FAILED: {ex}")
                clear_gpu_cache()
                val_cvar, val_mse = float("inf"), float("inf")

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
    study = optuna.create_study(direction="minimize", sampler=sampler,
                                study_name=f"{model_class}_stage1")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # Select best: primary = CVaR95 (study.best_trial), tie-break = MSE, prefer smaller LR
    # Re-sort trial_log for the final selection with tie-breaking
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
            ckpt_path = os.path.join(args.output_dir, "checkpoints",
                                     f"{model_class}_seed{seed}.pt")
            state = {"nn1": nn1.state_dict()}
            if ctrl is not None:
                state["controller"] = ctrl.state_dict()
            save_checkpoint(state, ckpt_path)

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

            ckpt_path = os.path.join(args.output_dir, "checkpoints",
                                     f"{model_class}_seed{seed}.pt")
            save_checkpoint(model.state_dict(), ckpt_path)

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

def main():
    args = parse_args()
    device = get_device()
    print(f"Device: {device}")
    print(f"Quick mode: {args.quick}")

    output_dir = "outputs"
    args.output_dir = output_dir
    for sub in ["plots", "plots_val", "plots_3d", "checkpoints", "run_configs"]:
        ensure_dir(os.path.join(output_dir, sub))

    # ── Step 1: Simulate market ─────────────────
    print("\n=== Step 1: Market Simulation ===")
    set_seed(args.seed_arch)
    S_tilde, dW, time_grid, sigma = simulate_market(
        args.paths, args.N, args.T, args.d_traded, args.m_brownian,
        args.r, [0.2] * args.d_traded, seed=args.seed_arch, device="cpu",
    )
    H_tilde = compute_european_put_payoff(S_tilde, args.K, args.r, args.T)
    Z_intrinsic = compute_intrinsic_process(S_tilde, args.K, args.r, time_grid)
    train_idx, val_idx, test_idx = split_data(args.paths, seed=args.seed_arch)

    split_hash = compute_split_hash(train_idx, val_idx, test_idx)
    print(f"  Paths={args.paths}  N={args.N}  Split hash={split_hash}")
    print(f"  Train={len(train_idx)}  Val={len(val_idx)}  Test={len(test_idx)}")

    # Save split indices
    save_json({"train": train_idx.tolist(), "val": val_idx.tolist(),
               "test": test_idx.tolist(), "hash": split_hash},
              os.path.join(output_dir, "run_configs", "split_indices.json"))

    # ── Step 2: Build features ──────────────────
    print("\n=== Step 2: Feature Construction ===")
    features_train, features_val, features_test, feat_dim = build_features(
        S_tilde, time_grid, args.T, train_idx, val_idx, test_idx,
        latent_dim=args.latent_dim, sig_level=args.sig_level,
        vae_epochs=50 if not args.quick else 10, device=device,
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

    # Prepare data dicts (train + val only; test deferred)
    hedger_train = prepare_hedger_data(features_train, S_tilde_train, Z_train, H_train, device)
    hedger_val = prepare_hedger_data(features_val, S_tilde_val, Z_val, H_val, device)

    bsde_train = prepare_bsde_data(features_train, dW_train, H_train, time_grid, device)
    bsde_val = prepare_bsde_data(features_val, dW_val, H_val, time_grid, device)

    # Save feature tensors
    data_dir = os.path.join(output_dir, "data")
    ensure_dir(data_dir)
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
    print("\n=== Step 3: Stage 1 – Optuna HP Search (TPE) ===")

    all_models = ["FNN", "LSTM", "DBSDE"]
    stage1_path = os.path.join(output_dir, "run_configs", "stage1_optuna.json")

    # Try to load existing Stage 1 results
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
                best, tlog = run_optuna_search(
                    model_class, feat_dim, args.d_traded, args.m_brownian, sigma,
                    hedger_train, hedger_val, args, device,
                )
            else:
                best, tlog = run_optuna_search(
                    model_class, feat_dim, args.d_traded, args.m_brownian, sigma,
                    bsde_train, bsde_val, args, device, time_grid=time_grid,
                )
            best_configs[model_class] = best
            trial_logs[model_class] = tlog

            # Incremental save after each model's search
            save_json({"best_configs": best_configs, "trial_logs": trial_logs},
                      stage1_path)
            print(f"  [saved Stage 1 progress: {list(best_configs.keys())}]")

    # ── Step 4: Stage 2 – Seed robustness (val only) ──
    print("\n=== Step 4: Stage 2 – Seed Robustness (Validation) ===")

    summary_path = os.path.join(output_dir, "metrics_summary.json")
    val_metrics_path = os.path.join(output_dir, "val_metrics.json")

    # Try to load existing Stage 2 results
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
                    feat_dim, args.d_traded, args.m_brownian, sigma,
                    hedger_train, hedger_val,
                    args, device,
                    S_tilde_val=S_tilde_val.to(device),
                    Z_val=Z_val.to(device),
                )
            else:
                seed_res, agg, comp_data = run_seed_robustness(
                    model_class, best_configs[model_class],
                    feat_dim, args.d_traded, args.m_brownian, sigma,
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

            # Incremental save after each model's seed robustness
            save_json(
                {"best_configs": best_configs,
                 "aggregated_val_metrics": {m: all_agg[m] for m in all_agg},
                 "config": {
                     "paths": args.paths, "N": args.N, "T": args.T,
                     "d_traded": args.d_traded, "m_brownian": args.m_brownian,
                     "K": args.K, "r": args.r, "seed_arch": args.seed_arch,
                     "seeds": args.seeds, "split_hash": split_hash,
                 }},
                summary_path,
            )
            _write_csv_summary(all_agg, os.path.join(output_dir, "val_metrics_summary.csv"))
            print(f"  [saved Stage 2 progress: {list(all_agg.keys())}]")

    # ── Step 5: Validation Analysis ────────────
    print("\n=== Step 5: Validation Analysis ===")

    plots_val_dir = os.path.join(output_dir, "plots_val")

    # Determine best model (lowest mean CVaR95_shortfall)
    best_model = min(
        all_agg.keys(),
        key=lambda m: (all_agg[m]["CVaR95_shortfall"]["mean"]
                        if isinstance(all_agg[m].get("CVaR95_shortfall"), dict)
                        else float("inf")),
    )
    print(f"  Best model: {best_model}")

    # Summary table (parameterized title)
    plot_summary_table(all_agg, output_dir=plots_val_dir,
                       title="Model Comparison (Validation Set)")

    # Highlighted validation summary
    plot_validation_summary(all_agg, best_model, output_dir=plots_val_dir)

    # Grouped bar chart
    plot_model_comparison_bars(all_agg, output_dir=plots_val_dir)

    # Comparison plots require comparison_data (only available if Stage 2 ran fresh)
    if all_comparison:
        # Overlay error histograms
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

    # Save val_metrics.json
    save_json(
        {"aggregated_val_metrics": all_agg, "best_model": best_model},
        val_metrics_path,
    )
    _write_csv_summary(all_agg, os.path.join(output_dir, "val_metrics_summary.csv"))
    print(f"  Saved val_metrics.json and val_metrics_summary.csv")

    # ── Step 6: DBSDE substeps study ────────────
    print("\n=== Step 6: DBSDE Substeps Study ===")

    # Check if substeps results already exist in saved summary
    sub_results = []
    if os.path.exists(summary_path):
        summary_saved = load_json(summary_path)
        sub_results = summary_saved.get("substeps_study", [])

    if sub_results:
        print("  Found completed substeps results, skipping.")
        for sr in sub_results:
            print(f"    substeps={sr['substeps']}: "
                  f"MSE={sr['val_MSE']:.6f}  CVaR95={sr['val_CVaR95']:.6f}")
    elif len(args.substeps) > 1:
        sub_results = run_substeps_study(
            best_configs["DBSDE"], feat_dim, args.d_traded, args.m_brownian,
            sigma, bsde_train, bsde_val, args.substeps, args, device, time_grid,
        )
        plot_substeps_convergence(
            sub_results, output_dir=os.path.join(output_dir, "plots"),
        )
    else:
        sub_results = []
        print("  Skipped (only 1 substep value).")

    # ── Step 7: Save final results (val metrics only) ──
    print("\n=== Step 7: Saving Final Results ===")

    # Final JSON summary (overwrites incremental with substeps added)
    summary = {
        "best_configs": best_configs,
        "aggregated_val_metrics": {},
        "best_model": best_model,
        "substeps_study": sub_results,
        "config": {
            "paths": args.paths, "N": args.N, "T": args.T,
            "d_traded": args.d_traded, "m_brownian": args.m_brownian,
            "K": args.K, "r": args.r, "seed_arch": args.seed_arch,
            "seeds": args.seeds, "split_hash": split_hash,
        },
    }
    for mc in all_models:
        summary["aggregated_val_metrics"][mc] = all_agg[mc]

    save_json(summary, summary_path)

    # Final CSV
    _write_csv_summary(all_agg, os.path.join(output_dir, "val_metrics_summary.csv"))

    print("\n=== Done ===")
    print(f"Results saved to {output_dir}/")


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
