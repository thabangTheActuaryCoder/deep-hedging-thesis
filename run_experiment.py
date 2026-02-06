#!/usr/bin/env python3
"""
Main experiment runner for deep hedging research.

Orchestrates:
1. Market simulation (incomplete market with correlated Brownian motion)
2. Feature construction (base + VAE + signature-like)
3. Model training with random HP search (FNN-5, LSTM-5, DBSDE)
4. Evaluation (portfolio simulation, metrics)
5. Plots (error histograms, convergence, function shape, summary)

Usage:
    python run_experiment.py --paths 20000 --N 50 --seeds 0 1 2 --n_trials 20
"""

import argparse
import json
import os
import sys
import time as time_module

import torch
import numpy as np
import csv

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.sim.simulate_market import simulate_market, compute_payoffs, split_data
from src.features.build_features import build_features
from src.models.fnn import FNN5Hedger
from src.models.lstm import LSTM5Hedger
from src.models.bsde import DeepBSDE
from src.training.train import train_hedger, optuna_search
from src.eval.portfolio import simulate_portfolio, simulate_portfolio_path, compute_hedging_errors
from src.eval.metrics import compute_metrics, format_metrics, aggregate_seed_metrics
from src.eval.plots import (
    generate_all_plots, plot_substeps_convergence,
    plot_function_shape, plot_summary_table,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Deep Hedging Experiment")

    # Market
    parser.add_argument("--paths", type=int, default=20000, help="Number of MC paths")
    parser.add_argument("--N", type=int, default=50, help="Number of exercise time steps")
    parser.add_argument("--T", type=float, default=1.0, help="Terminal time")
    parser.add_argument("--d_traded", type=int, default=2, help="Number of traded assets")
    parser.add_argument("--m_brownian", type=int, default=3, help="Number of Brownian drivers")
    parser.add_argument("--K", type=float, default=100.0, help="Strike price")
    parser.add_argument("--V0", type=float, default=0.0, help="Initial portfolio value")

    # Features
    parser.add_argument("--latent_dim", type=int, default=16, help="VAE latent dimension")
    parser.add_argument("--sig_level", type=int, default=2, help="Signature truncation level")
    parser.add_argument("--vae_epochs", type=int, default=100, help="VAE training epochs")

    # Model architecture (used as defaults / quick-mode fixed values)
    parser.add_argument("--hidden", type=int, default=128, help="Hidden layer dimension (quick mode)")
    parser.add_argument("--lstm_layers", type=int, default=5, help="LSTM layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate (quick mode)")

    # Regularization defaults (quick mode)
    parser.add_argument("--l1", type=float, default=1e-6, help="Default L1 regularization")
    parser.add_argument("--l2", type=float, default=1e-4, help="Default L2 regularization")

    # Training
    parser.add_argument("--epochs", type=int, default=1000, help="Max training epochs per trial")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clip norm")

    # Hyperparameter search
    parser.add_argument("--n_trials", type=int, default=20,
                        help="Number of random HP search trials per model per seed")

    # Experiment
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Random seeds")
    parser.add_argument("--substeps", type=int, nargs="+",
                        default=[0, 5, 10, 20, 30], help="Substeps for convergence study")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")

    # Quick mode (for testing)
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fewer paths, epochs, 1 seed, no search")

    return parser.parse_args()


# =========================================================================
# Model factories that accept (hidden, dropout) from the HP search
# =========================================================================

def make_fnn_factory(feature_dim: int, d_traded: int):
    """Return a factory fn(hidden, dropout) -> FNN5Hedger."""
    def factory(hidden: int = 128, dropout: float = 0.1):
        return FNN5Hedger(feature_dim, d_traded, hidden_dim=hidden,
                          n_layers=5, dropout=dropout)
    return factory


def make_lstm_factory(feature_dim: int, d_traded: int, n_layers: int = 5):
    """Return a factory fn(hidden, dropout) -> LSTM5Hedger."""
    def factory(hidden: int = 128, dropout: float = 0.1):
        return LSTM5Hedger(feature_dim, d_traded, hidden_dim=hidden,
                           n_layers=n_layers, dropout=dropout)
    return factory


def make_bsde_factory(feature_dim: int, d_traded: int, m_brownian: int):
    """Return a factory fn(hidden, dropout) -> DeepBSDE."""
    def factory(hidden: int = 128, dropout: float = 0.1):
        return DeepBSDE(feature_dim, d_traded, m_brownian,
                        hidden_dim=hidden, n_layers=4, dropout=dropout)
    return factory


# =========================================================================
# Single-seed run
# =========================================================================

def run_single_seed(
    args,
    seed: int,
    features_train: torch.Tensor,
    features_val: torch.Tensor,
    features_test: torch.Tensor,
    S_train: torch.Tensor,
    S_val: torch.Tensor,
    S_test: torch.Tensor,
    payoff_T_train: torch.Tensor,
    payoff_T_val: torch.Tensor,
    payoff_T_test: torch.Tensor,
    payoff_path_test: torch.Tensor,
    dW_train: torch.Tensor,
    dW_val: torch.Tensor,
    dW_test: torch.Tensor,
    time_grid: torch.Tensor,
) -> dict:
    """Train and evaluate all three models for a single seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    feature_dim = features_train.shape[-1]
    results = {}

    # Define model factories (accept hidden & dropout from search)
    fnn_factory = make_fnn_factory(feature_dim, args.d_traded)
    lstm_factory = make_lstm_factory(feature_dim, args.d_traded, args.lstm_layers)
    bsde_factory = make_bsde_factory(feature_dim, args.d_traded, args.m_brownian)

    # Map: name -> (factory, model_type, needs_dW)
    model_specs = {
        'FNN-5':  (fnn_factory,  'fnn',  False),
        'LSTM-5': (lstm_factory, 'lstm', False),
        'DBSDE':  (bsde_factory, 'bsde', True),
    }

    for model_name, (factory, mtype, needs_dw) in model_specs.items():
        print(f"\n  === {model_name} (seed={seed}) ===")

        dw_tr = dW_train if needs_dw else None
        dw_v = dW_val if needs_dw else None

        if args.quick:
            # Fixed config, no search
            torch.manual_seed(seed)
            model = factory(hidden=args.hidden, dropout=args.dropout)
            train_res = train_hedger(
                model, mtype,
                features_train, features_val,
                S_train, S_val,
                payoff_T_train, payoff_T_val,
                dW_train=dw_tr, dW_val=dw_v, time=time_grid,
                lr=1e-3, l1_lambda=args.l1, l2_lambda=args.l2,
                epochs=args.epochs, batch_size=args.batch_size,
                patience=args.patience, V0=args.V0, device=args.device,
            )
            best_hp = {'lr': 1e-3, 'l1': args.l1, 'l2': args.l2,
                       'dropout': args.dropout, 'hidden': args.hidden}
            all_trials = [best_hp]
        else:
            # Bayesian HP search (Optuna TPE + pruning)
            model, train_res, best_hp, all_trials = optuna_search(
                model_factory_fn=factory,
                model_type=mtype,
                feature_dim=feature_dim,
                features_train=features_train,
                features_val=features_val,
                S_train=S_train, S_val=S_val,
                payoff_T_train=payoff_T_train,
                payoff_T_val=payoff_T_val,
                dW_train=dw_tr, dW_val=dw_v,
                time=time_grid,
                n_trials=args.n_trials,
                epochs=args.epochs,
                batch_size=args.batch_size,
                patience=args.patience,
                V0=args.V0,
                device=args.device,
                seed=seed,
            )

        # Evaluate on test
        model.eval()
        with torch.no_grad():
            if mtype == 'bsde':
                deltas = model.compute_deltas_batch(
                    features_test.to(args.device), time_grid.to(args.device))
            else:
                deltas = model.compute_deltas(features_test.to(args.device))

        errors = compute_hedging_errors(
            deltas.cpu(), S_test, payoff_T_test, payoff_path_test, args.V0)
        metrics = compute_metrics(
            errors['V_T'], payoff_T_test, errors['total_error'],
            errors.get('worst_error'))
        print(format_metrics(metrics, model_name))

        results[model_name] = {
            'metrics': metrics,
            'train_result': train_res,
            'model': model,
            'errors': errors,
            'hparams': best_hp,
            'all_trials': all_trials,
        }

    return results


# =========================================================================
# Substeps convergence study
# =========================================================================

def run_substeps_study(args, substeps_list: list, seed: int = 0) -> dict:
    """Evaluate MSE/worst-error vs substeps for all models."""
    print("\n" + "=" * 60)
    print("SUBSTEPS CONVERGENCE STUDY")
    print("=" * 60)

    mse_results = {'FNN-5': [], 'LSTM-5': [], 'DBSDE': []}
    worst_results = {'FNN-5': [], 'LSTM-5': [], 'DBSDE': []}

    for substeps in substeps_list:
        print(f"\n--- Substeps: {substeps} ---")

        market = simulate_market(
            n_paths=min(args.paths, 10000), N=args.N, T=args.T,
            d_traded=args.d_traded, m_brownian=args.m_brownian,
            seed=seed, device=args.device, substeps=substeps,
        )
        payoff_T, payoff_path = compute_payoffs(market.S, K=args.K)
        (S_tr, dW_tr, pT_tr, pp_tr), \
        (S_v, dW_v, pT_v, pp_v), \
        (S_te, dW_te, pT_te, pp_te) = split_data(
            market.S, market.dW, payoff_T, payoff_path, seed=seed)

        feat_data = build_features(
            S_tr, S_v, S_te, market.time, market.T,
            latent_dim=args.latent_dim, sig_level=args.sig_level,
            vae_epochs=30, device=args.device, seed=seed)
        f_tr, f_v, f_te = feat_data['train'], feat_data['val'], feat_data['test']
        feature_dim = feat_data['feature_dim']

        sub_epochs = 200  # Enough for convergence study

        for model_name in ['FNN-5', 'LSTM-5', 'DBSDE']:
            torch.manual_seed(seed)
            if model_name == 'FNN-5':
                model = FNN5Hedger(feature_dim, args.d_traded, args.hidden, 5, args.dropout)
                mtype = 'fnn'
            elif model_name == 'LSTM-5':
                model = LSTM5Hedger(feature_dim, args.d_traded, args.hidden,
                                    args.lstm_layers, args.dropout)
                mtype = 'lstm'
            else:
                model = DeepBSDE(feature_dim, args.d_traded, args.m_brownian,
                                 args.hidden, 4, dropout=args.dropout)
                mtype = 'bsde'

            dw_tr = dW_tr if mtype == 'bsde' else None
            dw_v = dW_v if mtype == 'bsde' else None

            train_hedger(
                model, mtype, f_tr, f_v, S_tr, S_v, pT_tr, pT_v,
                dW_train=dw_tr, dW_val=dw_v, time=market.time,
                lr=1e-3, l1_lambda=args.l1, l2_lambda=args.l2,
                epochs=sub_epochs, batch_size=args.batch_size,
                patience=args.patience, V0=args.V0, device=args.device)

            model.eval()
            with torch.no_grad():
                if mtype == 'bsde':
                    deltas = model.compute_deltas_batch(
                        f_te.to(args.device), market.time.to(args.device))
                else:
                    deltas = model.compute_deltas(f_te.to(args.device))

            errors = compute_hedging_errors(deltas.cpu(), S_te, pT_te, pp_te, args.V0)
            metrics = compute_metrics(
                errors['V_T'], pT_te, errors['total_error'], errors.get('worst_error'))

            mse_results[model_name].append(metrics['MSE'])
            worst_results[model_name].append(metrics.get('worst_error_mean', 0.0))
            print(f"  {model_name}: MSE={metrics['MSE']:.6f}, "
                  f"Worst={metrics.get('worst_error_mean', 0.0):.6f}")

    return mse_results, worst_results


# =========================================================================
# Main
# =========================================================================

def main():
    args = parse_args()

    # Quick mode overrides
    if args.quick:
        args.paths = min(args.paths, 5000)
        args.epochs = 50
        args.seeds = [0]
        args.substeps = [0, 5]
        args.vae_epochs = 30
        args.n_trials = 1  # No search
        args.patience = 15
        print("*** QUICK MODE: reduced paths/epochs/seeds, no HP search ***")

    print("=" * 60)
    print("DEEP HEDGING EXPERIMENT")
    print("=" * 60)
    print(f"Paths: {args.paths}, N: {args.N}, T: {args.T}")
    print(f"d_traded: {args.d_traded}, m_brownian: {args.m_brownian}")
    print(f"Epochs: {args.epochs}, Patience: {args.patience}")
    print(f"HP search trials: {args.n_trials}")
    print(f"Seeds: {args.seeds}")
    print(f"Device: {args.device}")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    start_time = time_module.time()

    # =====================================================
    # STEP 1: Simulate market
    # =====================================================
    print("\n[1/5] Simulating market...")
    market = simulate_market(
        n_paths=args.paths, N=args.N, T=args.T,
        d_traded=args.d_traded, m_brownian=args.m_brownian,
        seed=42, device=args.device, substeps=0,
    )
    print(f"  S shape: {market.S.shape}")
    print(f"  dW shape: {market.dW.shape}")

    payoff_T, payoff_path = compute_payoffs(market.S, K=args.K, payoff_type="put")
    print(f"  Payoff T shape: {payoff_T.shape}, mean: {payoff_T.mean():.4f}")

    (S_train, dW_train, pT_train, pp_train), \
    (S_val, dW_val, pT_val, pp_val), \
    (S_test, dW_test, pT_test, pp_test) = split_data(
        market.S, market.dW, payoff_T, payoff_path, seed=42)
    print(f"  Train: {S_train.shape[0]}, Val: {S_val.shape[0]}, Test: {S_test.shape[0]}")

    # =====================================================
    # STEP 2: Build features
    # =====================================================
    print("\n[2/5] Building features...")
    feat_data = build_features(
        S_train, S_val, S_test, market.time, market.T,
        latent_dim=args.latent_dim, sig_level=args.sig_level,
        vae_epochs=args.vae_epochs, device=args.device, seed=42,
    )
    features_train = feat_data['train']
    features_val = feat_data['val']
    features_test = feat_data['test']
    feature_dim = feat_data['feature_dim']
    print(f"  Feature dim: {feature_dim}")
    print(f"  Features train shape: {features_train.shape}")

    # Save features
    data_dir = os.path.join(args.output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    for name, tensor in [
        ("features_train", features_train), ("features_val", features_val),
        ("features_test", features_test),
        ("payoff_T_train", pT_train), ("payoff_T_val", pT_val), ("payoff_T_test", pT_test),
        ("payoff_path_train", pp_train), ("payoff_path_val", pp_val), ("payoff_path_test", pp_test),
    ]:
        torch.save(tensor, os.path.join(data_dir, f"{name}.pt"))
    print("  Features and payoffs saved.")

    # =====================================================
    # STEP 3: Train models across seeds (with HP search)
    # =====================================================
    print("\n[3/5] Training models...")
    all_seed_results = {}
    all_trials_log = {}  # For saving HP search results

    for seed in args.seeds:
        print(f"\n{'='*40}")
        print(f"SEED {seed}")
        print(f"{'='*40}")

        seed_results = run_single_seed(
            args, seed,
            features_train, features_val, features_test,
            S_train, S_val, S_test,
            pT_train, pT_val, pT_test, pp_test,
            dW_train, dW_val, dW_test,
            market.time,
        )
        all_seed_results[seed] = seed_results

        # Collect trial logs
        all_trials_log[seed] = {}
        for model_name, res in seed_results.items():
            all_trials_log[seed][model_name] = res.get('all_trials', [])

        # Generate per-seed plots
        plot_dir = os.path.join(args.output_dir, "plots")
        for model_name, res in seed_results.items():
            generate_all_plots(
                model_name=f"{model_name}_seed{seed}",
                train_losses=res['train_result']['train_losses'],
                val_losses=res['train_result']['val_losses'],
                V_T=res['errors']['V_T'],
                H=pT_test,
                total_error=res['errors']['total_error'],
                worst_error=res['errors'].get('worst_error'),
                save_dir=plot_dir,
            )

        # Save checkpoints
        ckpt_dir = os.path.join(args.output_dir, "checkpoints")
        for model_name, res in seed_results.items():
            ckpt_path = os.path.join(ckpt_dir, f"{model_name}_seed{seed}.pt")
            torch.save(res['model'].state_dict(), ckpt_path)

    # =====================================================
    # STEP 4: Aggregate results + diagnostics
    # =====================================================
    print("\n[4/5] Aggregating results...")

    model_names = ['FNN-5', 'LSTM-5', 'DBSDE']
    summary_formatted = {}
    summary_raw = {}

    for model_name in model_names:
        seed_metrics = [all_seed_results[s][model_name]['metrics'] for s in args.seeds]
        formatted, raw = aggregate_seed_metrics(seed_metrics)
        summary_formatted[model_name] = formatted
        summary_raw[model_name] = raw
        print(f"\n{model_name} (across {len(args.seeds)} seeds):")
        for k, v in formatted.items():
            print(f"  {k:25s}: {v}")

    # Print best HPs found per model per seed
    print(f"\n{'='*60}")
    print("BEST HYPERPARAMETERS PER MODEL PER SEED")
    print(f"{'='*60}")
    for seed in args.seeds:
        for model_name in model_names:
            hp = all_seed_results[seed][model_name]['hparams']
            print(f"  seed={seed} {model_name:8s} | "
                  f"lr={hp['lr']:.1e}  l1={hp['l1']:.1e}  l2={hp['l2']:.1e}  "
                  f"dropout={hp['dropout']:.2f}  hidden={hp['hidden']}")

    # Summary table plot
    plot_dir = os.path.join(args.output_dir, "plots")
    plot_summary_table(summary_formatted, plot_dir)

    # Function shape plots
    last_seed = args.seeds[-1]
    S_range = torch.linspace(60.0, 140.0, 50)
    time_indices = [t for t in [0, 10, 20, 30, 40, min(49, args.N - 1)] if t < args.N]
    feat_template = features_test[:1, :, :]

    for mname in ['DBSDE', 'FNN-5']:
        m = all_seed_results[last_seed][mname]['model']
        m.eval()
        plot_function_shape(m, feat_template, market.time, S_range,
                            time_indices, mname, plot_dir, args.device)

    # =====================================================
    # STEP 5: Substeps convergence study
    # =====================================================
    if len(args.substeps) > 1:
        mse_conv, worst_conv = run_substeps_study(
            args, args.substeps, seed=args.seeds[0])
        plot_substeps_convergence(args.substeps, mse_conv, worst_conv, plot_dir)

    # =====================================================
    # Save final results
    # =====================================================
    print("\n[6/6] Saving results...")

    # CSV
    csv_path = os.path.join(args.output_dir, "metrics_summary.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        metric_keys = list(summary_formatted[model_names[0]].keys())
        writer.writerow(['Model'] + metric_keys)
        for model_name in model_names:
            row = [model_name] + [summary_formatted[model_name][k] for k in metric_keys]
            writer.writerow(row)
    print(f"  CSV saved: {csv_path}")

    # JSON (full results + HP search logs)
    json_path = os.path.join(args.output_dir, "metrics_summary.json")
    json_data = {
        'summary_formatted': summary_formatted,
        'summary_raw': summary_raw,
        'args': vars(args),
        'seeds': args.seeds,
        'best_hparams': {},
        'hp_search_trials': {},
    }
    for seed in args.seeds:
        json_data['best_hparams'][seed] = {}
        json_data['hp_search_trials'][seed] = {}
        for model_name in model_names:
            json_data['best_hparams'][seed][model_name] = \
                all_seed_results[seed][model_name].get('hparams', {})
            json_data['hp_search_trials'][seed][model_name] = \
                all_trials_log.get(seed, {}).get(model_name, [])

    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2, default=str)
    print(f"  JSON saved: {json_path}")

    # Final table
    elapsed = time_module.time() - start_time
    print(f"\n{'='*60}")
    print("FINAL TEST SET COMPARISON (mean +/- std across seeds)")
    print(f"{'='*60}")
    header = f"{'Model':12s} | {'MAE':22s} | {'MSE':22s} | {'R2':22s} | {'Worst Err':22s}"
    print(header)
    print("-" * len(header))
    for model_name in model_names:
        m = summary_formatted[model_name]
        print(f"{model_name:12s} | {m['MAE']:22s} | {m['MSE']:22s} | "
              f"{m['R2']:22s} | {m.get('worst_error_mean', 'N/A'):22s}")
    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"Results in: {args.output_dir}/")


if __name__ == "__main__":
    main()
