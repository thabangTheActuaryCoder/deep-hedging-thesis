"""Interactive 3D Plotly surface plots for hedge ratio visualization.

For each model, creates 3D surface of Delta^1(t_k, S1, S2)
at selected time steps k in {0, N//2, N-1}.
"""
import os
import numpy as np
import torch
from src.utils.io import ensure_dir


def generate_3d_plots(model_name, nn1, controller, feat_dim, d_traded,
                      N, T, use_controller=False,
                      output_dir="outputs/plots_3d",
                      grid_points=30, z_range=(-2.0, 2.0)):
    """Generate interactive 3D surface plots of Delta^1 vs (S1, S2).

    We evaluate the NN1 baseline delta (without controller, since controller
    depends on path-specific P/L history) on a grid of standardized features.

    Args:
        model_name: str
        nn1: baseline hedger model
        controller: NN2 controller (unused for grid plot)
        feat_dim: total feature dimension
        d_traded: number of traded assets
        N: number of time steps
        T: terminal time
        use_controller: ignored (always False for grid eval)
        output_dir: output directory
        grid_points: number of grid points per axis
        z_range: range of standardized feature values
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Plotly not installed, skipping 3D plots.")
        return

    ensure_dir(output_dir)
    nn1.eval()
    device = next(nn1.parameters()).device

    # Grid over standardized log-price features
    s1 = np.linspace(z_range[0], z_range[1], grid_points)
    s2 = np.linspace(z_range[0], z_range[1], grid_points)
    S1, S2 = np.meshgrid(s1, s2)

    time_steps = [0, N // 2, N - 1]
    dt = T / N

    for k in time_steps:
        tau = (T - k * dt)
        # Build synthetic standardized feature vectors
        # Features: [logS1, logS2, tau, vae(16 zeros), sig(5 zeros)]
        n_grid = grid_points * grid_points
        features = torch.zeros(n_grid, feat_dim, device=device)
        features[:, 0] = torch.tensor(S1.flatten(), dtype=torch.float32, device=device)
        features[:, 1] = torch.tensor(S2.flatten(), dtype=torch.float32, device=device)
        features[:, 2] = tau  # already standardized approx

        with torch.no_grad():
            delta = nn1(features)  # [n_grid, d_traded]
        delta1 = delta[:, 0].cpu().numpy().reshape(grid_points, grid_points)

        fig = go.Figure(data=[go.Surface(
            x=S1, y=S2, z=delta1,
            colorscale="Viridis",
            colorbar=dict(title="Delta^1"),
        )])
        fig.update_layout(
            title=f"{model_name}: Delta^1 at k={k} (tau={tau:.3f})",
            scene=dict(
                xaxis_title="log(S_tilde^1) (std)",
                yaxis_title="log(S_tilde^2) (std)",
                zaxis_title="Delta^1",
            ),
            width=800, height=700,
        )
        filename = f"{model_name}_delta_surface_k{k}.html"
        fig.write_html(os.path.join(output_dir, filename))


def generate_bsde_3d_plots(model_name, bsde_model, feat_dim, d_traded,
                           N, T, output_dir="outputs/plots_3d",
                           grid_points=30, z_range=(-2.0, 2.0)):
    """Generate 3D surface plots for DBSDE delta.

    Uses the z_net to compute Z, then projects to Delta via pseudo-inverse.
    Since S_tilde is needed for projection, we use exp(logS_std * train_std + train_mean)
    as approximate price; for standardized grid we use S_tilde = 1.0 as reference.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Plotly not installed, skipping 3D plots.")
        return

    ensure_dir(output_dir)
    bsde_model.eval()
    device = next(bsde_model.parameters()).device

    s1 = np.linspace(z_range[0], z_range[1], grid_points)
    s2 = np.linspace(z_range[0], z_range[1], grid_points)
    S1, S2 = np.meshgrid(s1, s2)

    time_steps = [0, N // 2, N - 1]
    dt = T / N

    for k in time_steps:
        tau = T - k * dt
        t_k = k * dt
        n_grid = grid_points * grid_points

        features = torch.zeros(n_grid, feat_dim, device=device)
        features[:, 0] = torch.tensor(S1.flatten(), dtype=torch.float32, device=device)
        features[:, 1] = torch.tensor(S2.flatten(), dtype=torch.float32, device=device)
        features[:, 2] = tau

        t_batch = torch.full((n_grid,), t_k, device=device)
        # Use S_tilde = 1.0 as reference for projection
        S_ref = torch.ones(n_grid, d_traded, device=device)

        with torch.no_grad():
            Z = bsde_model.compute_z(features, t_batch)
            delta = bsde_model.z_to_delta(Z, S_ref)
        delta1 = delta[:, 0].cpu().numpy().reshape(grid_points, grid_points)

        fig = go.Figure(data=[go.Surface(
            x=S1, y=S2, z=delta1,
            colorscale="Plasma",
            colorbar=dict(title="Delta^1"),
        )])
        fig.update_layout(
            title=f"{model_name}: Delta^1 at k={k} (tau={tau:.3f})",
            scene=dict(
                xaxis_title="log(S_tilde^1) (std)",
                yaxis_title="log(S_tilde^2) (std)",
                zaxis_title="Delta^1",
            ),
            width=800, height=700,
        )
        filename = f"{model_name}_delta_surface_k{k}.html"
        fig.write_html(os.path.join(output_dir, filename))
