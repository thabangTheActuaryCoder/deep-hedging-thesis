"""Interactive 3D Plotly surface plots for allocation visualization.

For FNN (output_dim=1): sigmoid(h(t_k, S1, S2)) surface.
For GRU/Regression (output_dim>1): raw hedge position h_1 surface.
At selected time steps k in {0, N//2, N-1}.
"""
import os
import numpy as np
import torch
from src.utils.io import ensure_dir


def _model_forward_2d(model, features):
    """Forward pass that handles both 2D-native and 3D-native models.

    GRU expects [batch, N, feat_dim]; FNN/Regression accept [batch, feat_dim].
    For GRU, wraps 2D input as single-step sequence and extracts step 0.
    """
    try:
        return model(features)
    except Exception:
        # GRU: wrap as [batch, 1, feat_dim] -> take [:, 0, :]
        out = model(features.unsqueeze(1))
        return out[:, 0, :]


def generate_3d_plots(model_name, model, feat_dim,
                      N, T, output_dir="outputs/plots_3d",
                      grid_points=30, z_range=(-2.0, 2.0)):
    """Generate interactive 3D surface plots.

    Evaluates the model on a grid of standardized features.
    FNN (output_dim=1): shows w1 = sigmoid(h).
    GRU/Regression (output_dim>1): shows raw h_1 (first asset position).

    Args:
        model_name: str
        model: hedger model (FNN, GRU, or Regression)
        feat_dim: total feature dimension
        N: number of time steps
        T: terminal time
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
    model.eval()
    # Use parameters first, fall back to buffers (RegressionHedger has no parameters)
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = next(model.buffers()).device

    s1 = np.linspace(z_range[0], z_range[1], grid_points)
    s2 = np.linspace(z_range[0], z_range[1], grid_points)
    S1, S2 = np.meshgrid(s1, s2)

    time_steps = [0, N // 2, N - 1]
    dt = T / N

    # Detect output_dim with a probe
    with torch.no_grad():
        probe = torch.zeros(1, feat_dim, device=device)
        probe_out = _model_forward_2d(model, probe)
        output_dim = probe_out.shape[-1]

    for k in time_steps:
        tau = (T - k * dt)
        n_grid = grid_points * grid_points
        features = torch.zeros(n_grid, feat_dim, device=device)
        features[:, 0] = torch.tensor(S1.flatten(), dtype=torch.float32, device=device)
        features[:, 1] = torch.tensor(S2.flatten(), dtype=torch.float32, device=device)
        if feat_dim > 2:
            features[:, 2] = tau

        with torch.no_grad():
            h = _model_forward_2d(model, features)  # [n_grid, output_dim]

        if output_dim == 1:
            # FNN: sigmoid allocation
            z_vals = torch.sigmoid(h).squeeze(-1).cpu().numpy()
            z_label = "w1 = sigmoid(h)"
            colorbar_title = "w1 = sigmoid(h)"
            title_str = f"{model_name}: Allocation w1 at k={k} (tau={tau:.3f})"
        else:
            # GRU/Regression: raw position for first asset
            z_vals = h[:, 0].cpu().numpy()
            z_label = "h_1 (hedge position)"
            colorbar_title = "h_1"
            title_str = f"{model_name}: Hedge Position h_1 at k={k} (tau={tau:.3f})"

        z_vals = z_vals.reshape(grid_points, grid_points)

        fig = go.Figure(data=[go.Surface(
            x=S1, y=S2, z=z_vals,
            colorscale="Viridis",
            colorbar=dict(title=colorbar_title),
        )])
        fig.update_layout(
            title=title_str,
            scene=dict(
                xaxis_title="log(S_tilde^1) (std)",
                yaxis_title="log(S_tilde^2) (std)",
                zaxis_title=z_label,
            ),
            width=800, height=700,
        )
        filename = f"{model_name}_allocation_surface_k{k}.html"
        fig.write_html(os.path.join(output_dir, filename))
