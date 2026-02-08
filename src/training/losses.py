"""Loss functions for hedging and BSDE training.

Primary objective: MSE + alpha * mean(shortfall) + beta * CVaR_q(shortfall)
Shortfall s = max(H_tilde - V_T, 0) measures under-hedging.
Negative terminal errors (V_T > H_tilde) are expected in incomplete markets.
"""
import torch


def terminal_error(V_T, H_tilde):
    """e_T = V_T - H_tilde (positive = over-hedge, negative = under-hedge)."""
    return V_T - H_tilde


def shortfall(V_T, H_tilde):
    """s = max(H_tilde - V_T, 0) = max(-e_T, 0)."""
    return torch.clamp(H_tilde - V_T, min=0.0)


def cvar(losses, q=0.95):
    """CVaR_q: mean of the top (1-q) fraction of losses.

    For q=0.95, this is the average of the worst 5% of shortfalls.
    """
    n = losses.shape[0]
    k = max(1, int((1 - q) * n))
    sorted_losses, _ = torch.sort(losses, descending=True)
    return sorted_losses[:k].mean()


def hedging_loss(V_T, H_tilde, alpha=1.0, beta=1.0, cvar_q=0.95):
    """Combined hedging loss: MSE + alpha*mean(shortfall) + beta*CVaR_q(shortfall).

    Args:
        V_T: [batch] terminal portfolio value
        H_tilde: [batch] discounted payoff target
        alpha: weight on mean shortfall
        beta: weight on CVaR shortfall
        cvar_q: CVaR quantile level

    Returns:
        loss: scalar
    """
    e = terminal_error(V_T, H_tilde)
    s = shortfall(V_T, H_tilde)
    mse = (e ** 2).mean()
    mean_s = s.mean()
    cvar_s = cvar(s, q=cvar_q)
    return mse + alpha * mean_s + beta * cvar_s


def bsde_loss(Y_T, H_tilde):
    """BSDE terminal loss: MSE(Y_N - H_tilde)."""
    return ((Y_T - H_tilde) ** 2).mean()


def elastic_net_penalty(model, l1=0.0, l2=1e-4):
    """Elastic net regularization on weight tensors only.

    Excludes biases, LayerNorm parameters, and BSDE Y0.
    """
    l1_reg = 0.0
    l2_reg = 0.0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Skip non-weight parameters
        if "bias" in name:
            continue
        if "LayerNorm" in name or "layer_norm" in name:
            continue
        if name == "Y0":
            continue
        if "weight" in name:
            l1_reg = l1_reg + param.abs().sum()
            l2_reg = l2_reg + (param ** 2).sum()
    return l1 * l1_reg + l2 * l2_reg
