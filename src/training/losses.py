"""Loss functions for super-hedging.

Primary objective: asymmetric shortfall penalisation.
L = lambda_short * mean(shortfall) + lambda_over * mean(over_hedge)
    + lambda_short * CVaR_q(shortfall)
"""
import torch


def terminal_error(V_T, H_tilde):
    """e_T = V_T - H_tilde (positive = over-hedge, negative = under-hedge)."""
    return V_T - H_tilde


def shortfall(V_T, H_tilde):
    """s = max(H_tilde - V_T, 0) = max(-e_T, 0)."""
    return torch.clamp(H_tilde - V_T, min=0.0)


def over_hedge(V_T, H_tilde):
    """o = max(V_T - H_tilde, 0)."""
    return torch.clamp(V_T - H_tilde, min=0.0)


def cvar(losses, q=0.95):
    """CVaR_q: mean of the top (1-q) fraction of losses.

    For q=0.95, this is the average of the worst 5% of shortfalls.
    """
    n = losses.shape[0]
    k = max(1, int((1 - q) * n))
    sorted_losses, _ = torch.sort(losses, descending=True)
    return sorted_losses[:k].mean()


def super_hedging_loss(V_T, H_tilde, lambda_short=10.0, lambda_over=1.0,
                       cvar_q=0.95):
    """Super-hedging loss: asymmetric penalisation of under- vs over-hedging.

    L = lambda_short * mean(shortfall) + lambda_over * mean(over_hedge)
        + lambda_short * CVaR_q(shortfall)

    Args:
        V_T: [batch] terminal portfolio value
        H_tilde: [batch] discounted payoff target
        lambda_short: weight on shortfall terms (default 10.0)
        lambda_over: weight on over-hedge term (default 1.0)
        cvar_q: CVaR quantile level

    Returns:
        loss: scalar
    """
    s = shortfall(V_T, H_tilde)
    o = over_hedge(V_T, H_tilde)
    mean_s = s.mean()
    mean_o = o.mean()
    cvar_s = cvar(s, q=cvar_q)
    return lambda_short * mean_s + lambda_over * mean_o + lambda_short * cvar_s
