"""
Early stopping with patience for validation-based model selection.
"""

import torch
import copy
from typing import Optional


class EarlyStopping:
    """Early stopping based on validation metric (lower is better)."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.best_state = None
        self.best_epoch = 0
        self.should_stop = False

    def step(self, val_metric: float, model: torch.nn.Module, epoch: int) -> bool:
        """Check if training should stop.

        Args:
            val_metric: Validation metric (lower is better)
            model: Current model
            epoch: Current epoch

        Returns:
            True if should stop
        """
        if self.best_score is None or val_metric < self.best_score - self.min_delta:
            self.best_score = val_metric
            self.best_state = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True
        return False

    def load_best(self, model: torch.nn.Module):
        """Load best model state."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
