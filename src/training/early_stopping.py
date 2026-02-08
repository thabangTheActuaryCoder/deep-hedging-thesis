"""Early stopping with best-model checkpointing.

Monitors validation CVaR95(shortfall) and stops training
when no improvement is seen for `patience` consecutive epochs.
"""
import copy


class EarlyStopping:
    def __init__(self, patience=10, mode="min"):
        """
        Args:
            patience: epochs without improvement before stopping
            mode: 'min' to minimize metric, 'max' to maximize
        """
        self.patience = patience
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.best_state = None
        self.should_stop = False

    def step(self, score, model):
        """Check if score improved; checkpoint if so.

        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            self.best_state = copy.deepcopy(model.state_dict())
            return False

        improved = (
            (score < self.best_score) if self.mode == "min"
            else (score > self.best_score)
        )

        if improved:
            self.best_score = score
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    def load_best(self, model):
        """Restore best checkpoint into model."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
