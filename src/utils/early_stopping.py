"""
Early stopping with best-model checkpointing.

Monitors a validation metric (default: val_loss, lower is better) and saves
the best checkpoint. Training is flagged to stop after `patience` epochs
without improvement.
"""

import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


class EarlyStopping:
    """
    Args:
        patience   : Epochs to wait without improvement before stopping
        min_delta  : Minimum change to qualify as an improvement
        mode       : 'min' (lower is better, e.g. loss) or
                     'max' (higher is better, e.g. accuracy)
        checkpoint_path : Where to save the best model weights (.pth)
        verbose    : Print messages on improvement / stop
    """

    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 1e-5,
        mode: str = "min",
        checkpoint_path: str = "checkpoints/best.pth",
        verbose: bool = True,
    ):
        assert mode in ("min", "max"), f"mode must be 'min' or 'max', got '{mode}'"
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.checkpoint_path = Path(checkpoint_path)
        self.verbose = verbose

        self.best_score: Optional[float] = None
        self.counter: int = 0
        self.stop: bool = False
        self.best_epoch: int = 0

        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    def _is_improvement(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "min":
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta

    def __call__(self, score: float, model: nn.Module, epoch: int) -> bool:
        """
        Call at the end of each epoch.

        Args:
            score : Validation metric value
            model : Model to checkpoint if improved
            epoch : Current epoch number (0-indexed)

        Returns:
            True if early stopping should be triggered, False otherwise
        """
        if self._is_improvement(score):
            if self.verbose:
                improvement = (
                    f"  EarlyStopping: {self.mode} metric improved"
                    f" from {self.best_score:.6f} → {score:.6f}"
                    if self.best_score is not None
                    else f"  EarlyStopping: first checkpoint at epoch {epoch+1}, metric={score:.6f}"
                )
                print(improvement)
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            torch.save(model.state_dict(), self.checkpoint_path)
        else:
            self.counter += 1
            if self.verbose:
                print(
                    f"  EarlyStopping: no improvement for {self.counter}/{self.patience} epochs"
                    f" (best={self.best_score:.6f} @ epoch {self.best_epoch+1})"
                )
            if self.counter >= self.patience:
                self.stop = True
                if self.verbose:
                    print(
                        f"  EarlyStopping: stopping. Best epoch was {self.best_epoch+1}"
                        f" with metric={self.best_score:.6f}"
                    )

        return self.stop

    def load_best(self, model: nn.Module) -> nn.Module:
        """Load the best saved weights back into model."""
        if self.checkpoint_path.exists():
            model.load_state_dict(torch.load(self.checkpoint_path, map_location="cpu"))
            if self.verbose:
                print(f"  Loaded best checkpoint from {self.checkpoint_path}")
        else:
            raise FileNotFoundError(f"No checkpoint found at {self.checkpoint_path}")
        return model
