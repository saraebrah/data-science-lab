"""A small CPU neural baseline for the prepared transaction features."""

from math import log

import numpy as np
import torch
from torch import nn


class NeuralBaseline(nn.Module):
    """Produce one logit per transaction, starting near training prevalence."""

    def __init__(self, n_features: int, *, positive_rate: float) -> None:
        super().__init__()
        if not 0 < positive_rate < 1:
            raise ValueError("positive_rate must lie strictly between 0 and 1.")

        self.layers = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        initial_bias = log(positive_rate / (1 - positive_rate))
        nn.init.constant_(self.layers[-1].bias, initial_bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs).squeeze(-1)


@torch.inference_mode()
def predict_scores(
    model: nn.Module,
    inputs: torch.Tensor,
    *,
    batch_size: int = 8192,
) -> np.ndarray:
    """Return CPU probability scores in input order; leave model in eval mode."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    model.eval()
    scores = np.empty(len(inputs), dtype=np.float32)
    for start in range(0, len(inputs), batch_size):
        batch = inputs[start : start + batch_size]
        scores[start : start + len(batch)] = torch.sigmoid(model(batch)).numpy()
    return scores
