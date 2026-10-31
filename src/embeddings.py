"""A CPU classifier that learns a separate embedding for each category column."""

import numpy as np
import torch
from torch import nn

from src.neural import NeuralBaseline


class EmbeddingClassifier(nn.Module):
    def __init__(
        self,
        *,
        n_numeric: int,
        vocabulary_sizes: tuple[int, ...],
        embedding_dims: tuple[int, ...],
        positive_rate: float,
    ) -> None:
        super().__init__()
        if len(vocabulary_sizes) != len(embedding_dims):
            raise ValueError("Vocabulary sizes and embedding dimensions need the same length.")

        self.embeddings = nn.ModuleList([
            nn.Embedding(size, dimension, padding_idx=0)
            for size, dimension in zip(vocabulary_sizes, embedding_dims, strict=True)
        ])
        self.classifier = NeuralBaseline(
            n_numeric + sum(embedding_dims), positive_rate=positive_rate
        )

    def forward(
        self, numeric: torch.Tensor, categories: torch.Tensor
    ) -> torch.Tensor:
        if categories.ndim != 2 or categories.shape[1] != len(self.embeddings):
            raise ValueError("Number of categorical columns must match the embedding tables.")
        vectors = [
            table(categories[:, column])
            for column, table in enumerate(self.embeddings)
        ]
        combined = torch.cat([numeric, *vectors], dim=1)
        return self.classifier(combined)


@torch.inference_mode()
def predict_embedding_scores(
    model: nn.Module,
    numeric: torch.Tensor,
    categories: torch.Tensor,
    *,
    batch_size: int = 8192,
) -> np.ndarray:
    """Return CPU scores in input order; leave the model in evaluation mode."""
    if len(numeric) != len(categories):
        raise ValueError("Numeric and categorical inputs need the same number of rows.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    model.eval()
    scores = np.empty(len(numeric), dtype=np.float32)
    for start in range(0, len(numeric), batch_size):
        stop = start + batch_size
        logits = model(numeric[start:stop], categories[start:stop])
        scores[start:stop] = torch.sigmoid(logits).numpy()
    return scores
