"""An embedding classifier with an optional categorical transformer block."""

import torch
from torch import nn

from src.embeddings import EmbeddingClassifier


class TabularTransformer(EmbeddingClassifier):
    def __init__(
        self,
        *,
        n_numeric: int,
        vocabulary_sizes: tuple[int, ...],
        positive_rate: float,
        token_dim: int = 8,
        num_heads: int = 2,
        feedforward_dim: int = 32,
        use_transformer: bool = True,
    ) -> None:
        # Initialize shared components first, so resetting the same seed gives
        # the control and transformer identical embedding/classifier weights.
        super().__init__(
            n_numeric=n_numeric,
            vocabulary_sizes=vocabulary_sizes,
            embedding_dims=(token_dim,) * len(vocabulary_sizes),
            positive_rate=positive_rate,
        )
        self.context = (
            nn.TransformerEncoderLayer(
                d_model=token_dim,
                nhead=num_heads,
                dim_feedforward=feedforward_dim,
                dropout=0.0,
                activation="relu",
                batch_first=True,
                norm_first=True,
            )
            if use_transformer else nn.Identity()
        )

    def forward(
        self, numeric: torch.Tensor, categories: torch.Tensor
    ) -> torch.Tensor:
        if categories.ndim != 2 or categories.shape[1] != len(self.embeddings):
            raise ValueError("Number of categorical columns must match the embedding tables.")
        tokens = torch.stack([
            table(categories[:, column])
            for column, table in enumerate(self.embeddings)
        ], dim=1)
        contextual = self.context(tokens)
        combined = torch.cat([numeric, contextual.flatten(start_dim=1)], dim=1)
        return self.classifier(combined)
