"""CPU masked-category pre-training; laundering labels are never inputs here."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.tabular_transformer import TabularTransformer


def mask_one_category(
    category_ids: torch.Tensor,
    mask_ids: torch.Tensor,
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hide one known value per row and return its column and zero-based target."""
    if category_ids.ndim != 2 or category_ids.shape[1] != len(mask_ids):
        raise ValueError("Supply one categorical column per mask ID.")
    if category_ids.dtype != torch.long or mask_ids.dtype != torch.long:
        raise TypeError("Category and mask IDs must use torch.long.")
    if ((category_ids < 1) | (category_ids >= mask_ids)).any():
        raise ValueError("Pre-training targets require original known category IDs.")

    rows = torch.arange(len(category_ids))
    columns = torch.randint(
        category_ids.shape[1], (len(category_ids),), generator=generator
    )
    targets = category_ids[rows, columns] - 1
    masked = category_ids.clone()
    masked[rows, columns] = mask_ids[columns]
    return masked, columns, targets


class MaskedCategoryModel(nn.Module):
    def __init__(
        self,
        *,
        n_numeric: int,
        known_counts: tuple[int, ...],
        token_dim: int = 8,
        num_heads: int = 2,
        feedforward_dim: int = 32,
    ) -> None:
        super().__init__()
        if not known_counts or any(count < 1 for count in known_counts):
            raise ValueError("Each categorical column needs at least one known value.")
        self.backbone = TabularTransformer(
            n_numeric=n_numeric,
            vocabulary_sizes=tuple(count + 2 for count in known_counts),
            positive_rate=0.5,
            token_dim=token_dim,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
        )
        # The backbone's laundering classifier is unused and gets no gradients.
        # Its shape preserves shared initialization with the supervised models.
        self.heads = nn.ModuleList([
            nn.Linear(token_dim, count) for count in known_counts
        ])
        self.register_buffer("mask_ids", torch.tensor(known_counts, dtype=torch.long) + 1)

    def forward(self, categories: torch.Tensor) -> tuple[torch.Tensor, ...]:
        tokens = self.backbone.encode_categories(categories)
        return tuple(head(tokens[:, column]) for column, head in enumerate(self.heads))


def pretrain_one_epoch(
    model: MaskedCategoryModel,
    batches: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    generator: torch.Generator,
) -> dict[str, float]:
    """Update from hidden categories only; return online training loss/accuracies."""
    model.train()
    loss_total = 0.0
    examples = 0
    counts = [0] * len(model.heads)
    correct = [0] * len(model.heads)

    for (categories,) in batches:
        masked, columns, targets = mask_one_category(
            categories, model.mask_ids, generator=generator
        )
        optimizer.zero_grad(set_to_none=True)
        logits = model(masked)
        losses = []
        for column, column_logits in enumerate(logits):
            selected = columns == column
            count = int(selected.sum().item())
            if count == 0:
                continue
            selected_logits = column_logits[selected]
            losses.append(F.cross_entropy(
                selected_logits, targets[selected], reduction="sum"
            ))
            predictions = selected_logits.detach().argmax(dim=1)
            correct[column] += int((predictions == targets[selected]).sum().item())
            counts[column] += count

        loss = sum(losses) / len(categories)
        if not torch.isfinite(loss).item():
            raise FloatingPointError("Pre-training loss is not finite.")
        loss.backward()
        optimizer.step()
        loss_total += loss.item() * len(categories)
        examples += len(categories)

    if examples == 0:
        raise ValueError("Cannot pre-train on an empty loader.")
    return {
        "loss": loss_total / examples,
        "accuracy": sum(correct) / examples,
        **{
            f"accuracy_{column}": hits / count if count else float("nan")
            for column, (hits, count) in enumerate(zip(correct, counts, strict=True))
        },
    }
