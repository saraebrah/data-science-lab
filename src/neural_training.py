"""Training operations shared by models with numeric and categorical inputs."""

import torch
from torch import nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    batches: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
) -> float:
    """Update the supplied model; loss_fn must return a scalar batch-mean loss."""
    model.train()
    total_loss = 0.0
    examples = 0

    for numeric, categories, labels in batches:
        optimizer.zero_grad(set_to_none=True)
        logits = model(numeric, categories)
        loss = loss_fn(logits, labels)
        if not torch.isfinite(loss).item():
            raise FloatingPointError("Training loss is not finite.")
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        examples += len(labels)

    if examples == 0:
        raise ValueError("Cannot train on an empty loader.")
    return total_loss / examples
