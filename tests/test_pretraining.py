import copy

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.neural_training import train_one_epoch
from src.pretraining import MaskedCategoryModel, mask_one_category, pretrain_one_epoch
from src.tabular_transformer import TabularTransformer


@pytest.fixture(autouse=True)
def limit_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def make_pretrainer():
    return MaskedCategoryModel(n_numeric=2, known_counts=(3, 3, 2))


def test_mask_preserves_original_and_targets_only_hidden_known_values():
    ids = torch.tensor([[1, 2, 1], [3, 1, 2], [2, 3, 1], [1, 1, 2]])
    original = ids.clone()
    masks = torch.tensor([4, 4, 3])
    result = mask_one_category(ids, masks, generator=torch.Generator().manual_seed(42))
    masked, columns, targets = result
    torch.testing.assert_close(ids, original)
    assert ((masked != original).sum(dim=1) == 1).all()
    rows = torch.arange(len(ids))
    torch.testing.assert_close(masked[rows, columns], masks[columns])
    torch.testing.assert_close(targets + 1, ids[rows, columns])
    repeated = mask_one_category(ids, masks, generator=torch.Generator().manual_seed(42))
    for first, second in zip(result, repeated, strict=True):
        torch.testing.assert_close(first, second)


@pytest.mark.parametrize("bad_value", [0, 4, -1])
def test_unknown_masked_or_invalid_original_values_are_rejected(bad_value):
    with pytest.raises(ValueError, match="known category IDs"):
        mask_one_category(
            torch.tensor([[bad_value, 1, 1]]), torch.tensor([4, 4, 3]),
            generator=torch.Generator().manual_seed(42),
        )


def test_loss_and_accuracy_count_only_hidden_positions_with_uneven_batches():
    torch.manual_seed(42)
    model = make_pretrainer()
    ids = torch.tensor([[1, 2, 1], [3, 1, 2], [2, 3, 1], [1, 1, 2],
                        [2, 2, 2], [1, 3, 1], [3, 3, 2]])
    loader = DataLoader(TensorDataset(ids), batch_size=4)
    generator = torch.Generator().manual_seed(42)
    expected_loss = 0.0
    expected_correct = 0
    counts = [0, 0, 0]
    correct = [0, 0, 0]
    model.train()
    with torch.no_grad():
        for (batch,) in loader:
            masked, columns, targets = mask_one_category(batch, model.mask_ids, generator=generator)
            logits = model(masked)
            for row, column in enumerate(columns.tolist()):
                expected_loss += F.cross_entropy(
                    logits[column][row:row + 1], targets[row:row + 1]
                ).item()
                hit = int(logits[column][row].argmax().item() == targets[row].item())
                expected_correct += hit
                counts[column] += 1
                correct[column] += hit

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0)
    metrics = pretrain_one_epoch(
        model, loader, optimizer, generator=torch.Generator().manual_seed(42)
    )
    assert metrics["loss"] == pytest.approx(expected_loss / len(ids), rel=1e-6)
    assert metrics["accuracy"] == expected_correct / len(ids)
    for column in range(3):
        if counts[column]:
            assert metrics[f"accuracy_{column}"] == correct[column] / counts[column]
    assert all(parameter.grad is None for parameter in model.backbone.classifier.parameters())


def test_pretraining_transfer_and_finetuning_update_only_the_intended_model():
    torch.manual_seed(42)
    model = make_pretrainer()
    ids = torch.stack([
        torch.randint(1, size + 1, (64,)) for size in (3, 3, 2)
    ], dim=1)
    original_classifier = copy.deepcopy(model.backbone.classifier.state_dict())
    original_attention = model.backbone.context.self_attn.in_proj_weight.detach().clone()
    original_mask = model.backbone.embeddings[0].weight[4].detach().clone()
    loader = DataLoader(TensorDataset(ids), batch_size=16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    metrics = pretrain_one_epoch(
        model, loader, optimizer, generator=torch.Generator().manual_seed(42)
    )
    assert np.isfinite(metrics["loss"])
    assert not torch.equal(model.backbone.context.self_attn.in_proj_weight, original_attention)
    assert not torch.equal(model.backbone.embeddings[0].weight[4], original_mask)
    for name, value in original_classifier.items():
        torch.testing.assert_close(model.backbone.classifier.state_dict()[name], value)
    for table in model.backbone.embeddings:
        torch.testing.assert_close(table.weight[0], torch.zeros(8))

    torch.manual_seed(42)
    target = TabularTransformer(
        n_numeric=2, vocabulary_sizes=(5, 5, 4), positive_rate=0.5
    )
    fresh_classifier = copy.deepcopy(target.classifier.state_dict())
    target.embeddings.load_state_dict(model.backbone.embeddings.state_dict())
    target.context.load_state_dict(model.backbone.context.state_dict())
    for name, value in fresh_classifier.items():
        torch.testing.assert_close(target.classifier.state_dict()[name], value)
    assert all(p.requires_grad for p in target.parameters())
    source_snapshot = copy.deepcopy(model.backbone.state_dict())
    before = target.context.self_attn.in_proj_weight.detach().clone()
    supervised = DataLoader(TensorDataset(
        torch.randn(64, 2), ids, (ids[:, 0] == 1).float()
    ), batch_size=16)
    train_one_epoch(
        target, supervised, torch.optim.AdamW(target.parameters(), lr=0.01),
        nn.BCEWithLogitsLoss(),
    )
    assert not torch.equal(target.context.self_attn.in_proj_weight, before)
    for name, value in source_snapshot.items():
        torch.testing.assert_close(model.backbone.state_dict()[name], value)


def test_empty_pretraining_loader_is_rejected():
    model = make_pretrainer()
    loader = DataLoader(TensorDataset(torch.empty(0, 3, dtype=torch.long)))
    with pytest.raises(ValueError, match="empty"):
        pretrain_one_epoch(
            model, loader, torch.optim.AdamW(model.parameters()),
            generator=torch.Generator().manual_seed(42),
        )
