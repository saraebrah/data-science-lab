import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.embeddings import EmbeddingClassifier, predict_embedding_scores
from src.neural_training import train_one_epoch
from src.tabular_transformer import TabularTransformer


@pytest.fixture(autouse=True)
def limit_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def make_model(use_transformer=True):
    return TabularTransformer(
        n_numeric=2,
        vocabulary_sizes=(4, 4, 3),
        positive_rate=0.2,
        use_transformer=use_transformer,
    )


def test_control_matches_existing_classifier_and_shared_initialization():
    torch.manual_seed(42)
    control = make_model(False)
    torch.manual_seed(42)
    transformer = make_model(True)
    for name, value in control.state_dict().items():
        torch.testing.assert_close(value, transformer.state_dict()[name], rtol=0, atol=0)

    torch.manual_seed(42)
    previous = EmbeddingClassifier(
        n_numeric=2,
        vocabulary_sizes=(4, 4, 3),
        embedding_dims=(8, 8, 8),
        positive_rate=0.2,
    )
    numeric = torch.randn(7, 2)
    categories = torch.randint(0, 3, (7, 3))
    torch.testing.assert_close(control(numeric, categories), previous(numeric, categories))


def test_predictions_are_independent_of_other_transactions_and_restore():
    torch.manual_seed(42)
    model = make_model()
    numeric = torch.randn(9, 2)
    categories = torch.randint(0, 3, (9, 3))
    scores = predict_embedding_scores(model, numeric, categories, batch_size=4)
    solo = predict_embedding_scores(model, numeric[:1], categories[:1], batch_size=1)
    np.testing.assert_allclose(solo, scores[:1], rtol=1e-5, atol=1e-7)

    changed_numeric = numeric.clone()
    changed_categories = categories.clone()
    changed_numeric[1:] = 100
    changed_categories[1:] = 2
    changed_scores = predict_embedding_scores(model, changed_numeric, changed_categories)
    np.testing.assert_allclose(changed_scores[:1], scores[:1], rtol=1e-5, atol=1e-7)

    restored = make_model()
    restored.load_state_dict(model.state_dict())
    np.testing.assert_allclose(
        predict_embedding_scores(restored, numeric, categories), scores,
        rtol=1e-5, atol=1e-7,
    )


def test_training_updates_attention_and_reports_weighted_mean_loss():
    torch.manual_seed(42)
    model = make_model()
    numeric = torch.randn(7, 2)
    categories = torch.tensor([[1, 1, 1], [2, 2, 2], [1, 2, 1],
                               [2, 1, 2], [1, 1, 2], [2, 2, 1], [1, 2, 2]])
    labels = torch.tensor([0., 1., 0., 1., 0., 1., 1.])
    loader = DataLoader(TensorDataset(numeric, categories, labels), batch_size=4)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    with torch.no_grad():
        expected = loss_fn(model(numeric, categories), labels).item()

    # With no parameter changes, uneven batches must match the full-data mean.
    stationary = torch.optim.AdamW(model.parameters(), lr=0.0)
    actual = train_one_epoch(model, loader, stationary, loss_fn)
    assert actual == pytest.approx(expected, rel=1e-6)

    before = model.context.self_attn.in_proj_weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    actual = train_one_epoch(model, loader, optimizer, loss_fn)
    assert np.isfinite(actual)
    assert not torch.equal(before, model.context.self_attn.in_proj_weight)
    for table in model.embeddings:
        torch.testing.assert_close(table.weight[0], torch.zeros(8))


def test_empty_training_loader_is_rejected():
    model = make_model()
    loader = DataLoader(TensorDataset(
        torch.empty(0, 2), torch.empty(0, 3, dtype=torch.long), torch.empty(0)
    ))
    optimizer = torch.optim.AdamW(model.parameters())
    with pytest.raises(ValueError, match="empty"):
        train_one_epoch(model, loader, optimizer, nn.BCEWithLogitsLoss())
