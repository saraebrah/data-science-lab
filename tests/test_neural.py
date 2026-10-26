import numpy as np
import pytest
import torch
from torch import nn

from src.neural import NeuralBaseline, predict_scores


@pytest.fixture(autouse=True)
def limit_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_single_row_keeps_batch_dimension_and_can_learn():
    torch.manual_seed(42)
    model = NeuralBaseline(3, positive_rate=0.5)
    assert model(torch.zeros(1, 3)).shape == (1,)

    X = torch.randn(128, 3)
    y = (X[:, 0] > 0).float()
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    initial_loss = loss_fn(model(X), y).item()

    for _ in range(30):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(X), y)
        loss.backward()
        optimizer.step()

    assert loss_fn(model(X), y).item() < initial_loss / 2


def test_predictions_preserve_row_order_and_do_not_depend_on_batch_size():
    torch.manual_seed(42)
    model = NeuralBaseline(3, positive_rate=0.01)
    X = torch.randn(17, 3)
    with torch.no_grad():
        expected = torch.sigmoid(model(X)).numpy()

    scores = predict_scores(model, X, batch_size=4)

    assert scores.shape == (17,)
    assert scores.dtype == np.float32
    assert np.isfinite(scores).all()
    np.testing.assert_allclose(scores, expected, rtol=1e-6)
    np.testing.assert_allclose(
        predict_scores(model, X, batch_size=17), scores, rtol=1e-6
    )
    assert not model.training


@pytest.mark.parametrize("positive_rate", [0.0, 1.0, -0.1, float("nan")])
def test_invalid_training_prevalence_is_rejected(positive_rate):
    with pytest.raises(ValueError, match="positive_rate"):
        NeuralBaseline(3, positive_rate=positive_rate)
