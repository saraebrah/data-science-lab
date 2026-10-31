import numpy as np
import pytest
import torch
from torch import nn

from src.embeddings import EmbeddingClassifier, predict_embedding_scores


@pytest.fixture(autouse=True)
def limit_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def make_model():
    return EmbeddingClassifier(
        n_numeric=2,
        vocabulary_sizes=(3, 3, 3),
        embedding_dims=(2, 2, 2),
        positive_rate=0.5,
    )


def test_model_learns_category_signal_and_keeps_unknown_vectors_zero():
    torch.manual_seed(42)
    model = make_model()
    numeric = torch.zeros(128, 2)
    categories = torch.tensor([[1, 1, 1], [2, 1, 1]]).repeat(64, 1)
    labels = torch.tensor([0.0, 1.0]).repeat(64)
    original_vector = model.embeddings[0].weight[1].detach().clone()
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    initial_loss = loss_fn(model(numeric, categories), labels).item()

    for _ in range(40):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(numeric, categories), labels)
        loss.backward()
        optimizer.step()

    assert loss_fn(model(numeric, categories), labels).item() < initial_loss / 2
    assert not torch.equal(model.embeddings[0].weight[1], original_vector)
    for table in model.embeddings:
        torch.testing.assert_close(table.weight[0], torch.zeros(2))
    unknown_logits = model(torch.zeros(1, 2), torch.zeros(1, 3, dtype=torch.long))
    assert unknown_logits.shape == (1,)
    assert torch.isfinite(unknown_logits).all()


def test_predictions_keep_order_and_checkpoint_restores_embeddings():
    torch.manual_seed(42)
    model = make_model()
    numeric = torch.randn(17, 2)
    categories = torch.randint(0, 3, (17, 3))
    with torch.no_grad():
        expected = torch.sigmoid(model(numeric, categories)).numpy()

    scores = predict_embedding_scores(model, numeric, categories, batch_size=4)
    np.testing.assert_allclose(scores, expected, rtol=1e-6)
    assert scores.shape == (17,)
    assert scores.dtype == np.float32
    assert not model.training

    restored = make_model()
    restored.load_state_dict(model.state_dict())
    np.testing.assert_allclose(
        predict_embedding_scores(restored, numeric, categories, batch_size=17),
        scores,
        rtol=1e-6,
    )


def test_mismatched_configuration_and_prediction_inputs_are_rejected():
    with pytest.raises(ValueError, match="same length"):
        EmbeddingClassifier(
            n_numeric=2,
            vocabulary_sizes=(3, 3),
            embedding_dims=(2,),
            positive_rate=0.5,
        )
    model = make_model()
    with pytest.raises(ValueError, match="same number of rows"):
        predict_embedding_scores(model, torch.zeros(2, 2), torch.ones(1, 3).long())
    with pytest.raises(ValueError, match="categorical columns"):
        model(torch.zeros(2, 2), torch.ones(2, 4).long())
    with pytest.raises(ValueError, match="batch_size"):
        predict_embedding_scores(
            model, torch.zeros(2, 2), torch.ones(2, 3).long(), batch_size=0
        )
