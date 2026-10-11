import pytest
import torch

from cs231n_practice.classifiers import BinaryImageAutoregressiveModel


def test_binary_autoregressive_model_shapes_and_gradients() -> None:
    model = BinaryImageAutoregressiveModel(embedding_dim=5, hidden_dim=7)
    token_ids = torch.tensor([[2, 0, 1, 0], [2, 1, 1, 0]])

    logits, hidden = model(token_ids)
    logits.square().mean().backward()

    assert logits.shape == (2, 4, 2)
    assert hidden.shape == (1, 2, 7)
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_binary_autoregressive_model_can_continue_from_hidden_state() -> None:
    model = BinaryImageAutoregressiveModel(embedding_dim=4, hidden_dim=6)
    first_tokens = torch.tensor([[2], [2]])
    next_tokens = torch.tensor([[0], [1]])

    _, hidden = model(first_tokens)
    logits, next_hidden = model(next_tokens, hidden)

    assert logits.shape == (2, 1, 2)
    assert next_hidden.shape == hidden.shape


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"embedding_dim": 0}, ValueError),
        ({"hidden_dim": 0}, ValueError),
        ({"hidden_dim": True}, TypeError),
    ],
)
def test_binary_autoregressive_model_rejects_invalid_configuration(
    kwargs: dict[str, object],
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        BinaryImageAutoregressiveModel(**kwargs)  # type: ignore[arg-type]


def test_binary_autoregressive_model_rejects_invalid_tokens() -> None:
    model = BinaryImageAutoregressiveModel()
    with pytest.raises(ValueError, match="shape"):
        model(torch.tensor([2, 0, 1]))
    with pytest.raises(TypeError, match="integer"):
        model(torch.tensor([[2.0, 0.0]]))
    with pytest.raises(ValueError, match="only"):
        model(torch.tensor([[2, 3]]))
