import pytest
import torch

from cs231n_practice.classifiers import CoordinateMLP


def test_coordinate_mlp_returns_one_scalar_per_query() -> None:
    model = CoordinateMLP(coordinate_dim=3, hidden_dim=16)

    predictions = model(torch.randn(7, 3))

    assert predictions.shape == (7,)


def test_coordinate_mlp_accepts_different_query_counts() -> None:
    model = CoordinateMLP(coordinate_dim=2, hidden_dim=8)

    assert model(torch.randn(1, 2)).shape == (1,)
    assert model(torch.randn(101, 2)).shape == (101,)


def test_coordinate_mlp_propagates_finite_gradients() -> None:
    model = CoordinateMLP(hidden_dim=12, number_hidden_layers=3)
    coordinates = torch.randn(5, 3, requires_grad=True)

    model(coordinates).square().mean().backward()

    assert coordinates.grad is not None
    assert torch.isfinite(coordinates.grad).all()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_coordinate_mlp_matches_notebook_parameter_count() -> None:
    model = CoordinateMLP()

    assert sum(parameter.numel() for parameter in model.parameters()) == 4_481


def test_coordinate_mlp_rejects_invalid_configuration_and_inputs() -> None:
    with pytest.raises(TypeError, match="integer"):
        CoordinateMLP(hidden_dim=True)
    with pytest.raises(ValueError, match="positive"):
        CoordinateMLP(number_hidden_layers=0)

    model = CoordinateMLP()
    with pytest.raises(ValueError, match="shape"):
        model(torch.zeros(2, 2))
    with pytest.raises(TypeError, match="floating-point"):
        model(torch.zeros(2, 3, dtype=torch.long))
