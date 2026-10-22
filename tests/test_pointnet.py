import pytest
import torch
from torch import nn

from cs231n_practice.classifiers import (
    TinyPointNet,
    point_cloud_classification_accuracy,
)


def test_tiny_pointnet_returns_scores_and_pool_indices() -> None:
    model = TinyPointNet(num_classes=5, point_feature_dim=12, hidden_dim=8)
    clouds = torch.randn(4, 17, 3)

    scores, indices = model(clouds, return_pool_indices=True)

    assert scores.shape == (4, 5)
    assert indices.shape == (4, 12)
    assert torch.all((indices >= 0) & (indices < 17))


def test_tiny_pointnet_is_permutation_invariant() -> None:
    torch.manual_seed(3)
    model = TinyPointNet()
    clouds = torch.randn(3, 20, 3)
    permutation = torch.randperm(20)

    torch.testing.assert_close(model(clouds), model(clouds[:, permutation]))


def test_tiny_pointnet_accepts_different_point_counts_and_backpropagates() -> None:
    model = TinyPointNet()
    short_clouds = torch.randn(2, 8, 3, requires_grad=True)
    long_clouds = torch.randn(2, 31, 3)

    model(short_clouds).square().mean().backward()

    assert model(long_clouds).shape == (2, 3)
    assert short_clouds.grad is not None
    assert torch.isfinite(short_clouds.grad).all()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_point_cloud_classification_accuracy_uses_model_predictions() -> None:
    class CoordinateClassifier(nn.Module):
        def forward(self, points: torch.Tensor) -> torch.Tensor:
            coordinate = points[:, 0, 0]
            return torch.stack((-coordinate, coordinate), dim=1)

    clouds = torch.zeros(4, 3, 3)
    clouds[:, 0, 0] = torch.tensor([-2.0, 1.0, -1.0, 3.0])
    labels = torch.tensor([0, 1, 1, 1])

    accuracy = point_cloud_classification_accuracy(
        CoordinateClassifier(), clouds, labels
    )

    assert accuracy == pytest.approx(0.75)


def test_tiny_pointnet_rejects_invalid_configuration_and_inputs() -> None:
    with pytest.raises(TypeError, match="integer"):
        TinyPointNet(num_classes=True)
    with pytest.raises(ValueError, match="positive"):
        TinyPointNet(point_feature_dim=0)

    model = TinyPointNet()
    with pytest.raises(ValueError, match="shape"):
        model(torch.zeros(2, 3))
    with pytest.raises(ValueError, match="nonempty"):
        model(torch.zeros(2, 0, 3))
    with pytest.raises(TypeError, match="floating-point"):
        model(torch.zeros(2, 4, 3, dtype=torch.long))
