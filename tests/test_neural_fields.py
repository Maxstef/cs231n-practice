import pytest
import torch
from torch import nn

from cs231n_practice.neural_fields import (
    coordinate_gradients,
    fourier_encode,
    sphere_signed_distance,
    union_signed_distance,
)


def test_sphere_signed_distance_has_expected_signs_and_values() -> None:
    coordinates = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    )

    distances = sphere_signed_distance(coordinates, torch.zeros(3), 1.0)

    torch.testing.assert_close(distances, torch.tensor([-1.0, 0.0, 1.0]))


def test_union_signed_distance_uses_pointwise_minimum() -> None:
    first = torch.tensor([-1.0, 0.3, 2.0])
    second = torch.tensor([0.4, -0.5, 1.0])

    union = union_signed_distance(first, second)

    torch.testing.assert_close(union, torch.tensor([-1.0, -0.5, 1.0]))


def test_fourier_encoding_has_expected_shape_order_and_values() -> None:
    coordinates = torch.tensor([[0.0, 0.5]])

    encoded = fourier_encode(coordinates, 2)

    assert encoded.shape == (1, 10)
    torch.testing.assert_close(encoded[:, :2], coordinates)
    expected_sines = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    expected_cosines = torch.tensor([[1.0, 0.0, 1.0, -1.0]])
    torch.testing.assert_close(encoded[:, 2:6], expected_sines, atol=1e-6, rtol=0)
    torch.testing.assert_close(encoded[:, 6:], expected_cosines, atol=1e-6, rtol=0)


def test_fourier_encoding_can_omit_raw_coordinates() -> None:
    coordinates = torch.randn(4, 3)

    encoded = fourier_encode(coordinates, 5, include_input=False)

    assert encoded.shape == (4, 2 * 5 * 3)


def test_coordinate_gradients_match_analytic_sphere_normals() -> None:
    class UnitSphereField(nn.Module):
        def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
            return torch.linalg.vector_norm(coordinates, dim=1) - 1.0

    coordinates = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 2.0]]
    )

    gradients = coordinate_gradients(
        UnitSphereField(), coordinates, normalize=True
    )

    torch.testing.assert_close(
        gradients,
        torch.tensor([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]),
    )


def test_neural_field_helpers_reject_invalid_inputs() -> None:
    class ConstantField(nn.Module):
        def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
            return (coordinates * 0.0).sum(dim=1)

    with pytest.raises(ValueError, match="shape"):
        sphere_signed_distance(torch.ones(2, 3), torch.ones(2), 1.0)
    with pytest.raises(ValueError, match="same shape"):
        union_signed_distance(torch.ones(2), torch.ones(3))
    with pytest.raises(TypeError, match="integer"):
        fourier_encode(torch.ones(2, 3), True)
    with pytest.raises(ValueError, match="zero"):
        coordinate_gradients(ConstantField(), torch.ones(2, 3), normalize=True)
