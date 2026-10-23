"""Coordinate-based MLPs for educational neural-field experiments."""

from numbers import Integral

import torch
from torch import nn


def _positive_integer(value: int, *, name: str) -> int:
    """Validate a positive integer without accepting Boolean values."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


class CoordinateMLP(nn.Module):
    """Map coordinate rows to scalar field values with a shared MLP.

    Args:
        coordinate_dim: Number of coordinates in every query point.
        hidden_dim: Width of each hidden layer.
        number_hidden_layers: Number of linear-plus-ReLU hidden blocks.

    Input shape is ``(N, coordinate_dim)`` and output shape is ``(N,)``. The
    model accepts any positive number of query rows; it has no fixed spatial
    grid resolution.
    """

    def __init__(
        self,
        coordinate_dim: int = 3,
        hidden_dim: int = 64,
        number_hidden_layers: int = 2,
    ) -> None:
        super().__init__()
        coordinate_dim = _positive_integer(
            coordinate_dim, name="coordinate_dim"
        )
        hidden_dim = _positive_integer(hidden_dim, name="hidden_dim")
        number_hidden_layers = _positive_integer(
            number_hidden_layers, name="number_hidden_layers"
        )
        self.coordinate_dim = coordinate_dim

        layers: list[nn.Module] = []
        input_dim = coordinate_dim
        for _ in range(number_hidden_layers):
            layers.extend((nn.Linear(input_dim, hidden_dim), nn.ReLU()))
            input_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Return one scalar field prediction per coordinate row."""
        if not isinstance(coordinates, torch.Tensor):
            raise TypeError("coordinates must be a PyTorch tensor")
        if (
            coordinates.ndim != 2
            or coordinates.shape[0] == 0
            or coordinates.shape[1] != self.coordinate_dim
        ):
            raise ValueError(
                "coordinates must have nonempty shape "
                f"(N, {self.coordinate_dim})"
            )
        if not coordinates.is_floating_point():
            raise TypeError("coordinates must contain floating-point values")
        return self.network(coordinates).squeeze(-1)
