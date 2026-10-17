"""Time-conditioned multilayer perceptron for vector-valued flow fields."""

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


class VelocityMLP(nn.Module):
    """Predict a vector field from positions and continuous time.

    Each input position is concatenated with one scalar time before entering
    the network. The output has the same width as the position, so it can be
    interpreted as the instantaneous velocity ``dx / dt``.

    Args:
        input_dim: Number of coordinates in each position and velocity.
        hidden_dim: Width of the three hidden layers.

    Shape:
        - Positions: ``(N, input_dim)``
        - Times: ``(N, 1)``
        - Returned velocities: ``(N, input_dim)``
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 128) -> None:
        super().__init__()
        self.input_dim = _positive_integer(input_dim, name="input_dim")
        self.hidden_dim = _positive_integer(hidden_dim, name="hidden_dim")
        self.network = nn.Sequential(
            nn.Linear(self.input_dim + 1, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.input_dim),
        )

    def forward(
        self,
        positions: torch.Tensor,
        times: torch.Tensor,
    ) -> torch.Tensor:
        """Return the predicted velocity for every position and time."""
        if not isinstance(positions, torch.Tensor):
            raise TypeError("positions must be a PyTorch tensor")
        if (
            positions.ndim != 2
            or positions.shape[0] == 0
            or positions.shape[1] != self.input_dim
        ):
            raise ValueError(
                f"positions must have nonempty shape (N, {self.input_dim})"
            )
        if not positions.is_floating_point():
            raise TypeError("positions must contain floating-point values")
        if not isinstance(times, torch.Tensor):
            raise TypeError("times must be a PyTorch tensor")
        if times.shape != (positions.shape[0], 1):
            raise ValueError("times must have shape (N, 1)")
        if not times.is_floating_point():
            raise TypeError("times must contain floating-point values")
        if times.device != positions.device or times.dtype != positions.dtype:
            raise ValueError("positions and times must share dtype and device")
        return self.network(torch.cat((positions, times), dim=1))
