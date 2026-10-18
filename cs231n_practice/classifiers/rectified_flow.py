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


class ConditionalVelocityMLP(nn.Module):
    """Predict a time-dependent vector field conditioned on discrete labels.

    The embedding table contains one row for every real condition plus one
    final row reserved for a null condition. Training the null row through
    condition dropout allows the same network to make both conditional and
    unconditional predictions for classifier-free guidance.

    Args:
        input_dim: Number of coordinates in each position and velocity.
        number_conditions: Number of real condition IDs. Valid IDs are
            ``0`` through ``number_conditions - 1``.
        condition_dim: Width of each learned condition embedding.
        hidden_dim: Width of the three hidden layers.

    Attributes:
        null_condition: Integer ID of the additional null embedding.

    Shape:
        - Positions: ``(N, input_dim)``
        - Times: ``(N, 1)``
        - Conditions: ``(N,)``
        - Returned velocities: ``(N, input_dim)``
    """

    def __init__(
        self,
        input_dim: int = 2,
        number_conditions: int = 8,
        condition_dim: int = 16,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.input_dim = _positive_integer(input_dim, name="input_dim")
        self.number_conditions = _positive_integer(
            number_conditions,
            name="number_conditions",
        )
        self.condition_dim = _positive_integer(
            condition_dim,
            name="condition_dim",
        )
        self.hidden_dim = _positive_integer(hidden_dim, name="hidden_dim")
        self.null_condition = self.number_conditions
        self.condition_embedding = nn.Embedding(
            self.number_conditions + 1,
            self.condition_dim,
        )
        self.network = nn.Sequential(
            nn.Linear(
                self.input_dim + 1 + self.condition_dim,
                self.hidden_dim,
            ),
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
        conditions: torch.Tensor,
    ) -> torch.Tensor:
        """Return velocities for positions, times, and condition IDs."""
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
        if not isinstance(conditions, torch.Tensor):
            raise TypeError("conditions must be a PyTorch tensor")
        if conditions.ndim != 1 or conditions.shape[0] != positions.shape[0]:
            raise ValueError("conditions must have shape (N,)")
        if conditions.dtype not in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            raise TypeError("conditions must contain integer IDs")
        if conditions.device != positions.device:
            raise ValueError("positions and conditions must share a device")
        if torch.any(conditions < 0) or torch.any(
            conditions > self.null_condition
        ):
            raise ValueError("conditions contain an index outside the embedding")

        condition_features = self.condition_embedding(conditions)
        model_inputs = torch.cat(
            (positions, times, condition_features),
            dim=1,
        )
        return self.network(model_inputs)


class SmallImageVelocityNet(nn.Module):
    """Predict rectified-flow velocities for small grayscale images.

    The compact U-Net-like structure downsamples once, injects continuous
    time into the compressed features, and upsamples to one velocity per
    pixel. A skip connection preserves fine spatial information.

    Shape:
        - Images: ``(N, image_size * image_size)``
        - Times: ``(N,)`` or ``(N, 1)``
        - Returned velocities: ``(N, image_size * image_size)``
    """

    def __init__(self, image_size: int = 16, base_channels: int = 32) -> None:
        super().__init__()
        self.image_size = _positive_integer(image_size, name="image_size")
        self.base_channels = _positive_integer(base_channels, name="base_channels")
        if self.image_size % 2 != 0:
            raise ValueError("image_size must be even")
        self.input_block = nn.Sequential(
            nn.Conv2d(1, self.base_channels, 3, padding=1),
            nn.SiLU(),
        )
        self.down = nn.Sequential(
            nn.Conv2d(self.base_channels, 2 * self.base_channels, 4, 2, 1),
            nn.SiLU(),
        )
        self.time_projection = nn.Sequential(
            nn.Linear(1, 2 * self.base_channels),
            nn.SiLU(),
            nn.Linear(2 * self.base_channels, 2 * self.base_channels),
        )
        self.middle = nn.Sequential(
            nn.Conv2d(
                2 * self.base_channels,
                2 * self.base_channels,
                3,
                padding=1,
            ),
            nn.SiLU(),
            nn.Conv2d(
                2 * self.base_channels,
                2 * self.base_channels,
                3,
                padding=1,
            ),
            nn.SiLU(),
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(
                2 * self.base_channels,
                self.base_channels,
                4,
                2,
                1,
            ),
            nn.SiLU(),
        )
        self.output = nn.Conv2d(self.base_channels, 1, 3, padding=1)

    def forward(
        self,
        noisy_images: torch.Tensor,
        times: torch.Tensor,
    ) -> torch.Tensor:
        """Return one predicted velocity value for every input pixel."""
        if not isinstance(noisy_images, torch.Tensor):
            raise TypeError("noisy_images must be a PyTorch tensor")
        expected_width = self.image_size**2
        if (
            noisy_images.ndim != 2
            or noisy_images.shape[0] == 0
            or noisy_images.shape[1] != expected_width
        ):
            raise ValueError(
                "noisy_images must have nonempty shape "
                f"(N, {expected_width})"
            )
        if not noisy_images.is_floating_point():
            raise TypeError("noisy_images must contain floating-point values")
        if not isinstance(times, torch.Tensor):
            raise TypeError("times must be a PyTorch tensor")
        if times.ndim == 1:
            times = times[:, None]
        if times.shape != (noisy_images.shape[0], 1):
            raise ValueError("times must have shape (N,) or (N, 1)")
        if not times.is_floating_point():
            raise TypeError("times must contain floating-point values")
        if times.device != noisy_images.device or times.dtype != noisy_images.dtype:
            raise ValueError("noisy_images and times must share dtype and device")

        images = noisy_images.reshape(-1, 1, self.image_size, self.image_size)
        skip = self.input_block(images)
        hidden = self.down(skip)
        time_features = self.time_projection(times).reshape(
            -1,
            hidden.shape[1],
            1,
            1,
        )
        hidden = self.middle(hidden + time_features)
        hidden = self.up(hidden) + skip
        return self.output(hidden).flatten(start_dim=1)
