"""Small multilayer perceptrons for introductory GAN experiments."""

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


class MLPGenerator(nn.Module):
    """Map latent vectors to generated feature vectors with a small MLP.

    The model intentionally places no activation after its final linear layer.
    Its outputs can therefore represent unrestricted real-valued data, such as
    the two-dimensional points in the introductory GAN notebook. Applications
    with a constrained data range may add a suitable output transformation.

    Args:
        latent_dim: Number of coordinates in every latent noise vector.
        output_dim: Number of coordinates in every generated sample.
        hidden_dim: Width of both hidden layers.

    Shape:
        - Input: ``(N, latent_dim)``
        - Output: ``(N, output_dim)``
    """

    def __init__(
        self,
        latent_dim: int = 2,
        output_dim: int = 2,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.latent_dim = _positive_integer(latent_dim, name="latent_dim")
        self.output_dim = _positive_integer(output_dim, name="output_dim")
        self.hidden_dim = _positive_integer(hidden_dim, name="hidden_dim")
        self.network = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

    def forward(self, latent_vectors: torch.Tensor) -> torch.Tensor:
        """Generate one feature vector from every latent vector."""
        if not isinstance(latent_vectors, torch.Tensor):
            raise TypeError("latent_vectors must be a PyTorch tensor")
        if (
            latent_vectors.ndim != 2
            or latent_vectors.shape[0] == 0
            or latent_vectors.shape[1] != self.latent_dim
        ):
            raise ValueError(
                "latent_vectors must have nonempty shape "
                f"(N, {self.latent_dim})"
            )
        if not latent_vectors.is_floating_point():
            raise TypeError("latent_vectors must contain floating-point values")
        return self.network(latent_vectors)


class MLPDiscriminator(nn.Module):
    """Assign one real-versus-generated logit to each feature vector.

    The returned value is a raw logit, not a probability. Keeping the sigmoid
    inside binary cross-entropy with logits gives a more numerically stable GAN
    objective.

    Args:
        input_dim: Number of coordinates in each real or generated sample.
        hidden_dim: Width of both hidden layers.

    Shape:
        - Input: ``(N, input_dim)``
        - Output: ``(N, 1)``
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 64) -> None:
        super().__init__()
        self.input_dim = _positive_integer(input_dim, name="input_dim")
        self.hidden_dim = _positive_integer(hidden_dim, name="hidden_dim")
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        """Return one realness logit for every input sample."""
        if not isinstance(samples, torch.Tensor):
            raise TypeError("samples must be a PyTorch tensor")
        if (
            samples.ndim != 2
            or samples.shape[0] == 0
            or samples.shape[1] != self.input_dim
        ):
            raise ValueError(
                f"samples must have nonempty shape (N, {self.input_dim})"
            )
        if not samples.is_floating_point():
            raise TypeError("samples must contain floating-point values")
        return self.network(samples)


class ConvolutionalGenerator(nn.Module):
    """Generate square grayscale images from latent vectors.

    Two transposed convolutions enlarge a learned low-resolution feature map
    by a factor of four. The final ``tanh`` bounds pixels to ``[-1, 1]``.

    Shape:
        - Input: ``(N, latent_dim)``
        - Output: ``(N, image_size * image_size)``
    """

    def __init__(
        self,
        latent_dim: int = 64,
        image_size: int = 16,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        self.latent_dim = _positive_integer(latent_dim, name="latent_dim")
        self.image_size = _positive_integer(image_size, name="image_size")
        self.base_channels = _positive_integer(base_channels, name="base_channels")
        if self.image_size % 4 != 0:
            raise ValueError("image_size must be divisible by four")
        self.initial_size = self.image_size // 4
        self.network = nn.Sequential(
            nn.Linear(
                self.latent_dim,
                2 * self.base_channels * self.initial_size**2,
            ),
            nn.Unflatten(
                1,
                (2 * self.base_channels, self.initial_size, self.initial_size),
            ),
            nn.BatchNorm2d(2 * self.base_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                2 * self.base_channels,
                self.base_channels,
                4,
                2,
                1,
            ),
            nn.BatchNorm2d(self.base_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.base_channels, 1, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, latent_vectors: torch.Tensor) -> torch.Tensor:
        """Return flattened generated images for a latent-vector batch."""
        if not isinstance(latent_vectors, torch.Tensor):
            raise TypeError("latent_vectors must be a PyTorch tensor")
        if (
            latent_vectors.ndim != 2
            or latent_vectors.shape[0] == 0
            or latent_vectors.shape[1] != self.latent_dim
        ):
            raise ValueError(
                "latent_vectors must have nonempty shape "
                f"(N, {self.latent_dim})"
            )
        if not latent_vectors.is_floating_point():
            raise TypeError("latent_vectors must contain floating-point values")
        return self.network(latent_vectors).flatten(start_dim=1)


class ConvolutionalDiscriminator(nn.Module):
    """Return one real-versus-generated logit for each grayscale image.

    Shape:
        - Input: ``(N, image_size * image_size)`` or
          ``(N, 1, image_size, image_size)``
        - Output: ``(N, 1)``
    """

    def __init__(self, image_size: int = 16, base_channels: int = 32) -> None:
        super().__init__()
        self.image_size = _positive_integer(image_size, name="image_size")
        self.base_channels = _positive_integer(base_channels, name="base_channels")
        if self.image_size % 4 != 0:
            raise ValueError("image_size must be divisible by four")
        final_size = self.image_size // 4
        self.network = nn.Sequential(
            nn.Conv2d(1, self.base_channels, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.base_channels, 2 * self.base_channels, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(2 * self.base_channels * final_size**2, 1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return one raw discriminator logit for every input image."""
        if not isinstance(images, torch.Tensor):
            raise TypeError("images must be a PyTorch tensor")
        flat_width = self.image_size**2
        valid_flat = (
            images.ndim == 2
            and images.shape[0] > 0
            and images.shape[1] == flat_width
        )
        valid_nchw = (
            images.ndim == 4
            and images.shape[0] > 0
            and images.shape[1:] == (1, self.image_size, self.image_size)
        )
        if not (valid_flat or valid_nchw):
            raise ValueError(
                "images must have nonempty shape "
                f"(N, {flat_width}) or "
                f"(N, 1, {self.image_size}, {self.image_size})"
            )
        if not images.is_floating_point():
            raise TypeError("images must contain floating-point values")
        images = images.reshape(-1, 1, self.image_size, self.image_size)
        return self.network(images)
