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
