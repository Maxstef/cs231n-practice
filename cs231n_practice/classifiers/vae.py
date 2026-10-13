"""A small variational autoencoder for square grayscale images."""

from numbers import Integral

import torch
from torch import nn
import torch.nn.functional as F

from cs231n_practice.generative import reparameterize


def _positive_integer(value: int, *, name: str) -> int:
    """Validate a positive integer without accepting Boolean values."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


class TinyVAE(nn.Module):
    """Model square grayscale images with a diagonal-Gaussian latent space.

    The encoder predicts a mean and log-variance for each image's approximate
    posterior. ``forward`` draws a reparameterized latent sample and decodes it
    to raw Bernoulli logits. Returning the distribution parameters alongside
    the logits lets a training loss combine reconstruction error with KL
    divergence from the standard-normal prior.

    Args:
        image_size: Height and width of each square input image.
        hidden_dim: Width of the encoder and decoder hidden layers.
        latent_dim: Number of dimensions in the latent distribution.

    Shape:
        - Input images: ``(N, image_size, image_size)``
        - Mean, log-variance, and sampled codes: ``(N, latent_dim)``
        - Reconstruction logits: ``(N, image_size, image_size)``
    """

    def __init__(
        self,
        image_size: int = 8,
        hidden_dim: int = 32,
        latent_dim: int = 2,
    ) -> None:
        super().__init__()
        self.image_size = _positive_integer(image_size, name="image_size")
        self.hidden_dim = _positive_integer(hidden_dim, name="hidden_dim")
        self.latent_dim = _positive_integer(latent_dim, name="latent_dim")
        self.input_dim = self.image_size * self.image_size

        self.encoder_hidden = nn.Linear(self.input_dim, self.hidden_dim)
        self.mean_output = nn.Linear(self.hidden_dim, self.latent_dim)
        self.log_variance_output = nn.Linear(self.hidden_dim, self.latent_dim)
        self.decoder_hidden = nn.Linear(self.latent_dim, self.hidden_dim)
        self.decoder_output = nn.Linear(self.hidden_dim, self.input_dim)

    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return approximate-posterior mean and log-variance for each image."""
        if not isinstance(images, torch.Tensor):
            raise TypeError("images must be a PyTorch tensor")
        expected_shape = (self.image_size, self.image_size)
        if (
            images.ndim != 3
            or images.shape[0] == 0
            or images.shape[1:] != expected_shape
        ):
            raise ValueError(
                f"images must have nonempty shape (N, {self.image_size}, "
                f"{self.image_size})"
            )
        if not images.is_floating_point():
            raise TypeError("images must contain floating-point values")

        flattened = images.reshape(images.shape[0], self.input_dim)
        hidden = F.relu(self.encoder_hidden(flattened))
        return self.mean_output(hidden), self.log_variance_output(hidden)

    def decode(self, latent_codes: torch.Tensor) -> torch.Tensor:
        """Map latent codes to square-image reconstruction logits."""
        if not isinstance(latent_codes, torch.Tensor):
            raise TypeError("latent_codes must be a PyTorch tensor")
        if (
            latent_codes.ndim != 2
            or latent_codes.shape[0] == 0
            or latent_codes.shape[1] != self.latent_dim
        ):
            raise ValueError(
                f"latent_codes must have nonempty shape (N, {self.latent_dim})"
            )
        if not latent_codes.is_floating_point():
            raise TypeError("latent_codes must contain floating-point values")

        hidden = F.relu(self.decoder_hidden(latent_codes))
        logits = self.decoder_output(hidden)
        return logits.reshape(
            latent_codes.shape[0],
            self.image_size,
            self.image_size,
        )

    def forward(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode, reparameterize, and return all values needed for training."""
        mean, log_variance = self.encode(images)
        latent_codes = reparameterize(mean, log_variance)
        reconstruction_logits = self.decode(latent_codes)
        return reconstruction_logits, mean, log_variance, latent_codes
