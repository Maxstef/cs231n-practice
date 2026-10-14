"""A small deterministic autoencoder for square grayscale images."""

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


class TinyAutoencoder(nn.Module):
    """Compress square grayscale images and reconstruct them from latent codes.

    The encoder flattens each image, applies one ReLU hidden layer, and returns
    a deterministic latent code. The decoder mirrors that path and returns raw
    reconstruction logits rather than probabilities. For binary-valued images,
    the logits can be passed directly to binary cross-entropy with logits.

    Args:
        image_size: Height and width of each square input image.
        hidden_dim: Width of the encoder and decoder hidden layers.
        latent_dim: Width of the compressed latent representation.

    Shape:
        - Input images: (N, image_size, image_size)
        - Latent codes: (N, latent_dim)
        - Reconstruction logits: (N, image_size, image_size)
    """

    def __init__(
        self,
        image_size: int = 8,
        hidden_dim: int = 32,
        latent_dim: int = 2,
    ) -> None:
        super().__init__()
        image_size = _positive_integer(image_size, name="image_size")
        hidden_dim = _positive_integer(hidden_dim, name="hidden_dim")
        latent_dim = _positive_integer(latent_dim, name="latent_dim")
        self.image_size = image_size
        self.input_dim = image_size * image_size
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.encoder_hidden = nn.Linear(self.input_dim, hidden_dim)
        self.encoder_output = nn.Linear(hidden_dim, latent_dim)
        self.decoder_hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder_output = nn.Linear(hidden_dim, self.input_dim)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Return one deterministic latent code for every image."""
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
        return self.encoder_output(hidden)

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

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images and return their reconstruction logits."""
        return self.decode(self.encode(images))


class ConvolutionalDecoder(nn.Module):
    """Decode latent vectors into logits for square color or grayscale images.

    Two transposed convolutions increase each spatial dimension by a factor of
    four. The projection establishes the initial ``base_channels * 2`` feature
    maps at one quarter of the requested output resolution.
    """

    def __init__(
        self,
        image_size: int = 16,
        output_channels: int = 3,
        base_channels: int = 16,
        latent_dim: int = 2,
    ) -> None:
        super().__init__()
        image_size = _positive_integer(image_size, name="image_size")
        output_channels = _positive_integer(
            output_channels,
            name="output_channels",
        )
        base_channels = _positive_integer(base_channels, name="base_channels")
        latent_dim = _positive_integer(latent_dim, name="latent_dim")
        if image_size % 4 != 0:
            raise ValueError("image_size must be divisible by four")

        self.image_size = image_size
        self.output_channels = output_channels
        self.base_channels = base_channels
        self.latent_dim = latent_dim
        self.feature_size = image_size // 4
        feature_dim = 2 * base_channels * self.feature_size**2
        self.projection = nn.Linear(latent_dim, feature_dim)
        self.network = nn.Sequential(
            nn.ConvTranspose2d(
                2 * base_channels,
                base_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                base_channels,
                output_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
        )

    def forward(self, latent_codes: torch.Tensor) -> torch.Tensor:
        """Return raw reconstruction logits for latent codes shaped ``(N,D)``."""
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

        hidden = F.relu(self.projection(latent_codes))
        hidden = hidden.reshape(
            latent_codes.shape[0],
            2 * self.base_channels,
            self.feature_size,
            self.feature_size,
        )
        return self.network(hidden)


class ConvolutionalAutoencoder(nn.Module):
    """Deterministic convolutional autoencoder for square images.

    The encoder downsamples twice and returns one latent vector per image. The
    decoder mirrors those spatial changes and returns raw reconstruction logits.
    """

    def __init__(
        self,
        image_size: int = 16,
        input_channels: int = 3,
        base_channels: int = 16,
        latent_dim: int = 2,
    ) -> None:
        super().__init__()
        self.image_size = _positive_integer(image_size, name="image_size")
        self.input_channels = _positive_integer(
            input_channels,
            name="input_channels",
        )
        self.base_channels = _positive_integer(base_channels, name="base_channels")
        self.latent_dim = _positive_integer(latent_dim, name="latent_dim")
        if self.image_size % 4 != 0:
            raise ValueError("image_size must be divisible by four")

        self.feature_size = self.image_size // 4
        self.feature_dim = 2 * self.base_channels * self.feature_size**2
        self.encoder_features = nn.Sequential(
            nn.Conv2d(
                self.input_channels,
                self.base_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                self.base_channels,
                2 * self.base_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
        )
        self.encoder_output = nn.Linear(self.feature_dim, self.latent_dim)
        self.decoder = ConvolutionalDecoder(
            self.image_size,
            self.input_channels,
            self.base_channels,
            self.latent_dim,
        )

    def _validate_images(self, images: torch.Tensor) -> None:
        """Validate images against the configured channel and spatial sizes."""
        if not isinstance(images, torch.Tensor):
            raise TypeError("images must be a PyTorch tensor")
        expected = (self.input_channels, self.image_size, self.image_size)
        if images.ndim != 4 or images.shape[0] == 0 or images.shape[1:] != expected:
            raise ValueError(
                "images must have nonempty shape "
                f"(N, {self.input_channels}, {self.image_size}, {self.image_size})"
            )
        if not images.is_floating_point():
            raise TypeError("images must contain floating-point values")

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Return one deterministic latent vector per image."""
        self._validate_images(images)
        features = self.encoder_features(images).flatten(start_dim=1)
        return self.encoder_output(features)

    def decode(self, latent_codes: torch.Tensor) -> torch.Tensor:
        """Return raw image logits for latent vectors."""
        return self.decoder(latent_codes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images and return their reconstruction logits."""
        return self.decode(self.encode(images))


class ConvolutionalVAE(nn.Module):
    """Convolutional VAE with a diagonal-Gaussian approximate posterior.

    ``encode`` returns posterior mean and log-variance. ``forward`` draws a
    reparameterized sample, decodes it, and returns all quantities needed by a
    standard reconstruction-plus-KL VAE objective.
    """

    def __init__(
        self,
        image_size: int = 16,
        input_channels: int = 3,
        base_channels: int = 16,
        latent_dim: int = 2,
    ) -> None:
        super().__init__()
        self.image_size = _positive_integer(image_size, name="image_size")
        self.input_channels = _positive_integer(
            input_channels,
            name="input_channels",
        )
        self.base_channels = _positive_integer(base_channels, name="base_channels")
        self.latent_dim = _positive_integer(latent_dim, name="latent_dim")
        if self.image_size % 4 != 0:
            raise ValueError("image_size must be divisible by four")

        self.feature_size = self.image_size // 4
        self.feature_dim = 2 * self.base_channels * self.feature_size**2
        self.encoder_features = nn.Sequential(
            nn.Conv2d(
                self.input_channels,
                self.base_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                self.base_channels,
                2 * self.base_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
        )
        self.mean_output = nn.Linear(self.feature_dim, self.latent_dim)
        self.log_variance_output = nn.Linear(self.feature_dim, self.latent_dim)
        self.decoder = ConvolutionalDecoder(
            self.image_size,
            self.input_channels,
            self.base_channels,
            self.latent_dim,
        )

    def _validate_images(self, images: torch.Tensor) -> None:
        """Validate images against the configured channel and spatial sizes."""
        if not isinstance(images, torch.Tensor):
            raise TypeError("images must be a PyTorch tensor")
        expected = (self.input_channels, self.image_size, self.image_size)
        if images.ndim != 4 or images.shape[0] == 0 or images.shape[1:] != expected:
            raise ValueError(
                "images must have nonempty shape "
                f"(N, {self.input_channels}, {self.image_size}, {self.image_size})"
            )
        if not images.is_floating_point():
            raise TypeError("images must contain floating-point values")

    def encode(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return posterior mean and log-variance for every input image."""
        self._validate_images(images)
        features = self.encoder_features(images).flatten(start_dim=1)
        return self.mean_output(features), self.log_variance_output(features)

    def decode(self, latent_codes: torch.Tensor) -> torch.Tensor:
        """Return raw image logits for latent vectors."""
        return self.decoder(latent_codes)

    def forward(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode, reparameterize, and decode an image batch."""
        mean, log_variance = self.encode(images)
        latent_codes = reparameterize(mean, log_variance)
        reconstruction_logits = self.decode(latent_codes)
        return reconstruction_logits, mean, log_variance, latent_codes


class VAEEncoderView(nn.Module):
    """Expose a convolutional VAE's posterior mean as deterministic features."""

    def __init__(self, vae: ConvolutionalVAE) -> None:
        super().__init__()
        if not isinstance(vae, ConvolutionalVAE):
            raise TypeError("vae must be a ConvolutionalVAE")
        self.vae = vae
        self.feature_dim = vae.latent_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return posterior means without sampling latent noise."""
        mean, _ = self.vae.encode(images)
        return mean
