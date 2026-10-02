"""Small image models for educational self-supervised experiments."""

from numbers import Integral

import torch
from torch import nn


def _positive_integer(value: int, name: str) -> int:
    """Validate a positive integer without accepting Boolean values."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


class SmallEncoder(nn.Module):
    """Map small NCHW images to fixed-width feature vectors.

    Two max-pooling stages reduce spatial resolution, while adaptive average
    pooling makes the final representation independent of the input height and
    width. Inputs must be at least 4 by 4 pixels.
    """

    def __init__(self, input_channels: int = 3, feature_dim: int = 64) -> None:
        super().__init__()
        input_channels = _positive_integer(input_channels, "input_channels")
        feature_dim = _positive_integer(feature_dim, "feature_dim")
        self.input_channels = input_channels
        self.feature_dim = feature_dim
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return one ``feature_dim``-wide vector for each image."""
        if not isinstance(images, torch.Tensor):
            raise TypeError("images must be a PyTorch tensor")
        if images.ndim != 4 or any(size == 0 for size in images.shape):
            raise ValueError("images must have nonempty shape (N, C, H, W)")
        if images.shape[1] != self.input_channels:
            raise ValueError(f"images must contain {self.input_channels} channels")
        if images.shape[-2] < 4 or images.shape[-1] < 4:
            raise ValueError("image height and width must be at least 4")
        if not images.is_floating_point():
            raise TypeError("images must contain floating-point values")
        return self.features(images).flatten(1)


class RotationPredictionModel(nn.Module):
    """Predict an image's applied quarter turn using a small CNN encoder."""

    def __init__(
        self,
        input_channels: int = 3,
        feature_dim: int = 64,
        num_rotations: int = 4,
    ) -> None:
        super().__init__()
        num_rotations = _positive_integer(num_rotations, "num_rotations")
        self.encoder = SmallEncoder(input_channels, feature_dim)
        self.rotation_head = nn.Linear(self.encoder.feature_dim, num_rotations)
        self.num_rotations = num_rotations

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return one rotation-score vector per input image."""
        return self.rotation_head(self.encoder(images))


class ProjectionHead(nn.Module):
    """Map encoder representations into a contrastive projection space."""

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 64,
        projection_dim: int = 32,
    ) -> None:
        super().__init__()
        input_dim = _positive_integer(input_dim, "input_dim")
        hidden_dim = _positive_integer(hidden_dim, "hidden_dim")
        projection_dim = _positive_integer(projection_dim, "projection_dim")
        self.input_dim = input_dim
        self.projection_dim = projection_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, representations: torch.Tensor) -> torch.Tensor:
        """Return projected rows with shape ``(N, projection_dim)``."""
        if not isinstance(representations, torch.Tensor):
            raise TypeError("representations must be a PyTorch tensor")
        if representations.ndim != 2 or representations.shape[1] != self.input_dim:
            raise ValueError(
                f"representations must have shape (N, {self.input_dim})"
            )
        if representations.shape[0] == 0:
            raise ValueError("representations must contain at least one row")
        if not representations.is_floating_point():
            raise TypeError("representations must contain floating-point values")
        return self.network(representations)


class SmallSimCLR(nn.Module):
    """Return encoder representations and contrastive projections for images."""

    def __init__(
        self,
        input_channels: int = 3,
        feature_dim: int = 64,
        projection_hidden_dim: int = 64,
        projection_dim: int = 32,
    ) -> None:
        super().__init__()
        self.encoder = SmallEncoder(input_channels, feature_dim)
        self.projector = ProjectionHead(
            feature_dim,
            projection_hidden_dim,
            projection_dim,
        )

    def forward(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return reusable representations ``h`` and contrastive projections ``z``."""
        representations = self.encoder(images)
        projections = self.projector(representations)
        return representations, projections


class FineTunedClassifier(nn.Module):
    """Attach a trainable linear classifier to an encoder with ``feature_dim``."""

    def __init__(self, encoder: nn.Module, num_classes: int = 10) -> None:
        super().__init__()
        if not isinstance(encoder, nn.Module):
            raise TypeError("encoder must be a torch.nn.Module")
        if not hasattr(encoder, "feature_dim"):
            raise ValueError("encoder must expose a feature_dim attribute")
        feature_dim = _positive_integer(encoder.feature_dim, "encoder.feature_dim")
        num_classes = _positive_integer(num_classes, "num_classes")
        self.encoder = encoder
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return class scores after updating both encoder and classifier."""
        return self.classifier(self.encoder(images))
