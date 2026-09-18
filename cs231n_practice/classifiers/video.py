"""Small PyTorch classifiers for educational video experiments."""

from numbers import Integral

import torch
from torch import nn

from cs231n_practice.video import NonLocalBlock3D


def _positive_integer(value: int, name: str) -> int:
    """Validate a positive integer without accepting Boolean values."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _floating_tensor(
    value: torch.Tensor,
    name: str,
    *,
    ndim: int,
    channels: int,
) -> torch.Tensor:
    """Validate a nonempty floating-point feature tensor."""
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a PyTorch tensor")
    if value.ndim != ndim or any(size == 0 for size in value.shape):
        layout = "(N, C, T, H, W)" if ndim == 5 else "(N, C, H, W)"
        raise ValueError(f"{name} must have nonempty shape {layout}")
    if value.shape[1] != channels:
        raise ValueError(f"{name} must contain {channels} channels")
    if not value.is_floating_point():
        raise TypeError(f"{name} must contain floating-point values")
    return value


class _SpatialEncoder(nn.Module):
    """Encode one image-like tensor into a fixed-width feature vector."""

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        feature_channels: int,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, feature_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).flatten(1)


class Small3DVideoClassifier(nn.Module):
    """Classify ``(N, C, T, H, W)`` clips with a compact 3D CNN.

    The first convolution preserves time while spatially downsampling. An
    optional non-local block adds global interaction before a second 3D
    convolution, global pooling, and the linear classifier.
    """

    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 4,
        stem_channels: int = 6,
        feature_channels: int = 8,
        spatial_stride: int = 2,
        use_nonlocal: bool = False,
        attention_channels: int | None = None,
    ) -> None:
        super().__init__()
        input_channels = _positive_integer(input_channels, "input_channels")
        num_classes = _positive_integer(num_classes, "num_classes")
        stem_channels = _positive_integer(stem_channels, "stem_channels")
        feature_channels = _positive_integer(feature_channels, "feature_channels")
        spatial_stride = _positive_integer(spatial_stride, "spatial_stride")
        if not isinstance(use_nonlocal, bool):
            raise TypeError("use_nonlocal must be Boolean")
        if attention_channels is not None:
            attention_channels = _positive_integer(
                attention_channels, "attention_channels"
            )

        self.input_channels = input_channels
        self.stem = nn.Sequential(
            nn.Conv3d(
                input_channels,
                stem_channels,
                kernel_size=3,
                padding=1,
                stride=(1, spatial_stride, spatial_stride),
            ),
            nn.ReLU(),
        )
        self.context = (
            NonLocalBlock3D(stem_channels, attention_channels)
            if use_nonlocal
            else nn.Identity()
        )
        self.head = nn.Sequential(
            nn.Conv3d(stem_channels, feature_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
        )
        self.classifier = nn.Linear(feature_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return one class score vector per input clip."""
        x = _floating_tensor(
            x, "x", ndim=5, channels=self.input_channels
        )
        features = self.stem(x)
        features = self.context(features)
        features = self.head(features).flatten(1)
        return self.classifier(features)


class TwoStreamVideoClassifier(nn.Module):
    """Fuse 2D appearance and stacked-motion feature streams."""

    def __init__(
        self,
        appearance_channels: int = 3,
        motion_channels: int = 12,
        num_classes: int = 4,
        hidden_channels: int = 6,
        feature_channels: int = 8,
    ) -> None:
        super().__init__()
        appearance_channels = _positive_integer(
            appearance_channels, "appearance_channels"
        )
        motion_channels = _positive_integer(motion_channels, "motion_channels")
        num_classes = _positive_integer(num_classes, "num_classes")
        hidden_channels = _positive_integer(hidden_channels, "hidden_channels")
        feature_channels = _positive_integer(feature_channels, "feature_channels")

        self.appearance_channels = appearance_channels
        self.motion_channels = motion_channels
        self.appearance_encoder = _SpatialEncoder(
            appearance_channels, hidden_channels, feature_channels
        )
        self.motion_encoder = _SpatialEncoder(
            motion_channels, hidden_channels, feature_channels
        )
        self.classifier = nn.Linear(2 * feature_channels, num_classes)

    def forward(
        self,
        appearance: torch.Tensor,
        motion: torch.Tensor,
    ) -> torch.Tensor:
        """Return scores after concatenating appearance and motion features."""
        appearance = _floating_tensor(
            appearance,
            "appearance",
            ndim=4,
            channels=self.appearance_channels,
        )
        motion = _floating_tensor(
            motion, "motion", ndim=4, channels=self.motion_channels
        )
        if appearance.shape[0] != motion.shape[0]:
            raise ValueError("appearance and motion batch sizes must match")
        if appearance.shape[2:] != motion.shape[2:]:
            raise ValueError("appearance and motion spatial dimensions must match")

        appearance_features = self.appearance_encoder(appearance)
        motion_features = self.motion_encoder(motion)
        fused_features = torch.cat((appearance_features, motion_features), dim=1)
        return self.classifier(fused_features)
