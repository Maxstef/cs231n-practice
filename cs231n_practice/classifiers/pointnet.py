"""Small PointNet-style classifiers for educational point-cloud experiments."""

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


class TinyPointNet(nn.Module):
    """Classify unordered point clouds with shared features and max pooling.

    The shared MLP maps every input point independently from three coordinates
    to ``point_feature_dim`` features. Coordinate-wise max pooling over the
    point axis produces one fixed-width representation per object, so the
    result is invariant to point order and accepts any positive point count.

    Args:
        num_classes: Number of output class scores.
        point_feature_dim: Width of each point's learned feature vector.
        hidden_dim: Hidden width used by the point MLP and classifier.
    """

    def __init__(
        self,
        num_classes: int = 3,
        point_feature_dim: int = 64,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        num_classes = _positive_integer(num_classes, name="num_classes")
        point_feature_dim = _positive_integer(
            point_feature_dim, name="point_feature_dim"
        )
        hidden_dim = _positive_integer(hidden_dim, name="hidden_dim")
        self.point_feature_dim = point_feature_dim
        self.point_network = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, point_feature_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(point_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        point_clouds: torch.Tensor,
        *,
        return_pool_indices: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Return class scores and, optionally, max-pooling winner indices.

        Args:
            point_clouds: Floating-point coordinates shaped ``(N, P, 3)``.
            return_pool_indices: Also return the winning point index for every
                object and feature channel, shaped ``(N, point_feature_dim)``.
        """
        if not isinstance(point_clouds, torch.Tensor):
            raise TypeError("point_clouds must be a PyTorch tensor")
        if (
            point_clouds.ndim != 3
            or point_clouds.shape[1] == 0
            or point_clouds.shape[2] != 3
        ):
            raise ValueError("point_clouds must have nonempty shape (N, P, 3)")
        if point_clouds.shape[0] == 0:
            raise ValueError("point_clouds must contain at least one object")
        if not point_clouds.is_floating_point():
            raise TypeError("point_clouds must contain floating-point values")

        point_features = self.point_network(point_clouds)
        global_features, pool_indices = point_features.max(dim=1)
        logits = self.classifier(global_features)
        if return_pool_indices:
            return logits, pool_indices
        return logits


def point_cloud_classification_accuracy(
    model: nn.Module,
    point_clouds: torch.Tensor,
    labels: torch.Tensor,
    *,
    device: torch.device | str = "cpu",
) -> float:
    """Evaluate classification accuracy without recording gradients."""
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a PyTorch module")
    if not isinstance(point_clouds, torch.Tensor):
        raise TypeError("point_clouds must be a PyTorch tensor")
    if not isinstance(labels, torch.Tensor):
        raise TypeError("labels must be a PyTorch tensor")
    if labels.ndim != 1 or labels.shape[0] != point_clouds.shape[0]:
        raise ValueError("labels must have shape (N,) matching point_clouds")
    if labels.dtype != torch.long:
        raise TypeError("labels must have torch.long dtype")

    model.eval()
    with torch.no_grad():
        logits = model(point_clouds.to(device))
        if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
            raise ValueError("model must return class scores with shape (N, C)")
        predictions = logits.argmax(dim=1).cpu()
    return float((predictions == labels.cpu()).float().mean().item())
