"""PyTorch helpers for coordinate-based neural fields."""

from numbers import Integral, Real

import torch
from torch import nn


def _coordinate_tensor(coordinates: torch.Tensor, *, name: str) -> torch.Tensor:
    """Validate a nonempty floating-point coordinate matrix ``(N, D)``."""
    if not isinstance(coordinates, torch.Tensor):
        raise TypeError(f"{name} must be a PyTorch tensor")
    if (
        coordinates.ndim != 2
        or coordinates.shape[0] == 0
        or coordinates.shape[1] == 0
    ):
        raise ValueError(f"{name} must have nonempty shape (N, D)")
    if not coordinates.is_floating_point():
        raise TypeError(f"{name} must contain floating-point values")
    return coordinates


def _positive_integer(value: int, *, name: str) -> int:
    """Validate a positive integer without accepting Boolean values."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def sphere_signed_distance(
    coordinates: torch.Tensor,
    center: torch.Tensor,
    radius: float,
) -> torch.Tensor:
    """Return signed distances from coordinates to a sphere.

    Negative values are inside, zero is on the surface, and positive values
    are outside. ``coordinates`` has shape ``(N, D)`` and ``center`` must have
    shape ``(D,)`` on the same device with the same dtype.
    """
    coordinates = _coordinate_tensor(coordinates, name="coordinates")
    if not isinstance(center, torch.Tensor):
        raise TypeError("center must be a PyTorch tensor")
    if center.shape != (coordinates.shape[1],):
        raise ValueError("center must have shape (D,) matching coordinates")
    if not center.is_floating_point():
        raise TypeError("center must contain floating-point values")
    if center.device != coordinates.device or center.dtype != coordinates.dtype:
        raise ValueError("center must match coordinates dtype and device")
    if isinstance(radius, bool) or not isinstance(radius, Real):
        raise TypeError("radius must be numeric")
    radius = float(radius)
    if not torch.isfinite(torch.tensor(radius)) or radius <= 0.0:
        raise ValueError("radius must be finite and positive")
    return torch.linalg.vector_norm(coordinates - center, dim=1) - radius


def union_signed_distance(*distances: torch.Tensor) -> torch.Tensor:
    """Combine identically shaped signed-distance fields using minimum.

    The minimum represents the union under the convention that negative
    values are inside. It gives the union's zero level set, although values in
    overlapping interiors need not satisfy every strict distance property.
    """
    if not distances:
        raise ValueError("at least one distance tensor is required")
    first = distances[0]
    if not isinstance(first, torch.Tensor):
        raise TypeError("distances must be PyTorch tensors")
    for distance in distances[1:]:
        if not isinstance(distance, torch.Tensor):
            raise TypeError("distances must be PyTorch tensors")
        if distance.shape != first.shape:
            raise ValueError("all distance tensors must have the same shape")
        if distance.device != first.device or distance.dtype != first.dtype:
            raise ValueError("all distance tensors must share dtype and device")
    return torch.stack(distances, dim=0).amin(dim=0)


def fourier_encode(
    coordinates: torch.Tensor,
    number_frequencies: int,
    *,
    include_input: bool = True,
) -> torch.Tensor:
    """Encode coordinates with powers-of-two sine and cosine features.

    For input shape ``(N, D)`` and ``L`` frequencies, the result has width
    ``2 * L * D`` plus ``D`` when ``include_input`` is true. Features are
    ordered as the optional raw coordinates, all frequency-major sine values,
    then all frequency-major cosine values.
    """
    coordinates = _coordinate_tensor(coordinates, name="coordinates")
    number_frequencies = _positive_integer(
        number_frequencies, name="number_frequencies"
    )
    if not isinstance(include_input, bool):
        raise TypeError("include_input must be Boolean")

    frequencies = 2.0 ** torch.arange(
        number_frequencies,
        dtype=coordinates.dtype,
        device=coordinates.device,
    )
    angles = (
        torch.pi
        * coordinates[:, None, :]
        * frequencies[None, :, None]
    )
    encoded_parts = (
        torch.sin(angles).flatten(start_dim=1),
        torch.cos(angles).flatten(start_dim=1),
    )
    if include_input:
        encoded_parts = (coordinates, *encoded_parts)
    return torch.cat(encoded_parts, dim=1)


def coordinate_gradients(
    model: nn.Module,
    coordinates: torch.Tensor,
    *,
    normalize: bool = False,
    create_graph: bool = False,
) -> torch.Tensor:
    """Differentiate independent scalar field predictions by coordinates.

    The model must map ``(N, D)`` coordinates to one value per row with shape
    ``(N,)`` or ``(N, 1)``. Summing those independent outputs lets autograd
    produce one ``D``-dimensional input gradient per row.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a PyTorch module")
    coordinates = _coordinate_tensor(coordinates, name="coordinates")
    if not isinstance(normalize, bool):
        raise TypeError("normalize must be Boolean")
    if not isinstance(create_graph, bool):
        raise TypeError("create_graph must be Boolean")

    if not coordinates.requires_grad:
        coordinates = coordinates.detach().requires_grad_(True)
    predictions = model(coordinates)
    if not isinstance(predictions, torch.Tensor):
        raise TypeError("model must return a PyTorch tensor")
    if predictions.shape == (coordinates.shape[0], 1):
        predictions = predictions[:, 0]
    if predictions.shape != (coordinates.shape[0],):
        raise ValueError("model must return one scalar per coordinate row")

    gradients = torch.autograd.grad(
        predictions.sum(),
        coordinates,
        create_graph=create_graph,
    )[0]
    if normalize:
        lengths = torch.linalg.vector_norm(gradients, dim=1, keepdim=True)
        if torch.any(lengths == 0):
            raise ValueError("cannot normalize a zero coordinate gradient")
        gradients = gradients / lengths
    return gradients
