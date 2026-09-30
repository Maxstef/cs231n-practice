"""Utilities for self-supervised pretraining and representation evaluation."""

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


def make_rotation_batch(
    images: torch.Tensor,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a four-way rotation-prediction minibatch.

    Every NCHW image is independently rotated by zero, one, two, or three
    quarter turns. The returned integer target records the applied number of
    counter-clockwise quarter turns.

    Args:
        images: Nonempty square images with shape ``(N, C, H, W)``.
        generator: Optional random generator for reproducible target sampling.

    Returns:
        Rotated images with the same shape and dtype as ``images``, followed by
        targets with shape ``(N,)`` and dtype ``torch.long``.
    """
    if not isinstance(images, torch.Tensor):
        raise TypeError("images must be a PyTorch tensor")
    if images.ndim != 4 or any(size == 0 for size in images.shape):
        raise ValueError("images must have nonempty shape (N, C, H, W)")
    if images.shape[-2] != images.shape[-1]:
        raise ValueError("images must be square so quarter turns preserve shape")
    if generator is not None and not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator or None")

    # Generate targets on the generator's device when one is supplied. This
    # also lets a seeded CPU generator control images that are later on a GPU.
    target_device = generator.device if generator is not None else images.device
    rotation_targets = torch.randint(
        0,
        4,
        (images.shape[0],),
        generator=generator,
        device=target_device,
    ).to(images.device)

    rotated_images = images.clone()
    for quarter_turns in range(4):
        selected = rotation_targets == quarter_turns
        rotated_images[selected] = torch.rot90(
            images[selected],
            k=quarter_turns,
            dims=(-2, -1),
        )
    return rotated_images, rotation_targets


def freeze_encoder(encoder: nn.Module) -> nn.Module:
    """Disable parameter gradients, put an encoder in evaluation mode, and return it."""
    if not isinstance(encoder, nn.Module):
        raise TypeError("encoder must be a torch.nn.Module")
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    encoder.eval()
    return encoder


@torch.no_grad()
def extract_features(
    encoder: nn.Module,
    images: torch.Tensor,
    batch_size: int = 256,
) -> torch.Tensor:
    """Extract frozen features in batches and return them on the CPU.

    Images are moved to the encoder's current device. The encoder is switched
    to evaluation mode, and the returned tensor is detached from autograd.
    """
    if not isinstance(encoder, nn.Module):
        raise TypeError("encoder must be a torch.nn.Module")
    if not isinstance(images, torch.Tensor):
        raise TypeError("images must be a PyTorch tensor")
    if images.ndim < 1 or images.shape[0] == 0:
        raise ValueError("images must contain a nonempty batch axis")
    batch_size = _positive_integer(batch_size, "batch_size")

    encoder.eval()
    parameter_or_buffer = next(
        iter((*encoder.parameters(), *encoder.buffers())),
        None,
    )
    device = (
        parameter_or_buffer.device
        if parameter_or_buffer is not None
        else torch.device("cpu")
    )

    feature_batches = []
    for start in range(0, images.shape[0], batch_size):
        batch = images[start : start + batch_size].to(device)
        features = encoder(batch)
        if not isinstance(features, torch.Tensor):
            raise TypeError("encoder must return a PyTorch tensor")
        if features.ndim == 0 or features.shape[0] != batch.shape[0]:
            raise ValueError("encoder output must preserve the batch axis")
        feature_batches.append(features.detach().cpu())
    return torch.cat(feature_batches, dim=0)
