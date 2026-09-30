"""Utilities for self-supervised pretraining and representation evaluation."""

from numbers import Integral, Real

import torch
from torch import nn
import torch.nn.functional as F


def _positive_integer(value: int, name: str) -> int:
    """Validate a positive integer without accepting Boolean values."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _positive_float(value: float, name: str) -> float:
    """Validate a positive finite floating-point value."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be numeric")
    value = float(value)
    if not torch.isfinite(torch.tensor(value)) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def _embedding_matrix(embeddings: torch.Tensor) -> torch.Tensor:
    """Validate and return a nonempty floating-point ``(N, D)`` matrix."""
    if not isinstance(embeddings, torch.Tensor):
        raise TypeError("embeddings must be a PyTorch tensor")
    if embeddings.ndim != 2 or any(size == 0 for size in embeddings.shape):
        raise ValueError("embeddings must have nonempty shape (N, D)")
    if not embeddings.is_floating_point():
        raise TypeError("embeddings must contain floating-point values")
    return embeddings


def l2_normalize(
    embeddings: torch.Tensor,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """Normalize each embedding row to unit L2 length.

    A zero row remains zero because its denominator is clamped to ``epsilon``.
    """
    embeddings = _embedding_matrix(embeddings)
    epsilon = _positive_float(epsilon, "epsilon")
    norms = torch.linalg.vector_norm(embeddings, ord=2, dim=1, keepdim=True)
    return embeddings / norms.clamp_min(epsilon)


def pairwise_cosine_similarity(embeddings: torch.Tensor) -> torch.Tensor:
    """Return the ``(N, N)`` cosine-similarity matrix between embedding rows."""
    normalized = l2_normalize(embeddings)
    return normalized @ normalized.transpose(0, 1)


def positive_pair_indices(
    number_of_embeddings: int,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return positive columns for embeddings ordered as ``[view_a; view_b]``.

    ``number_of_embeddings`` must equal ``2N`` for at least two source
    examples. The result maps every view in the first half to the corresponding
    view in the second half and vice versa.
    """
    number_of_embeddings = _positive_integer(
        number_of_embeddings, "number_of_embeddings"
    )
    if number_of_embeddings < 4 or number_of_embeddings % 2 != 0:
        raise ValueError(
            "number_of_embeddings must be even and represent at least two sources"
        )
    number_of_sources = number_of_embeddings // 2
    indices = torch.arange(number_of_embeddings, device=device)
    return (indices + number_of_sources) % number_of_embeddings


def info_nce_loss(
    embeddings: torch.Tensor,
    temperature: float = 0.2,
) -> torch.Tensor:
    """Return symmetric InfoNCE for embeddings ordered as ``[view_a; view_b]``.

    Rows are L2-normalized before their pairwise similarities are calculated.
    Every embedding is an anchor once. Its other augmented view is the target,
    its self-similarity is excluded, and all remaining rows are negatives.
    """
    embeddings = _embedding_matrix(embeddings)
    temperature = _positive_float(temperature, "temperature")
    targets = positive_pair_indices(
        embeddings.shape[0],
        device=embeddings.device,
    )

    logits = pairwise_cosine_similarity(embeddings) / temperature
    self_mask = torch.eye(
        embeddings.shape[0],
        dtype=torch.bool,
        device=embeddings.device,
    )
    logits = logits.masked_fill(self_mask, -torch.inf)
    return F.cross_entropy(logits, targets)


def nt_xent_loss(
    embeddings: torch.Tensor,
    temperature: float = 0.2,
) -> torch.Tensor:
    """Return normalized temperature-scaled cross-entropy (InfoNCE)."""
    return info_nce_loss(embeddings, temperature)


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
