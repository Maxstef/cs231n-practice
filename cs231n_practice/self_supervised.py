"""Utilities for self-supervised pretraining and representation evaluation."""

from collections.abc import Callable
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


def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Convert NCHW images into row-major sequences of flattened patches.

    An input with shape ``(N, C, H, W)`` becomes
    ``(N, (H/P) * (W/P), C * P * P)`` for patch size ``P``.
    """
    if not isinstance(images, torch.Tensor):
        raise TypeError("images must be a PyTorch tensor")
    if images.ndim != 4 or any(size == 0 for size in images.shape):
        raise ValueError("images must have nonempty shape (N, C, H, W)")
    patch_size = _positive_integer(patch_size, "patch_size")
    number_images, channels, height, width = images.shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("image height and width must be divisible by patch_size")

    windows = images.unfold(2, patch_size, patch_size).unfold(
        3, patch_size, patch_size
    )
    return windows.permute(0, 2, 3, 1, 4, 5).reshape(
        number_images,
        -1,
        channels * patch_size * patch_size,
    )


def unpatchify(
    patches: torch.Tensor,
    patch_size: int,
    channels: int,
    image_height: int,
    image_width: int,
) -> torch.Tensor:
    """Restore row-major flattened patches to an NCHW image batch."""
    if not isinstance(patches, torch.Tensor):
        raise TypeError("patches must be a PyTorch tensor")
    if patches.ndim != 3 or any(size == 0 for size in patches.shape):
        raise ValueError("patches must have nonempty shape (N, L, patch_dim)")
    patch_size = _positive_integer(patch_size, "patch_size")
    channels = _positive_integer(channels, "channels")
    image_height = _positive_integer(image_height, "image_height")
    image_width = _positive_integer(image_width, "image_width")
    if image_height % patch_size != 0 or image_width % patch_size != 0:
        raise ValueError("image height and width must be divisible by patch_size")

    number_images, number_patches, patch_dim = patches.shape
    grid_height = image_height // patch_size
    grid_width = image_width // patch_size
    if number_patches != grid_height * grid_width:
        raise ValueError("patch count does not match the requested image grid")
    expected_patch_dim = channels * patch_size * patch_size
    if patch_dim != expected_patch_dim:
        raise ValueError("patch_dim does not match channels * patch_size squared")

    patch_grid = patches.reshape(
        number_images,
        grid_height,
        grid_width,
        channels,
        patch_size,
        patch_size,
    )
    return patch_grid.permute(0, 3, 1, 4, 2, 5).reshape(
        number_images,
        channels,
        image_height,
        image_width,
    )


def random_mask_patches(
    patches: torch.Tensor,
    mask_ratio: float,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Randomly retain a patch subset independently for every batch item.

    Returns visible patches in shuffled order, a binary original-order mask,
    original indices of kept patches, and the inverse shuffle used to restore
    a full sequence. Mask values are zero for visible patches and one for
    masked patches.
    """
    if not isinstance(patches, torch.Tensor):
        raise TypeError("patches must be a PyTorch tensor")
    if patches.ndim != 3 or any(size == 0 for size in patches.shape):
        raise ValueError("patches must have nonempty shape (N, L, patch_dim)")
    if isinstance(mask_ratio, bool) or not isinstance(mask_ratio, Real):
        raise TypeError("mask_ratio must be numeric")
    mask_ratio = float(mask_ratio)
    if not torch.isfinite(torch.tensor(mask_ratio)) or not 0.0 < mask_ratio < 1.0:
        raise ValueError("mask_ratio must be finite and strictly between 0 and 1")
    if generator is not None and not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator or None")

    number_images, number_patches, patch_dim = patches.shape
    number_visible = max(1, int(number_patches * (1.0 - mask_ratio)))
    noise_device = generator.device if generator is not None else patches.device
    noise = torch.rand(
        number_images,
        number_patches,
        generator=generator,
        device=noise_device,
    ).to(patches.device)
    ids_shuffle = noise.argsort(dim=1)
    ids_restore = ids_shuffle.argsort(dim=1)
    ids_keep = ids_shuffle[:, :number_visible]
    visible_patches = patches.gather(
        1,
        ids_keep.unsqueeze(-1).expand(-1, -1, patch_dim),
    )

    mask = torch.ones(
        number_images,
        number_patches,
        dtype=torch.float32,
        device=patches.device,
    )
    mask[:, :number_visible] = 0.0
    mask = mask.gather(1, ids_restore)
    return visible_patches, mask, ids_keep, ids_restore


def restore_mask_tokens(
    visible_tokens: torch.Tensor,
    ids_restore: torch.Tensor,
    mask_token: torch.Tensor,
) -> torch.Tensor:
    """Insert mask tokens and restore a shuffled token sequence to spatial order."""
    if not isinstance(visible_tokens, torch.Tensor):
        raise TypeError("visible_tokens must be a PyTorch tensor")
    if visible_tokens.ndim != 3 or any(size == 0 for size in visible_tokens.shape):
        raise ValueError("visible_tokens must have nonempty shape (N, K, D)")
    if not isinstance(ids_restore, torch.Tensor):
        raise TypeError("ids_restore must be a PyTorch tensor")
    if ids_restore.ndim != 2 or ids_restore.shape[0] != visible_tokens.shape[0]:
        raise ValueError("ids_restore must have shape (N, L)")
    if ids_restore.dtype != torch.long:
        raise TypeError("ids_restore must have dtype torch.long")
    if not isinstance(mask_token, torch.Tensor):
        raise TypeError("mask_token must be a PyTorch tensor")

    number_images, number_visible, feature_dim = visible_tokens.shape
    number_patches = ids_restore.shape[1]
    if number_visible > number_patches:
        raise ValueError("visible token count cannot exceed the full patch count")
    if mask_token.shape != (1, 1, feature_dim):
        raise ValueError(f"mask_token must have shape (1, 1, {feature_dim})")
    if mask_token.device != visible_tokens.device:
        raise ValueError("mask_token and visible_tokens must be on the same device")
    if mask_token.dtype != visible_tokens.dtype:
        raise TypeError("mask_token and visible_tokens must have the same dtype")

    number_masked = number_patches - number_visible
    mask_tokens = mask_token.expand(number_images, number_masked, feature_dim)
    shuffled_tokens = torch.cat((visible_tokens, mask_tokens), dim=1)
    return shuffled_tokens.gather(
        1,
        ids_restore.to(visible_tokens.device).unsqueeze(-1).expand(
            -1, -1, feature_dim
        ),
    )


def masked_reconstruction_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Return mean squared error over only positions marked one in ``mask``."""
    if not isinstance(predictions, torch.Tensor) or not isinstance(
        targets, torch.Tensor
    ):
        raise TypeError("predictions and targets must be PyTorch tensors")
    if predictions.ndim != 3 or predictions.shape != targets.shape:
        raise ValueError("predictions and targets must share shape (N, L, D)")
    if any(size == 0 for size in predictions.shape):
        raise ValueError("predictions and targets must be nonempty")
    if not predictions.is_floating_point() or not targets.is_floating_point():
        raise TypeError("predictions and targets must contain floating-point values")
    if not isinstance(mask, torch.Tensor):
        raise TypeError("mask must be a PyTorch tensor")
    if mask.shape != predictions.shape[:2]:
        raise ValueError("mask must have shape (N, L)")
    if mask.device != predictions.device or targets.device != predictions.device:
        raise ValueError("predictions, targets, and mask must be on the same device")
    if not bool(torch.all((mask == 0) | (mask == 1))):
        raise ValueError("mask values must be binary")

    numeric_mask = mask.to(dtype=predictions.dtype)
    number_masked = numeric_mask.sum()
    if number_masked.item() == 0:
        raise ValueError("mask must select at least one position")
    squared_error = (predictions - targets).square()
    masked_error = squared_error * numeric_mask.unsqueeze(-1)
    return masked_error.sum() / (number_masked * predictions.shape[-1])


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


def make_contrastive_views(
    images: torch.Tensor,
    transform: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a stochastic transform twice and independently to every image.

    The transform receives one image without a batch axis. Its output shape may
    differ from the input shape, but it must be consistent across all calls.
    """
    if not isinstance(images, torch.Tensor):
        raise TypeError("images must be a PyTorch tensor")
    if images.ndim < 2 or images.shape[0] == 0:
        raise ValueError("images must contain a nonempty batch axis")
    if not callable(transform):
        raise TypeError("transform must be callable")

    views_a = [transform(image) for image in images]
    views_b = [transform(image) for image in images]
    outputs = (*views_a, *views_b)
    if not all(isinstance(view, torch.Tensor) for view in outputs):
        raise TypeError("transform must return a PyTorch tensor")
    reference_shape = views_a[0].shape
    if any(view.shape != reference_shape for view in outputs):
        raise ValueError("transform must return one consistent output shape")
    return torch.stack(views_a), torch.stack(views_b)


def simclr_loss(
    model: nn.Module,
    view_a: torch.Tensor,
    view_b: torch.Tensor,
    temperature: float = 0.2,
) -> torch.Tensor:
    """Return InfoNCE for two already prepared view batches.

    The model must return ``(representations, projections)``. Input
    normalization and augmentation remain the caller's responsibility.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not isinstance(view_a, torch.Tensor) or not isinstance(view_b, torch.Tensor):
        raise TypeError("view_a and view_b must be PyTorch tensors")
    if view_a.ndim < 1 or view_b.ndim < 1 or view_a.shape[0] == 0:
        raise ValueError("views must contain nonempty batch axes")
    if view_a.shape[0] != view_b.shape[0]:
        raise ValueError("view_a and view_b batch sizes must match")

    output_a = model(view_a)
    output_b = model(view_b)
    if (
        not isinstance(output_a, tuple)
        or not isinstance(output_b, tuple)
        or len(output_a) != 2
        or len(output_b) != 2
    ):
        raise TypeError("model must return (representations, projections)")
    projections_a = _embedding_matrix(output_a[1])
    projections_b = _embedding_matrix(output_b[1])
    if projections_a.shape != projections_b.shape:
        raise ValueError("both projection batches must have the same shape")
    projections = torch.cat((projections_a, projections_b), dim=0)
    return info_nce_loss(projections, temperature)


@torch.no_grad()
def contrastive_similarity_metrics(
    projections: torch.Tensor,
) -> tuple[float, float]:
    """Return mean positive and negative cosine similarities for ``[A; B]``."""
    projections = _embedding_matrix(projections)
    number_of_embeddings = projections.shape[0]
    positives = positive_pair_indices(
        number_of_embeddings,
        device=projections.device,
    )
    rows = torch.arange(number_of_embeddings, device=projections.device)
    similarities = pairwise_cosine_similarity(projections)
    excluded = torch.eye(
        number_of_embeddings,
        dtype=torch.bool,
        device=projections.device,
    )
    excluded[rows, positives] = True
    positive_similarity = similarities[rows, positives].mean()
    negative_similarity = similarities[~excluded].mean()
    return positive_similarity.item(), negative_similarity.item()


def train_linear_probe(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    evaluation_features: torch.Tensor,
    evaluation_labels: torch.Tensor,
    *,
    num_classes: int,
    epochs: int = 50,
    batch_size: int = 128,
    learning_rate: float = 5e-2,
    device: torch.device | str = "cpu",
    seed: int = 0,
) -> tuple[nn.Linear, dict[str, list[float]]]:
    """Train a linear classifier on fixed features and report epoch accuracies."""
    train_features = _embedding_matrix(train_features)
    evaluation_features = _embedding_matrix(evaluation_features)
    if train_features.shape[1] != evaluation_features.shape[1]:
        raise ValueError("train and evaluation feature dimensions must match")
    for labels, features, name in (
        (train_labels, train_features, "train_labels"),
        (evaluation_labels, evaluation_features, "evaluation_labels"),
    ):
        if not isinstance(labels, torch.Tensor):
            raise TypeError(f"{name} must be a PyTorch tensor")
        if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
            raise ValueError(f"{name} must have shape ({features.shape[0]},)")
        if labels.dtype != torch.long:
            raise TypeError(f"{name} must have dtype torch.long")

    num_classes = _positive_integer(num_classes, "num_classes")
    epochs = _positive_integer(epochs, "epochs")
    batch_size = _positive_integer(batch_size, "batch_size")
    learning_rate = _positive_float(learning_rate, "learning_rate")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError("seed must be an integer")
    device = torch.device(device)

    with torch.random.fork_rng():
        torch.manual_seed(int(seed))
        head = nn.Linear(train_features.shape[1], num_classes)
    head = head.to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=learning_rate)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_features, train_labels),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(int(seed)),
    )
    train_features_device = train_features.to(device)
    evaluation_features_device = evaluation_features.to(device)
    history = {"train_accuracy": [], "evaluation_accuracy": []}

    for _ in range(epochs):
        head.train()
        for feature_batch, label_batch in loader:
            logits = head(feature_batch.to(device))
            loss = F.cross_entropy(logits, label_batch.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        head.eval()
        with torch.no_grad():
            train_predictions = head(train_features_device).argmax(dim=1).cpu()
            evaluation_predictions = head(evaluation_features_device).argmax(dim=1).cpu()
        history["train_accuracy"].append(
            (train_predictions == train_labels.cpu()).float().mean().item()
        )
        history["evaluation_accuracy"].append(
            (evaluation_predictions == evaluation_labels.cpu()).float().mean().item()
        )
    return head.cpu(), history


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
