"""Reusable helpers for image–text embedding alignment and retrieval."""

from numbers import Integral, Real

import torch
import torch.nn.functional as F


def _embedding_batch(value: torch.Tensor, *, name: str) -> torch.Tensor:
    """Validate a nonempty floating-point embedding matrix ``(N, D)``."""
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a PyTorch tensor")
    if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] == 0:
        raise ValueError(f"{name} must have nonempty shape (N, D)")
    if not value.is_floating_point():
        raise TypeError(f"{name} must contain floating-point values")
    return value


def _positive_integer(value: int, *, name: str) -> int:
    """Validate a positive integer without accepting Boolean values."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def image_text_similarity_matrix(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
) -> torch.Tensor:
    """Return all pairwise cosine similarities between two modalities.

    Args:
        image_embeddings: Image representations shaped ``(N_image, D)``.
        text_embeddings: Text representations shaped ``(N_text, D)``.

    Returns:
        Matrix shaped ``(N_image, N_text)``. Row ``i`` compares image ``i``
        with every text; column ``j`` compares text ``j`` with every image.
    """
    image_embeddings = _embedding_batch(
        image_embeddings, name="image_embeddings"
    )
    text_embeddings = _embedding_batch(
        text_embeddings, name="text_embeddings"
    )
    if image_embeddings.shape[1] != text_embeddings.shape[1]:
        raise ValueError("image and text embedding widths must match")
    if image_embeddings.device != text_embeddings.device:
        raise ValueError("image and text embeddings must share a device")
    if image_embeddings.dtype != text_embeddings.dtype:
        raise ValueError("image and text embeddings must share a dtype")

    normalized_images = F.normalize(image_embeddings, dim=1)
    normalized_texts = F.normalize(text_embeddings, dim=1)
    return normalized_images @ normalized_texts.T


def clip_contrastive_loss(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    temperature: float = 0.07,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return symmetric CLIP loss and normalized similarity matrix.

    Rows with matching indices are treated as positive image–text pairs. The
    function therefore requires the same positive batch size for both
    modalities. Image-to-text cross-entropy is calculated over matrix rows;
    text-to-image cross-entropy is calculated over its transpose.

    Args:
        image_embeddings: Floating-point matrix shaped ``(N, D)``.
        text_embeddings: Floating-point matrix shaped ``(N, D)``.
        temperature: Positive finite divisor controlling logit sharpness.

    Returns:
        A scalar mean of both directional losses and the unscaled cosine-
        similarity matrix shaped ``(N, N)``.
    """
    if isinstance(temperature, bool) or not isinstance(temperature, Real):
        raise TypeError("temperature must be numeric")
    temperature = float(temperature)
    if not torch.isfinite(torch.tensor(temperature)) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")

    similarities = image_text_similarity_matrix(
        image_embeddings, text_embeddings
    )
    if similarities.shape[0] != similarities.shape[1]:
        raise ValueError(
            "image and text batches must have the same size for paired loss"
        )

    targets = torch.arange(similarities.shape[0], device=similarities.device)
    logits = similarities / temperature
    image_to_text_loss = F.cross_entropy(logits, targets)
    text_to_image_loss = F.cross_entropy(logits.T, targets)
    loss = 0.5 * (image_to_text_loss + text_to_image_loss)
    return loss, similarities


def retrieve_top_k(
    similarity_matrix: torch.Tensor,
    query_index: int,
    k: int,
    *,
    direction: str,
) -> torch.Tensor:
    """Return descending top-k candidate indices for one cross-modal query.

    ``direction="image_to_text"`` selects a row; ``"text_to_image"`` selects
    a column. The result is a one-dimensional integer tensor on the same
    device as ``similarity_matrix``.
    """
    if not isinstance(similarity_matrix, torch.Tensor):
        raise TypeError("similarity_matrix must be a PyTorch tensor")
    if (
        similarity_matrix.ndim != 2
        or similarity_matrix.shape[0] == 0
        or similarity_matrix.shape[1] == 0
    ):
        raise ValueError("similarity_matrix must have nonempty shape (I, T)")
    if not similarity_matrix.is_floating_point():
        raise TypeError("similarity_matrix must contain floating-point values")
    if isinstance(query_index, bool) or not isinstance(query_index, Integral):
        raise TypeError("query_index must be an integer")
    query_index = int(query_index)
    k = _positive_integer(k, name="k")

    if direction == "image_to_text":
        if not 0 <= query_index < similarity_matrix.shape[0]:
            raise IndexError("image query_index is out of range")
        scores = similarity_matrix[query_index]
    elif direction == "text_to_image":
        if not 0 <= query_index < similarity_matrix.shape[1]:
            raise IndexError("text query_index is out of range")
        scores = similarity_matrix[:, query_index]
    else:
        raise ValueError(
            "direction must be 'image_to_text' or 'text_to_image'"
        )
    if k > scores.numel():
        raise ValueError("k cannot exceed the number of retrieval candidates")
    return torch.topk(scores, k=k, largest=True, sorted=True).indices


def prompt_ensemble_prototypes(
    prompt_embeddings: torch.Tensor,
) -> torch.Tensor:
    """Combine several text-prompt embeddings into one prototype per class.

    Each prompt is normalized before averaging so that its magnitude does not
    determine its influence. The average is normalized again because averaging
    unit vectors does not generally produce another unit vector.

    Args:
        prompt_embeddings: Floating-point tensor shaped
            (number_classes, number_prompts, embedding_width).

    Returns:
        Unit-length class prototypes shaped
        (number_classes, embedding_width).
    """
    if not isinstance(prompt_embeddings, torch.Tensor):
        raise TypeError("prompt_embeddings must be a PyTorch tensor")
    if (
        prompt_embeddings.ndim != 3
        or any(size == 0 for size in prompt_embeddings.shape)
    ):
        raise ValueError(
            "prompt_embeddings must have nonempty shape "
            "(number_classes, number_prompts, embedding_width)"
        )
    if not prompt_embeddings.is_floating_point():
        raise TypeError("prompt_embeddings must contain floating-point values")

    normalized_prompts = F.normalize(prompt_embeddings, dim=2)
    mean_prompts = normalized_prompts.mean(dim=1)
    return F.normalize(mean_prompts, dim=1)


def zero_shot_classify(
    image_embeddings: torch.Tensor,
    class_prototypes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Classify images by cosine similarity to text-derived prototypes.

    No image-specific classifier is fitted: each prototype acts like one
    class-weight vector produced by a text encoder.

    Args:
        image_embeddings: Image representations shaped (N, D).
        class_prototypes: Text-derived class representations shaped (C, D).

    Returns:
        A pair containing cosine-similarity scores shaped (N, C) and the
        highest-scoring class index for each image, shaped (N,).
    """
    scores = image_text_similarity_matrix(
        image_embeddings, class_prototypes
    )
    predictions = scores.argmax(dim=1)
    return scores, predictions

