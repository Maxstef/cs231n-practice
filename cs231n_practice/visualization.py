"""Reusable PyTorch attribution methods for understanding vision models."""

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def _single_image(image: torch.Tensor) -> None:
    """Validate an image batch containing exactly one example."""
    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a PyTorch tensor")
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("image must have shape (1, C, H, W)")
    if not image.is_floating_point():
        raise TypeError("image must have a floating-point dtype")


def _selected_class(logits: torch.Tensor, class_id: int | None) -> int:
    """Return a valid selected class, defaulting to the prediction."""
    if logits.ndim != 2 or logits.shape[0] != 1 or logits.shape[1] == 0:
        raise ValueError("model output must have shape (1, number_of_classes)")
    if class_id is None:
        return int(logits[0].argmax())
    if isinstance(class_id, (bool, np.bool_)) or not isinstance(
        class_id, (int, np.integer)
    ):
        raise TypeError("class_id must be an integer or None")
    class_id = int(class_id)
    if class_id < 0 or class_id >= logits.shape[1]:
        raise ValueError("class_id is outside the model's class range")
    return class_id


def _normalized_positive_map(values: torch.Tensor) -> torch.Tensor:
    """Keep positive evidence and normalize it to the interval [0, 1]."""
    values = torch.relu(values.detach())
    return values / values.max().clamp_min(1e-12)


def input_gradient_saliency(
    model: nn.Module,
    image: torch.Tensor,
    class_id: int | None = None,
) -> tuple[torch.Tensor, int]:
    """Return input-gradient saliency and the explained class ID.

    The returned ``(H, W)`` map contains the largest absolute class-score
    derivative across the input channels at each spatial position, normalized
    to ``[0, 1]``.
    """
    _single_image(image)
    was_training = model.training
    model.eval()
    try:
        saliency_input = image.detach().clone().requires_grad_(True)
        logits = model(saliency_input)
        class_id = _selected_class(logits, class_id)
        (input_gradient,) = torch.autograd.grad(
            logits[0, class_id], saliency_input
        )
        saliency = input_gradient.abs().amax(dim=1)[0]
        saliency = saliency / saliency.max().clamp_min(1e-12)
        return saliency.detach(), class_id
    finally:
        model.train(was_training)


def occlusion_sensitivity(
    model: nn.Module,
    image: torch.Tensor,
    class_id: int | None = None,
    *,
    patch_size: int = 4,
    stride: int = 2,
) -> tuple[torch.Tensor, int]:
    """Return class-score drops from sliding a mean-valued square patch.

    The output spatial axes enumerate valid patch starting positions. Positive
    values mean that covering the patch reduced the selected class score.
    """
    _single_image(image)
    for value, name in ((patch_size, "patch_size"), (stride, "stride")):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise TypeError(f"{name} must be an integer")
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if patch_size > image.shape[-2] or patch_size > image.shape[-1]:
        raise ValueError("patch_size must fit inside the image")

    rows = range(0, image.shape[-2] - patch_size + 1, stride)
    columns = range(0, image.shape[-1] - patch_size + 1, stride)
    score_drops = image.new_empty((len(rows), len(columns)))
    replacement = image.mean(dim=(2, 3), keepdim=True)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            logits = model(image)
            class_id = _selected_class(logits, class_id)
            baseline_score = logits[0, class_id]
            for output_row, row in enumerate(rows):
                for output_column, column in enumerate(columns):
                    occluded = image.detach().clone()
                    occluded[
                        :, :, row : row + patch_size, column : column + patch_size
                    ] = replacement
                    occluded_score = model(occluded)[0, class_id]
                    score_drops[output_row, output_column] = (
                        baseline_score - occluded_score
                    )
        return score_drops, class_id
    finally:
        model.train(was_training)


def class_activation_map(
    model: nn.Module,
    image: torch.Tensor,
    feature_layer: nn.Module,
    classifier: nn.Linear,
    class_id: int | None = None,
) -> tuple[torch.Tensor, int]:
    """Return CAM for a GAP-linear model and the explained class ID.

    ``feature_layer`` must output ``(1, K, Hf, Wf)`` activations, and the
    classifier must receive the ``K`` globally averaged feature channels.
    """
    _single_image(image)
    activation_cache: dict[str, torch.Tensor] = {}

    def save_activation(
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        activation_cache["value"] = output.detach()

    was_training = model.training
    model.eval()
    hook = feature_layer.register_forward_hook(save_activation)
    try:
        with torch.no_grad():
            logits = model(image)
        class_id = _selected_class(logits, class_id)
    finally:
        hook.remove()
        model.train(was_training)

    if "value" not in activation_cache:
        raise ValueError("feature_layer did not run during the model forward pass")
    activation = activation_cache["value"]
    if activation.ndim != 4 or activation.shape[0] != 1:
        raise ValueError("feature_layer must output shape (1, K, Hf, Wf)")
    if classifier.weight.ndim != 2 or (
        classifier.weight.shape[1] != activation.shape[1]
    ):
        raise ValueError("feature channels must match classifier input features")

    class_weights = classifier.weight.detach()[class_id]
    cam = (activation[0] * class_weights[:, None, None]).sum(dim=0)
    return _normalized_positive_map(cam), class_id


def grad_cam(
    model: nn.Module,
    image: torch.Tensor,
    target_layer: nn.Module,
    class_id: int | None = None,
) -> tuple[torch.Tensor, int]:
    """Return Grad-CAM for a selected feature layer and explained class ID."""
    _single_image(image)
    activation_cache: dict[str, torch.Tensor] = {}

    def save_activation(
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        activation_cache["value"] = output

    was_training = model.training
    model.eval()
    hook = target_layer.register_forward_hook(save_activation)
    try:
        logits = model(image)
        class_id = _selected_class(logits, class_id)
        if "value" not in activation_cache:
            raise ValueError("target_layer did not run during the model forward pass")
        activation = activation_cache["value"]
        if activation.ndim != 4 or activation.shape[0] != 1:
            raise ValueError("target_layer must output shape (1, K, Hf, Wf)")
        (activation_gradient,) = torch.autograd.grad(
            logits[0, class_id], activation
        )
        channel_weights = activation_gradient.mean(dim=(0, 2, 3))
        heatmap = (
            activation.detach()[0] * channel_weights[:, None, None]
        ).sum(dim=0)
        return _normalized_positive_map(heatmap), class_id
    finally:
        hook.remove()
        model.train(was_training)


def plot_attribution_overlay(
    image: torch.Tensor,
    attribution: torch.Tensor,
    *,
    title: str = "Attribution",
    class_names: Sequence[str] | None = None,
    class_id: int | None = None,
    cmap: str = "jet",
    alpha: float = 0.55,
):
    """Plot an input beside an attribution map overlaid on that input."""
    _single_image(image)
    if not isinstance(attribution, torch.Tensor) or attribution.ndim != 2:
        raise ValueError("attribution must be a two-dimensional tensor")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if class_id is not None:
        if class_names is None or class_id < 0 or class_id >= len(class_names):
            raise ValueError("class_id requires a matching entry in class_names")
        title = f"{title}: {class_names[class_id]}"

    import matplotlib.pyplot as plt

    resized = F.interpolate(
        attribution.detach()[None, None].float(),
        size=image.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )[0, 0].cpu().numpy()
    display_image = image[0].detach().cpu().permute(1, 2, 0).numpy()
    if display_image.shape[-1] == 1:
        display_image = display_image[..., 0]

    figure, axes = plt.subplots(1, 2, figsize=(7, 3))
    axes[0].imshow(display_image, cmap="gray" if image.shape[1] == 1 else None)
    axes[0].set_title("Input")
    axes[1].imshow(display_image, cmap="gray" if image.shape[1] == 1 else None)
    axes[1].imshow(resized, cmap=cmap, alpha=alpha, vmin=0, vmax=1)
    axes[1].set_title(title)
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    return figure, axes
