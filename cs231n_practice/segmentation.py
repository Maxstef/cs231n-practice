"""Reusable NumPy operations for semantic, instance, and panoptic segmentation."""

import numpy as np


def _class_count(value: int) -> int:
    """Return a strictly positive integer number of classes."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError("number_of_classes must be an integer")
    value = int(value)
    if value <= 0:
        raise ValueError("number_of_classes must be positive")
    return value


def _integer_array(value: np.ndarray, *, name: str) -> np.ndarray:
    """Return an array containing integer class IDs."""
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.integer) or np.issubdtype(
        array.dtype, np.bool_
    ):
        raise TypeError(f"{name} must contain integer class IDs")
    return array


def semantic_cross_entropy(
    logits: np.ndarray,
    targets: np.ndarray,
    ignore_index: int = -1,
) -> tuple[float, np.ndarray]:
    """Return mean valid-pixel cross-entropy and its logit gradient.

    Args:
        logits: Class scores with shape ``(N, C, H, W)``.
        targets: Integer class IDs with shape ``(N, H, W)``.
        ignore_index: Target value excluded from the loss and gradient.

    Returns:
        The scalar loss and ``dlogits`` with the same shape as ``logits``.
        Both are averaged over valid pixels, and ignored positions have zero
        gradient.
    """
    logits = np.asarray(logits)
    targets = _integer_array(targets, name="targets")
    if not np.issubdtype(logits.dtype, np.number) or np.issubdtype(
        logits.dtype, np.complexfloating
    ):
        raise TypeError("logits must contain real numeric values")
    if logits.ndim != 4:
        raise ValueError("logits must have shape (N, C, H, W)")
    if logits.shape[1] == 0 or targets.shape != (
        logits.shape[0],
        logits.shape[2],
        logits.shape[3],
    ):
        raise ValueError("targets must match the (N, H, W) logit axes")
    if not np.all(np.isfinite(logits)):
        raise ValueError("logits must contain finite values")
    if isinstance(ignore_index, (bool, np.bool_)) or not isinstance(
        ignore_index, (int, np.integer)
    ):
        raise TypeError("ignore_index must be an integer")

    valid = targets != ignore_index
    if not np.any(valid):
        raise ValueError("at least one target pixel must be valid")

    valid_targets = targets[valid]
    number_of_classes = logits.shape[1]
    if np.any(valid_targets < 0) or np.any(
        valid_targets >= number_of_classes
    ):
        raise ValueError("targets contain a class index outside [0, C)")

    calculation_dtype = np.result_type(logits.dtype, np.float32)
    logits_last = np.moveaxis(
        logits.astype(calculation_dtype, copy=False), 1, -1
    )
    valid_logits = logits_last[valid]
    shifted = valid_logits - np.max(valid_logits, axis=1, keepdims=True)
    exp_shifted = np.exp(shifted)
    normalizers = np.sum(exp_shifted, axis=1, keepdims=True)
    probabilities = exp_shifted / normalizers

    rows = np.arange(valid_targets.size)
    losses = np.log(normalizers[:, 0]) - shifted[rows, valid_targets]
    loss = np.mean(losses)

    dvalid_logits = probabilities
    dvalid_logits[rows, valid_targets] -= 1.0
    dvalid_logits /= valid_targets.size
    dlogits_last = np.zeros(logits_last.shape, dtype=calculation_dtype)
    dlogits_last[valid] = dvalid_logits
    dlogits = np.moveaxis(dlogits_last, -1, 1)
    return float(loss), dlogits


def segmentation_confusion_matrix(
    targets: np.ndarray,
    predictions: np.ndarray,
    number_of_classes: int,
    ignore_index: int = -1,
) -> np.ndarray:
    """Count valid pixels for every ``(target, prediction)`` class pair."""
    targets = _integer_array(targets, name="targets")
    predictions = _integer_array(predictions, name="predictions")
    number_of_classes = _class_count(number_of_classes)
    if targets.shape != predictions.shape:
        raise ValueError("targets and predictions must have the same shape")
    if targets.ndim == 0:
        raise ValueError("targets and predictions must have at least one axis")
    if isinstance(ignore_index, (bool, np.bool_)) or not isinstance(
        ignore_index, (int, np.integer)
    ):
        raise TypeError("ignore_index must be an integer")

    valid = targets != ignore_index
    valid_targets = targets[valid]
    valid_predictions = predictions[valid]
    if np.any(valid_targets < 0) or np.any(
        valid_targets >= number_of_classes
    ):
        raise ValueError("targets contain a class index outside the valid range")
    if np.any(valid_predictions < 0) or np.any(
        valid_predictions >= number_of_classes
    ):
        raise ValueError(
            "predictions contain a class index outside the valid range"
        )

    combined = valid_targets * number_of_classes + valid_predictions
    counts = np.bincount(
        combined, minlength=number_of_classes * number_of_classes
    )
    return counts.reshape(number_of_classes, number_of_classes)


def segmentation_metrics(
    confusion: np.ndarray,
) -> tuple[float, np.ndarray, float]:
    """Return pixel accuracy, per-class IoU, and mean class IoU.

    A class absent from both targets and predictions has an undefined IoU and
    receives ``nan``. Mean IoU averages only the classes with a defined union.
    """
    confusion = np.asarray(confusion)
    if not np.issubdtype(confusion.dtype, np.number) or np.issubdtype(
        confusion.dtype, np.complexfloating
    ):
        raise TypeError("confusion must contain real numeric counts")
    if confusion.ndim != 2 or confusion.shape[0] != confusion.shape[1]:
        raise ValueError("confusion must be a square matrix")
    if confusion.shape[0] == 0:
        raise ValueError("confusion must contain at least one class")
    if not np.all(np.isfinite(confusion)) or np.any(confusion < 0):
        raise ValueError("confusion must contain finite nonnegative counts")

    true_positive = np.diag(confusion)
    target_count = np.sum(confusion, axis=1)
    predicted_count = np.sum(confusion, axis=0)
    total = np.sum(target_count)
    pixel_accuracy = float(true_positive.sum() / total) if total > 0 else np.nan

    union = target_count + predicted_count - true_positive
    class_iou = np.divide(
        true_positive,
        union,
        out=np.full(union.shape, np.nan, dtype=float),
        where=union > 0,
    )
    valid_iou = class_iou[~np.isnan(class_iou)]
    mean_iou = float(np.mean(valid_iou)) if valid_iou.size else np.nan
    return pixel_accuracy, class_iou, mean_iou


def binary_mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Return pixel IoU between two same-shape Boolean instance masks.

    Two empty masks return zero because neither mask describes an object to
    match.
    """
    mask_a = np.asarray(mask_a)
    mask_b = np.asarray(mask_b)
    if not np.issubdtype(mask_a.dtype, np.bool_) or not np.issubdtype(
        mask_b.dtype, np.bool_
    ):
        raise TypeError("mask_a and mask_b must contain Boolean values")
    if mask_a.shape != mask_b.shape:
        raise ValueError("mask_a and mask_b must have the same shape")
    if mask_a.ndim == 0:
        raise ValueError("mask_a and mask_b must have at least one axis")

    intersection = np.count_nonzero(mask_a & mask_b)
    union = np.count_nonzero(mask_a | mask_b)
    return float(intersection / union) if union else 0.0


def select_class_mask_logits(
    mask_logits: np.ndarray,
    class_ids: np.ndarray,
) -> np.ndarray:
    """Select one class-specific spatial mask-logit grid for every RoI.

    Args:
        mask_logits: Mask-head output with shape ``(R, C, H, W)``.
        class_ids: Foreground-class channel for each RoI, with shape ``(R,)``.

    Returns:
        Selected mask logits with shape ``(R, H, W)``.
    """
    mask_logits = np.asarray(mask_logits)
    class_ids = _integer_array(class_ids, name="class_ids")
    if not np.issubdtype(mask_logits.dtype, np.number) or np.issubdtype(
        mask_logits.dtype, np.complexfloating
    ):
        raise TypeError("mask_logits must contain real numeric values")
    if mask_logits.ndim != 4:
        raise ValueError("mask_logits must have shape (R, C, H, W)")
    if class_ids.shape != (mask_logits.shape[0],):
        raise ValueError("class_ids must have shape (R,)")
    if mask_logits.shape[1] == 0:
        raise ValueError("mask_logits must contain at least one class")
    if not np.all(np.isfinite(mask_logits)):
        raise ValueError("mask_logits must contain finite values")
    if np.any(class_ids < 0) or np.any(class_ids >= mask_logits.shape[1]):
        raise ValueError("class_ids contain an index outside [0, C)")

    return mask_logits[np.arange(mask_logits.shape[0]), class_ids]


def compose_instance_map(
    masks: np.ndarray,
    scores: np.ndarray,
) -> np.ndarray:
    """Compose Boolean masks by letting higher-scoring instances claim first.

    The result uses zero for unassigned pixels and ``k + 1`` for pixels
    claimed by input mask ``k``. Equal scores retain input order.
    """
    masks = np.asarray(masks)
    scores = np.asarray(scores)
    if not np.issubdtype(masks.dtype, np.bool_):
        raise TypeError("masks must contain Boolean values")
    if masks.ndim != 3 or masks.shape[1] == 0 or masks.shape[2] == 0:
        raise ValueError("masks must have shape (K, H, W) with H and W positive")
    if not np.issubdtype(scores.dtype, np.number) or np.issubdtype(
        scores.dtype, np.complexfloating
    ):
        raise TypeError("scores must contain real numeric values")
    if scores.shape != (masks.shape[0],):
        raise ValueError("scores must have shape (K,)")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores must contain finite values")

    instance_map = np.zeros(masks.shape[1:], dtype=np.int64)
    for prediction_index in np.argsort(-scores, kind="stable"):
        claim = masks[prediction_index] & (instance_map == 0)
        instance_map[claim] = prediction_index + 1
    return instance_map


def _panoptic_divisor(value: int) -> int:
    """Return a strictly positive integer panoptic ID divisor."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError("divisor must be an integer")
    value = int(value)
    if value <= 0:
        raise ValueError("divisor must be positive")
    return value


def encode_panoptic_ids(
    semantic_ids: np.ndarray,
    instance_ids: np.ndarray,
    divisor: int = 1000,
) -> np.ndarray:
    """Combine nonnegative semantic and instance IDs in one segment-ID map."""
    semantic_ids = _integer_array(semantic_ids, name="semantic_ids")
    instance_ids = _integer_array(instance_ids, name="instance_ids")
    divisor = _panoptic_divisor(divisor)
    if semantic_ids.shape != instance_ids.shape:
        raise ValueError("semantic_ids and instance_ids must have the same shape")
    if np.any(semantic_ids < 0) or np.any(instance_ids < 0):
        raise ValueError("semantic_ids and instance_ids must be nonnegative")
    if np.any(instance_ids >= divisor):
        raise ValueError("every instance ID must be smaller than divisor")
    return semantic_ids * divisor + instance_ids


def decode_panoptic_ids(
    panoptic_ids: np.ndarray,
    divisor: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """Separate a combined segment-ID map into semantic and instance IDs."""
    panoptic_ids = _integer_array(panoptic_ids, name="panoptic_ids")
    divisor = _panoptic_divisor(divisor)
    if np.any(panoptic_ids < 0):
        raise ValueError("panoptic_ids must be nonnegative")
    return panoptic_ids // divisor, panoptic_ids % divisor


def _nonnegative_count(value: int, *, name: str) -> int:
    """Return a nonnegative integer count."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def panoptic_quality(
    matched_ious: np.ndarray,
    false_positives: int,
    false_negatives: int,
) -> tuple[float, float, float]:
    """Return segmentation quality, recognition quality, and panoptic quality."""
    matched_ious = np.asarray(matched_ious)
    if not np.issubdtype(matched_ious.dtype, np.number) or np.issubdtype(
        matched_ious.dtype, np.complexfloating
    ):
        raise TypeError("matched_ious must contain real numeric values")
    if matched_ious.ndim != 1:
        raise ValueError("matched_ious must be one-dimensional")
    if not np.all(np.isfinite(matched_ious)) or np.any(
        (matched_ious < 0) | (matched_ious > 1)
    ):
        raise ValueError("matched_ious must contain finite values in [0, 1]")
    false_positives = _nonnegative_count(
        false_positives, name="false_positives"
    )
    false_negatives = _nonnegative_count(
        false_negatives, name="false_negatives"
    )

    true_positives = matched_ious.size
    segmentation_quality = (
        float(np.mean(matched_ious)) if true_positives else 0.0
    )
    denominator = true_positives + 0.5 * (
        false_positives + false_negatives
    )
    recognition_quality = (
        float(true_positives / denominator) if denominator else 0.0
    )
    panoptic_quality_value = segmentation_quality * recognition_quality
    return segmentation_quality, recognition_quality, panoptic_quality_value
